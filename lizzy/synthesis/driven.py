"""Numerical time-dependent synthesis in small Pauli Lie algebras.

This implements numerical Wei–Norman coordinates, not a new decomposition theorem:
Qvarfort & Pikovski, PRX Quantum 6, 010201 (2025). Unlike the static router, this
opt-in route has no certified global error budget. No Hilbert-space matrices or
symbolic equation engine are needed: Pauli adjoint actions are planar rotations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.integrate import solve_ivp

from lizzy.hamiltonian import Circuit, fold_phases, terms_of, weight
from lizzy.synthesis._numerical import (
    _positive_integer,
    _time_span,
    _validate_chart_controls,
)


def _real_vector(values, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if values.shape != (size,):
        raise ValueError(f"coefficients must have shape ({size},)")
    if not np.all(np.isfinite(values)) or np.any(values.imag != 0):
        raise ValueError("coefficients must be finite and real (Hermitian Hamiltonian)")
    return values.real.copy()


def _pauli_words(paulis: Sequence[str | PauliString]) -> tuple[str, ...]:
    words = tuple(str(p) for p in paulis)
    if not words or any(not w or set(w) - set("IXYZ") for w in words):
        raise ValueError("supply nonempty Pauli words using only I, X, Y, Z")
    if len({len(w) for w in words}) != 1:
        raise ValueError("all Pauli words must have the same width")
    if len(set(words)) != len(words):
        raise ValueError("Pauli words must be distinct; combine duplicate controls")
    return words


@dataclass(frozen=True)
class DrivenHamiltonian:
    """Fixed, distinct Pauli words with real coefficients ``coefficients(t)``.

    The callback must return one coefficient per word in the supplied order, at
    absolute time ``t``. Words must have the same nonzero width. Include *all*
    potentially active controls, even if their coefficients initially vanish.
    Identity words are supported and retain global phase. Smooth controls are
    expected; split discontinuous pulses into separate synthesis calls.
    """

    paulis: Sequence[str | PauliString]
    coefficients: Callable[[float], Sequence[float]]

    def __post_init__(self) -> None:
        words = _pauli_words(self.paulis)
        if not callable(self.coefficients):
            raise TypeError("coefficients must be a callable of absolute time")
        # Keep immutable words, not mutable user-owned PauLie objects.
        object.__setattr__(self, "paulis", words)

    @property
    def n_qubits(self) -> int:
        return len(self.paulis[0])

    def at(self, time: float) -> np.ndarray:
        """Evaluate and validate the real coefficient vector."""
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        return _real_vector(self.coefficients(float(time)), len(self.paulis))

    @classmethod
    def from_static(cls, hamiltonian: PauliStringLinear) -> "DrivenHamiltonian":
        """Embed a static Hamiltonian for experiments, without changing its router."""
        terms = terms_of(hamiltonian)
        coefficients = _real_vector([c for c, _ in terms], len(terms))
        return cls([p for _, p in terms], lambda t: coefficients.copy())


class AlgebraTooLarge(ValueError):
    """The explicit Pauli closure exceeded the requested dimension budget."""


class IntegrationFailure(RuntimeError):
    """Numerical integration or its work/chart budget failed; no circuit returned."""


@dataclass
class DrivenResult:
    """A numerical circuit and diagnostics, not a global accuracy certificate.

    ``basis`` is the full Pauli closure, including a central identity if supplied.
    ``intervals`` records accepted chart intervals; later circuits are appended as
    left increments. Restarts can make circuit length grow with evolution time.
    ``rhs_evaluations`` includes rejected attempts. ``max_observed_condition`` is
    sampled at ODE evaluations and accepted mesh states, including rejected
    attempts; it is not a rigorous bound over continuous time.
    """

    circuit: Circuit
    basis: tuple[str, ...]
    intervals: tuple[tuple[float, float], ...]
    rhs_evaluations: int
    rejected_intervals: int
    max_observed_condition: float
    error_guaranteed: bool = field(default=False, init=False)

    @property
    def dimension(self) -> int:
        return len(self.basis)

    @property
    def chart_restarts(self) -> int:
        return max(0, len(self.intervals) - 1)


def _closure(words: Sequence[str], limit: int) -> list[PauliString]:
    """Budgeted closure using PauLie products; stop *before* building large arrays."""
    if len(words) > limit:
        raise AlgebraTooLarge(f"Pauli closure exceeds max_dimension={limit}")
    basis = [get_pauli_string(w) for w in words]
    known = set(words)
    current = 0
    while current < len(basis):
        for previous in range(current):
            left, right = basis[previous], basis[current]
            if left.commutes_with(right):
                continue
            product = left @ right
            word = str(product)
            if word not in known:
                if len(basis) == limit:
                    raise AlgebraTooLarge(f"Pauli closure exceeds max_dimension={limit}")
                known.add(word)
                basis.append(product)
        current += 1
    # A deterministic, low-weight-first order, not an optimal-order claim.
    return sorted(basis, key=lambda p: (weight(p), str(p)))


def _adjoint_pairs(basis: Sequence[PauliString]) -> list[tuple[np.ndarray, ...]]:
    """Disjoint planes for Ad(exp(-i theta P)); P Q = i sign R."""
    positions = {str(p): j for j, p in enumerate(basis)}
    pairs = []
    for p in basis:
        first, second, signs = [], [], []
        for j, q in enumerate(basis):
            if not p.commutes_with(q):
                k = positions[str(p @ q)]
                if j < k:
                    first.append(j)
                    second.append(k)
                    signs.append(float(p.sign(q).imag))
        pairs.append((np.array(first, dtype=int), np.array(second, dtype=int),
                      np.array(signs)))
    return pairs


def _coordinate_matrix(
    theta: np.ndarray, pairs: Sequence[tuple[np.ndarray, ...]],
) -> np.ndarray:
    """Real Jacobian M for U = product_j exp(-i theta_j P_j), M theta_dot = h.

    Column j expands prefix_j P_j prefix_j^dagger in the Hermitian Pauli
    basis. Differentiating U and cancelling -i in U_dot = -i H U gives this
    convention directly; M(0) = I. See docs/methods.md for the
    correspondence with the paper's differently normalized coefficients.
    """
    prefix = np.eye(len(theta))
    matrix = np.empty_like(prefix)
    for column, (angle, (first, second, signs)) in enumerate(zip(theta, pairs)):
        matrix[:, column] = prefix[:, column]
        if not len(first):
            continue
        # Multiply the prefix on the right by this adjoint rotation. Exact trig,
        # not a truncated BCH series: finite closure does not imply nilpotence.
        left, right = prefix[:, first].copy(), prefix[:, second].copy()
        cosine, sine = np.cos(2 * angle), signs * np.sin(2 * angle)
        prefix[:, first] = cosine * left + sine * right
        prefix[:, second] = cosine * right - sine * left
    return matrix


@dataclass(frozen=True, init=False)
class WeiNormanBasis:
    """Reusable ordered Pauli closure and adjoint planes, without solver state.

    Construct from all declared controls, never sampled coefficient values.
    ``basis_order`` may permute the complete closure. The frozen payload owns
    immutable words and bytes-backed arrays; numerical charts, controls and work
    counters belong to each integration, not to this reusable preparation.
    """

    controls: tuple[str, ...]
    words: tuple[str, ...]
    _pairs: tuple[tuple[np.ndarray, ...], ...] = field(repr=False, compare=False)
    _positions: np.ndarray = field(repr=False, compare=False)
    _noncentral: np.ndarray = field(repr=False, compare=False)

    def __init__(
        self, paulis: Sequence[str | PauliString], *, max_dimension: int = 32,
        basis_order: Sequence[str] | None = None,
    ) -> None:
        controls = _pauli_words(paulis)
        limit = _positive_integer(max_dimension, "max_dimension")
        words = tuple(str(p) for p in _closure(controls, limit))
        if basis_order is not None:
            order = tuple(str(p) for p in basis_order)
            if len(order) != len(words) or set(order) != set(words):
                raise ValueError("basis_order must be a permutation of the full Pauli closure")
            words = order
        self._initialize(controls, words)

    @classmethod
    def _from_closure(cls, controls: tuple[str, ...], words: tuple[str, ...]):
        """Use the wrapper's already budget-checked and ordered component closure."""
        prepared = object.__new__(cls)
        prepared._initialize(controls, words)
        return prepared

    def _initialize(self, controls: tuple[str, ...], words: tuple[str, ...]) -> None:
        def immutable(array):
            # A merely write-protected owning ndarray can be made writable again.
            return np.frombuffer(array.tobytes(), dtype=array.dtype)

        pairs = tuple(tuple(immutable(array) for array in pair)
                      for pair in _adjoint_pairs([get_pauli_string(w) for w in words]))
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_positions", immutable(
            np.array([words.index(word) for word in controls], dtype=int)))
        object.__setattr__(self, "_noncentral", immutable(
            np.array([bool(len(first)) for first, _, _ in pairs])))

    @property
    def dimension(self) -> int:
        return len(self.words)

    def jacobian(self, angles: Sequence[float]) -> np.ndarray:
        """Wei–Norman coordinate matrix in this preparation's fixed basis order."""
        theta = np.asarray(angles, dtype=float)
        if theta.shape != (self.dimension,):
            raise ValueError(f"angles must have shape ({self.dimension},)")
        return _coordinate_matrix(theta, self._pairs)


class _ChartLimit(Exception):
    pass


def synthesize_driven(
    hamiltonian: DrivenHamiltonian,
    time_span: tuple[float, float],
    *,
    max_dimension: int = 32,
    basis_order: Sequence[str] | None = None,
    prepared_basis: WeiNormanBasis | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = np.inf,
    chart_radius: float | None = None,
    condition_limit: float = 100.0,
    max_segments: int = 1024,
    max_rhs_evaluations: int = 100_000,
) -> DrivenResult:
    """Integrate Wei–Norman angles and emit an ordinary Lizzy Pauli circuit.

    This is an explicit numerical route, not automatic cost-based routing.
    All declared controls generate the closure; no coefficients are sampled to
    decide structural eligibility. ``basis_order`` may permute the entire closure.
    ``prepared_basis`` reuses that structural work for the same ordered controls;
    its dimension must fit ``max_dimension`` and any ``basis_order`` must match it.
    Every invocation still starts with fresh coordinates and work counters.

    By default, try one compact product for the full interval. A trial interval
    is bisected when sampled coordinates/Jacobians become nonfinite or the
    Jacobian exceeds ``condition_limit``. Accepted increments are appended in
    application order; restarts can still increase circuit length. Set
    ``chart_radius=0.5`` to retain the earlier conservative L1 noncentral-angle
    cap (smaller positive radii are also supported). ``None`` removes only that
    radius cap, not the condition/work guards.

    Conditioning is checked at RHS evaluations and accepted ODE mesh states.
    These samples can miss intervening singularities: this is not a certified
    atlas or a globally valid fixed-product parameterization. Closure, segment,
    and global RHS limits are enforced; failure returns no partial circuit.

    ``rtol``/``atol`` are local coordinate tolerances, NOT a bound on final operator
    error. Set ``max_step`` to resolve the fastest control timescale; an adaptive
    solver can miss narrow or aliased pulses. Both time directions are supported.
    No Magnus/Fer approximation or symbolic Symdyn engine is included here.
    """
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    max_segments = _positive_integer(max_segments, "max_segments")
    max_rhs_evaluations = _positive_integer(max_rhs_evaluations, "max_rhs_evaluations")
    start, end = _time_span(time_span)
    _validate_chart_controls(rtol=rtol, atol=atol, max_step=max_step,
                             chart_radius=chart_radius, condition_limit=condition_limit)

    if prepared_basis is None:
        prepared_basis = WeiNormanBasis(
            hamiltonian.paulis, max_dimension=max_dimension, basis_order=basis_order,
        )
    elif not isinstance(prepared_basis, WeiNormanBasis):
        raise TypeError("prepared_basis must be a WeiNormanBasis or None")
    else:
        if prepared_basis.controls != tuple(hamiltonian.paulis):
            raise ValueError("prepared_basis must match the ordered Hamiltonian controls")
        if prepared_basis.dimension > max_dimension:
            raise AlgebraTooLarge(f"Pauli closure exceeds max_dimension={max_dimension}")
        if basis_order is not None and tuple(map(str, basis_order)) != prepared_basis.words:
            raise ValueError("basis_order must match prepared_basis")
    words = prepared_basis.words
    basis = [get_pauli_string(word) for word in words]
    evaluations, rejected, peak_condition = 0, 0, 1.0

    def check_chart(theta):
        nonlocal peak_condition
        if not np.all(np.isfinite(theta)):
            raise _ChartLimit("nonfinite coordinates")
        # Central generators (including identity) never change the Jacobian.
        if chart_radius is not None and np.sum(np.abs(theta[prepared_basis._noncentral])) > chart_radius:
            raise _ChartLimit("chart_radius exceeded")
        matrix = prepared_basis.jacobian(theta)
        if not np.all(np.isfinite(matrix)):
            raise _ChartLimit("nonfinite coordinate Jacobian")
        try:
            condition = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError as exc:
            raise _ChartLimit("condition estimation failed") from exc
        peak_condition = max(peak_condition, condition) if np.isfinite(condition) else np.inf
        if not np.isfinite(condition) or condition > condition_limit:
            raise _ChartLimit("condition_limit exceeded")
        return matrix

    def rhs(time, theta):
        nonlocal evaluations
        if evaluations >= max_rhs_evaluations:
            raise IntegrationFailure("max_rhs_evaluations exhausted")
        evaluations += 1
        coefficients = np.zeros(len(basis))
        coefficients[prepared_basis._positions] = hamiltonian.at(time)
        matrix = check_chart(theta)
        try:
            derivative = np.linalg.solve(matrix, coefficients)
        except np.linalg.LinAlgError as exc:
            raise _ChartLimit("coordinate solve failed") from exc
        if not np.all(np.isfinite(derivative)):
            raise _ChartLimit("nonfinite coordinate derivative")
        return derivative

    circuit, intervals = Circuit(), []
    current, target = start, end
    direction = 1 if end >= start else -1
    while current != end:
        if len(intervals) >= max_segments:
            raise IntegrationFailure("max_segments exhausted; no partial circuit returned")
        try:
            solution = solve_ivp(rhs, (current, target), np.zeros(len(basis)),
                                 method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
            if not solution.success:
                raise IntegrationFailure(solution.message)
            # Accepted mesh states need not coincide with checked RHS stages.
            for mesh_angles in solution.y.T:
                check_chart(mesh_angles)
            angles = solution.y[:, -1]
        except _ChartLimit as exc:
            rejected += 1
            midpoint = current + (target - current) / 2
            if midpoint == current or midpoint == target:
                raise IntegrationFailure(
                    f"chart interval cannot be resolved at this time scale: {exc}"
                ) from exc
            target = midpoint
            continue
        # Circuit application order is opposite to the displayed WN product.
        for pauli, angle in reversed(list(zip(basis, angles))):
            if angle != 0:
                circuit.add(pauli, float(angle), "wei-norman")
        intervals.append((current, target))
        duration = target - current
        current = target
        remaining = abs(end - current)
        target = end if abs(duration) >= remaining / 2 else current + 2 * duration
        if current != end and direction * (target - current) <= 0:
            raise IntegrationFailure("integration made no time progress")

    # Merge exactly commuting factors but do not drop small computed angles.
    return DrivenResult(fold_phases(circuit, tolerance=0), words, tuple(intervals),
                        evaluations, rejected, peak_condition)
