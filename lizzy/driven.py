"""Experimental time-dependent synthesis in small Pauli Lie algebras.

This implements numerical Wei–Norman coordinates, not a new decomposition theorem:
Qvarfort & Pikovski, PRX Quantum 6, 010201 (2025). Unlike the static router, this
opt-in route has no certified global error budget. No Hilbert-space matrices or
symbolic equation engine are needed: Pauli adjoint actions are planar rotations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from operator import index

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.integrate import solve_ivp

from lizzy.hamiltonian import Circuit, fold_phases, terms_of, weight


def _real_vector(values, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if values.shape != (size,):
        raise ValueError(f"coefficients must have shape ({size},)")
    if not np.all(np.isfinite(values)) or np.any(values.imag != 0):
        raise ValueError("coefficients must be finite and real (Hermitian Hamiltonian)")
    return values.real.copy()


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
        words = tuple(str(p) for p in self.paulis)
        if not words or any(not w or set(w) - set("IXYZ") for w in words):
            raise ValueError("supply nonempty Pauli words using only I, X, Y, Z")
        if len({len(w) for w in words}) != 1:
            raise ValueError("all Pauli words must have the same width")
        if len(set(words)) != len(words):
            raise ValueError("Pauli words must be distinct; combine duplicate controls")
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
    sampled at ODE evaluations, not a rigorous bound over continuous time.
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


def _positive_integer(value: int, name: str) -> int:
    try:
        parsed = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


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


def _coordinate_matrix(theta: np.ndarray, pairs: list) -> np.ndarray:
    """Right-trivialized Jacobian for U = exp(-i theta_0 P_0) ... exp(-i theta_d P_d)."""
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


class _ChartLimit(Exception):
    pass


def synthesize_driven(
    hamiltonian: DrivenHamiltonian,
    time_span: tuple[float, float],
    *,
    max_dimension: int = 32,
    basis_order: Sequence[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = np.inf,
    chart_radius: float = 0.5,
    condition_limit: float = 100.0,
    max_segments: int = 1024,
    max_rhs_evaluations: int = 100_000,
) -> DrivenResult:
    """Integrate Wei–Norman angles and emit an ordinary Lizzy Pauli circuit.

    This is an explicit experimental route, not automatic cost-based routing.
    All declared controls generate the closure; no coefficients are sampled to
    decide structural eligibility. ``basis_order`` may permute the entire closure.

    A trial interval is bisected when the sum of absolute noncentral chart angles exceeds
    ``chart_radius`` or its Jacobian becomes ill-conditioned. Successful intervals
    restart at zero coordinates, appending a new left increment. Limits bound
    closure size, accepted segments, and total RHS calls; failures raise instead
    of returning a partial circuit. The angle cap is a conservative heuristic,
    not a proof that every point of the numerical path is well conditioned.

    ``rtol``/``atol`` are local coordinate tolerances, NOT a bound on final operator
    error. Set ``max_step`` to resolve the fastest control timescale; an adaptive
    solver can miss narrow or aliased pulses. Both time directions are supported.
    No Magnus/Fer approximation or symbolic Symdyn engine is included here.
    """
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    max_segments = _positive_integer(max_segments, "max_segments")
    max_rhs_evaluations = _positive_integer(max_rhs_evaluations, "max_rhs_evaluations")
    span = np.asarray(time_span, dtype=float)
    if span.shape != (2,) or not np.all(np.isfinite(span)):
        raise ValueError("time_span must contain two finite times")
    start, end = map(float, span)
    if not np.isfinite(end - start):
        raise ValueError("time_span duration must be finite")
    for value, name in ((rtol, "rtol"), (atol, "atol"), (chart_radius, "chart_radius")):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if chart_radius > 0.5:
        raise ValueError("chart_radius must be at most 0.5 for this prototype")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    if np.isnan(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive")

    basis = _closure(hamiltonian.paulis, max_dimension)
    if basis_order is not None:
        order = tuple(str(p) for p in basis_order)
        if len(order) != len(basis) or set(order) != {str(p) for p in basis}:
            raise ValueError("basis_order must be a permutation of the full Pauli closure")
        by_word = {str(p): p for p in basis}
        basis = [by_word[w] for w in order]
    words = tuple(str(p) for p in basis)
    positions = [words.index(w) for w in hamiltonian.paulis]
    pairs = _adjoint_pairs(basis)
    noncentral = np.array([len(first) > 0 for first, _, _ in pairs])
    evaluations, rejected, peak_condition = 0, 0, 1.0

    def check_chart(theta):
        nonlocal peak_condition
        # Central generators (including identity) never change the Jacobian.
        if not np.all(np.isfinite(theta)) or np.sum(np.abs(theta[noncentral])) > chart_radius:
            raise _ChartLimit
        matrix = _coordinate_matrix(theta, pairs)
        condition = float(np.linalg.cond(matrix))
        peak_condition = max(peak_condition, condition)
        if not np.isfinite(condition) or condition > condition_limit:
            raise _ChartLimit
        return matrix

    def rhs(time, theta):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_rhs_evaluations:
            raise IntegrationFailure("max_rhs_evaluations exhausted")
        coefficients = np.zeros(len(basis))
        coefficients[positions] = hamiltonian.at(time)
        matrix = check_chart(theta)
        try:
            return np.linalg.solve(matrix, coefficients)
        except np.linalg.LinAlgError as exc:
            raise _ChartLimit from exc

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
            angles = solution.y[:, -1]
            check_chart(angles)
        except _ChartLimit:
            rejected += 1
            midpoint = current + (target - current) / 2
            if midpoint == current or midpoint == target:
                raise IntegrationFailure("chart interval cannot be resolved at this time scale")
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
