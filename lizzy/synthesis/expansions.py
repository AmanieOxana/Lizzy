"""Fourth-order Magnus and Fer time stepping in a bounded Pauli Lie algebra.

The algorithms are equations (22)–(24) of Blanes, Casas, Oteo & Ros, *The Fer and
Magnus expansions*, https://personales.upv.es/serblaza/2011EncyclopediaFerMagnus.pdf.
The 1998 convergence paper is https://doi.org/10.1088/0305-4470/31/1/023.

Expansion and gate synthesis are separate: an exponential of a Pauli *sum* is not
one Pauli rotation. Noncommuting factors use Lizzy's numerical Wei–Norman compiler;
all emitted rotations, including its chart restarts, are charged. Neither the
finite-step approximation nor its numerical synthesis is globally certified.
"""

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.hamiltonian import Circuit, fold_phases
from lizzy.synthesis._numerical import (
    _positive_integer,
    _time_span,
    _validate_ode_tolerances,
)
from lizzy.synthesis.driven import (
    DrivenHamiltonian,
    IntegrationFailure,
    _closure,
    synthesize_driven,
)


@dataclass(frozen=True)
class ExpansionPlan:
    """Approximate propagator before gate synthesis, including absolute phase.

    Each vector in ``factors`` means ``exp(-i sum_j factors[k][j] basis[j])``.
    Factors are in CIRCUIT APPLICATION order: the final matrix is the product
    in reverse order. These are Pauli-sum exponentials, not individual gates.
    ``intervals`` records time steps, not the internal numerical compiler charts.
    The default explicit-closure cap is 32; central identity is included if given.
    """

    method: str
    basis: tuple[str, ...]
    time_span: tuple[float, float]
    intervals: tuple[tuple[float, float], ...]
    factors: tuple[tuple[float, ...], ...]
    control_evaluations: int
    order: int = field(default=4, init=False)
    error_guaranteed: bool = field(default=False, init=False)

    @property
    def dimension(self) -> int:
        return len(self.basis)

    @property
    def steps(self) -> int:
        return len(self.intervals)

    @property
    def exponentials(self) -> int:
        return len(self.factors)


@dataclass
class ExpansionResult:
    """Compiled expansion with distinct approximation and synthesis diagnostics.

    ``plan`` exposes the truncated propagator independently of the emitted circuit.
    ``compilation_segments`` and ``compilation_rhs_evaluations`` count the complete
    numerical factor-synthesis work (including rejected RHS evaluations).
    ``direct_exponentials`` counts factors with mutually commuting active terms,
    emitted directly without ODE integration. Tolerances do not certify total error.
    """

    circuit: Circuit
    plan: ExpansionPlan
    compilation_rhs_evaluations: int
    compilation_segments: int
    direct_exponentials: int
    error_guaranteed: bool = field(default=False, init=False)

    @property
    def dimension(self) -> int:
        return self.plan.dimension

    @property
    def steps(self) -> int:
        return self.plan.steps

    @property
    def exponentials(self) -> int:
        return self.plan.exponentials


class _PauliBracket:
    """Sparse structure constants for T(a)=-i sum a_j P_j, [T(a),T(b)]=T(c)."""

    def __init__(self, basis):
        positions = {str(p): j for j, p in enumerate(basis)}
        left, right, output, scales = [], [], [], []
        for j, p in enumerate(basis):
            for k in range(j + 1, len(basis)):
                q = basis[k]
                if not p.commutes_with(q):
                    left.append(j)
                    right.append(k)
                    output.append(positions[str(p @ q)])
                    scales.append(2 * p.sign(q).imag)
        self.left = np.array(left, dtype=int)
        self.right = np.array(right, dtype=int)
        self.output = np.array(output, dtype=int)
        self.scales = np.array(scales, dtype=float)
        self.dimension = len(basis)

    def __call__(self, left, right):
        coefficients = self.scales * (
            left[self.left] * right[self.right] - left[self.right] * right[self.left]
        )
        return np.bincount(self.output, weights=coefficients, minlength=self.dimension)


def expand_driven(
    hamiltonian: DrivenHamiltonian,
    time_span: tuple[float, float],
    *,
    method: str = "magnus4",
    steps: int = 1,
    max_dimension: int = 32,
    max_exponentials: int = 4096,
) -> ExpansionPlan:
    """Build fourth-order Gauss–Legendre Magnus or two-factor Fer time steps.

    For the scaled moments a=h/2(A_early+A_late),
    b=h*sqrt(3)/12(A_late-A_early), and c=[b,a], the matrix formulas are
    Magnus: exp(a+c); Fer: exp(a) exp(c-[a,c]/2). We store the latter as
    [c-[a,c]/2, a] because circuits apply listed factors from left to right.

    ``steps`` is a user-selected mesh, NOT sized from a guaranteed error bound.
    Smoothness and adequate resolution of the controls are required; quadrature
    can miss narrow pulses. Both quadrature and expansion truncation contribute
    error. Reverse-time intervals are supported; Fer4 need not be exactly
    time-reversal symmetric at finite step size, despite being fourth order.
    No dense Hilbert-space matrices are built. For constant H the commutator
    correction vanishes, but exp(-itH) still needs synthesis into gates.

    All declared controls generate the budgeted closure, including initially zero
    controls. ``max_exponentials`` caps the worst-case number of factors (two per
    Fer step) before evaluating controls; exactly zero factors are omitted.
    """
    if method not in ("magnus4", "fer4"):
        raise ValueError("method must be 'magnus4' or 'fer4'")
    steps = _positive_integer(steps, "steps")
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    max_exponentials = _positive_integer(max_exponentials, "max_exponentials")
    start, end = _time_span(time_span)
    if start != end and steps * (2 if method == "fer4" else 1) > max_exponentials:
        raise IntegrationFailure("max_exponentials exceeded before control evaluation")
    basis = _closure(hamiltonian.paulis, max_dimension)
    words = tuple(str(p) for p in basis)
    if start == end:
        return ExpansionPlan(method, words, (start, end), (), (), 0)
    bracket = _PauliBracket(basis)
    positions = [words.index(w) for w in hamiltonian.paulis]
    grid = np.linspace(start, end, steps + 1)
    direction = 1 if end > start else -1
    intervals, factors = [], []
    for lower, upper in pairwise(grid):
        duration = upper - lower
        early_time = lower + (0.5 - np.sqrt(3) / 6) * duration
        late_time = lower + (0.5 + np.sqrt(3) / 6) * duration
        if not all(direction * gap > 0 for gap in
                   (early_time - lower, late_time - early_time, upper - late_time)):
            raise IntegrationFailure("quadrature nodes cannot be resolved at this time scale")
        early, late = np.zeros(len(basis)), np.zeros(len(basis))
        early[positions] = hamiltonian.at(early_time)
        late[positions] = hamiltonian.at(late_time)
        with np.errstate(over="raise", invalid="raise"):
            try:
                average = (duration / 2) * early + (duration / 2) * late
                moment = (duration * np.sqrt(3) / 12) * (late - early)
                correction = bracket(moment, average)
                if method == "magnus4":
                    step_factors = (average + correction,)
                else:
                    step_factors = (correction - 0.5 * bracket(average, correction), average)
                if not all(np.all(np.isfinite(f)) for f in step_factors):
                    raise FloatingPointError
            except FloatingPointError as exc:
                raise IntegrationFailure("nonfinite expansion; reduce step size or control scale") from exc
        factors.extend(tuple(map(float, f)) for f in step_factors if np.any(f != 0))
        intervals.append((float(lower), float(upper)))
    return ExpansionPlan(method, words, (start, end), tuple(intervals), tuple(factors), 2 * steps)


def synthesize_expansion(
    hamiltonian: DrivenHamiltonian,
    time_span: tuple[float, float],
    *,
    method: str = "magnus4",
    steps: int = 1,
    max_dimension: int = 32,
    max_exponentials: int = 4096,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_rhs_evaluations: int = 100_000,
    max_compilation_segments: int = 4096,
) -> ExpansionResult:
    """Expand and compile every factor, retaining total circuit cost and phase.

    Commuting active terms emit directly. Other Pauli-sum exponentials are compiled
    via numerical Wei–Norman evolution of a constant generator over [0,1]. ``rtol``
    and ``atol`` apply to that *internal factor synthesis*, not to time stepping
    or total operator error. Both compilation work limits apply across the whole
    result, not separately per exponential; failure never returns a partial circuit.
    This remains opt-in and does not alter Lizzy's existing static route selection.
    """
    max_rhs_evaluations = _positive_integer(max_rhs_evaluations, "max_rhs_evaluations")
    max_compilation_segments = _positive_integer(max_compilation_segments, "max_compilation_segments")
    _validate_ode_tolerances(rtol, atol)
    plan = expand_driven(hamiltonian, time_span, method=method, steps=steps,
                         max_dimension=max_dimension, max_exponentials=max_exponentials)
    paulis = [get_pauli_string(w) for w in plan.basis]
    circuit = Circuit()
    evaluations, segments, direct = 0, 0, 0
    for factor in plan.factors:
        active = [(p, angle) for p, angle in zip(paulis, factor) if angle != 0]
        if all(p.commutes_with(q) for j, (p, _) in enumerate(active) for q, _ in active[j + 1:]):
            for pauli, angle in active:
                circuit.add(pauli, angle, f"{method}:commuting")
            direct += 1
            continue
        if evaluations >= max_rhs_evaluations or segments >= max_compilation_segments:
            raise IntegrationFailure("expansion compilation work budget exhausted")
        constant = DrivenHamiltonian(plan.basis, lambda t, values=factor: values)
        try:
            compiled = synthesize_driven(
                constant, (0, 1), max_dimension=max_dimension, basis_order=plan.basis,
                rtol=rtol, atol=atol, max_rhs_evaluations=max_rhs_evaluations - evaluations,
                max_segments=max_compilation_segments - segments,
            )
        except IntegrationFailure as exc:
            raise IntegrationFailure(f"{method} factor synthesis failed: {exc}") from exc
        evaluations += compiled.rhs_evaluations
        segments += len(compiled.intervals)
        for pauli, angle in compiled.circuit.rotations:
            circuit.add(pauli, angle, f"{method}:wei-norman")
    return ExpansionResult(fold_phases(circuit, tolerance=0), plan, evaluations, segments, direct)
