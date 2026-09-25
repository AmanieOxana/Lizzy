"""End-to-end numerical Wei–Norman synthesis of static and driven Pauli sums.

The established Wei–Norman equations are implemented in :mod:`lizzy.driven`.
This module adds commuting-component reduction, explicit pulse boundaries,
shared resource budgets and phase-preserving, concrete native gate emission.
No dense Hilbert-space matrices are used and no global error certificate is
implied by the local ODE tolerances.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    DrivenResult,
    IntegrationFailure,
    _closure,
    _positive_integer,
    synthesize_driven,
)
from lizzy.emit import EmissionQuote
from lizzy.hamiltonian import Circuit, fold_phases
from lizzy.native import ladder_circuit, native_frame_candidate, native_frame_circuit


@dataclass
class WeiNormanResult:
    """Logical rotations, actual emitted gates and numerical diagnostics.

    ``segments`` has one core result per commuting component per pulse interval,
    in circuit application order. ``charts`` counts accepted compilation segments
    (ODE charts plus one segment for each directly emitted static component/pulse);
    ``chart_restarts`` counts only numerical retries after accepted charts, not
    explicit pulse boundaries. Closure sizes include declared central terms.
    ``emission=None`` only when explicitly requested with ``emission='none'``.
    """

    circuit: Circuit
    width: int
    time_span: tuple[float, float]
    component_bases: tuple[tuple[str, ...], ...]
    segments: tuple[DrivenResult, ...]
    emission: EmissionQuote | None
    error_guaranteed: bool = field(default=False, init=False)

    @property
    def component_dimensions(self) -> tuple[int, ...]:
        return tuple(map(len, self.component_bases))

    @property
    def dimension(self) -> int:
        return len({word for basis in self.component_bases for word in basis})

    @property
    def charts(self) -> int:
        return sum(len(segment.intervals) for segment in self.segments)

    @property
    def chart_restarts(self) -> int:
        return sum(segment.chart_restarts for segment in self.segments)

    @property
    def rhs_evaluations(self) -> int:
        return sum(segment.rhs_evaluations for segment in self.segments)

    @property
    def rejected_intervals(self) -> int:
        return sum(segment.rejected_intervals for segment in self.segments)

    @property
    def max_observed_condition(self) -> float:
        return max((segment.max_observed_condition for segment in self.segments), default=1.0)

    @property
    def logical_two_qubit_gates(self) -> int:
        """Existing block-aware logical cost, not a concrete native gate count."""
        return self.circuit.two_qubit_gates

    @property
    def two_qubit_gates(self) -> int:
        return (self.logical_two_qubit_gates if self.emission is None
                else self.emission.two_qubit_gates)

    @property
    def emission_backend(self) -> str:
        return "none" if self.emission is None else self.emission.backend

    @property
    def emitted_circuit(self) -> object:
        return self.circuit if self.emission is None else self.emission.circuit


def _components(words: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Anticommutation-connected controls; different components commute at ALL times."""
    paulis = [get_pauli_string(word) for word in words]
    remaining = set(range(len(words)))
    components = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        pending, found = [root], [root]
        while pending:
            current = pending.pop()
            neighbors = sorted(j for j in remaining if not paulis[current].commutes_with(paulis[j]))
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
            found.extend(neighbors)
        components.append(tuple(sorted(found)))
    return tuple(components)


def _time_pieces(time_span, breakpoints):
    span = np.asarray(time_span, dtype=float)
    if span.ndim == 0:
        span = np.array([0.0, float(span)])
    if span.shape != (2,) or not np.all(np.isfinite(span)):
        raise ValueError("time_span must be a finite duration or two finite times")
    start, end = map(float, span)
    if not np.isfinite(end - start):
        raise ValueError("time_span duration must be finite")
    points = np.asarray(breakpoints, dtype=float)
    if points.ndim != 1 or not np.all(np.isfinite(points)):
        raise ValueError("breakpoints must be a finite sequence in integration order")
    direction = 1 if end >= start else -1
    boundaries = (start, *map(float, points), end)
    if (len(points) or start != end) and any(
        direction * (b - a) <= 0 for a, b in pairwise(boundaries)
    ):
        raise ValueError("breakpoints must be strictly inside time_span, in integration order")
    return (start, end), (() if start == end else tuple(pairwise(boundaries)))


def _emit(circuit, width, emission):
    if emission == "none":
        return None
    candidates = [("native-ladder", ladder_circuit(circuit, width))]
    if emission == "native" and native_frame_candidate(circuit, width):
        candidates.append(("native-frame", native_frame_circuit(circuit, width)))
    backend, emitted = min(candidates, key=lambda item: (item[1].two_qubit_gates, len(item[1].gates)))
    return EmissionQuote(backend, emitted.two_qubit_gates, emitted)


def synthesize_wei_norman(
    hamiltonian: DrivenHamiltonian | PauliStringLinear,
    time_span: tuple[float, float] | float,
    *,
    split_components: bool = True,
    breakpoints: Sequence[float] = (),
    max_dimension: int = 32,
    max_total_dimension: int = 1024,
    basis_order: Sequence[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = np.inf,
    chart_radius: float | None = None,
    condition_limit: float = 100.0,
    max_segments: int = 1024,
    max_rhs_evaluations: int = 100_000,
    emission: str = "native",
) -> WeiNormanResult:
    """Compile a Pauli Hamiltonian all the way to phase-preserving hardware gates.

    A scalar time means ``(0, time)``. Driven callbacks use absolute time and must
    be deterministic. Supply known discontinuities as strictly ordered interior
    ``breakpoints`` (descending for backwards evolution). At each pulse boundary,
    controls are evaluated from the interior of that pulse, not across its jump.

    Disconnected anticommutation components commute even at different times, so
    they can be integrated separately without a product-formula error. The closure
    cap is per component; ``max_total_dimension`` also caps the union. All closures
    are checked BEFORE evaluating a driven callback. No control is removed merely
    because a sampled coefficient vanishes. ``basis_order`` optionally permutes
    the entire closure; each component uses its induced order.

    RHS and segment budgets are shared across every component and pulse; a directly
    emitted static component/pulse consumes one segment but no RHS evaluations. Any failed
    component raises, never returning a partial evolution. Static commuting
    components emit their exact angles directly. Other components use local
    numerical charts; their circuit length can grow with time and their integration
    cost with algebra dimension. This is NOT an always-better or globally optimal
    synthesis method, nor do ``rtol``/``atol`` bound final operator error.

    ``chart_radius=None`` (default) keeps a compact product while sampled
    Jacobian conditioning permits it, with bounded restarts on rejected trials.
    An explicit ``chart_radius=0.5`` restores the conservative noncentral-angle
    cap. RHS and accepted-mesh checks are not a continuous nonsingularity
    guarantee; set ``max_step`` to resolve the controls' fastest timescale.

    ``emission='native'`` selects the smaller concrete ladder or eligible shared
    Clifford-frame artifact by (CX count, total gate count); it is independent of
    optional SDKs. ``'ladder'`` forces ladder emission, ``'none'`` retains only
    logical rotations. Global phase is preserved in both native artifacts.
    """
    if emission not in {"native", "ladder", "none"}:
        raise ValueError("emission must be 'native', 'ladder', or 'none'")
    if not isinstance(split_components, (bool, np.bool_)):
        raise TypeError("split_components must be boolean")
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    max_total_dimension = _positive_integer(max_total_dimension, "max_total_dimension")
    max_segments = _positive_integer(max_segments, "max_segments")
    max_rhs_evaluations = _positive_integer(max_rhs_evaluations, "max_rhs_evaluations")
    span, pieces = _time_pieces(time_span, breakpoints)
    is_static = isinstance(hamiltonian, PauliStringLinear)
    if not is_static and not isinstance(hamiltonian, DrivenHamiltonian):
        raise TypeError("hamiltonian must be a PauliStringLinear or DrivenHamiltonian")
    drive = DrivenHamiltonian.from_static(hamiltonian) if is_static else hamiltonian
    if len(drive.paulis) > max_total_dimension:
        raise AlgebraTooLarge(f"declared controls exceed max_total_dimension={max_total_dimension}")
    groups = _components(drive.paulis) if split_components else (tuple(range(len(drive.paulis))),)
    bases, known = [], set()
    for group in groups:
        words = tuple(str(p) for p in _closure([drive.paulis[j] for j in group], max_dimension))
        bases.append(words)
        known.update(words)
        if len(known) > max_total_dimension:
            raise AlgebraTooLarge(f"Pauli closure exceeds max_total_dimension={max_total_dimension}")
    if basis_order is not None:
        order = tuple(str(p) for p in basis_order)
        if len(order) != len(known) or set(order) != known:
            raise ValueError("basis_order must be a permutation of the full Pauli closure")
        bases = [tuple(word for word in order if word in basis) for basis in bases]

    options = {"max_dimension": max_dimension, "rtol": rtol, "atol": atol,
               "max_step": max_step, "chart_radius": chart_radius,
               "condition_limit": condition_limit}
    # Validate solver controls even for a zero-time or directly emitted static run.
    probe = DrivenHamiltonian([drive.paulis[0]], lambda t: [0.0])
    synthesize_driven(probe, (span[0], span[0]), **options)
    static_values = drive.at(span[0]) if is_static else None
    circuit, segments = Circuit(), []
    evaluations, charts = 0, 0
    for start, end in pieces:
        lower, upper = sorted((start, end))
        interior_low, interior_high = np.nextafter(lower, upper), np.nextafter(upper, lower)
        if len(pieces) > 1 and interior_low > interior_high:
            raise IntegrationFailure("pulse interval has no representable interior time")
        for group, basis in zip(groups, bases):
            if charts >= max_segments:
                raise IntegrationFailure("max_segments exhausted across components/pulses")
            words = tuple(drive.paulis[j] for j in group)
            indices = np.array(group)

            def coefficients(time, indices=indices, low=interior_low, high=interior_high):
                # Clipping is only necessary at explicitly split discontinuities.
                sample = float(np.clip(time, low, high)) if len(pieces) > 1 else time
                return drive.at(sample)[indices]

            component = DrivenHamiltonian(words, coefficients)
            if is_static and len(basis) == len(words) and all(
                get_pauli_string(p).commutes_with(get_pauli_string(q))
                for j, p in enumerate(words) for q in words[j + 1:]
            ):
                direct = Circuit()
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        angles = (end - start) * static_values[indices]
                    except FloatingPointError as exc:
                        raise IntegrationFailure("nonfinite static rotation angle") from exc
                for word, angle in zip(words, angles):
                    if angle != 0:
                        direct.add(get_pauli_string(word), float(angle), "wei-norman:commuting")
                part = DrivenResult(direct, basis, ((start, end),), 0, 0, 1.0)
            else:
                if evaluations >= max_rhs_evaluations:
                    raise IntegrationFailure("max_rhs_evaluations exhausted across components/pulses")
                part = synthesize_driven(
                    component, (start, end), basis_order=basis,
                    max_segments=max_segments - charts,
                    max_rhs_evaluations=max_rhs_evaluations - evaluations, **options,
                )
            segments.append(part)
            circuit.extend(part.circuit)
            evaluations += part.rhs_evaluations
            charts += len(part.intervals)
    circuit = fold_phases(circuit, tolerance=0)
    return WeiNormanResult(circuit, drive.n_qubits, span, tuple(bases), tuple(segments),
                           _emit(circuit, drive.n_qubits, emission))
