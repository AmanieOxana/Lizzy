"""Ancilla-free Clifford+T emission, with explicit numerical error accounting.

Generic rotations are delegated to the optional ``pygridsynth`` dependency.
The dependency-free cost estimate is a search heuristic, never a T-count bound.
"""

from dataclasses import dataclass, field
from math import ceil, fsum, isfinite, log2, pi, ulp
from operator import index as integer_index

import numpy as np

from lizzy.emission.native import (
    NativeCircuit,
    NativeGate,
    _qasm3,
    ladder_circuit,
    native_frame_candidates,
)
from lizzy.hamiltonian import Circuit

_FRAME_MAX_WIDTH = 8
_FRAME_MAX_ROTATIONS = 256


def _positive_precision(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0 < value < 1:
        raise ValueError("The error tolerance must be finite and strictly between 0 and 1.")
    return value


def _special_rotation(theta: float, precision: float):
    """Return a phase-aware pi/4 Rz approximation, charged against its tolerance."""
    quotient = theta / (pi / 4)
    if not isfinite(quotient):
        return None
    multiple = round(quotient)
    # Include floating-point pi and arithmetic uncertainty, not only the visible
    # difference. This also prevents treating an almost-Clifford angle as exact.
    error = abs(theta - multiple * (pi / 4)) / 2
    if multiple:
        error += abs(multiple) * ulp(pi) + ulp(theta)
    if error > precision:
        return None
    remainder = multiple % 8
    words = ((), ("t",), ("s",), ("s", "t"), ("z",),
             ("z", "t"), ("sdg",), ("tdg",))
    return words[remainder], -multiple * pi / 8, error


def estimate_rotation_t_count(angle: float, precision: float) -> int:
    r"""Estimate the T cost of ``exp(-i angle P)`` for a nonidentity Pauli.

    Clifford/T angles (including affordable snapping) cost zero/one T; other
    rotations use ``ceil(3 log2(1/precision) + 4)``. This typical-case heuristic
    is neither a worst-case guarantee nor a replacement for concrete compilation.
    ``precision`` is the per-rotation operator-norm error allowance.
    """
    precision = _positive_precision(precision)
    angle = float(angle)
    if not isfinite(angle) or not isfinite(2 * angle):
        raise ValueError("A rotation angle must be finite.")
    special = _special_rotation(2 * abs(angle), precision)
    if special is not None:
        return sum(kind in ("t", "tdg") for kind in special[0])
    return ceil(-3 * log2(precision) + 4)


@dataclass(frozen=True)
class CliffordTGate:
    """One ideal Clifford+T gate, in circuit application order."""

    kind: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"h", "s", "sdg", "x", "z", "cx", "t", "tdg"}:
            raise ValueError(f"Unsupported Clifford+T gate {self.kind!r}.")
        # Reuse the native gate's arity and strict integer validation.
        native_kind = self.kind if self.kind in {"h", "s", "sdg", "cx"} else "h"
        validated = NativeGate(native_kind, self.qubits)
        object.__setattr__(self, "qubits", validated.qubits)

    def inverse(self) -> "CliffordTGate":
        inverse_kind = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        return CliffordTGate(inverse_kind.get(self.kind, self.kind), self.qubits)


@dataclass(frozen=True)
class CliffordTCircuit:
    """Materialized ancilla-free circuit, including absolute global phase.

    ``rotation_error_bound`` is the sum of checked local numerical bounds plus
    phase-bookkeeping roundoff. It compares this artifact to the supplied logical
    rotation sequence, NOT to its Hamiltonian or ODE solution. It is not an
    interval-arithmetic certificate. ``global_phase`` is bookkeeping; controlling
    the circuit requires separately implementing that phase on the control.
    """

    width: int
    gates: tuple[CliffordTGate, ...]
    global_phase: float = 0.0
    error_budget: float = 1e-6
    rotation_error_bound: float = 0.0
    error_bound_kind: str = field(default="numerical triangle bound", init=False)

    def __post_init__(self) -> None:
        width = integer_index(self.width)
        validated = NativeCircuit(self.width, global_phase=self.global_phase)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "global_phase", validated.global_phase)
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "error_budget", _positive_precision(self.error_budget))
        bound = float(self.rotation_error_bound)
        if not isfinite(bound) or not 0 <= bound <= self.error_budget:
            raise ValueError("The rotation error bound must lie within the error budget.")
        object.__setattr__(self, "rotation_error_bound", bound)
        for gate in self.gates:
            if not isinstance(gate, CliffordTGate):
                raise TypeError("Clifford+T artifacts contain CliffordTGate objects.")
            if any(qubit >= width for qubit in gate.qubits):
                raise ValueError("A gate addresses a qubit outside the circuit width.")

    @property
    def t_count(self) -> int:
        """Actual occurrences of T and T-dagger in the emitted artifact."""
        return sum(gate.kind in {"t", "tdg"} for gate in self.gates)

    @property
    def two_qubit_gates(self) -> int:
        return sum(gate.kind == "cx" for gate in self.gates)

    def n_2qb_gates(self) -> int:
        return self.two_qubit_gates

    def inverse(self) -> "CliffordTCircuit":
        """Exactly invert the emitted word, without running synthesis again."""
        return CliffordTCircuit(
            self.width, tuple(gate.inverse() for gate in reversed(self.gates)),
            -self.global_phase, self.error_budget, self.rotation_error_bound,
        )

    def get_unitary(self) -> np.ndarray:
        """Dense absolute-phase verification, capped at ten qubits."""
        if self.width > 10:
            raise ValueError("Dense Clifford+T verification is capped at 10 qubits.")
        native = NativeCircuit(self.width)
        total = np.exp(1j * self.global_phase) * np.eye(2**self.width, dtype=complex)
        local_extra = {
            "t": np.diag([1, np.exp(1j * pi / 4)]),
            "tdg": np.diag([1, np.exp(-1j * pi / 4)]),
            "x": np.array([[0, 1], [1, 0]], dtype=complex),
            "z": np.diag([1, -1]),
        }
        for gate in self.gates:
            if gate.kind not in local_extra:
                matrix = native._gate_matrix(NativeGate(gate.kind, gate.qubits))
            else:
                matrix = np.array([[1]], dtype=complex)
                for qubit in range(self.width):
                    local = local_extra[gate.kind] if qubit == gate.qubits[0] else np.eye(2)
                    matrix = np.kron(matrix, local)
            total = matrix @ total
        return total

    def to_qasm3(self) -> str:
        return _qasm3(self.width, self.gates, self.global_phase)


def _gridsynth_rotation(theta: float, precision: float):
    """Compile positive Rz(theta), adapting pygridsynth's matrix-product order."""
    try:
        import mpmath as mp
        from pygridsynth.gridsynth import gridsynth_gates
    except ImportError as exc:
        raise ImportError(
            "Generic Clifford+T rotations require the optional dependency; "
            "install Lizzy with `pip install 'lizzy[ft]'`."
        ) from exc

    dps = max(50, ceil(-np.log10(precision)) + 35)
    with mp.workdps(dps):
        # mp.mpf(float) retains the exact binary input. A decimal repr alone would
        # quietly change the target at sub-double requested tolerances.
        target_angle = mp.mpf(theta)
        word = gridsynth_gates(
            theta=target_angle, epsilon=mp.mpf(precision) / 4,
            dps=dps, up_to_phase=False, seed=0,
        )
        if not isinstance(word, str) or set(word) - set("HTSXW"):
            raise ValueError("pygridsynth returned an unsupported gate word.")
        phase = float(word.count("W") * mp.pi / 4)
        kinds = tuple(symbol.lower() for symbol in reversed(word) if symbol != "W")
        root = mp.exp(1j * mp.pi / 4)
        matrices = {
            "h": mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2),
            "t": mp.matrix([[1, 0], [0, root]]),
            "s": mp.matrix([[1, 0], [0, 1j]]),
            "x": mp.matrix([[0, 1], [1, 0]]),
        }
        approximate = mp.exp(1j * mp.mpf(phase)) * mp.eye(2)
        for kind in kinds:
            approximate = matrices[kind] * approximate
        target = mp.matrix([
            [mp.exp(-1j * target_angle / 2), 0],
            [0, mp.exp(1j * target_angle / 2)],
        ])
        # Frobenius norm upper-bounds the operator norm and avoids spectral
        # cancellation. The upward float rounding does not make this an interval
        # certificate; the high-precision computation itself is still numerical.
        numerical = mp.sqrt(sum(abs(x) ** 2 for x in approximate - target))
        bound = float(np.nextafter(float(numerical), np.inf))
        if bound > precision:
            raise ValueError("The synthesized rotation failed its numerical error budget.")
    return kinds, phase, bound


def _append_reduced(gates: list[CliffordTGate], gate: CliffordTGate) -> None:
    if gates and gates[-1] == gate.inverse():
        gates.pop()
    elif gates and gates[-1] == gate and gate.kind in {"t", "tdg"}:
        gates.pop()
        _append_reduced(gates, CliffordTGate("s" if gate.kind == "t" else "sdg", gate.qubits))
    else:
        gates.append(gate)


def compile_clifford_t(
    circuit: Circuit, width: int, *, error: float = 1e-6,
) -> CliffordTCircuit:
    """Compile a logical Pauli circuit to actual ancilla-free Clifford+T gates.

    First, affordable pi/4 Rz snapping is identified using ``error/(2N)`` for
    ``N`` input rotations. After charging these special rotations' errors, the
    remaining half-budget is divided only among generic rotation occurrences.
    Thus zero/Clifford padding does not artificially tighten their tolerances.
    The other half is reserved for phase bookkeeping. Positive/negative copies
    use exactly inverse cached words, preserving mirrored Cartan wings. No
    Hamiltonian/decomposition error is included in ``error``; failed numerical
    checks reject compilation.
    """
    native = ladder_circuit(circuit, width)
    native.global_phase = fsum(
        -float(angle) for pauli, angle in circuit.rotations if not pauli.get_support()
    )
    return compile_native_clifford_t(native, error=error)


def compile_frame_candidates(
    circuit: Circuit, width: int, *, error: float = 1e-6,
) -> tuple[
    tuple[tuple[str, CliffordTCircuit], ...],
    tuple[tuple[str, str], ...],
]:
    """Compile bounded exact frame alternatives without selecting by native CX.

    Returns ``(successful_candidates, rejected_candidates)`` with strategy names
    and either a concrete Clifford+T circuit or a diagnostic string. The caller
    retains its independently compiled ladder reference. Public input errors
    raise; optional frame build/compilation failures are isolated and reported.

    At most eight qubits and 256 logical rotations are compared. Larger inputs
    return an explicit ``frame-cap`` rejection. Fixed-frame alternatives are
    generated once, or rolling frames with lookahead zero and eight. Identical
    native artifacts are deduplicated before compilation. Every surviving
    candidate receives the same total ``error`` budget and unchanged downstream
    approximation policy; no dense verification or angle truncation is used.
    """
    error = _positive_precision(error)
    width = NativeCircuit(width).width
    if not isinstance(circuit, Circuit):
        raise TypeError("Expected a logical Circuit for frame candidate compilation.")
    scalar_angles = []
    for pauli, angle in circuit.rotations:
        if len(str(pauli)) != width:
            raise ValueError("A logical Pauli width does not match the circuit width.")
        angle = float(angle)
        if not isfinite(angle) or not isfinite(2 * angle):
            raise ValueError("A rotation angle must be finite.")
        if not pauli.get_support():
            scalar_angles.append(-angle)
    phase = fsum(scalar_angles)
    if not isfinite(phase):
        raise ValueError("The logical scalar phase must be finite.")
    if width > _FRAME_MAX_WIDTH or len(circuit.rotations) > _FRAME_MAX_ROTATIONS:
        return (), ((
            "frame-cap",
            f"Skipped: frame comparison is capped at {_FRAME_MAX_WIDTH} qubits and "
            f"{_FRAME_MAX_ROTATIONS} logical rotations; received {width} qubits and "
            f"{len(circuit.rotations)} rotations.",
        ),)

    compiled, rejected, seen = [], [], set()
    optional_failures = (ValueError, ImportError, RuntimeError, np.linalg.LinAlgError)
    for lookahead in (0, 8):
        try:
            alternatives = native_frame_candidates(circuit, width, lookahead=lookahead)
        except optional_failures as exc:
            rejected.append((f"frame-build-{lookahead}", f"{type(exc).__name__}: {exc}"))
            continue
        for name, candidate in alternatives:
            strategy = f"frame-{name}-{lookahead}" if name == "rolling" else f"frame-{name}"
            try:
                if candidate.width != width:
                    raise ValueError("A frame candidate changed the circuit width.")
                # Frame/unframe Cliffords cancel their absolute phase. Match the
                # ladder API's compensated scalar sum, not incremental += sums.
                native = NativeCircuit(width, list(candidate.gates), phase)
                key = (native.width, tuple(native.gates), native.global_phase)
                if key in seen:
                    continue
                seen.add(key)
                emitted = compile_native_clifford_t(native, error=error)
            except optional_failures as exc:
                rejected.append((strategy, f"{type(exc).__name__}: {exc}"))
            else:
                compiled.append((strategy, emitted))
        if not any(name == "rolling" for name, _ in alternatives):
            break  # Lookahead does not affect fixed or empty frames.
    return tuple(compiled), tuple(rejected)


def compile_native_clifford_t(
    circuit: NativeCircuit, *, error: float = 1e-6,
) -> CliffordTCircuit:
    """Compile an existing H/S/Sdg/CX/Rz artifact through the same backend.

    This entry point keeps an upstream compiler's fixed Clifford gates intact.
    The numerical error budget is relative to this input artifact, excluding
    any earlier matrix decomposition, Hamiltonian approximation or SDK rounding.
    Its allocation, angle snapping and inverse-word reuse are identical to
    :func:`compile_clifford_t`.
    """
    error = _positive_precision(error)
    if not isinstance(circuit, NativeCircuit):
        raise TypeError("Expected a NativeCircuit for native Clifford+T compilation.")
    native = NativeCircuit(circuit.width, list(circuit.gates), circuit.global_phase)
    rotations = [gate for gate in native.gates if gate.kind == "rz"]
    snapping_precision = error / (2 * max(len(rotations), 1))
    if snapping_precision == 0:
        raise ValueError("The per-rotation tolerance underflowed; increase the error budget.")
    # Snapping decisions are frozen before the generic budget is redistributed.
    # Charge every occurrence, including mirrored copies, but do not make an
    # exactly zero/Clifford Rz consume a generic synthesis precision slot.
    special = {}
    special_errors = []
    generic_occurrences = 0
    for gate in rotations:
        assert gate.angle is not None
        angle = abs(gate.angle)
        if angle not in special:
            special[angle] = _special_rotation(angle, snapping_precision)
        compiled = special[angle]
        if compiled is None:
            generic_occurrences += 1
        else:
            special_errors.append(compiled[2])
    precision = (error / 2 - fsum(special_errors)) / max(generic_occurrences, 1)
    if generic_occurrences and precision <= 0:
        raise ValueError("The generic-rotation tolerance underflowed; increase the error budget.")
    phase_parts = [native.global_phase] if native.global_phase else []
    cache = {angle: compiled for angle, compiled in special.items() if compiled is not None}
    gates: list[CliffordTGate] = []
    errors = []
    for gate in native.gates:
        if gate.kind != "rz":
            _append_reduced(gates, CliffordTGate(gate.kind, gate.qubits))
            continue
        assert gate.angle is not None
        angle = abs(gate.angle)
        if angle not in cache:
            cache[angle] = _gridsynth_rotation(angle, precision)
        kinds, phase, bound = cache[angle]
        local = tuple(CliffordTGate(kind, gate.qubits) for kind in kinds)
        if gate.angle < 0:
            local = tuple(item.inverse() for item in reversed(local))
            phase = -phase
        for item in local:
            _append_reduced(gates, item)
        phase_parts.append(phase)
        errors.append(bound)
    phase = fsum(phase_parts)
    # Sum absolute ulps, so cancellation of large phases cannot hide roundoff.
    phase_roundoff = fsum(ulp(part) for part in phase_parts if part) + (ulp(phase) if phase else 0)
    bound = float(np.nextafter(fsum(errors) + phase_roundoff, np.inf)) if errors or phase_roundoff else 0.0
    if bound > error:
        raise ValueError("Phase bookkeeping or rotation synthesis exceeded the error budget.")
    return CliffordTCircuit(native.width, tuple(gates), phase, error, bound)
