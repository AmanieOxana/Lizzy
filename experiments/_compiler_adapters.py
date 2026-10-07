"""Strict-phase SDK baselines for matched Clifford+T synthesis comparisons.

All targets use Lizzy's big-endian tensor order (qubit zero on the left). These
adapters produce exact-arbitrary-angle ``NativeCircuit`` artifacts, not T counts.
The benchmark then uses the same ``compile_native_clifford_t`` for every method.
"""

from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from lizzy.native import NativeCircuit, NativeGate

if TYPE_CHECKING:
    from lizzy.clifford_t import CliffordTCircuit


@dataclass(frozen=True)
class CompilerCandidate:
    """One upstream variant; ``native=None`` records an individual adapter failure."""

    name: str
    native: NativeCircuit | None
    metadata: dict = field(default_factory=dict)
    # Public automatic T synthesis has already selected and compiled its output.
    # Benchmark that retained artifact, not an oracle's reselection/recompilation.
    compiled: "CliffordTCircuit | None" = None


def native_append_clifford(
    circuit: NativeCircuit, kind: str, qubits: tuple[int, ...],
) -> None:
    """Append a Clifford without disguising it as approximate rotations."""
    if kind in {"h", "s", "sdg", "cx"}:
        circuit.append(NativeGate(kind, qubits))
    elif kind == "id":
        return
    elif kind in {"x", "y", "z"}:
        if kind == "y":
            native_append_clifford(circuit, "sdg", qubits)
        if kind in {"x", "y"}:
            native_append_clifford(circuit, "h", qubits)
        native_append_clifford(circuit, "s", qubits)
        native_append_clifford(circuit, "s", qubits)
        if kind in {"x", "y"}:
            native_append_clifford(circuit, "h", qubits)
        if kind == "y":
            native_append_clifford(circuit, "s", qubits)
    elif kind == "cz":
        control, target = qubits
        native_append_clifford(circuit, "h", (target,))
        native_append_clifford(circuit, "cx", (control, target))
        native_append_clifford(circuit, "h", (target,))
    elif kind == "swap":
        first, second = qubits
        for pair in ((first, second), (second, first), (first, second)):
            native_append_clifford(circuit, "cx", pair)
    else:
        raise ValueError(f"Unsupported Clifford gate {kind!r}.")


def native_append_rotation(
    circuit: NativeCircuit, axis: str, angle: float, qubit: int,
) -> None:
    """Append hardware RX/RY/RZ(angle)=exp(-i angle P/2), without approximation."""
    if axis not in {"rx", "ry", "rz"}:
        raise ValueError(f"Unsupported rotation axis {axis!r}.")
    if axis == "ry":
        native_append_clifford(circuit, "sdg", (qubit,))
    if axis != "rz":
        native_append_clifford(circuit, "h", (qubit,))
    circuit.append(NativeGate("rz", (qubit,), float(angle)))
    if axis != "rz":
        native_append_clifford(circuit, "h", (qubit,))
    if axis == "ry":
        native_append_clifford(circuit, "s", (qubit,))


def _target_width(target: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = np.asarray(target, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The target must be a square unitary matrix.")
    dimension = matrix.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("The target dimension must be a positive-qubit power of two.")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix.conj().T @ matrix, np.eye(dimension), atol=1e-10, rtol=0,
    ):
        raise ValueError("The target must be finite and unitary.")
    return matrix, dimension.bit_length() - 1


def _bit_reversed(matrix: np.ndarray, width: int) -> np.ndarray:
    indices = [int(f"{index:0{width}b}"[::-1], 2) for index in range(2**width)]
    return matrix[np.ix_(indices, indices)]


def qiskit_to_native(circuit) -> NativeCircuit:
    """Interpret Qiskit wire zero as Lizzy wire zero (leftmost tensor factor)."""
    if circuit.num_clbits:
        raise ValueError("Only unitary circuits without classical bits are supported.")
    native = NativeCircuit(circuit.num_qubits, global_phase=float(circuit.global_phase))
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if operation.name in {"barrier", "id"}:
            continue
        if operation.name in {"rx", "ry", "rz"}:
            native_append_rotation(native, operation.name, float(operation.params[0]), qubits[0])
        else:
            native_append_clifford(native, operation.name, qubits)
    return native


def qiskit_qsd_candidates(target: np.ndarray) -> list[CompilerCandidate]:
    """Upstream exact QSD/block-ZXZ, with fixed level-zero and level-three variants.

    Qiskit >=2.5 uses the newer block-ZXZ implementation behind ``qs_decomposition``;
    metadata records the installed version instead of claiming a frozen algorithm.
    A bit reversal at the input boundary accounts for Qiskit's little-endian basis.
    No approximate unitary synthesis or hardware layout is requested.
    """
    from qiskit import transpile
    from qiskit.synthesis import qs_decomposition

    matrix, width = _target_width(target)
    upstream = qs_decomposition(_bit_reversed(matrix, width))
    basis = ["rz", "rx", "ry", "h", "s", "sdg", "x", "z", "cx"]
    candidates = []
    for level in (0, 3):
        metadata = {
            "package": "qiskit", "version": version("qiskit"),
            "algorithm": "qiskit.synthesis.qs_decomposition",
            "optimization_level": level, "input_tensor_order": "big-endian",
            "basis_conversion": "input bit reversal; wire indices preserved",
        }
        try:
            emitted = transpile(
                upstream, basis_gates=basis, optimization_level=level,
                approximation_degree=1.0, seed_transpiler=0,
            )
            native = qiskit_to_native(emitted)
            metadata["upstream_cx"] = int(emitted.count_ops().get("cx", 0))
            metadata["status"] = "OK"
        except Exception as exc:
            # A failed optional optimization must not suppress the raw candidate.
            native = None
            metadata.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        candidates.append(CompilerCandidate(f"qiskit-qsd-opt{level}", native, metadata))
    return candidates


def pytket_to_native(circuit) -> NativeCircuit:
    """Convert rebased pytket gates; pytket angles and phase are in half turns."""
    if circuit.bits:
        raise ValueError("Only unitary circuits without classical bits are supported.")
    if any(first != last for first, last in circuit.implicit_qubit_permutation().items()):
        raise ValueError("Implicit pytket wire permutations must be made explicit.")
    indices = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    native = NativeCircuit(circuit.n_qubits, global_phase=float(circuit.phase) * np.pi)
    for command in circuit.get_commands():
        kind = command.op.type.name.lower()
        qubits = tuple(indices[qubit] for qubit in command.qubits)
        if kind in {"rx", "ry", "rz"}:
            native_append_rotation(native, kind, float(command.op.params[0]) * np.pi, qubits[0])
        elif kind == "phase":
            native.add_global_phase(float(command.op.params[0]) * np.pi)
        elif kind not in {"barrier", "noop"}:
            native_append_clifford(native, kind, qubits)
    return native


def pytket_candidates(target: np.ndarray) -> list[CompilerCandidate]:
    """Genuine upstream 1/2/3-qubit unitary synthesis, with optional peephole pass.

    There is no generic dense n>3 unitary-box baseline here. Such targets raise
    NotImplementedError, rather than quietly borrowing Lizzy's decomposition.
    """
    from pytket.circuit import Circuit, OpType, Unitary1qBox, Unitary2qBox, Unitary3qBox
    from pytket.passes import (
        AutoRebase,
        DecomposeBoxes,
        FullPeepholeOptimise,
        SynthesiseTket,
    )

    matrix, width = _target_width(target)
    if width > 3:
        raise NotImplementedError("The pytket dense-unitary baseline supports only 1-3 qubits.")
    circuit = Circuit(width)
    if width == 1:
        circuit.add_unitary1qbox(Unitary1qBox(matrix), 0)
    elif width == 2:
        circuit.add_unitary2qbox(Unitary2qBox(matrix), 0, 1)
    else:
        circuit.add_unitary3qbox(Unitary3qBox(matrix), 0, 1, 2)
    DecomposeBoxes().apply(circuit)
    SynthesiseTket().apply(circuit)
    candidates = []
    for optimize in (False, True):
        metadata = {
            "package": "pytket", "version": version("pytket"),
            "algorithm": f"Unitary{width}qBox/DecomposeBoxes/SynthesiseTket",
            "peephole": optimize, "input_tensor_order": "big-endian",
        }
        try:
            emitted = circuit.copy()
            if optimize:
                FullPeepholeOptimise(allow_swaps=False).apply(emitted)
            AutoRebase({OpType.CX, OpType.Rz, OpType.Rx}, allow_swaps=False).apply(emitted)
            native = pytket_to_native(emitted)
            metadata.update(status="OK", upstream_cx=emitted.n_gates_of_type(OpType.CX))
        except Exception as exc:
            native = None
            metadata.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        candidates.append(CompilerCandidate(
            "pytket-unitary-peephole" if optimize else "pytket-unitary-base",
            native, metadata,
        ))
    return candidates
