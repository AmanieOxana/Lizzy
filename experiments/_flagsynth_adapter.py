"""Benchmark adapter for the authors' FlagSynth SDM implementation.

The optional ``flagsynth`` dependency constructs the circuit. Scoped, explicitly
reported repairs make two upstream helpers robust to degenerate spectra and
zero angles; no decomposition algorithm is copied or target perturbed here.
PennyLane and Lizzy both number wire zero as the most significant tensor factor.
"""

from functools import wraps
from importlib import import_module
from importlib.metadata import distribution, version
from inspect import signature
from json import loads
from threading import Lock

import numpy as np

from experiments._compiler_adapters import (
    CompilerCandidate,
    native_append_clifford,
    native_append_rotation,
)
from lizzy.emission.native import NativeCircuit

_SDM_LOCK = Lock()
_KEYWORD_BRIDGE = "recursive_flag_decomp_cliff_rz: selective_demux -> use_sdm"
_PHASE_BRIDGE = "zyz_rotation_angles: return_global_phase=True"
_DEMUX_REPAIR = "de_mux: complex Schur fallback for nonorthonormal eigvectors"
_MOTTONEN_REPAIR = "Mottönen: retain zero rotations through PennyLane's trainable-angle path"
_DEMUX_TOLERANCE = 1e-10


def _checked_demux(original, k0, k1):
    """Keep valid upstream factors; repair only a nonunitary eigenvector basis.

    A unitary matrix with repeated eigenvalues need not receive orthogonal
    eigenvectors from ``numpy.linalg.eig``. Its complex Schur basis is unitary;
    for a normal matrix the triangular factor must also be diagonal. Validate
    that condition and both demultiplexed blocks instead of projecting the input.
    """
    from scipy.linalg import schur

    identity = np.eye(len(k0))

    def unitary_error(matrix):
        return float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2))

    if max(unitary_error(k0), unitary_error(k1)) > _DEMUX_TOLERANCE:
        raise ValueError("FlagSynth de_mux received nonunitary blocks")
    m0, theta, m1 = original(k0, k1)
    repaired = unitary_error(m0) > _DEMUX_TOLERANCE
    if repaired:
        triangular, m0 = schur(k0 @ k1.conj().T, output="complex")
        diagonal = np.diag(triangular)
        if np.linalg.norm(triangular - np.diag(diagonal), ord=2) > _DEMUX_TOLERANCE:
            raise ValueError("FlagSynth Schur fallback did not diagonalize the unitary")
        phases = np.exp(0.5j * np.angle(diagonal))
        theta = -2 * np.angle(phases)
        m1 = (phases[:, None] * m0.conj().T) @ k1

    phases = np.exp(-0.5j * theta)
    errors = (
        unitary_error(m0), unitary_error(m1),
        np.linalg.norm((m0 * phases) @ m1 - k0, ord=2),
        np.linalg.norm((m0 * phases.conj()) @ m1 - k1, ord=2),
    )
    if not np.isfinite(errors).all() or max(errors) > _DEMUX_TOLERANCE:
        raise ValueError(f"FlagSynth de_mux failed factor/reconstruction checks: {errors}")
    return (m0, theta, m1), repaired


def _upstream_sdm(matrix, wires):
    """Run upstream SDM with scoped API bridges and reported robustness repairs.

    FlagSynth's March 2026 rename changed the recursive function's argument to
    ``use_sdm``, but its SDM caller still passes ``selective_demux``. The official
    August 2026 revision retains that mismatch and expects four Euler outputs
    without requesting the optional phase from PennyLane. Keep the dependency
    unmodified and restore every temporary binding even after a failed call.
    The two robustness repairs are deliberately distinct from API bridges in
    metadata: these results must not be called an unmodified upstream baseline.
    The lock serializes calls through this adapter, not unrelated SDK calls.
    """
    import pennylane as qml

    module = import_module("flagsynth.sdm")
    recursive = import_module("flagsynth.recursive_flag_decomp")
    linalg = import_module("flagsynth.linalg")
    with _SDM_LOCK:
        original = module.recursive_flag_decomp_cliff_rz
        original_angles = recursive.zyz_rotation_angles
        original_demux = linalg.de_mux
        original_rotations = linalg._uniform_rotation_dagger_ops
        bindings = [(owner, "de_mux", owner.de_mux) for owner in (linalg, module, recursive)]
        counts = {"schur_fallbacks": 0, "retained_zero_rotations": 0}
        parameters = signature(original).parameters
        bridge = "use_sdm" in parameters and "selective_demux" not in parameters
        phase_option = signature(original_angles).parameters.get("return_global_phase")
        phase_bridge = phase_option is not None and phase_option.default is False

        @wraps(original)
        def renamed_keyword(*args, **kwargs):
            if "selective_demux" in kwargs:
                if "use_sdm" in kwargs:
                    raise TypeError("Both FlagSynth SDM keyword spellings were supplied.")
                kwargs["use_sdm"] = kwargs.pop("selective_demux")
            return original(*args, **kwargs)

        @wraps(original_angles)
        def angles_with_phase(*args, **kwargs):
            kwargs.setdefault("return_global_phase", True)
            return original_angles(*args, **kwargs)

        @wraps(original_demux)
        def orthonormal_demux(k0, k1):
            factors, repaired = _checked_demux(original_demux, k0, k1)
            counts["schur_fallbacks"] += int(repaired)
            return factors

        @wraps(original_rotations)
        def fixed_rotation_layout(gate, alpha, controls, target):
            # PennyLane keeps the complete Gray-code layout for trainable
            # angles. Only the metadata changes: every angle value is retained,
            # and its own decomposition, not a copied implementation, is used.
            angles = qml.numpy.array(alpha, requires_grad=True)
            operations = original_rotations(gate, angles, controls, target)
            epsilon = np.finfo(np.asarray(angles).dtype).eps
            counts["retained_zero_rotations"] += int(sum(
                isinstance(operation, gate)
                and abs(float(operation.parameters[0])) <= epsilon
                for operation in operations
            ))
            # Trainability is only a switch for this helper's pruning policy;
            # do not let it alter subsequent upstream decomposition branches.
            return [
                gate(float(operation.parameters[0]), wires=operation.wires)
                if isinstance(operation, gate) else operation
                for operation in operations
            ]

        try:
            if bridge:
                module.recursive_flag_decomp_cliff_rz = renamed_keyword
            if phase_bridge:
                recursive.zyz_rotation_angles = angles_with_phase
            for owner, name, _ in bindings:
                setattr(owner, name, orthonormal_demux)
            linalg._uniform_rotation_dagger_ops = fixed_rotation_layout
            operations = list(module.sdm(matrix, wires))
        finally:
            module.recursive_flag_decomp_cliff_rz = original
            recursive.zyz_rotation_angles = original_angles
            for owner, name, binding in bindings:
                setattr(owner, name, binding)
            linalg._uniform_rotation_dagger_ops = original_rotations
    bridges = []
    if bridge:
        bridges.append(_KEYWORD_BRIDGE)
    if phase_bridge:
        bridges.append(_PHASE_BRIDGE)
    return operations, bridges, counts


def pennylane_operations_to_native(operations, width: int) -> NativeCircuit:
    """Preserve gate order, wire order, and absolute phase of a PennyLane list.

    Fixed Clifford gates remain fixed gates, not arbitrary rotations that would
    inflate a shared Clifford+T compiler's approximation budget. Unsupported
    operations fail explicitly; they are never ignored or numerically refitted.
    """
    native = NativeCircuit(width)
    cliffords = {
        "Hadamard": "h", "S": "s", "Adjoint(S)": "sdg",
        "PauliX": "x", "PauliY": "y", "PauliZ": "z",
        "CNOT": "cx", "CZ": "cz", "SWAP": "swap", "Identity": "id",
    }
    for operation in operations:
        name = operation.name
        if name == "GlobalPhase":
            # PennyLane uses exp(-i phi), while NativeCircuit stores exp(+i phi).
            native.add_global_phase(-float(operation.parameters[0]))
        elif name in {"RX", "RY", "RZ"}:
            wires = tuple(operation.wires)
            if len(wires) != 1:
                raise ValueError(f"Expected one wire for PennyLane {name}.")
            native_append_rotation(
                native, name.lower(), float(operation.parameters[0]), wires[0],
            )
        elif name in cliffords:
            native_append_clifford(native, cliffords[name], tuple(operation.wires))
        else:
            raise ValueError(f"Unsupported PennyLane operation {name!r} in FlagSynth output.")
    return native


def flagsynth_sdm_candidates(target: np.ndarray) -> list[CompilerCandidate]:
    """Return locally repaired SDM for a dense, big-endian unitary (>= 2 qubits).

    Global phase is retained for strict diagnostics even when the benchmark's
    acceptance metric is phase-aligned. No private simplification is applied:
    all candidates must receive the same subsequent native/Clifford+T passes.
    """
    matrix = np.asarray(target, dtype=complex)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] < 4 or matrix.shape[0] & (matrix.shape[0] - 1)):
        raise ValueError("FlagSynth SDM needs a square 2**n matrix with n >= 2.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The target unitary must contain only finite entries.")
    if np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(matrix)), ord=2) > 1e-8:
        raise ValueError("The FlagSynth target must be unitary.")
    try:
        import_module("flagsynth")
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "This benchmark baseline requires the official FlagSynth package "
            "(https://github.com/PennyLaneAI/flagsynth) and PennyLane."
        ) from exc

    width = len(matrix).bit_length() - 1
    with qml.QueuingManager.stop_recording():
        operations, compatibility_bridges, repair_counts = _upstream_sdm(matrix, list(range(width)))
    native = pennylane_operations_to_native(operations, width)
    direct_url = distribution("flagsynth").read_text("direct_url.json")
    installation = loads(direct_url) if direct_url else {}
    metadata = {
        "implementation": "PennyLaneAI/flagsynth.sdm",
        "flagsynth_version": version("flagsynth"),
        "pennylane_version": qml.__version__,
        "upstream_commit": installation.get("vcs_info", {}).get("commit_id"),
        "compatibility_bridges": compatibility_bridges,
        "baseline_variant": "locally patched upstream SDM",
        "robustness_repairs": [_DEMUX_REPAIR, _MOTTONEN_REPAIR],
        "repair_counts": repair_counts,
        "source_rotation_count": sum(op.name in {"RX", "RY", "RZ"} for op in operations),
        "source_entangler_count": sum(op.name in {"CNOT", "CZ"} for op in operations),
        "source_global_phase_preserved": True,
    }
    return [CompilerCandidate("flagsynth-sdm-patched", native, metadata)]
