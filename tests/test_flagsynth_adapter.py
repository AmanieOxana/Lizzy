"""Cross-backend matrix contracts for the optional FlagSynth benchmark adapter."""

from importlib import import_module

import numpy as np
import pytest

from experiments._flagsynth_adapter import (
    _checked_demux,
    flagsynth_sdm_candidates,
    pennylane_operations_to_native,
)


def _upstream_bindings():
    """Snapshot every binding temporarily replaced by this adapter."""
    module = import_module("flagsynth.sdm")
    recursive = import_module("flagsynth.recursive_flag_decomp")
    linalg = import_module("flagsynth.linalg")
    return (
        module.recursive_flag_decomp_cliff_rz, recursive.zyz_rotation_angles,
        linalg.de_mux, module.de_mux, recursive.de_mux,
        linalg._uniform_rotation_dagger_ops,
    )


def test_pennylane_gate_lowering_preserves_phase_and_wire_order() -> None:
    qml = pytest.importorskip("pennylane")
    operations = [
        qml.GlobalPhase(0.37), qml.RX(0.23, 2), qml.RY(-0.57, 0),
        qml.RZ(1.13, 1), qml.CNOT([2, 0]), qml.CZ([0, 1]),
        qml.Hadamard(1), qml.S(2), qml.adjoint(qml.S(0)),
        qml.PauliX(1), qml.PauliY(2), qml.PauliZ(0), qml.SWAP([0, 2]),
        qml.Identity(1),
    ]
    expected = qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(3))
    native = pennylane_operations_to_native(operations, 3)
    assert np.linalg.norm(native.get_unitary() - expected, ord=2) < 2e-14
    assert sum(gate.kind == "rz" for gate in native.gates) == 3


@pytest.mark.parametrize("width", [2, 3])
def test_upstream_sdm_matches_dense_target_including_global_phase(width: int) -> None:
    pytest.importorskip("flagsynth")
    pytest.importorskip("pennylane")
    rng = np.random.default_rng(319 + width)
    random_matrix = rng.normal(size=(2**width, 2**width)) + 1j * rng.normal(
        size=(2**width, 2**width),
    )
    unitary, _ = np.linalg.qr(random_matrix)
    target = np.exp(0.41j) * unitary
    originals = _upstream_bindings()
    candidates = flagsynth_sdm_candidates(target)
    assert _upstream_bindings() == originals
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.name == "flagsynth-sdm-patched"
    assert np.linalg.norm(candidate.native.get_unitary() - target, ord=2) < 1e-10
    assert candidate.metadata["source_rotation_count"] == 4**width - 1
    assert candidate.native.two_qubit_gates == candidate.metadata["source_entangler_count"]
    assert candidate.metadata["baseline_variant"] == "locally patched upstream SDM"
    assert len(candidate.metadata["robustness_repairs"]) == 2
    assert candidate.metadata["repair_counts"] == {
        "schur_fallbacks": 0, "retained_zero_rotations": 0,
    }


def test_demux_repairs_degenerate_basis_but_keeps_valid_upstream_factors() -> None:
    pytest.importorskip("flagsynth")
    original = import_module("flagsynth.linalg").de_mux
    rng = np.random.default_rng(710)
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    identity = np.eye(4)
    angles = np.array([0.1, 0.1, 0.4, 0.4])
    k0 = (basis * np.exp(1j * angles)) @ basis.conj().T
    # Force a legitimate nonorthogonal eigenbasis in a repeated eigenspace,
    # independently of the platform's LAPACK choice of eigenvectors.
    nonorthogonal = basis.copy()
    nonorthogonal[:, 1] = (basis[:, 0] + basis[:, 1]) / np.sqrt(2)
    upstream_factors = (
        nonorthogonal, -angles,
        np.exp(0.5j * angles)[:, None] * nonorthogonal.conj().T,
    )
    (m0, theta, m1), repaired = _checked_demux(lambda *_: upstream_factors, k0, identity)
    assert repaired
    phases = np.exp(-0.5j * theta)
    for matrix in (m0, m1):
        assert np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2) < 1e-13
    assert np.linalg.norm((m0 * phases) @ m1 - k0, ord=2) < 1e-13
    assert np.linalg.norm((m0 * phases.conj()) @ m1 - identity, ord=2) < 1e-13

    nondegenerate = np.diag(np.exp(1j * np.arange(4)))
    valid = original(nondegenerate, identity)
    retained, repaired = _checked_demux(lambda *_: valid, nondegenerate, identity)
    assert not repaired
    assert all(actual is expected for actual, expected in zip(retained, valid, strict=True))
    with pytest.raises(ValueError, match="nonunitary blocks"):
        _checked_demux(original, 0.9 * identity, identity)


def test_formerly_failing_structured_targets_reconstruct_without_perturbation() -> None:
    pytest.importorskip("flagsynth")
    from experiments.compiler_comparison import cases
    from lizzy.dense import evolution

    # One free and one interacting representative cover the repeated-eigenspace
    # repair at different recursion depths; exact zero angles are checked below.
    regression_cases = {"tfim3", "heisenberg_all_to_all4"}
    for case in cases():
        if case.name not in regression_cases:
            continue
        target = evolution(case.operator, case.time)
        unchanged = target.copy()
        candidate, = flagsynth_sdm_candidates(target)
        assert np.array_equal(target, unchanged), case.name
        assert np.linalg.norm(candidate.native.get_unitary() - target, ord=2) < 1e-10, case.name
        assert candidate.metadata["source_rotation_count"] == len(target)**2 - 1
        assert all(isinstance(count, int) and count >= 0
                   for count in candidate.metadata["repair_counts"].values())
    # Exact zero-angle and repeated-phase edge cases, not only random models.
    for target in (np.eye(8), np.diag(np.exp(1j * np.repeat([0.1, 0.3, 0.4, 0.6], 2)))):
        candidate, = flagsynth_sdm_candidates(target)
        assert np.linalg.norm(candidate.native.get_unitary() - target, ord=2) < 1e-10


def test_repaired_sdm_passes_final_clifford_t_check() -> None:
    pytest.importorskip("flagsynth")
    pytest.importorskip("pygridsynth")
    from experiments.compiler_comparison import cases, evaluate_candidate
    from lizzy.dense import evolution

    case = next(case for case in cases() if case.name == "tfim3")
    target = evolution(case.operator, case.time)
    candidate, = flagsynth_sdm_candidates(target)
    result = evaluate_candidate(candidate, target, epsilon=1e-6)
    assert result["status"] == "PASS"
    assert result["final_error"]["strict"] < 1e-6
    assert result["t_count"] > 0


def test_upstream_binding_is_restored_after_synthesis_failure(monkeypatch) -> None:
    pytest.importorskip("flagsynth")
    module = import_module("flagsynth.sdm")
    originals = _upstream_bindings()

    def fail(*args, **kwargs):
        assert _upstream_bindings() != originals
        raise RuntimeError("upstream decomposition failure")

    monkeypatch.setattr(module, "sdm", fail)
    with pytest.raises(RuntimeError, match="upstream decomposition failure"):
        flagsynth_sdm_candidates(np.eye(4))
    assert _upstream_bindings() == originals


def test_adapter_rejects_invalid_targets_and_unknown_operations() -> None:
    for target in (np.eye(2), np.eye(6), np.ones((4, 4)), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            flagsynth_sdm_candidates(target)
    qml = pytest.importorskip("pennylane")
    with pytest.raises(ValueError, match="Unsupported PennyLane operation"):
        pennylane_operations_to_native([qml.ControlledPhaseShift(0.2, [0, 1])], 2)
