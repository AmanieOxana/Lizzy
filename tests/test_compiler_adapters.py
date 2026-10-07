"""SDK baselines must preserve wire order, global phase and genuine CX gates."""

import numpy as np
import pytest
from scipy.linalg import expm

from experiments._compiler_adapters import (
    pytket_candidates,
    qiskit_qsd_candidates,
)


def _asymmetric_target(width):
    rng = np.random.default_rng(914 + width)
    raw = rng.normal(size=(2**width, 2**width)) + 1j * rng.normal(size=(2**width, 2**width))
    return expm(-0.19j * (raw + raw.conj().T)) * np.exp(0.217j)


def test_qiskit_qsd_preserves_big_endian_phase_and_cx() -> None:
    pytest.importorskip("qiskit")
    for width in (1, 2, 3):
        target = _asymmetric_target(width)
        candidates = qiskit_qsd_candidates(target)
        assert {candidate.name for candidate in candidates} == {"qiskit-qsd-opt0", "qiskit-qsd-opt3"}
        for candidate in candidates:
            assert np.linalg.norm(candidate.native.get_unitary() - target, 2) < 2e-12
            assert candidate.native.n_2qb_gates() == candidate.metadata["upstream_cx"]
            assert candidate.metadata["package"] == "qiskit"


def test_pytket_boxes_preserve_phase_and_mark_wider_targets_unsupported() -> None:
    pytest.importorskip("pytket")
    for width in (1, 2, 3):
        target = _asymmetric_target(width)
        candidates = pytket_candidates(target)
        assert {candidate.name for candidate in candidates} == {"pytket-unitary-base", "pytket-unitary-peephole"}
        for candidate in candidates:
            assert np.linalg.norm(candidate.native.get_unitary() - target, 2) < 3e-11
            assert candidate.native.n_2qb_gates() == candidate.metadata["upstream_cx"]
    with pytest.raises(NotImplementedError, match="1-3 qubits"):
        pytket_candidates(np.eye(16))


def test_failed_qiskit_optimization_does_not_hide_raw_candidate(monkeypatch) -> None:
    qiskit = pytest.importorskip("qiskit")
    original = qiskit.transpile

    def fail_level_three(*args, **kwargs):
        if kwargs.get("optimization_level") == 3:
            raise RuntimeError("simulated upstream optimization failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(qiskit, "transpile", fail_level_three)
    raw, optimized = qiskit_qsd_candidates(_asymmetric_target(1))
    assert raw.native is not None and raw.metadata["status"] == "OK"
    assert optimized.native is None and optimized.metadata["status"] == "ERROR"
    assert "simulated upstream" in optimized.metadata["error"]

