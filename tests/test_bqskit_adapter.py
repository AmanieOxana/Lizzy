"""Exact conversion and bounded-search contracts for the optional BQSKit adapter."""

import numpy as np
import pytest

from experiments import _bqskit_adapter as adapter


def test_u3_conversion_preserves_phase_and_nonadjacent_control_order() -> None:
    pytest.importorskip("bqskitrs")
    pytest.importorskip("bqskit")
    from bqskit.ir import Circuit
    from bqskit.ir.gates import CNOTGate, U3Gate

    circuit = Circuit(3)
    circuit.append_gate(U3Gate(), [2], [0.37, -1.23, 2.41])
    circuit.append_gate(CNOTGate(), [2, 0])
    circuit.append_gate(U3Gate(), [0], [-0.91, 0.73, -0.27])
    circuit.append_gate(CNOTGate(), [0, 1])
    native = adapter.bqskit_to_native(circuit)
    assert np.linalg.norm(native.get_unitary() - np.asarray(circuit.get_unitary()), ord=2) < 3e-14
    assert native.two_qubit_gates == 2


def test_seeded_two_qubit_search_preserves_target_up_to_global_phase() -> None:
    pytest.importorskip("bqskitrs")
    pytest.importorskip("bqskit")
    rng = np.random.default_rng(5719)
    target, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    candidate, = adapter.bqskit_candidates(target)
    actual = candidate.native.get_unitary()
    phase = np.trace(target.conj().T @ actual)
    phase /= abs(phase)
    assert np.linalg.norm(actual - phase * target, ord=2) < 5e-7
    assert candidate.metadata["seed"] == 0
    assert candidate.metadata["optimization_level"] == 1
    assert candidate.metadata["synthesis_epsilon"] == 1e-14
    assert candidate.metadata["adapter_strict_error"] < 1e-10


def test_width_cap_precedes_any_search_and_is_not_unsupported(monkeypatch) -> None:
    def unexpected_search(*args):
        pytest.fail("A capped width must never start the search.")

    monkeypatch.setattr(adapter, "_run_bounded", unexpected_search)
    with pytest.raises(adapter.BQSKitResourceCap) as caught:
        adapter.bqskit_candidates(np.eye(16))
    assert isinstance(caught.value, TimeoutError)
    assert caught.value.metadata["width_cap"] == 3
    for target in (np.eye(3), np.ones((4, 4)), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            adapter.bqskit_candidates(target)


def test_timeout_stops_only_the_owned_worker_group(monkeypatch) -> None:
    class Worker:
        pid = 7654321

        def communicate(self, *args, **kwargs):
            raise adapter.subprocess.TimeoutExpired("owned-worker", 120)

    worker = Worker()
    stopped = []
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *args, **kwargs: worker)
    monkeypatch.setattr(adapter, "_stop_owned_group", lambda process: stopped.append(process))
    with pytest.raises(adapter.BQSKitResourceCap):
        adapter._run_bounded(np.eye(4), {"timeout_seconds": 120})
    assert stopped == [worker]
