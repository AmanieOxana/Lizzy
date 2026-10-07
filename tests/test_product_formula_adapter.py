"""The optional Hamiltonian baseline must approximate the full target, not a state."""

import json

import numpy as np
import pytest

from experiments import _product_formula_adapter as adapter
from lizzy.dense import evolution
from lizzy.hamiltonian import hamiltonian


def test_both_lowerings_preserve_wire_order_phase_and_first_passing_policy() -> None:
    pytest.importorskip("qiskit")
    terms = {"YXI": 0.7, "ZII": -0.31, "IIX": 0.23, "III": 0.19}
    original_terms = terms.copy()
    target = evolution(hamiltonian(terms), 0.37)
    original_target = target.copy()
    budget = 2e-4
    candidates = adapter.qiskit_product_formula_candidates(terms, 0.37, target, budget)
    assert len(candidates) == 8
    assert terms == original_terms
    assert np.array_equal(target, original_target)
    for order in (2, 4):
        variants = [candidate for candidate in candidates if candidate.metadata["order"] == order]
        default = variants[0]
        assert {(candidate.metadata["lowering"], candidate.metadata["optimization_level"])
                for candidate in variants} == {("default", 0), ("default", 3), ("rustiq", 0), ("rustiq", 3)}
        for candidate in variants:
            assert default.metadata["reps"] == candidate.metadata["reps"]
            assert np.linalg.norm(default.native.get_unitary() - candidate.native.get_unitary(), ord=2) < 1e-10
            assert np.linalg.norm(candidate.native.get_unitary() - target, ord=2) <= budget
            assert candidate.metadata["term_order"] == list(terms)
            assert candidate.metadata["precheck_error"]["phase_aligned"] <= budget
            json.dumps(candidate.metadata, allow_nan=False)
        search = next(item for item in default.metadata["search"] if item["order"] == order)
        assert search["attempts"][-1]["status"] == "PASS"
        assert all(item["status"] == "REJECT_ERROR" for item in search["attempts"][:-1])
        assert search["selected_reps"] == search["attempts"][-1]["reps"]


def test_final_clifford_t_artifacts_match_original_hamiltonian() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("pygridsynth")
    from lizzy.clifford_t import compile_native_clifford_t

    terms = {"YZ": 0.31, "II": -0.17}
    target = evolution(hamiltonian(terms), -0.43)
    candidates = adapter.qiskit_product_formula_candidates(terms, -0.43, target, 5e-7)
    for candidate in candidates:
        compiled = compile_native_clifford_t(candidate.native, error=5e-7)
        assert np.linalg.norm(compiled.get_unitary() - target, ord=2) < 1e-6
        assert compiled.t_count > 0


def test_caps_and_upstream_failures_remain_visible_without_relaxing_accuracy(monkeypatch) -> None:
    pytest.importorskip("qiskit")
    from qiskit import synthesis

    terms = {"XI": 0.7, "ZI": 0.4}
    target = evolution(hamiltonian(terms), 0.8)
    monkeypatch.setattr(adapter, "MAX_EXPANDED_ROTATIONS", 3)
    with pytest.raises(adapter.ProductFormulaBudgetExceeded) as caught:
        adapter.qiskit_product_formula_candidates(terms, 0.8, target, 1e-14)
    searches = caught.value.metadata["search"]
    assert [item["status"] for item in searches] == ["CAP", "CAP"]
    assert searches[0]["attempts"][0]["status"] == "REJECT_ERROR"
    assert searches[0]["attempts"][0]["error"]["phase_aligned"] > 1e-14
    assert all(item["attempts"][-1]["status"] == "CAP" for item in searches)
    json.dumps(caught.value.metadata, allow_nan=False)

    def fail(*args, **kwargs):
        raise RuntimeError("upstream Rustiq failure")

    original_lower = adapter._lower

    def fail_optimization(circuit, optimization_level=0):
        if optimization_level == 3:
            raise RuntimeError("upstream level-three failure")
        return original_lower(circuit, optimization_level)

    monkeypatch.setattr(adapter, "MAX_EXPANDED_ROTATIONS", 4096)
    monkeypatch.setattr(synthesis, "synth_pauli_network_rustiq", fail)
    monkeypatch.setattr(adapter, "_lower", fail_optimization)
    commuting = {"YZ": 0.31}
    candidates = adapter.qiskit_product_formula_candidates(
        commuting, 0.2, evolution(hamiltonian(commuting), 0.2), 1e-8,
    )
    assert len(candidates) == 8
    for candidate in candidates:
        if candidate.metadata["lowering"] == "rustiq":
            assert candidate.native is None
            assert candidate.metadata["status"] == "ERROR"
            assert "upstream Rustiq failure" in candidate.metadata["error"]
        elif candidate.metadata["optimization_level"] == 3:
            assert candidate.native is None
            assert candidate.metadata["status"] == "ERROR"
            assert "upstream level-three failure" in candidate.metadata["error"]
        else:
            assert candidate.native is not None
            assert candidate.metadata["status"] == "OK"
