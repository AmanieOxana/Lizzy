"""Matched-accuracy comparison contracts, without optional synthesis SDKs."""

import numpy as np
import pytest

from experiments import compiler_comparison as benchmark
from experiments._compiler_adapters import CompilerCandidate
from lizzy.emission.clifford_t import CliffordTCircuit, CliffordTGate
from lizzy.emission.native import NativeCircuit, NativeGate


def test_common_phase_contract_half_budget_and_both_accuracy_checks(monkeypatch):
    target = np.eye(2, dtype=complex)
    # Zero trace overlap uses phase one; nonfinite artifacts cannot pass by
    # producing NaN comparisons. Every harness now shares these conventions.
    assert benchmark.operator_errors(np.diag([1, -1]), target) == {
        "strict": 2.0, "phase_aligned": 2.0,
    }
    with pytest.raises(ValueError, match="Nonfinite output"):
        benchmark.operator_errors(np.full((2, 2), np.nan), target)
    candidate = CompilerCandidate("phase-only", NativeCircuit(1, global_phase=0.37))
    original_compile = benchmark.compile_native_clifford_t
    budgets = []

    def observed_compile(native, *, error):
        budgets.append(error)
        return original_compile(native, error=error)

    monkeypatch.setattr(benchmark, "compile_native_clifford_t", observed_compile)
    for epsilon in (1e-3, 1e-6):
        result = benchmark.evaluate_candidate(candidate, target, epsilon)
        assert result["status"] == "PASS"
        assert result["decomposition_error"]["strict"] > epsilon
        assert result["final_error"]["strict"] > epsilon
        assert result["decomposition_error"]["phase_aligned"] < 1e-14
        assert result["final_error"]["phase_aligned"] < 1e-14
    assert budgets == [5e-4, 5e-7]

    wrong_native = NativeCircuit(1, [NativeGate("rz", (0,), 0.2)])
    rejected = benchmark.evaluate_candidate(CompilerCandidate("wrong", wrong_native), target, 1e-3)
    assert rejected["status"] == "FAIL_DECOMPOSITION_ERROR"
    assert budgets == [5e-4, 5e-7]  # Invalid decomposition never reaches T synthesis.

    # Even a zero-T backend output cannot pass merely because its input passed.
    monkeypatch.setattr(
        benchmark, "compile_native_clifford_t",
        lambda native, *, error: CliffordTCircuit(1, (CliffordTGate("h", (0,)),), error_budget=error),
    )
    rejected = benchmark.evaluate_candidate(candidate, target, 1e-3)
    assert rejected["t_count"] == 0
    assert rejected["status"] == "FAIL_FINAL_ERROR"


def test_selection_ignores_invalid_cheap_candidates_and_estimated_costs(monkeypatch):
    epsilon = 1e-6
    case = benchmark.Case("selection", "test", {"Z": 0.3})
    candidates = [
        CompilerCandidate("invalid-cheap", None),
        CompilerCandidate("expensive-valid", None, {"estimated_t_count": 1}),
        CompilerCandidate("cheapest-valid", None, {"estimated_t_count": 99}),
    ]

    def build(actual_case, method, target, rotation_error):
        assert actual_case is case and method == "lizzy-bdi"
        assert rotation_error == epsilon / 2
        return candidates

    def evaluate(candidate, target, final_error):
        assert final_error == epsilon
        count = {"invalid-cheap": 0, "expensive-valid": 7, "cheapest-valid": 3}[candidate.name]
        return {
            "name": candidate.name, "settings": candidate.metadata,
            "status": "FAIL_FINAL_ERROR" if candidate.name == "invalid-cheap" else "PASS",
            "t_count": count, "cx_count": 0, "t_depth": count,
            "final_error": {"strict": 0.0, "phase_aligned": 0.0},
        }

    monkeypatch.setattr(benchmark, "build_candidates", build)
    monkeypatch.setattr(benchmark, "evaluate_candidate", evaluate)
    result = benchmark.measure(case, "lizzy-bdi", np.eye(2), epsilon)
    assert result["status"] == "PASS"
    assert result["selected"] == "cheapest-valid"
    assert result["t_count"] == 3
    assert [row["name"] for row in result["candidates"]] == [item.name for item in candidates]
    assert result["candidates"][0]["status"] == "FAIL_FINAL_ERROR"


def test_pairwise_results_use_only_common_passing_cases_at_each_accuracy():
    def row(case, method, count, *, status="PASS", epsilon=1e-6):
        return {"case": case, "method": method, "epsilon": epsilon,
                "status": status, "t_count": count}

    rows = [
        row("first-win", "a", 10), row("first-win", "b", 20),
        row("tie", "a", 5), row("tie", "b", 5),
        row("unsupported", "a", 0, status="UNSUPPORTED"), row("unsupported", "b", 100),
        row("failed", "a", 9), row("failed", "b", 0, status="FAIL"),
        row("missing", "a", 1),
        row("first-win", "a", 30, epsilon=1e-4),
        row("first-win", "b", 20, epsilon=1e-4),
    ]
    tight, loose = benchmark.pairwise_summary(rows, ["a", "b"], [1e-6, 1e-4])
    assert tight["common_cases"] == ["first-win", "tie"]
    assert (tight["first_wins"], tight["ties"], tight["second_wins"]) == (1, 1, 0)
    assert loose["common_cases"] == ["first-win"]
    assert (loose["first_wins"], loose["ties"], loose["second_wins"]) == (0, 0, 1)
    assert tight["epsilon"] == pytest.approx(1e-6)
    assert loose["epsilon"] == pytest.approx(1e-4)


def test_public_auto_output_is_checked_as_delivered_without_recompilation(monkeypatch):
    epsilon = 1e-6
    artifact = CliffordTCircuit(1, (), error_budget=epsilon / 2)
    candidate = CompilerCandidate("auto", NativeCircuit(1), compiled=artifact)

    def no_recompile(*args, **kwargs):
        pytest.fail("The public selector's retained artifact must be measured as delivered")

    monkeypatch.setattr(benchmark, "compile_native_clifford_t", no_recompile)
    result = benchmark.evaluate_candidate(candidate, np.eye(2), epsilon)
    assert result["status"] == "PASS" and result["t_count"] == 0
    assert benchmark.evaluate_candidate(candidate, np.eye(2), 1e-4)["status"] == "FAIL_PRECOMPILED_CONTRACT"
    wrong = CompilerCandidate(
        "wrong-delivered-artifact", NativeCircuit(1),
        compiled=CliffordTCircuit(1, (CliffordTGate("h", (0,)),), error_budget=epsilon / 2),
    )
    assert benchmark.evaluate_candidate(wrong, np.eye(2), epsilon)["status"] == "FAIL_FINAL_ERROR"


def test_search_caps_preserve_settings_and_do_not_count_as_compiler_failure(monkeypatch):
    case = benchmark.Case("capped", "test", {"Z": 0.3})

    def capped(*args):
        error = TimeoutError("declared search budget exhausted")
        error.metadata = {"search": [{"reps": 128, "status": "CAP"}]}
        raise error

    monkeypatch.setattr(benchmark, "build_candidates", capped)
    result = benchmark.measure(case, "qiskit-pf", np.eye(2), 1e-6)
    assert result["status"] == "CAP"
    assert result["settings"]["search"][0]["reps"] == 128
