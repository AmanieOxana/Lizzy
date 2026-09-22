"""The broader comparison must expose exclusions, costs, and measured errors."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lizzy import synthesis_bench as bench
from lizzy.driven import DrivenHamiltonian, _closure
from lizzy.hamiltonian import Circuit


def test_case_families_are_reproducible_and_include_a_full_closure_cap():
    first = {case.name: case for case in bench.benchmark_cases()}
    second = {case.name: case for case in bench.benchmark_cases()}
    assert len(first) >= 12
    assert {"short-tfim3", "long-tfim3", "static-su4", "driven-su4",
            "su8-closure-cap", "commuting-su2", "driven-commuting"} <= first.keys()
    for name, case in first.items():
        assert case.hamiltonian.n_qubits <= 3
        assert np.array_equal(case.target, second[name].target)
    assert len(_closure(first["static-su4"].hamiltonian.paulis, 64)) == 15
    assert len(_closure(first["su8-closure-cap"].hamiltonian.paulis, 64)) == 63


def test_commuting_and_short_time_cases_pass_all_available_routes():
    selected = [case for case in bench.benchmark_cases()
                if case.name in {"static-commuting", "short-tfim3"}]
    rows = bench.run_suite(selected)
    assert len(rows) == 6
    assert all(row.status == "PASS" for row in rows)
    assert all(row.op_error <= 1e-6 for row in rows)
    assert all(row.emitted_gates >= row.emitted_cx >= 0 for row in rows)
    assert all(row.compile_seconds >= 0 and row.verification_seconds >= 0 for row in rows)
    short = {row.route: row for row in rows if row.case == "short-tfim3"}
    assert short["midpoint2"].steps == 1
    assert short["midpoint2"].emitted_cx < short["wei-norman"].emitted_cx


def test_closure_and_search_caps_and_unsupported_route_are_explicit():
    case = next(case for case in bench.benchmark_cases() if case.name == "su8-closure-cap")
    rows = bench.run_suite([case], max_dimension=32, max_steps=1)
    assert [row.status for row in rows] == ["CAP", "FAIL_STEP_CAP", "UNSUPPORTED"]
    assert rows[0].rotations is None and rows[0].emitted_cx is None
    assert "max_dimension" in rows[0].detail
    assert rows[1].op_error > 1e-6


def test_rhs_budget_failure_does_not_suppress_other_methods():
    case = bench._static_case("small", ["X", "Z"], [0.6, -0.2], 0.1)
    rows = bench.run_suite([case], max_rhs_evaluations=1)
    assert rows[0].status == "FAIL_COMPILE"
    assert all(row.status == "PASS" for row in rows[1:])


def test_driven_case_has_explicit_unsupported_static_route():
    driven = DrivenHamiltonian(["Z"], lambda t: [0.3 * t])
    case = bench.BenchmarkCase("linear", driven, (0.0, 1.0),
                               np.diag(np.exp([-0.15j, 0.15j])), "analytic")
    rows = bench.run_suite([case])
    assert [row.status for row in rows] == ["PASS", "PASS", "UNSUPPORTED"]


def test_only_old_exact_route_can_pass_with_an_unmatched_global_phase():
    case = bench.BenchmarkCase("phase", DrivenHamiltonian(["Z"], lambda t: [0.0]),
                               (0.0, 1.0), -np.eye(2), "phase regression")
    exact = bench._assess(case, "static-exact", Circuit(), 1, 1e-6, 0.0)
    wn = bench._assess(case, "wei-norman", Circuit(), 1, 1e-6, 0.0)
    assert exact.status == "PASS" and exact.op_error == 0 and exact.strict_error == 2
    assert wn.status == "FAIL_ACCURACY"


def test_emission_mismatch_cannot_be_hidden_by_target_agreement(monkeypatch):
    case = bench.BenchmarkCase("wrong-emitter", DrivenHamiltonian(["X"], lambda t: [0.0]),
                               (0.0, 1.0), -np.eye(2), "test")
    fake = SimpleNamespace(gates=[], two_qubit_gates=0, get_unitary=lambda: -np.eye(2))
    monkeypatch.setattr(bench, "_emission", lambda *args: ("fake", fake))
    row = bench._assess(case, "midpoint2", Circuit(), 1, 1e-6, 0.0)
    assert row.status == "FAIL_ACCURACY"
    assert "emission/logical mismatch" in row.detail


def test_json_reports_costs_and_time_fields(monkeypatch, capsys):
    case = bench._static_case("small", ["Z"], [0.3], 0.1)
    monkeypatch.setattr(bench, "benchmark_cases", lambda: iter([case]))
    assert bench.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["rows"]) == 3
    assert payload["configuration"]["seed"] == 20260921
    assert "python" in payload["versions"] and "numpy" in payload["versions"]
    assert {"emitted_cx", "emitted_gates", "strict_error", "op_error", "compile_seconds",
            "search_seconds", "verification_seconds"} <= payload["rows"][0].keys()


def test_known_exact_embedding_refusal_is_unsupported(monkeypatch):
    case = bench._static_case("small", ["X", "Z"], [0.6, -0.2], 0.1)
    def refuse(*args, **kwargs):
        raise ValueError("BDI embedding unavailable")
    monkeypatch.setattr(bench.exact, "decompose", refuse)
    rows = bench.run_suite([case])
    assert rows[-1].status == "UNSUPPORTED"
    assert "BDI embedding unavailable" in rows[-1].detail


def test_product_search_rechecks_final_artifact_accuracy(monkeypatch):
    case = bench._static_case("small", ["Z"], [0.3], 0.1)
    checked = []
    def assess(case, route, circuit, dimension, error, elapsed, *, steps, **kwargs):
        checked.append(steps)
        return bench.BenchmarkRow(case.name, route, "PASS" if steps == 2 else "FAIL_ACCURACY",
                                  dimension, steps=steps, detail="accuracy" if steps == 1 else "")
    monkeypatch.setattr(bench, "_assess", assess)
    result = bench._product_formula(case, 1, 1e-6, 2)
    assert checked == [1, 2]
    assert result.status == "PASS" and result.steps == 2


def test_benchmark_rejects_non_small_dense_workloads():
    case = bench.BenchmarkCase("too-wide", DrivenHamiltonian(["XXXX"], lambda t: [1.0]),
                               (0.0, 1.0), np.eye(16), "unused")
    with pytest.raises(ValueError, match="three qubits"):
        bench.run_suite([case])


@pytest.mark.parametrize("kwargs", [{"error": 0}, {"error": np.nan},
    {"max_steps": 0}, {"max_steps": 1.5}, {"max_dimension": 0}, {"max_dimension": True}])
def test_invalid_search_settings_raise(kwargs):
    with pytest.raises(ValueError):
        bench.run_suite([], **kwargs)


def test_cli_unknown_case_is_an_error(monkeypatch):
    monkeypatch.setattr(bench, "benchmark_cases", lambda: iter([]))
    with pytest.raises(SystemExit):
        bench.main(["--case", "not-a-case"])
