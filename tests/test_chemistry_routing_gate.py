"""A missing or censored result must never manufacture a routing improvement."""

import json
from pathlib import Path

import pytest

from experiments.chemistry_routing_gate import (
    CASES,
    TIMES,
    oracle_gate,
    target_accuracy_sensitivity,
    tasks_from_measurements,
)


def _tasks():
    return [{"case": case, "time": time, "costs": {"jw": 100, "bk": 105, "df": 200}}
            for case in CASES for time in TIMES]


def _measurements():
    return {"configuration": {"threshold": .001, "backend_timeout_seconds": 20.0,
                              "steps": [1, 2, 3]},
            "rows": [{"case": case, "time": time, "route": route, "steps": step,
                      "status": "NO_PASSING_BACKEND", "two_qubit_gates": None,
                      "max_infidelity": None}
                     for case in CASES for time in TIMES for route in ("jw", "bk", "df")
                     for step in (1, 2, 3)]}


def test_marginal_oracle_gain_does_not_trigger_model_building():
    tasks = _tasks()
    tasks[0]["costs"]["bk"] = 90
    result = oracle_gate(tasks)
    assert len(result["strict_winner_routes"]) == 2
    assert 0 < result["oracle_saving"] < 0.10
    assert result["verdict"] == "NO_GO_HEADROOM"
    assert not result["selector_fitted"]


def test_large_headroom_only_permits_an_ablation_not_a_positive_claim():
    tasks = _tasks()
    for task in tasks[:4]:
        task["costs"]["bk"] = 200
    for task in tasks[4:]:
        task["costs"] = {"jw": 200, "bk": 50, "df": 300}
    result = oracle_gate(tasks)
    assert result["oracle_saving"] > 0.10
    assert result["verdict"] == "PROCEED_TO_FROZEN_ABLATION"
    assert not result["selector_fitted"]


def test_missing_tasks_remain_in_coverage_denominator():
    result = oracle_gate(_tasks()[:5])
    assert result["expected_tasks"] == 8
    assert len(result["tasks"]) == 8
    assert result["common_coverage"] == 5
    assert result["verdict"] == "INCONCLUSIVE_COVERAGE"
    assert result["oracle_saving"] is None


def test_capped_route_is_not_treated_as_an_infinite_cost_win():
    tasks = _tasks()
    for task in tasks[4:]:
        task["costs"]["df"] = None
    result = oracle_gate(tasks)
    assert result["common_coverage"] == 4
    assert result["verdict"] == "INCONCLUSIVE_COVERAGE"


def test_missing_holdout_stops_even_with_six_covered_tasks():
    result = oracle_gate([task for task in _tasks() if task["case"] != "BH-10"])
    assert result["common_coverage"] == 6
    assert result["verdict"] == "INCONCLUSIVE_COVERAGE"


def test_missing_training_molecule_stops_even_with_six_covered_tasks():
    result = oracle_gate([task for task in _tasks() if task["case"] != "H2-4"])
    assert result["common_coverage"] == 6
    assert result["verdict"] == "INCONCLUSIVE_COVERAGE"


def test_reducer_never_picks_a_cheaper_inaccurate_candidate():
    rows = [{"case": "H2-4", "time": .25, "route": "jw", "steps": step,
             "status": status, "two_qubit_gates": cost, "max_infidelity": error}
            for step, status, cost, error in [(1, "NO_PASSING_BACKEND", 5, .5),
                                              (2, "PASS", 40, .0001), (3, "PASS", 60, .00001)]]
    data = _measurements()
    data["rows"][:3] = rows
    tasks = tasks_from_measurements(data)
    assert tasks[0]["costs"] == {"jw": 40, "bk": None, "df": None}
    assert oracle_gate(tasks)["verdict"] == "INCONCLUSIVE_COVERAGE"


def test_reducer_rejects_false_pass_and_modified_threshold():
    data = _measurements()
    data["rows"][0].update(status="PASS", two_qubit_gates=5, max_infidelity=.1)
    with pytest.raises(ValueError, match="accuracy"):
        tasks_from_measurements(data)
    data["configuration"]["threshold"] = .2
    with pytest.raises(ValueError, match="threshold"):
        tasks_from_measurements(data)


def test_reducer_requires_failed_candidates_not_silent_omission():
    data = _measurements()
    data["rows"].pop()
    with pytest.raises(ValueError, match="all 72"):
        tasks_from_measurements(data)


def test_posthoc_diagnostic_keeps_primary_unchanged_and_never_admits_bad_accuracy():
    data = _measurements()
    data["cases"] = [{"name": case, "factorization_fingerprint": "same"} for case in CASES]
    for row in data["rows"]:
        row["backends"] = [{"status": "FAIL_VERIFICATION", "two_qubit_gates": 100,
                            "max_infidelity": 1e-5, "state_infidelities": [1e-5]*4,
                            "normalization_error": 1e-14,
                            "emission_mismatch_infidelity": 1e-7,
                            "factorization_fingerprint": "same"}]
    before = oracle_gate(tasks_from_measurements(data))
    secondary = target_accuracy_sensitivity(data)
    assert secondary["common_coverage"] == 8
    assert secondary["oracle_saving"] == pytest.approx(0)
    assert secondary["additional_backend_artifacts_admitted"] == 72
    assert "verdict" not in secondary and not secondary["selector_fitted"]
    assert oracle_gate(tasks_from_measurements(data)) == before
    assert before["verdict"] == "INCONCLUSIVE_COVERAGE"
    for row in data["rows"][:9]:
        row["backends"][0]["max_infidelity"] = .1
    assert target_accuracy_sensitivity(data)["common_coverage"] == 7


@pytest.mark.parametrize("change", [{"normalization_error": .01},
    {"factorization_fingerprint": "changed"}, {"status": "TIMEOUT"},
    {"max_infidelity": float("nan")}, {"emission_mismatch_infidelity": 0}])
def test_posthoc_diagnostic_preserves_other_safety_checks(change):
    data = _measurements()
    data["cases"] = [{"name": case, "factorization_fingerprint": "same"} for case in CASES]
    for row in data["rows"]:
        row["backends"] = [{"status": "FAIL_VERIFICATION", "two_qubit_gates": 10,
                            "max_infidelity": 1e-5, "state_infidelities": [1e-5]*4,
                            "normalization_error": 1e-14,
                            "emission_mismatch_infidelity": 1e-7,
                            "factorization_fingerprint": "same"}]
        if row["route"] == "df":
            row["backends"][0].update(change)
    assert target_accuracy_sensitivity(data)["common_coverage"] == 0


def test_saved_analysis_reproduces_from_unchanged_raw_artifacts():
    directory = Path(__file__).parents[1] / "experiments"
    data = json.loads((directory / "chemistry_routing_results.json").read_text())
    saved = json.loads((directory / "chemistry_routing_analysis.json").read_text())
    assert oracle_gate(tasks_from_measurements(data)) == saved["primary"]
    secondary = target_accuracy_sensitivity(data)
    assert secondary["tasks"] == saved["posthoc_target_accuracy_only"]["tasks"]
    assert secondary["oracle_saving"] == pytest.approx(saved["posthoc_target_accuracy_only"]["oracle_saving"])
    assert saved["primary"]["verdict"] == "INCONCLUSIVE_COVERAGE"
    assert not secondary["selector_fitted"]
