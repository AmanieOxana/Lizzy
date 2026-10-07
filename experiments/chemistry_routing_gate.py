"""Analysis-only oracle stop gate for the frozen chemistry routing experiment.

No predictor is fitted here. Input tasks have ``case``, ``time``, and ``costs``
(route -> minimum validated CX, or None if no candidate passed). Missing tasks
remain part of the expected denominator. See chemistry_routing_protocol.md.
"""

import argparse
import json
import math
from pathlib import Path

ROUTES = ("jw", "bk", "df")
CASES = ("H2-4", "LiH-8", "BH-10", "LiH-12")
TIMES = (0.25, 1.0)


def tasks_from_measurements(data):
    """Collapse the frozen step grid, retaining absent/non-passing routes as None."""
    if data["configuration"]["threshold"] != 1e-3:
        raise ValueError("measurement threshold differs from the frozen protocol")
    if data["configuration"].get("backend_timeout_seconds") != 20.0:
        raise ValueError("backend timeout differs from the frozen 20-second budget")
    if data["configuration"].get("steps") != [1, 2, 3]:
        raise ValueError("step grid differs from the frozen protocol")
    tasks = {(case, time): {"case": case, "time": time, "costs": dict.fromkeys(ROUTES)}
             for case in CASES for time in TIMES}
    seen = set()
    for row in data["rows"]:
        key = (row["case"], float(row["time"]))
        candidate = (*key, row["route"], row["steps"])
        if key not in tasks or row["route"] not in ROUTES or row["steps"] not in (1, 2, 3):
            raise ValueError("candidate outside the frozen grid")
        if candidate in seen:
            raise ValueError("duplicate formula candidate")
        seen.add(candidate)
        if row["status"] != "PASS":
            continue
        error = row["max_infidelity"]
        if error is None or not math.isfinite(error) or not 0 <= error <= 1e-3:
            raise ValueError("PASS row violates the frozen accuracy threshold")
        cost = row["two_qubit_gates"]
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
            raise ValueError("PASS row lacks a valid concrete CX count")
        previous = tasks[key]["costs"][row["route"]]
        tasks[key]["costs"][row["route"]] = cost if previous is None else min(cost, previous)
    expected = {(case, time, route, steps) for case in CASES for time in TIMES
                for route in ROUTES for steps in (1, 2, 3)}
    if seen != expected:
        raise ValueError("incomplete measurement grid: retain all 72 candidates, including failures")
    return list(tasks.values())


def oracle_gate(tasks):
    """Evaluate the predeclared coverage and oracle-headroom stop criteria."""
    by_key = {}
    for task in tasks:
        key = (task["case"], float(task["time"]))
        if key in by_key:
            raise ValueError(f"duplicate task: {key}")
        if key not in {(name, time) for name in CASES for time in TIMES}:
            raise ValueError(f"outside the frozen corpus: {key}")
        costs = {route: task["costs"].get(route) for route in ROUTES}
        for cost in costs.values():
            if cost is not None and (isinstance(cost, bool) or not isinstance(cost, int) or cost < 0):
                raise ValueError("CX costs must be non-negative integers or None")
        by_key[key] = {"case": key[0], "time": key[1], "costs": costs}

    table, common = [], []
    for name in CASES:
        for time in TIMES:
            task = by_key.get((name, time), {"case": name, "time": time,
                                            "costs": dict.fromkeys(ROUTES)})
            feasible = all(task["costs"][route] is not None for route in ROUTES)
            table.append({**task, "common_feasible": feasible})
            if feasible:
                common.append(task)
    result = {"tasks": table, "expected_tasks": 8, "common_coverage": len(common),
              "selector_fitted": False, "zero_cost_floor_used": False,
              "best_fixed_route": None, "oracle_saving": None,
              "strict_winner_routes": []}
    holdouts = {row["case"] for row in common} >= {"BH-10", "LiH-12"}
    training = sum(row["case"] in {"H2-4", "LiH-8"} for row in common) == 4
    if len(common) < 6 or not holdouts or not training:
        return {**result, "verdict": "INCONCLUSIVE_COVERAGE",
                "reason": "Insufficient common coverage: need six tasks, all four training rows and both holdouts; stop at the fixed budget."}

    def log_cost(cost):
        return math.log(max(1, cost))

    means = {route: sum(log_cost(row["costs"][route]) for row in common) / len(common)
             for route in ROUTES}
    best = min(ROUTES, key=lambda route: means[route])
    oracle_mean = sum(log_cost(min(row["costs"].values())) for row in common) / len(common)
    saving = 1 - math.exp(oracle_mean - means[best])
    strict_winners = set()
    for row in common:
        costs = row["costs"]
        winner = min(ROUTES, key=lambda route: costs[route])
        if sum(cost == costs[winner] for cost in costs.values()) == 1:
            strict_winners.add(winner)
    result.update(best_fixed_route=best, oracle_saving=saving,
                  fixed_geomean_cx={route: math.exp(value) for route, value in means.items()},
                  oracle_geomean_cx=math.exp(oracle_mean),
                  strict_winner_routes=[route for route in ROUTES if route in strict_winners],
                  zero_cost_floor_used=any(cost == 0 for row in common for cost in row["costs"].values()))
    if saving <= 0.10 or len(strict_winners) < 2:
        return {**result, "verdict": "NO_GO_HEADROOM",
                "reason": "Even oracle route choice fails the >10% headroom/two-winner gate; no feature model fitted."}
    return {**result, "verdict": "PROCEED_TO_FROZEN_ABLATION",
            "reason": "Headroom exists; this alone is not a successful selector or a novelty result."}


def target_accuracy_sensitivity(data):
    """Post-hoc diagnostic, never a replacement for the strict primary gate.

    Admit emission-mismatch failures only when the measured emitted circuit
    still passes the original reference-state target and normalization check.
    No timeout, missing artifact, factorization change, or accuracy failure is
    converted to a win. No selector is fitted.
    """
    tasks_from_measurements(data)  # Reject altered budgets and incomplete grids.
    tasks = {(case, time): {"case": case, "time": time, "costs": dict.fromkeys(ROUTES)}
             for case in CASES for time in TIMES}
    fingerprints = {case["name"]: case["factorization_fingerprint"] for case in data["cases"]}
    admitted = 0
    for row in data["rows"]:
        for record in row["backends"]:
            if record["status"] not in {"PASS", "FAIL_VERIFICATION"}:
                continue
            numbers = [record.get(key) for key in
                       ("max_infidelity", "normalization_error", "emission_mismatch_infidelity")]
            if any(value is None or not math.isfinite(value) or value < 0 for value in numbers):
                continue
            if numbers[0] > 1e-3 or numbers[1] > 1e-8:
                continue
            errors = record.get("state_infidelities", [])
            if len(errors) != 4 or any(not math.isfinite(error) or not 0 <= error <= 1e-3 for error in errors):
                continue
            if max(errors) != numbers[0]:
                continue
            if record["status"] == "FAIL_VERIFICATION" and numbers[2] <= 1e-9:
                continue  # Only the explicitly relaxed mismatch guard may fail.
            if row["route"] == "df" and record.get("factorization_fingerprint") != fingerprints[row["case"]]:
                continue
            cost = record.get("two_qubit_gates")
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                continue
            admitted += record["status"] == "FAIL_VERIFICATION"
            task = tasks[(row["case"], float(row["time"]))]
            previous = task["costs"][row["route"]]
            task["costs"][row["route"]] = cost if previous is None else min(previous, cost)
    comparison = oracle_gate(list(tasks.values()))
    comparison.pop("verdict")
    comparison.pop("reason")
    return {"label": "POST_HOC_TARGET_ACCURACY_ONLY_NOT_PRIMARY",
            "caveat": "Ignores the logical/emission mismatch threshold; cannot authorize fitting or establish feature value.",
            "additional_backend_artifacts_admitted": admitted, **comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    data = json.loads(args.measurements.read_text())
    print(json.dumps({"primary": oracle_gate(tasks_from_measurements(data)),
                      "posthoc_target_accuracy_only": target_accuracy_sensitivity(data)}, indent=2))


if __name__ == "__main__":
    main()
