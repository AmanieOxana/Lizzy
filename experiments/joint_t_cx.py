"""Bounded joint-cost ablation; does not change Lizzy's public selector.

Every stage retains the preceding stage. Compare actual emitted (T, CX) pairs
at the cross-compiler benchmark's accuracy, phase and no-ancilla contract.
Dense matrices are verification references, never inputs to decomposition.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import islice, product
from math import fsum, prod
from pathlib import Path

import numpy as np

from experiments.compiler_comparison import cases, operator_errors
from lizzy import exact, gaussian
from lizzy.classify import summands
from lizzy.clifford_t import compile_native_clifford_t
from lizzy.dense import circuit_matrix, evolution
from lizzy.hamiltonian import Circuit, fold_phases, n_qubits
from lizzy.native import ladder_circuit, native_frame_candidates
from lizzy.synthesize import synthesize

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("delivered", "same-frames", "route-frames", "gauge-frames")
GAUGE_COMBINATION_CAP = 32


def frontier(rows):
    """Keep nondominated passing outputs, with a stable representative per pair."""
    unique = {}
    for row in rows:
        if row["status"] == "PASS":
            unique.setdefault((row["t_count"], row["cx_count"]), row)
    answer, lowest_cx = [], float("inf")
    for (_, cx), row in sorted(unique.items()):
        if cx < lowest_cx:
            answer.append(row)
            lowest_cx = cx
    return answer


def choose(rows, *, priority="t", t_cap=None, cx_cap=None):
    """An explicit resource constraint never changes the accuracy threshold."""
    if priority not in {"t", "cx"}:
        raise ValueError("priority must be 't' or 'cx'")
    eligible = [r for r in frontier(rows)
                if (t_cap is None or r["t_count"] <= t_cap)
                and (cx_cap is None or r["cx_count"] <= cx_cap)]
    key = (lambda r: (r["t_count"], r["cx_count"])) if priority == "t" else (
        lambda r: (r["cx_count"], r["t_count"]))
    return min(eligible, key=key, default=None)


def _point(row):
    return {key: row[key] for key in ("name", "t_count", "cx_count", "qasm_sha256")}


def summarize(rows):
    reference = next(r for r in rows if r["stage"] == "delivered" and r["status"] == "PASS")
    summaries = {}
    for stage_index, stage in enumerate(STAGES):
        eligible = [r for r in rows if STAGES.index(r["stage"]) <= stage_index]
        safe = choose(eligible, t_cap=reference["t_count"], cx_cap=reference["cx_count"])
        summaries[stage] = {
            "frontier": [_point(r) for r in frontier(eligible)],
            "t_first": _point(choose(eligible)),
            "cx_first": _point(choose(eligible, priority="cx")),
            "no_compromise": _point(safe),
        }
    return summaries


def _join(plans, time):
    logical = Circuit()
    for plan in plans:
        logical.extend(plan.circuit(time))
    return fold_phases(logical, tolerance=0.0)


def measure(case, epsilon):
    """Retain actual artifacts in memory; save all frontier/reference QASM words."""
    operator, width = case.operator, n_qubits(case.operator)
    target, budget = evolution(operator, case.time), epsilon / 2
    baseline = synthesize(operator, case.time, error=budget, objective="t")
    rows, artifacts, cached = [], {}, {}
    notes = []

    def evaluate(name, stage, logical, native, compiled=None):
        # Match the production identity-phase convention, including cancellation
        # of large scalar phases. No final Clifford frame is discarded.
        native.global_phase = fsum(-float(a) for p, a in logical.rotations if not p.get_support())
        digest = hashlib.sha256(native.to_qasm3().encode()).hexdigest()
        record = {"name": name, "stage": stage, "native_sha256": digest}
        try:
            actual = native.get_unitary()
            record["frame_error"] = operator_errors(actual, circuit_matrix(logical, width))
            record["decomposition_error"] = operator_errors(actual, target)
            if record["frame_error"]["strict"] > min(1e-10, budget):
                record["status"] = "FAIL_FRAME"
            elif record["decomposition_error"]["phase_aligned"] > budget:
                record["status"] = "FAIL_DECOMPOSITION"
            elif compiled is None and digest in cached:
                prior = cached[digest]
                # Recheck each logical-to-native mapping even if the identical
                # native artifact has already been compiled against this target.
                record = {**prior, **record, "same_native_as": prior["name"]}
            else:
                emitted = compiled or compile_native_clifford_t(native, error=budget)
                if emitted.error_budget != budget or emitted.width != width:
                    raise ValueError("Wrong retained-artifact width or precision")
                qasm = emitted.to_qasm3()
                qasm_digest = hashlib.sha256(qasm.encode()).hexdigest()
                record.update(
                    t_count=emitted.t_count, cx_count=emitted.n_2qb_gates(),
                    rotation_error_bound=emitted.rotation_error_bound,
                    error_bound_kind=emitted.error_bound_kind,
                    final_error=operator_errors(emitted.get_unitary(), target),
                    qasm_sha256=qasm_digest,
                )
                record["status"] = ("PASS" if record["final_error"]["phase_aligned"] <= epsilon
                                    else "FAIL_FINAL_ERROR")
                artifacts[qasm_digest] = qasm
        except Exception as exc:
            record.update(status="FAIL_COMPILE", detail=f"{type(exc).__name__}: {exc}")
        rows.append(record)
        if "qasm_sha256" in record:
            cached[digest] = record

    def variants(name, stage, logical, *, include_ladder=True):
        logical = fold_phases(logical, tolerance=0.0)
        if include_ladder:
            evaluate(name + "/ladder", stage, logical, ladder_circuit(logical, width))
        seen = set()
        for lookahead in (0, 8):
            try:
                alternatives = native_frame_candidates(logical, width, lookahead=lookahead)
            except Exception as exc:
                notes.append({"stage": stage, "candidate": name, "lookahead": lookahead,
                              "status": "FAIL_FRAME_BUILD", "detail": str(exc)})
                continue
            for label, native in alternatives:
                digest = hashlib.sha256(native.to_qasm3().encode()).hexdigest()
                if digest not in seen:
                    evaluate(f"{name}/{label}-lookahead{lookahead}", stage, logical, native)
                    seen.add(digest)

    logical = baseline.circuit
    evaluate("public-auto", "delivered", logical, ladder_circuit(logical, width), baseline.emitted_circuit)
    if rows[0]["status"] != "PASS":
        return {"case": asdict(case), "status": "FAIL_BASELINE", "candidates": rows}
    variants("selected", "same-frames", logical, include_ladder=False)

    parts, plans = summands(operator), None
    try:
        plans = [exact.prepare_bdi(part) for part in parts]
        variants("bdi-reference", "route-frames", _join(plans, case.time))
        precision = budget / max(1, sum(plan.parameter_bound for plan in plans))
        optimized = [exact.prepare_bdi(part, optimize="t", rotation_error=precision) for part in parts]
        variants("bdi-t-gauge", "route-frames", _join(optimized, case.time))
    except Exception as exc:
        notes.append({"stage": "route-frames", "candidate": "bdi", "detail": str(exc)})
    try:
        givens = Circuit()
        for part in parts:
            if not exact.is_decomposable(part):
                raise NotImplementedError("Givens not eligible for this summand")
            givens.extend(exact.decompose(part, case.time, method="givens"))
        variants("givens-reference", "route-frames", givens)
    except Exception as exc:
        notes.append({"stage": "route-frames", "candidate": "givens", "detail": str(exc)})
    if gaussian.is_gaussian(operator):
        try:
            variants("gaussian-reference", "route-frames", gaussian.decompose(
                operator, case.time, error_tolerance=min(1e-8, 0.1 * budget)))
        except Exception as exc:
            notes.append({"stage": "route-frames", "candidate": "gaussian", "detail": str(exc)})

    search = {"possible_combinations": 0, "evaluated_combinations": 0, "truncated": False}
    if plans is not None:
        try:
            alternatives = [exact.bdi_gauge_candidates(plan) for plan in plans]
            possible = prod(len(items) for items in alternatives)
            search = {"possible_combinations": possible,
                      "evaluated_combinations": 0,
                      "truncated": possible > GAUGE_COMBINATION_CAP}
            # Only a deterministic bounded set is explored. Never discard a
            # component on a local cost: all costs below are whole-circuit costs.
            for combination in islice(product(*alternatives), GAUGE_COMBINATION_CAP):
                labels, selected = zip(*combination, strict=True)
                variants("bdi/" + "+".join(labels), "gauge-frames", _join(selected, case.time))
                search["evaluated_combinations"] += 1
        except Exception as exc:
            notes.append({"stage": "gauge-frames", "candidate": "bdi-gauges", "detail": str(exc)})
    summaries = summarize(rows)
    retained = {p["qasm_sha256"] for stage in summaries.values() for p in stage["frontier"]}
    failed = any(r["status"].startswith("FAIL") for r in rows) or bool(notes)
    return {
        "case": asdict(case), "status": "PARTIAL" if failed else "PASS",
        "public_selection": asdict(baseline.t_selection),
        "gauge_search": search, "notes": notes, "candidates": rows, "stages": summaries,
        "artifacts": {digest: artifacts[digest] for digest in sorted(retained)},
    }


def run(selected, epsilon):
    sources = [Path(__file__), ROOT / "experiments/compiler_comparison.py",
               *sorted((ROOT / "lizzy").glob("*.py"))]
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources}
    records = []
    for case in selected:
        try:
            record = measure(case, epsilon)
        except Exception as exc:
            status = "UNSUPPORTED" if isinstance(exc, NotImplementedError) else "FAIL"
            record = {"case": asdict(case), "status": status, "detail": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(json.dumps({"case": case.name, "status": record["status"],
                          "frontiers": {name: [(p["t_count"], p["cx_count"]) for p in s["frontier"]]
                                        for name, s in record.get("stages", {}).items()}}), flush=True)
    changed = [str(p.relative_to(ROOT)) for p in sources
               if hashlib.sha256(p.read_bytes()).hexdigest() != fingerprints[str(p.relative_to(ROOT))]]
    versions = {}
    for package in ("numpy", "scipy", "paulie", "kak_tools", "pygridsynth", "openfermion"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        "complete": not changed, "measured_at": datetime.now(UTC).isoformat(),
        "epsilon": epsilon, "rotation_error_budget": epsilon / 2,
        "contract": "Full unitary, phase-aligned operator error; strict frame equivalence; "
                    "no ancillas; all-to-all CX; same total gridsynth rotation budget. "
                    "Public selector unchanged. Not a new external compiler comparison.",
        "stages": list(STAGES), "lookaheads": [0, 8],
        "gauge_combination_cap": GAUGE_COMBINATION_CAP,
        "versions": versions,
        "source_sha256": fingerprints, "source_changed_during_run": changed,
        "results": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="names")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declared = list(cases())
    if not np.isfinite(args.epsilon) or not 0 < args.epsilon < 1:
        parser.error("epsilon must be finite and between zero and one")
    unknown = set(args.names or ()) - {c.name for c in declared}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    selected = [c for c in declared if not args.names or c.name in args.names]
    report = run(selected, args.epsilon)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["complete"]
               or any(r["status"] not in {"PASS", "UNSUPPORTED"} for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
