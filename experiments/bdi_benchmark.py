"""Compare paper-ordered horizontal BDI and Givens on the same Hamiltonians.

Run ``python -m experiments.bdi_benchmark --json``. Dense references are only
verification oracles, never inputs to synthesis. Each route retains its own
mapping and index ordering; both emit independent native Pauli ladders. BDI
prepares a time-independent Cartan factorization and reuses it across evolution
times. Preparation and prepared evaluation have separate clocks. This fixed
corpus measures an empirical trade-off, not a universal winner.
"""

import argparse
import json
import os
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy import exact
from lizzy.classify import summands
from lizzy.dense import circuit_matrix, evolution
from lizzy.hamiltonian import (
    Circuit,
    fold_phases,
    hamiltonian,
    model,
    n_qubits,
    terms_of,
)
from lizzy.native import ladder_circuit, native_frame_circuit

from ._validation import _errors, _versions

SEED = 20261007
METHODS = ("givens", "bdi")
ERROR_THRESHOLD = 1e-8


@dataclass(frozen=True)
class Case:
    name: str
    hamiltonian: PauliStringLinear
    time: float
    seed: int | None = SEED
    note: str = ""


def cases():
    """Fixed families, widths, coefficients and times, including adverse regimes."""
    for family in ("tfim", "tfxy", "xy"):
        for width in (3, 4, 5, 6):
            operator = model(family, width, seed=SEED)
            for time in (0.1, 1.0, 5.0):
                yield Case(f"{family}{width}-t{time:g}", operator, time)
    for family, width in (("tfim", 4), ("tfxy", 4), ("xy", 5)):
        yield Case(f"uniform-{family}{width}", model(family, width), 1.0,
                   seed=None, note="All nonzero coefficients are one.")
    yield Case("negative-tfim4", model("tfim", 4, seed=SEED), -1.0)
    terms = [(str(p), c.real) for c, p in terms_of(model("tfim", 3, seed=SEED))]
    terms.append(("XII", 0.41))
    yield Case("odd-so7-boundary", hamiltonian(terms), 1.0,
               note="A boundary field extends the quadratic so(6) algebra to so(7).")


def _prepare(case):
    """Only the commuting-summand split is shared between the two methods."""
    start = perf_counter()
    width = n_qubits(case.hamiltonian)
    if width > 6:
        raise ValueError("Dense comparison is capped at six qubits.")
    parts = summands(case.hamiltonian)
    return parts, {
        "shared_summand_seconds": perf_counter() - start,
        "summands": len(parts),
    }


def compile_parts(parts, time, method):
    """Use the public exact compiler with the same pre-split components."""
    circuit = Circuit()
    for part in parts:
        circuit.extend(exact.decompose(part, time, method=method))
    return circuit


def _failure(status, exc):
    return {"status": status, "detail": f"{type(exc).__name__}: {exc}"}


def _prepare_routes(parts, width):
    """Keep method-specific mapping/factorization preparation outside evaluation."""
    outcomes = {method: {"prepared_evaluation_samples_seconds": []} for method in METHODS}
    compilers = {}
    for method in METHODS:
        start = perf_counter()
        try:
            if method == "givens":
                mappings = []
                for part in parts:
                    words = tuple(str(p) for _, p in terms_of(part))
                    # A process cache from earlier times must not turn preparation
                    # into a cache lookup in one route but actual work in the other.
                    exact._irrep_cache.pop(words, None)
                    _, _, order, size = exact._irrep(words, width)
                    mappings.append({
                        "mapping_kind": "givens-low-weight", "irrep_size": size,
                        "order": list(order), "cache_hit": False,
                        "parameter_bound": size * (size - 1) // 2,
                    })

                def compile_givens(time):
                    return compile_parts(parts, time, "givens")

                compilers[method] = compile_givens
            else:
                plans = [exact.prepare_bdi(part, cache=False) for part in parts]
                mappings = [{
                    "mapping_kind": plan.mapping_kind,
                    "irrep_size": plan.irrep_size,
                    "partition": None if plan.partition is None else list(plan.partition),
                    "parameter_bound": plan.parameter_bound,
                    "cache_hit": False,
                    "phase_preserving": plan.phase_preserving,
                } for plan in plans]

                def compile_bdi(time):
                    circuit = Circuit()
                    for plan in plans:
                        circuit.extend(plan.circuit(time, route="exact-bdi"))
                    return circuit

                compilers[method] = compile_bdi
            outcomes[method]["mappings"] = mappings
            outcomes[method]["phase_preserving"] = (
                method == "bdi" and all(plan.phase_preserving for plan in plans)
            )
        except Exception as exc:
            outcomes[method].update(_failure("FAIL_PREPARATION", exc))
        outcomes[method]["preparation_seconds"] = perf_counter() - start
    return outcomes, compilers


def _compile_pair(outcomes, compilers, time, repeats):
    """Measure first evaluation, then warmed reuse; setup/emission are excluded."""
    circuits = {}
    for method in METHODS:
        row = outcomes[method]
        if "status" in row:
            continue
        start = perf_counter()
        try:
            circuits[method] = compilers[method](time)
        except Exception as exc:  # A failed backend must remain visible in the report.
            row.update(_failure("FAIL_COMPILE", exc))
        else:
            row["first_evaluation_seconds"] = perf_counter() - start
            row["preparation_plus_first_evaluation_seconds"] = (
                row["preparation_seconds"] + row["first_evaluation_seconds"]
            )
    for repeat in range(repeats):
        order = METHODS if repeat % 2 == 0 else tuple(reversed(METHODS))
        for method in order:
            row = outcomes[method]
            if "status" in row:
                continue
            start = perf_counter()
            try:
                circuit = compilers[method](time)
            except Exception as exc:
                row.update(_failure("FAIL_COMPILE", exc))
            else:
                row["prepared_evaluation_samples_seconds"].append(perf_counter() - start)
                circuits[method] = circuit
    for method, row in outcomes.items():
        if row["prepared_evaluation_samples_seconds"]:
            row["prepared_evaluation_median_seconds"] = median(
                row["prepared_evaluation_samples_seconds"]
            )
    return outcomes, circuits


def assess(circuit, width, target, *, require_strict_phase=False):
    """Count concrete gates; check each route's target-phase contract and emission."""
    start = perf_counter()
    raw_rotations = len(circuit)
    circuit = fold_phases(circuit)
    postprocessing_seconds = perf_counter() - start
    start = perf_counter()
    emitted = ladder_circuit(circuit, width)
    emission_seconds = perf_counter() - start
    start = perf_counter()
    logical = circuit_matrix(circuit, width)
    actual = emitted.get_unitary()
    discrepancy = float(np.linalg.norm(actual - logical, 2))
    aligned, strict = _errors(actual, target)
    target_error = strict if require_strict_phase else aligned
    passed = (np.isfinite(discrepancy) and discrepancy <= ERROR_THRESHOLD
              and np.isfinite(target_error) and target_error <= ERROR_THRESHOLD)
    result = {
        "status": "PASS" if passed else "FAIL_ACCURACY",
        "raw_rotations": raw_rotations,
        "rotations": len(circuit),
        "cx": emitted.two_qubit_gates,
        "native_gates": len(emitted.gates),
        "emission": "native-ladder",
        "target_phase_contract": "strict" if require_strict_phase else "up-to-global-phase",
        "aligned_error": aligned if np.isfinite(aligned) else None,
        "strict_error": strict if np.isfinite(strict) else None,
        "emission_discrepancy": discrepancy if np.isfinite(discrepancy) else None,
        "postprocessing_seconds": postprocessing_seconds,
        "emission_seconds": emission_seconds,
        "verification_seconds": perf_counter() - start,
    }
    # The same existing generic optimization is available to both methods. It
    # does not implement the paper's specialized two-qubit XY/YX gate fusion.
    start = perf_counter()
    framed = native_frame_circuit(circuit, width, lookahead=8, discount_rate=0.9)
    frame_emission_seconds = perf_counter() - start
    start = perf_counter()
    framed_matrix = framed.get_unitary()
    frame_discrepancy = float(np.linalg.norm(framed_matrix - logical, 2))
    frame_aligned, frame_strict = _errors(framed_matrix, target)
    frame_error = frame_strict if require_strict_phase else frame_aligned
    frame_passed = (
        np.isfinite(frame_discrepancy) and frame_discrepancy <= ERROR_THRESHOLD
        and np.isfinite(frame_error) and frame_error <= ERROR_THRESHOLD
    )
    result["native_frame"] = {
        "status": "PASS" if frame_passed else "FAIL_ACCURACY",
        "emission": "native-frame", "cx": framed.two_qubit_gates,
        "native_gates": len(framed.gates),
        "aligned_error": frame_aligned if np.isfinite(frame_aligned) else None,
        "strict_error": frame_strict if np.isfinite(frame_strict) else None,
        "emission_discrepancy": frame_discrepancy if np.isfinite(frame_discrepancy) else None,
        "emission_seconds": frame_emission_seconds,
        "verification_seconds": perf_counter() - start,
    }
    if passed and frame_passed:
        best = min((result, result["native_frame"]),
                   key=lambda item: (item["cx"], item["native_gates"]))
        result["native_portfolio"] = {
            key: best[key] for key in ("emission", "cx", "native_gates")
        }
    else:
        result["status"] = "FAIL_ACCURACY"
    return result


def run(selected=None, *, repeats=3):
    """Return all paired outcomes, preserving setup, compile and accuracy failures."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    selected = list(cases()) if selected is None else list(selected)
    rows, inventory = [], []
    for case in selected:
        metadata = {
            "name": case.name, "width": n_qubits(case.hamiltonian),
            "time": case.time, "seed": case.seed, "note": case.note,
            "terms": [[str(p), float(c.real)] for c, p in terms_of(case.hamiltonian)],
        }
        inventory.append(metadata)
        try:
            parts, setup = _prepare(case)
            metadata.update(setup)
        except Exception as exc:
            rows.extend({"case": case.name, "method": method,
                         **_failure("FAIL_SETUP", exc)} for method in METHODS)
            continue
        outcomes, compilers = _prepare_routes(parts, metadata["width"])
        outcomes, circuits = _compile_pair(outcomes, compilers, case.time, repeats)
        start = perf_counter()
        try:
            target = evolution(case.hamiltonian, case.time)
        except Exception as exc:
            for row in outcomes.values():
                if "status" not in row:
                    row.update(_failure("FAIL_REFERENCE", exc))
        else:
            metadata["reference_seconds"] = perf_counter() - start
            for method, row in outcomes.items():
                if "status" in row:
                    continue
                try:
                    row.update(assess(circuits[method], metadata["width"], target,
                                      require_strict_phase=row["phase_preserving"]))
                except Exception as exc:
                    row.update(_failure("FAIL_ASSESSMENT", exc))
        rows.extend({"case": case.name, "method": method, **outcomes[method]}
                    for method in METHODS)
    return {
        "configuration": {
            "seed": SEED, "repeats": repeats, "error_threshold": ERROR_THRESHOLD,
            "methods": list(METHODS),
            "emission": "same native-ladder and native-frame (lookahead=8, discount_rate=0.9); separate counts and best-of-two portfolio",
            "mapping": "Givens low-weight order; BDI generic paper graph mapping and horizontal partition, without Hamiltonian-family special cases",
            "accuracy": "strict operator norm for horizontal BDI; trace-phase-aligned operator norm for Givens or any nonhorizontal BDI fallback",
            "emission_accuracy": "strict operator norm against the logical circuit",
            "timing": "cold method preparation and first evaluation recorded separately; then median of warmed alternating prepared evaluations; excludes folding, emission and dense verification",
            "reuse": "BDI evaluations reuse K and Cartan rates; Givens evaluations reuse its mapping but recompute the matrix exponential and elimination; these are different reuse strategies, not isolated factorization timings",
            "internal_validation": "BDI preparation includes checked recursive_bdi decomposition of time-independent K; prepared evaluation does not repeat it",
            "setup": "only summand splitting is shared; Givens mapping cache entries evicted before each preparation; BDI prepare_bdi(cache=False) bypasses plan cache",
            "postprocessing": "same fold_phases(tolerance=1e-12) on both; no optional SDK optimization",
            "paper_gate_counts": "native-frame is a generic existing emitter, not specialized paper XY/YX fusion or hardware routing",
            "historical_baseline": "experiments/bdi_results.json is the unchanged earlier shared-order endpoint-factorization experiment, not this corrected implementation",
            "scope": "fixed <=6-qubit corpus, not a universal ranking or scaling claim",
            "thread_environment": {
                name: os.environ.get(name) for name in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
                )
            },
        },
        "versions": _versions(), "cases": inventory, "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--case", action="append", help="repeat to select named cases")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    selected = list(cases())
    if args.case:
        unknown = set(args.case) - {case.name for case in selected}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        selected = [case for case in selected if case.name in args.case]
    try:
        report = run(selected, repeats=args.repeats)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("Same Hamiltonians; method-specific mapping and ordering; native ladders and frames.")
        print("Cold preparation and prepared evaluation are separate; BDI reuses K across times.")
        for row in report["rows"]:
            elapsed = row.get("prepared_evaluation_median_seconds")
            timing = "-" if elapsed is None else f"{elapsed:.6f}"
            preparation = row.get("preparation_seconds")
            setup = "-" if preparation is None else f"{preparation:.6f}"
            frame_cx = row.get("native_frame", {}).get("cx", "-")
            print(f"{row['case']:24} {row['method']:7} {row['status']:17} "
                  f"rot={row.get('rotations', '-'):>4} CX={row.get('cx', '-'):>4} "
                  f"frame_CX={frame_cx:>4} prepare_s={setup} reuse_s={timing} "
                  f"error={row.get('aligned_error', '-')} {row.get('detail', '')}")
    return int(any(row["status"] != "PASS" for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
