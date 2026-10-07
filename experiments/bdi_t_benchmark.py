"""Small, reproducible T-cost comparison of reference and gauge-optimized BDI.

Run with the optional ``ft`` extra. Both candidates keep the paper recursion;
the gauge search is estimated, but selection and reporting use real T gates.
Dense matrices verify the output and are never synthesis inputs.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from lizzy.dense import evolution
from lizzy.hamiltonian import hamiltonian, model, n_qubits, terms_of
from lizzy.synthesize import synthesize


def cases():
    """Include favorable unbalanced inputs and no-search comparison controls."""
    star = {"XI": 0.37, "YI": -0.61, "ZX": 0.83, "ZY": 1.13, "ZZ": -0.29}
    yield "unbalanced-so6", hamiltonian(star), 0.713, True
    yield "negative-so6", hamiltonian(star), -2.7, True
    yield "unbalanced-so5", hamiltonian({"XI": 0.2, "YI": -0.31, "ZX": 0.47,
                                        "ZY": 0.63}), 0.81, True
    yield "balanced-tfim3", model("tfim", 3, seed=20261007), 0.81, True
    words = ["".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")]
    yield "general-so6", hamiltonian({word: (i + 1) / 29 for i, word in enumerate(words)}), \
        0.81, False


def measure(name, operator, time, phase_preserving, error):
    start = perf_counter()
    result = synthesize(operator, time, error=error, method="bdi", objective="t")
    seconds = perf_counter() - start
    emitted = result.emitted_circuit
    actual, target = emitted.get_unitary(), evolution(operator, time)
    phase = np.trace(target.conj().T @ actual)
    phase = phase / abs(phase) if abs(phase) else 1.0
    strict = float(np.linalg.norm(actual - target, ord=2))
    aligned = float(np.linalg.norm(actual - phase * target, ord=2))
    achieved = strict if phase_preserving else aligned
    selection = result.t_selection
    counts = dict(selection.candidate_t_counts)
    return {
        "case": name,
        "width": n_qubits(operator),
        "terms": {str(p): float(c.real) for c, p in terms_of(operator)},
        "time": time,
        "phase_preserving": phase_preserving,
        "candidate_t_counts": counts,
        "selected": selection.selected,
        "t_count": result.t_count,
        "cx_count": result.two_qubit_gates,
        "logical_rotations": len(result.circuit),
        "gauge_search": [asdict(record) for record in selection.bdi_optimizations],
        "rotation_error_bound": emitted.rotation_error_bound,
        "strict_operator_error": strict,
        "phase_aligned_operator_error": aligned,
        "end_to_end_seconds": seconds,
        "status": "PASS" if achieved <= error and result.t_count <= counts["reference"]
        else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="names", help="Case name; repeatable")
    parser.add_argument("--error", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, help="Write the full JSON measurement record")
    args = parser.parse_args()
    selected = [case for case in cases() if not args.names or case[0] in args.names]
    unknown = set(args.names or ()) - {case[0] for case in cases()}
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    records = []
    for case in selected:
        try:
            row = measure(*case, args.error)
        except Exception as exc:  # Record failures instead of removing difficult cases.
            row = {"case": case[0], "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}
        records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    versions = {}
    for package in ("numpy", "scipy", "kak_tools", "paulie", "pygridsynth"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        "rotation_error_budget": args.error,
        "versions": versions,
        "protocol": "Reference and gauge-optimized BDI; same recursive order and same total "
                    "rotation budget. Actual compiled T/Tdg counts; numerical dense validation. "
                    "Times include warm caches and are not a speed comparison. No formal global "
                    "error certificate or FlagSynth comparison.",
        "results": records,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if all(row["status"] == "PASS" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
