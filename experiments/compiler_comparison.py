"""Matched, ancilla-free Clifford+T comparison on a fixed heterogeneous corpus.

See compiler_comparison_protocol.md. All candidates use the same rotation
compiler, final operator-norm threshold and phase convention. Failures remain
in the record. This is not a runtime or asymptotic-scaling competition.
"""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments._bqskit_adapter import bqskit_candidates
from experiments._compiler_adapters import (
    CompilerCandidate,
    pytket_candidates,
    qiskit_qsd_candidates,
)
from experiments._flagsynth_adapter import flagsynth_sdm_candidates
from experiments._product_formula_adapter import qiskit_product_formula_candidates
from lizzy import exact
from lizzy.classify import summands
from lizzy.clifford_t import compile_native_clifford_t
from lizzy.dense import evolution, operator_errors
from lizzy.driven import AlgebraTooLarge, IntegrationFailure
from lizzy.hamiltonian import (
    Circuit,
    fold_phases,
    hamiltonian,
    model,
    n_qubits,
    terms_of,
)
from lizzy.native import ladder_circuit
from lizzy.synthesize import synthesize
from lizzy.wei_norman import synthesize_wei_norman

SEED = 20261007
METHODS = ("lizzy-auto", "lizzy-bdi", "lizzy-givens", "lizzy-wei-norman",
           "flagsynth-sdm", "qiskit-qsd", "pytket", "bqskit", "qiskit-pf")
WN_OPTIONS = dict(max_dimension=32, max_segments=128, max_rhs_evaluations=20_000,
                  rtol=1e-10, atol=1e-12, max_step=0.05, emission="none")
# Offline extraction from the actual HamLib molecular tensor, not a toy fixture.
H2_TERMS = {
    "XXXX": 0.014034099995004047, "XXYY": 0.014034099995004047,
    "YYXX": 0.014034099995004047, "YYYY": 0.014034099995004047,
    "ZIII": 0.28235088851107365, "ZZII": 0.08211612483661979,
    "ZIZI": 0.16462320552994025, "ZIIZ": 0.09615022483162383,
    "IZII": -0.0039867476924421025, "IZZI": 0.09615022483162383,
    "IZIZ": 0.08366738652351667, "IIZI": 0.28235088851107365,
    "IIZZ": 0.08211612483661979, "IIIZ": -0.003986747692442109,
}


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    terms: dict[str, float]
    time: float = 0.7
    metadata: dict = field(default_factory=dict)

    @property
    def operator(self):
        return hamiltonian(self.terms)


def cases():
    yield Case("commuting-z2", "structured", {"ZI": 0.31, "IZ": -0.47, "ZZ": 0.23})
    yield Case("encoded-su2", "structured", {"XX": 0.31, "XY": -0.47, "IZ": 0.23})
    rng = np.random.default_rng(SEED)
    yield Case("anticommuting-star2", "structured",
               dict(zip(("XI", "YI", "ZX", "ZY", "ZZ"), map(float, rng.normal(size=5)), strict=True)))
    for width in (2, 3):
        words = ["".join(word) for word in product("IXYZ", repeat=width) if set(word) != {"I"}]
        rng = np.random.default_rng(SEED)
        yield Case(f"generic-su{2**width}", "generic",
                   dict(zip(words, map(float, rng.normal(size=len(words)) / np.sqrt(len(words))), strict=True)))
    for family, width in (("tfim", 3), ("tfxy", 3), ("heisenberg", 3),
                          ("tfim", 4), ("tfxy", 4), ("heisenberg_all_to_all", 4)):
        operator = model(family, width, seed=SEED)
        yield Case(f"{family}{width}", "interacting" if "heisenberg" in family else "structured",
                   {str(p): float(c.real) for c, p in terms_of(operator)})
    yield Case("H2-JW4", "chemistry", H2_TERMS, metadata={
        "source": "HamLib chemistry/electronic/standard/H2.hdf5",
        "dataset": "ham_molec-4", "encoding": "OpenFermion JW alpha-then-beta",
        "scalar_identity_removed": True, "geometry": "unavailable",
        "source_sha256": "91b56636b915f939602107b86970711a9a058161a08ea27bda5eb1d3b2416a95",
    })


def t_depth(circuit):
    """Dependency-constrained T layers of this gate order; Clifford weight zero."""
    layers = [0] * circuit.width
    for gate in circuit.gates:
        level = max(layers[q] for q in gate.qubits) + int(gate.kind in {"t", "tdg"})
        for qubit in gate.qubits:
            layers[qubit] = level
    return max(layers, default=0)


def _logical_candidate(name, circuit, width, metadata=None):
    folded = fold_phases(circuit, tolerance=0.0)
    return CompilerCandidate(name, ladder_circuit(folded, width),
                             {"logical_rotations": len(folded), **(metadata or {})})


def build_candidates(case, method, target, rotation_error):
    if method == "bqskit":
        return bqskit_candidates(target)
    if method == "qiskit-pf":
        return qiskit_product_formula_candidates(case.terms, case.time, target, rotation_error)
    if method == "flagsynth-sdm":
        return flagsynth_sdm_candidates(target)
    if method == "qiskit-qsd":
        return qiskit_qsd_candidates(target)
    if method == "pytket":
        return pytket_candidates(target)
    operator = case.operator
    width = n_qubits(operator)
    if method == "lizzy-auto":
        result = synthesize(operator, case.time, error=rotation_error, objective="t")
        return [CompilerCandidate(
            "lizzy-auto-selected", ladder_circuit(result.circuit, width),
            {"public_api": "synthesize(method='auto', objective='t')",
             "selection": asdict(result.t_selection), "routes": result.routes},
            compiled=result.emitted_circuit,
        )]
    if method == "lizzy-wei-norman":
        result = synthesize_wei_norman(operator, case.time, **WN_OPTIONS)
        return [_logical_candidate(method, result.circuit, width,
                                   {"charts": result.charts, "dimension": result.dimension,
                                    "solver_options": WN_OPTIONS})]
    parts = summands(operator)
    circuit = Circuit()
    if method == "lizzy-givens":
        for part in parts:
            if not exact.is_decomposable(part):
                raise NotImplementedError("Givens requires supported so(m) summands")
            circuit.extend(exact.decompose(part, case.time, method="givens"))
        return [_logical_candidate(method, circuit, width)]
    if method != "lizzy-bdi":
        raise ValueError(f"Unknown compiler {method}")
    plans = [exact.prepare_bdi(part) for part in parts]
    precision = rotation_error / max(1, sum(plan.parameter_bound for plan in plans))
    for plan in plans:
        circuit.extend(plan.circuit(case.time))
    candidates = [_logical_candidate("bdi-reference", circuit, width)]
    optimized, diagnostics = Circuit(), []
    try:
        for part in parts:
            candidate = exact.prepare_bdi(part, optimize="t", rotation_error=precision)
            optimized.extend(candidate.circuit(case.time))
            diagnostics.append(asdict(candidate.optimization))
    except Exception as exc:
        candidates.append(CompilerCandidate("bdi-nullspace", None, {
            "status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
        }))
        return candidates
    if any(row["optimized"] for row in diagnostics):
        candidates.append(_logical_candidate("bdi-nullspace", optimized, width,
                                              {"gauge_search": diagnostics}))
    return candidates


def evaluate_candidate(candidate, target, epsilon):
    record = {"name": candidate.name, "settings": candidate.metadata}
    if candidate.native is None:
        return {**record, "status": "FAIL_ADAPTER"}
    native = candidate.native
    if 2**native.width != len(target):
        return {**record, "status": "FAIL_WIDTH"}
    record.update(ancillas=0, input_rz=sum(g.kind == "rz" for g in native.gates),
                  native_cx=native.n_2qb_gates())
    start = perf_counter()
    record["decomposition_error"] = operator_errors(native.get_unitary(), target)
    record["decomposition_check_seconds"] = perf_counter() - start
    if record["decomposition_error"]["phase_aligned"] > epsilon / 2:
        return {**record, "status": "FAIL_DECOMPOSITION_ERROR"}
    start = perf_counter()
    if candidate.compiled is None:
        emitted = compile_native_clifford_t(native, error=epsilon / 2)
    else:
        # Auto must be measured as delivered, not retrospectively selected from
        # the independently checked BDI/Givens benchmark columns.
        emitted = candidate.compiled
        if emitted.width != native.width or emitted.error_budget != epsilon / 2:
            return {**record, "status": "FAIL_PRECOMPILED_CONTRACT"}
    record["t_compile_seconds"] = perf_counter() - start
    record.update(t_count=emitted.t_count, cx_count=emitted.n_2qb_gates(),
                  t_depth=t_depth(emitted), gates=len(emitted.gates),
                  rotation_error_bound=emitted.rotation_error_bound,
                  error_bound_kind=emitted.error_bound_kind,
                  qasm_sha256=hashlib.sha256(emitted.to_qasm3().encode()).hexdigest())
    start = perf_counter()
    record["final_error"] = operator_errors(emitted.get_unitary(), target)
    record["final_check_seconds"] = perf_counter() - start
    record["status"] = "PASS" if record["final_error"]["phase_aligned"] <= epsilon else "FAIL_FINAL_ERROR"
    return record


def measure(case, method, target, epsilon):
    row = {"case": case.name, "method": method, "epsilon": epsilon}
    start = perf_counter()
    try:
        candidates = build_candidates(case, method, target, epsilon / 2)
    except AlgebraTooLarge as exc:
        return {**row, "status": "CAP", "detail": f"{type(exc).__name__}: {exc}"}
    except TimeoutError as exc:
        return {**row, "status": "CAP", "detail": f"{type(exc).__name__}: {exc}",
                "settings": getattr(exc, "metadata", {})}
    except IntegrationFailure as exc:
        status = "CAP" if "exhausted" in str(exc) else "FAIL_INTEGRATION"
        return {**row, "status": status, "detail": f"{type(exc).__name__}: {exc}"}
    except NotImplementedError as exc:
        return {**row, "status": "UNSUPPORTED", "detail": str(exc)}
    except ImportError as exc:
        return {**row, "status": "UNAVAILABLE", "detail": str(exc)}
    except Exception as exc:
        return {**row, "status": "FAIL_BUILD", "detail": f"{type(exc).__name__}: {exc}"}
    row["decomposition_seconds"] = perf_counter() - start
    evaluated = []
    for candidate in candidates:
        try:
            evaluated.append(evaluate_candidate(candidate, target, epsilon))
        except Exception as exc:
            evaluated.append({"name": candidate.name, "settings": candidate.metadata,
                              "status": "UNAVAILABLE" if isinstance(exc, ImportError) else "FAIL_COMPILE",
                              "detail": f"{type(exc).__name__}: {exc}"})
    row["candidates"] = evaluated
    passing = [candidate for candidate in evaluated if candidate["status"] == "PASS"]
    if not passing:
        status = "UNAVAILABLE" if evaluated and all(c["status"] == "UNAVAILABLE" for c in evaluated) else "FAIL"
        return {**row, "status": status}
    best = min(passing, key=lambda item: item["t_count"])
    row.update(status="PASS", selected=best["name"], t_count=best["t_count"],
               cx_count=best["cx_count"], t_depth=best["t_depth"], final_error=best["final_error"])
    return row


def pairwise_summary(rows, methods, epsilons):
    summary = []
    for epsilon in epsilons:
        entries = {(row["case"], row["method"]): row for row in rows if row["epsilon"] == epsilon}
        names = sorted({name for name, _ in entries})
        for first, second in combinations(methods, 2):
            pairs = [(entries[(name, first)], entries[(name, second)]) for name in names
                     if (name, first) in entries and (name, second) in entries
                     and entries[(name, first)]["status"] == entries[(name, second)]["status"] == "PASS"]
            summary.append({"epsilon": epsilon, "first": first, "second": second,
                            "common_cases": [left["case"] for left, _ in pairs],
                            "first_wins": sum(a["t_count"] < b["t_count"] for a, b in pairs),
                            "ties": sum(a["t_count"] == b["t_count"] for a, b in pairs),
                            "second_wins": sum(a["t_count"] > b["t_count"] for a, b in pairs)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="names")
    parser.add_argument("--method", action="append", choices=METHODS, dest="methods")
    parser.add_argument("--epsilon", action="append", type=float, dest="epsilons")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = [case for case in cases() if not args.names or case.name in args.names]
    if set(args.names or ()) - {case.name for case in cases()}:
        parser.error("Unknown case name")
    methods, epsilons = args.methods or list(METHODS), args.epsilons or [1e-6, 1e-4]
    if any(not np.isfinite(eps) or not 0 < eps < 1 for eps in epsilons):
        parser.error("epsilon must be finite and between zero and one")
    versions = {}
    for package in ("numpy", "scipy", "kak_tools", "paulie", "pygridsynth", "flagsynth",
                    "pennylane", "qiskit", "pytket", "bqskit", "bqskitrs",
                    "openfermion", "cirq-core"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    try:
        flag_source = json.loads(distribution("flagsynth").read_text("direct_url.json") or "null")
    except PackageNotFoundError:
        flag_source = None
    report = {"measured_at": datetime.now(UTC).isoformat(), "complete": False,
              "revision": "shared-frame-no-regression-automatic-exact-selection",
              "previous_record": "experiments/compiler_comparison_before_shared_frames.json",
              "baseline_variants": {"flagsynth-sdm": "locally patched upstream SDM"},
              "seed": SEED, "epsilons": epsilons, "methods": methods,
              "protocol": "experiments/compiler_comparison_protocol.md",
              "versions": versions, "flagsynth_source": flag_source,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: os.environ.get(name) for name in (
                                  "PYTHONHASHSEED", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                  "VECLIB_MAXIMUM_THREADS")}},
              "cases": [asdict(case) for case in corpus], "rows": [], "targets": {}}
    root = Path(__file__).resolve().parents[1]
    sources = ("experiments/compiler_comparison.py", "experiments/compiler_comparison_protocol.md",
               "experiments/_compiler_adapters.py", "experiments/_flagsynth_adapter.py",
               "experiments/_bqskit_adapter.py", "experiments/_product_formula_adapter.py",
               "lizzy/synthesize.py", "lizzy/gaussian.py", "lizzy/clifford_t.py", "lizzy/exact.py", "lizzy/native.py", "lizzy/wei_norman.py",
               "lizzy/driven.py", "lizzy/_orthogonal_mapping.py", "lizzy/dense.py", "lizzy/emit.py")
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in sources}

    def save():
        report["pairwise"] = pairwise_summary(report["rows"], methods, epsilons)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for case in corpus:
        start = perf_counter()
        target = evolution(case.operator, case.time)
        report["targets"][case.name] = {"construction_seconds": perf_counter() - start,
                                       "width": n_qubits(case.operator)}
        for epsilon in epsilons:
            for method in methods:
                row = measure(case, method, target, epsilon)
                report["rows"].append(row)
                print(json.dumps({key: row[key] for key in ("case", "epsilon", "method", "status", "t_count")
                                  if key in row}), flush=True)
                save()
    report["complete"] = True
    save()
    return int(any(row["status"].startswith("FAIL") for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
