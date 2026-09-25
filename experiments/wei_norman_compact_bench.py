"""Bounded, phase-preserving test of compact Wei–Norman coordinates.

Run python -m experiments.wei_norman_compact_bench [--json]. The historical
"production" arm pins chart_radius=0.5 even after compact-policy integration.
A scoped patch substitutes only the low-level coordinate solver, so
all three variants retain the same component splitting, pulse handling, static
commuting shortcut, resource budgets, and concrete native emission portfolio.
Dense references assess completed circuits; they never guide synthesis.
"""

import argparse
import json
from dataclasses import dataclass, field
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from experiments.wei_norman_compact import synthesize_compact
from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import AlgebraTooLarge, DrivenHamiltonian, IntegrationFailure
from lizzy.driven_bench import _dense_reference
from lizzy.hamiltonian import hamiltonian
from lizzy.kernels import compile_layer
from lizzy.synthesis_bench import (
    BenchmarkCase,
    _emission,
    _errors,
    _exact_static,
    _static_case,
    _versions,
    benchmark_cases,
)
from lizzy.wei_norman import synthesize_wei_norman


@dataclass
class Trial:
    case: BenchmarkCase
    options: dict = field(default_factory=dict)
    note: str = ""


def trials():
    """Fixed cases, seed and settings; no success-dependent order search."""
    original = list(benchmark_cases())
    yield from (Trial(case) for case in original)
    by_name = {case.name: case for case in original}
    cap = by_name["su8-closure-cap"]
    yield Trial(BenchmarkCase("su8-enabled", cap.hamiltonian, cap.time_span,
                             cap.target, cap.reference, cap.static_coefficients),
                {"max_dimension": 64}, "Explicit separate raised-cap experiment.")
    static = by_name["static-tfim3"]
    yield Trial(BenchmarkCase("reverse-tfim3", static.hamiltonian, (1.0, 0.0),
                             static.target.conj().T, "dense expm, reversed",
                             static.static_coefficients))
    yield Trial(_static_case("very-long-tfim3", static.hamiltonian.paulis,
                             static.static_coefficients, 20.0))
    driven = by_name["driven-tfim3"]
    shifted = DrivenHamiltonian([*driven.hamiltonian.paulis, "III"],
                                lambda t: [*driven.hamiltonian.at(t), 1000.0])
    duration = driven.time_span[1] - driven.time_span[0]
    yield Trial(BenchmarkCase("driven-tfim3-identity", shifted, driven.time_span,
                             np.exp(-1000j * duration) * driven.target,
                             "independent dense drive + analytic central phase"))
    for name, strength, duration in (("euler-singularity", 1.0, 2.0),
                                     ("fast-euler-crossing", 100.0, 0.01)):
        drive = DrivenHamiltonian(["X", "Y", "Z"], lambda t, s=strength: [0, s, 0])
        yield Trial(BenchmarkCase(name, drive, (0.0, duration),
                                 expm(-1j * strength * duration * pauli_matrix("Y")),
                                 "analytic single-axis exponential"),
                    {"basis_order": ["X", "Y", "Z"]},
                    "Exact XYZ coordinate singularity is crossed. Sampled condition "
                    "guards can miss it; endpoint PASS does not certify chart regularity.")
    pulse = DrivenHamiltonian(["X", "Z"],
                              lambda t: [1.0, 0.0] if t < 0.5 else [0.0, 0.7])
    pulse_target = expm(-0.35j * pauli_matrix("Z")) @ expm(-0.5j * pauli_matrix("X"))
    yield Trial(BenchmarkCase("pulse-breakpoint", pulse, (0.0, 1.0), pulse_target,
                             "analytic ordered pulse product"), {"breakpoints": [0.5]})
    rng = np.random.default_rng(20260925)
    for index in range(6):
        width = 2 if index < 3 else 3
        words = (["XI", "ZI", "IX", "IZ", "ZZ"] if width == 2
                 else ["ZZI", "IZZ", "XII", "IXI", "IIX"])
        base = rng.uniform(-0.7, 0.7, len(words))
        amplitude = rng.uniform(-0.2, 0.2, len(words))
        frequency = rng.uniform(0.3, 2.0, len(words))
        duration = (0.4, 1.0, 2.5)[index % 3]
        if index % 2 == 0:
            yield Trial(_static_case(f"seeded-static-{index}", words, base, duration))
        else:
            drive = DrivenHamiltonian(words, lambda t, b=base, a=amplitude, f=frequency:
                                      b + a * np.sin(f * t))
            span = (0.0, duration)
            yield Trial(BenchmarkCase(f"seeded-driven-{index}", drive, span,
                                     _dense_reference(drive, span), "independent dense DOP853"))


def run_variant(trial, variant):
    case = trial.case
    compiler_input = (case.hamiltonian if case.static_coefficients is None else
                      hamiltonian(dict(zip(case.hamiltonian.paulis, case.static_coefficients))))
    options = {"max_dimension": 32, "rtol": 1e-10, "atol": 1e-12, "max_step": 0.025,
               "condition_limit": 100.0, "max_rhs_evaluations": 20000,
               "max_segments": 1024, "emission": "none"}
    options.update(trial.options)
    if variant == "production":
        # Preserve the pre-integration comparator for this archived experiment.
        options["chart_radius"] = 0.5
        return synthesize_wei_norman(compiler_input, case.time_span, **options)

    def replacement(drive, span, **kwargs):
        kwargs.pop("chart_radius", None)
        return synthesize_compact(drive, span, restart=variant == "condition-restart", **kwargs)

    if variant not in {"single-product", "condition-restart"}:
        raise ValueError(f"unknown variant: {variant}")
    # Changes this process's binding only, not a repository file or the runtime
    # of another process. The context manager restores it even on failure.
    with patch("lizzy.wei_norman.synthesize_driven", replacement):
        return synthesize_wei_norman(compiler_input, case.time_span, **options)


def assess(case, circuit, *, strict=True):
    backend, emitted = _emission(circuit, case.hamiltonian.n_qubits)
    actual = emitted.get_unitary()
    logical = circuit_matrix(circuit, case.hamiltonian.n_qubits)
    discrepancy = float(np.linalg.norm(actual - logical, 2))
    aligned, error = _errors(actual, case.target)
    valid = np.isfinite(discrepancy) and discrepancy < 1e-8
    passed = valid and np.isfinite(error) and (error if strict else aligned) <= 1e-6
    return {"status": "PASS" if passed else "FAIL_ACCURACY", "rotations": len(circuit),
            "cx": emitted.two_qubit_gates, "native_gates": len(emitted.gates),
            "backend": backend, "strict_error": error, "aligned_error": aligned,
            "emission_discrepancy": discrepancy, "strict_required": strict}


def run(selected=None):
    selected = list(trials()) if selected is None else list(selected)
    rows = []
    for trial in selected:
        case = trial.case
        for variant in ("production", "single-product", "condition-restart"):
            row = {"case": case.name, "variant": variant}
            start = perf_counter()
            try:
                result = run_variant(trial, variant)
                row["compile_seconds"] = perf_counter() - start
                row.update(assess(case, result.circuit))
                row.update(dimension=result.dimension, charts=result.charts,
                           rhs_evaluations=result.rhs_evaluations,
                           rejected_intervals=result.rejected_intervals,
                           peak_sampled_condition=result.max_observed_condition)
            except AlgebraTooLarge as exc:
                row.update(status="CAP", detail=str(exc))
            except (IntegrationFailure, ValueError, RuntimeError) as exc:
                row.update(status="FAIL_COMPILE", detail=f"{type(exc).__name__}: {exc}")
            row["total_seconds"] = perf_counter() - start
            rows.append(row)
        # These are named existing candidates, not the complete auto router.
        if case.static_coefficients is not None:
            for name in ("existing-orthogonal", "existing-pair-kernel"):
                if name == "existing-pair-kernel" and case.hamiltonian.n_qubits != 2:
                    continue
                row = {"case": case.name, "variant": name}
                try:
                    if name == "existing-orthogonal":
                        circuit = _exact_static(case)
                    else:
                        static = hamiltonian(dict(zip(case.hamiltonian.paulis,
                                                      case.static_coefficients)))
                        circuit = compile_layer(static, case.time_span[1] - case.time_span[0])
                    row.update(assess(case, circuit, strict=False))
                except (NotImplementedError, ValueError, StopIteration) as exc:
                    row.update(status="UNSUPPORTED", detail=f"{type(exc).__name__}: {exc}")
                rows.append(row)
    return {"configuration": {
        "seed": 20260925, "error_threshold": 1e-6, "rtol": 1e-10, "atol": 1e-12,
        "max_step": 0.025, "condition_limit": 100, "max_rhs_evaluations": 20000,
        "max_segments": 1024, "default_dimension_cap": 32,
        "factor_order": "production weight-first; explicit XYZ in singularity cases",
        "production_comparator": "historical radius-limited policy, chart_radius=0.5",
        "emission": "same concrete native-ladder / eligible native-frame for every row",
        "accuracy": "WN strict operator norm including global phase; existing exact "
                    "candidates phase-aligned, both errors reported",
        "scope": "empirical <=3 qubit experiments, not chemistry or a certified router",
        "conditioning": "sampled, not a continuous nonsingularity guarantee",
        "timings": "single runs, normal caches, compile excludes emission/reference/verification",
        }, "versions": _versions(),
        "cases": [{"name": t.case.name, "paulis": t.case.hamiltonian.paulis,
                   "time_span": t.case.time_span, "reference": t.case.reference,
                   "static_coefficients": t.case.static_coefficients,
                   "options": t.options, "note": t.note} for t in selected], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    selected = list(trials())
    if args.case:
        unknown = set(args.case) - {trial.case.name for trial in selected}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        selected = [trial for trial in selected if trial.case.name in args.case]
    report = run(selected)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        for row in report["rows"]:
            print(f"{row['case']:26} {row['variant']:23} {row['status']:14} "
                  f"CX={row.get('cx', '-'):>5} charts={row.get('charts', '-'):>4} "
                  f"strict_error={row.get('strict_error', '-')} {row.get('detail', '')}")


if __name__ == "__main__":
    main()
