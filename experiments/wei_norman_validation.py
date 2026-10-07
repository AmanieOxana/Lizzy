"""Validate production Wei–Norman synthesis on a fixed small-instance corpus.

Run python -m experiments.wei_norman_validation --json. Both modes call the
real public compiler; no duplicate solver or runtime monkeypatch is involved.
Dense references are validation oracles, never inputs to synthesis.
"""

import argparse
import json
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import AlgebraTooLarge, DrivenHamiltonian, IntegrationFailure
from lizzy.hamiltonian import hamiltonian
from lizzy.wei_norman import synthesize_wei_norman

from ._validation import (
    BenchmarkCase,
    _dense_reference,
    _emission,
    _errors,
    _static_case,
    _versions,
    benchmark_cases,
)

MODES = ("resolved-step", "adaptive-default-step")


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


def compile_trial(trial, mode):
    """Compile with the public API and fixed tolerances/shared work limits."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    case = trial.case
    compiler_input = (case.hamiltonian if case.static_coefficients is None else
                      hamiltonian(dict(zip(case.hamiltonian.paulis, case.static_coefficients))))
    options = {
        "max_dimension": 32, "rtol": 1e-10, "atol": 1e-12,
        "chart_radius": None, "condition_limit": 100.0,
        "max_rhs_evaluations": 20000, "max_segments": 1024, "emission": "none",
    }
    if mode == "resolved-step":
        options["max_step"] = 0.025
    options.update(trial.options)
    return synthesize_wei_norman(compiler_input, case.time_span, **options)


def assess(case, circuit):
    """Count a concrete artifact and require strict, phase-sensitive accuracy."""
    backend, emitted = _emission(circuit, case.hamiltonian.n_qubits)
    actual = emitted.get_unitary()
    logical = circuit_matrix(circuit, case.hamiltonian.n_qubits)
    discrepancy = float(np.linalg.norm(actual - logical, 2))
    aligned, error = _errors(actual, case.target)
    passed = (np.isfinite(discrepancy) and discrepancy < 1e-8
              and np.isfinite(error) and error <= 1e-6)
    return {
        "status": "PASS" if passed else "FAIL_ACCURACY",
        "rotations": len(circuit), "cx": emitted.two_qubit_gates,
        "native_gates": len(emitted.gates), "backend": backend,
        "strict_error": error, "aligned_error": aligned,
        "emission_discrepancy": discrepancy, "strict_required": True,
    }


def run(selected=None):
    """Retain failures and the explicitly expected default-dimension refusals."""
    selected = list(trials()) if selected is None else list(selected)
    rows = []
    for trial in selected:
        for mode in MODES:
            row = {"case": trial.case.name, "mode": mode}
            start = perf_counter()
            try:
                result = compile_trial(trial, mode)
                row["compile_seconds"] = perf_counter() - start
                row.update(assess(trial.case, result.circuit))
                row.update(
                    dimension=result.dimension, charts=result.charts,
                    rhs_evaluations=result.rhs_evaluations,
                    rejected_intervals=result.rejected_intervals,
                    peak_sampled_condition=result.max_observed_condition,
                )
            except AlgebraTooLarge as exc:
                row.update(status="CAP" if trial.case.expected_cap else "FAIL_CAP",
                           detail=str(exc))
            except (IntegrationFailure, ValueError, RuntimeError) as exc:
                row.update(status="FAIL_COMPILE", detail=f"{type(exc).__name__}: {exc}")
            row["total_seconds"] = perf_counter() - start
            rows.append(row)
    return {
        "configuration": {
            "seed": 20260925, "error_threshold": 1e-6,
            "rtol": 1e-10, "atol": 1e-12, "chart_radius": None,
            "modes": {"resolved-step": {"max_step": 0.025},
                      "adaptive-default-step": {"max_step": "infinity (API default)"}},
            "condition_limit": 100, "max_rhs_evaluations": 20000,
            "max_segments": 1024, "default_dimension_cap": 32,
            "emission": "concrete native-ladder / eligible native-frame",
            "accuracy": "strict operator norm including global phase",
            "scope": "empirical <=3 qubit validation, not a certified router",
            "conditioning": "sampled, not a continuous nonsingularity guarantee",
            "timings": "single run; compile excludes reference/emission/verification",
        },
        "versions": _versions(),
        "cases": [
            {"name": t.case.name, "paulis": t.case.hamiltonian.paulis,
             "time_span": t.case.time_span, "reference": t.case.reference,
             "static_coefficients": t.case.static_coefficients,
             "expected_cap": t.case.expected_cap, "options": t.options, "note": t.note}
            for t in selected
        ],
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--case", action="append")
    args = parser.parse_args(argv)
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
            print(f"{row['case']:26} {row['mode']:23} {row['status']:14} "
                  f"CX={row.get('cx', '-'):>5} charts={row.get('charts', '-'):>4} "
                  f"strict_error={row.get('strict_error', '-')} {row.get('detail', '')}")
    return int(any(row["status"].startswith("FAIL") for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
