"""Accuracy-matched small-instance synthesis experiments, not a universal claim.

Run ``python -m lizzy.synthesis_bench``. Static references use dense ``expm``;
driven references independently integrate the full Schrodinger equation. These
exponentially sized oracles are validation/search tools ONLY, not scalable
compiler components. Product-formula steps are searched on a doubling grid.
The fixed emission portfolio contains concrete native ladders and eligible
Clifford-frame circuits, never environment-dependent optional SDK backends.
PASS compares trace-phase-aligned operator norms; Wei-Norman additionally must
pass the strict norm because, unlike the legacy exact spin-cover route, it
retains global phase. Both norms are reported; phase alignment is not claimed
to minimize operator norm over all global phases.
"""

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from lizzy import exact
from lizzy.classify import summands
from lizzy.dense import circuit_matrix, operator_errors, pauli_matrix
from lizzy.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    IntegrationFailure,
    _closure,
)
from lizzy.driven_bench import _dense_reference, _midpoint_formula
from lizzy.emit import native_emission_candidates
from lizzy.hamiltonian import Circuit, fold_phases, hamiltonian


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    hamiltonian: DrivenHamiltonian
    time_span: tuple[float, float]
    target: np.ndarray
    reference: str
    static_coefficients: tuple[float, ...] | None = None
    expected_cap: bool = False


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    route: str
    status: str
    dimension: int
    component_dimensions: tuple[int, ...] = ()
    charts: int | None = None
    steps: int | None = None
    rotations: int | None = None
    logical_cx_estimate: int | None = None
    emitted_cx: int | None = None
    emitted_gates: int | None = None
    backend: str | None = None
    op_error: float | None = None
    strict_error: float | None = None
    compile_seconds: float = 0.0
    search_seconds: float = 0.0
    verification_seconds: float = 0.0
    detail: str = ""


def _static_case(name, words, coefficients, duration, *, expected_cap=False):
    coefficients = tuple(float(c) for c in coefficients)
    driven = DrivenHamiltonian(words, lambda t: coefficients)
    generator = sum(c * pauli_matrix(p) for p, c in zip(words, coefficients))
    return BenchmarkCase(
        name, driven, (0.0, duration), expm(-1j * duration * generator),
        "dense expm", coefficients, expected_cap,
    )


def benchmark_cases():
    """Fixed families/seeds; at most three qubits, including unfavorable regimes."""
    yield _static_case("static-commuting", ["ZZI", "IZZ", "ZIZ"], [0.7, -0.4, 0.3], 1.2)
    span = (0.2, 1.4)
    commuting = DrivenHamiltonian(["ZZI", "IZZ", "ZIZ"], lambda t: [t, -0.4, 0.3 * t])
    integral = [0.5 * (span[1]**2 - span[0]**2), -0.4 * (span[1] - span[0])]
    generator = (integral[0] * pauli_matrix("ZZI") + integral[1] * pauli_matrix("IZZ")
                 + 0.3 * integral[0] * pauli_matrix("ZIZ"))
    yield BenchmarkCase("driven-commuting", commuting, span, expm(-1j * generator), "analytic integral")
    yield _static_case("static-spin", ["X", "Z"], [0.8, -0.3], 1.2)
    yield _static_case("static-encoded", ["XXX", "XXY", "IIZ"], [0.8, 0.3, -0.2], 1.2)
    yield _static_case("commuting-su2", ["XII", "ZII", "IXI", "IZI", "IIX", "IIZ"],
                       [0.7, 0.2, -0.4, 0.5, 0.3, -0.6], 1.2)

    omega, detuning = 1.7, -0.24
    for name, words in (("driven-spin", ["X", "Y", "Z"]),
                        ("driven-encoded", ["XXX", "XXY", "IIZ"])):
        driven = DrivenHamiltonian(words, lambda t: [np.cos(omega*t), np.sin(omega*t), detuning])
        x_axis, _, z_axis = map(pauli_matrix, words)
        target = (expm(-0.5j * omega * span[1] * z_axis)
                  @ expm(-1j * (span[1] - span[0]) * (x_axis + (detuning - omega/2)*z_axis))
                  @ expm(0.5j * omega * span[0] * z_axis))
        yield BenchmarkCase(name, driven, span, target, "analytic rotating frame")

    words = ["ZZI", "IZZ", "XII", "IXI", "IIX"]
    coefficients = [0.7, -0.45, 0.35, 0.5, -0.4]
    for name, duration in (("short-tfim3", 0.003), ("static-tfim3", 1.0), ("long-tfim3", 5.0)):
        yield _static_case(name, words, coefficients, duration)
    driven = DrivenHamiltonian(words, lambda t: [0.7 + 0.12*np.sin(0.9*t),
        -0.45 + 0.08*np.cos(1.1*t), 0.35 + 0.11*np.cos(0.7*t),
        0.5 + 0.13*np.sin(1.3*t), -0.4 + 0.09*np.cos(1.7*t)])
    yield BenchmarkCase("driven-tfim3", driven, (0.0, 1.2),
                        _dense_reference(driven, (0.0, 1.2)), "dense DOP853, max_step=.005")

    rng = np.random.default_rng(20260921)
    words = ["".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I")]
    yield _static_case("static-su4", words, rng.uniform(-0.3, 0.3, len(words)), 0.8)
    words = ["XI", "ZI", "IX", "IZ", "ZZ"]
    driven = DrivenHamiltonian(words, lambda t: [0.4 + 0.1*np.sin(t), 0.2*np.cos(1.3*t),
        -0.3 + 0.07*np.cos(t), 0.25 + 0.04*np.sin(0.8*t), 0.6 + 0.11*np.sin(1.1*t)])
    yield BenchmarkCase("driven-su4", driven, (0.1, 0.9),
                        _dense_reference(driven, (0.1, 0.9)), "dense DOP853, max_step=.005")
    yield _static_case("su8-closure-cap", ["XII", "ZII", "IXI", "IZI", "IIX", "IIZ", "ZZI", "IZZ"],
                       [0.3, -0.4, 0.2, 0.25, -0.35, 0.1, 0.45, -0.3], 0.4, expected_cap=True)


def _errors(achieved, target):
    """Strict norm and trace-phase-aligned norm (not an asserted phase optimum)."""
    errors = operator_errors(achieved, target)
    return errors["phase_aligned"], errors["strict"]


def _emission(circuit, width):
    """Concrete, fixed portfolio: counts are from artifacts, not block estimates."""
    candidates = native_emission_candidates(circuit, width)
    return min(candidates, key=lambda item: (item[1].two_qubit_gates, len(item[1].gates)))


def _assess(case, route, circuit, dimension, error, compile_seconds, *, steps=None,
            charts=None, component_dimensions=(), search_seconds=0.0, verification_seconds=0.0):
    start = perf_counter()
    backend, emitted = _emission(circuit, case.hamiltonian.n_qubits)
    compile_seconds += perf_counter() - start
    start = perf_counter()
    logical = circuit_matrix(circuit, case.hamiltonian.n_qubits)
    actual = emitted.get_unitary()
    discrepancy = float(np.linalg.norm(actual - logical, 2))
    aligned_error, strict_error = _errors(actual, case.target)
    verification_seconds += perf_counter() - start
    valid = np.isfinite(discrepancy) and discrepancy <= 1e-8
    passed = valid and np.isfinite(aligned_error) and aligned_error <= error
    # Unlike the older exact spin-cover route, WN promises to retain global phase.
    if route == "wei-norman":
        passed = passed and strict_error <= error
    detail = "" if valid else f"emission/logical mismatch {discrepancy:.3e}"
    if not passed and not detail:
        detail = "requested achieved accuracy was not reached"
    return BenchmarkRow(case.name, route, "PASS" if passed else "FAIL_ACCURACY", dimension,
        tuple(component_dimensions), charts, steps, len(circuit), circuit.two_qubit_gates,
        emitted.two_qubit_gates, len(emitted.gates), backend, aligned_error, strict_error,
        compile_seconds, search_seconds, verification_seconds, detail)


def _exact_static(case):
    if case.static_coefficients is None:
        raise NotImplementedError("the existing exact route accepts static Hamiltonians only")
    static = hamiltonian(dict(zip(case.hamiltonian.paulis, case.static_coefficients)))
    parts = summands(static)
    if not all(exact.is_decomposable(part) for part in parts):
        raise NotImplementedError("outside the existing orthogonal/summand exact route")
    circuit = Circuit()
    try:
        for part in parts:
            circuit.extend(exact.decompose(part, case.time_span[1] - case.time_span[0]))
    except (StopIteration, ValueError) as exc:
        # As in the static router, an algebra label does not guarantee the
        # upstream BDI generator embedding exists. Keep the full refusal reason.
        raise NotImplementedError(f"{type(exc).__name__}: {exc}") from exc
    return fold_phases(circuit, tolerance=0)


def _product_formula(case, dimension, error, max_steps):
    steps, verification_time = 1, 0.0
    start_search = perf_counter()
    while True:
        start = perf_counter()
        circuit = _midpoint_formula(case.hamiltonian, case.time_span, steps)
        elapsed = perf_counter() - start
        start = perf_counter()
        aligned_error, _ = _errors(circuit_matrix(circuit, case.hamiltonian.n_qubits), case.target)
        verification_time += perf_counter() - start
        at_cap = steps * 2 > max_steps
        if aligned_error <= error or at_cap:
            row = _assess(case, "midpoint2", circuit, dimension, error, elapsed,
                steps=steps, search_seconds=perf_counter() - start_search,
                verification_seconds=verification_time)
            if row.status == "PASS" or "emission/logical mismatch" in row.detail:
                return row
            if at_cap:
                values = asdict(row)
                values.update(status="FAIL_STEP_CAP", detail=f"doubling-grid search exhausted max_steps={max_steps}")
                return BenchmarkRow(**values)
        steps *= 2


def run_suite(cases=None, *, error=1e-6, max_dimension=32, max_steps=4096,
              max_rhs_evaluations=100_000):
    """Return evidence rows; an unsupported/capped route is never silently omitted.

    ``compile_seconds`` measures final logical synthesis and emission, without
    dense validation. ``search_seconds`` separately includes the PF accuracy
    search; ``verification_seconds`` includes its dense checks and final replay.
    Times are single-run wall times with normal process caches, not isolated
    performance measurements. Exact/global-phase agreement is reported explicitly.
    """
    from lizzy.wei_norman import synthesize_wei_norman

    if not np.isfinite(error) or error <= 0:
        raise ValueError("error must be finite and positive")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 8192:
        raise ValueError("max_steps must be an integer between 1 and 8192")
    if not isinstance(max_dimension, int) or isinstance(max_dimension, bool) or not 1 <= max_dimension <= 64:
        raise ValueError("max_dimension must be an integer between 1 and 64")
    rows = []
    for case in benchmark_cases() if cases is None else cases:
        if case.hamiltonian.n_qubits > 3:
            raise ValueError("this dense validation benchmark is capped at three qubits")
        # Tiny dense-suite diagnostic only: the compiler retains its own cap.
        dimension = len(_closure(case.hamiltonian.paulis, 4**case.hamiltonian.n_qubits))
        start = perf_counter()
        try:
            compiler_input = (case.hamiltonian if case.static_coefficients is None
                else hamiltonian(dict(zip(case.hamiltonian.paulis, case.static_coefficients))))
            result = synthesize_wei_norman(compiler_input, case.time_span,
                max_dimension=max_dimension, rtol=1e-10, atol=1e-12,
                max_step=0.025, max_rhs_evaluations=max_rhs_evaluations, emission="none")
            elapsed = perf_counter() - start
            rows.append(_assess(case, "wei-norman", result.circuit, dimension, error, elapsed,
                charts=result.charts, component_dimensions=result.component_dimensions))
        except AlgebraTooLarge as exc:
            rows.append(BenchmarkRow(case.name, "wei-norman", "CAP", dimension,
                compile_seconds=perf_counter()-start, detail=str(exc)))
        except (IntegrationFailure, ValueError, RuntimeError) as exc:
            rows.append(BenchmarkRow(case.name, "wei-norman", "FAIL_COMPILE", dimension,
                compile_seconds=perf_counter()-start, detail=f"{type(exc).__name__}: {exc}"))

        start = perf_counter()
        try:
            rows.append(_product_formula(case, dimension, error, max_steps))
        except (ValueError, RuntimeError) as exc:
            rows.append(BenchmarkRow(case.name, "midpoint2", "FAIL_COMPILE", dimension,
                compile_seconds=perf_counter()-start, detail=f"{type(exc).__name__}: {exc}"))

        start = perf_counter()
        try:
            circuit = _exact_static(case)
            rows.append(_assess(case, "static-exact", circuit, dimension, error, perf_counter()-start))
        except NotImplementedError as exc:
            rows.append(BenchmarkRow(case.name, "static-exact", "UNSUPPORTED", dimension,
                compile_seconds=perf_counter()-start, detail=str(exc)))
        except (StopIteration, ValueError, RuntimeError) as exc:
            rows.append(BenchmarkRow(case.name, "static-exact", "FAIL_COMPILE", dimension,
                compile_seconds=perf_counter()-start, detail=f"{type(exc).__name__}: {exc}"))
    return rows


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("lizzy", "numpy", "scipy", "paulie", "kak_tools"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error", type=float, default=1e-6)
    parser.add_argument("--max-dimension", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--case", action="append", dest="names", help="repeat to select named cases")
    parser.add_argument("--json", action="store_true", help="machine-readable rows including all timings")
    args = parser.parse_args(argv)
    cases = list(benchmark_cases())
    if args.names:
        unknown = set(args.names) - {case.name for case in cases}
        if unknown:
            parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.name in args.names]
    try:
        rows = run_suite(cases, error=args.error, max_dimension=args.max_dimension, max_steps=args.max_steps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps({"error_threshold": args.error,
            "configuration": {"suite_version": 2, "seed": 20260921,
                "max_dimension": args.max_dimension, "max_steps": args.max_steps,
                "wn_rtol": 1e-10, "wn_atol": 1e-12, "wn_max_step": 0.025,
                "wn_chart_radius": None, "wn_chart_policy": "condition-only restarts",
                "max_rhs_evaluations": 100_000,
                "emission_portfolio": ["native-ladder", "eligible native-frame"],
                "accuracy_policy": "trace-phase-aligned norm; WN also strict norm",
                "timing_policy": "single run, normal caches; reference generation excluded"},
            "versions": _versions(),
            "cases": [{"name": case.name, "time_span": case.time_span,
                "paulis": case.hamiltonian.paulis, "reference": case.reference,
                "static_coefficients": case.static_coefficients} for case in cases],
            "rows": [asdict(row) for row in rows]}, indent=2))
    else:
        print(f"Empirical operator-norm threshold {args.error:.1e}; not certified; finite cases prove no universal ranking.")
        print("PASS uses trace-phase-aligned norm; Wei-Norman must ALSO pass strict norm. Both errors are printed.")
        print("Concrete fixed emission portfolio: native ladders + eligible native-frame; no optional SDKs.")
        print("CX and gates count emitted artifacts. compile_s excludes dense verification; search_s includes PF grid search.")
        print("Single-run wall times include normal caches and exclude reference generation. Dense search is a small-case oracle.")
        print("PF uses first passing doubling-grid point, not the minimum step count; CAP/UNSUPPORTED are explicit exclusions.")
        print("case                 route         d charts steps  rots   CX gates compile_s search_s  op_error strict_err status")
        for case in cases:
            print(f"# {case.name}: t={case.time_span}; reference={case.reference}")
            for row in (r for r in rows if r.case == case.name):
                def number(value):
                    return "-" if value is None else str(value)
                op = "-" if row.op_error is None else f"{row.op_error:.2e}"
                strict = "-" if row.strict_error is None else f"{row.strict_error:.2e}"
                print(f"{row.case:<20} {row.route:<12} {row.dimension:>2} {number(row.charts):>6} "
                      f"{number(row.steps):>5} {number(row.rotations):>5} {number(row.emitted_cx):>4} "
                      f"{number(row.emitted_gates):>5} {row.compile_seconds:>9.3f} {row.search_seconds:>8.3f} "
                      f"{op:>9} {strict:>10} {row.status}")
                if row.detail:
                    print(f"# {row.route}: {row.detail}")
        comparisons = []
        for case in cases:
            passing = [row for row in rows if row.case == case.name and row.status == "PASS"]
            wn = next((row for row in passing if row.route == "wei-norman"), None)
            if wn:
                comparisons.extend((case.name, row.route) for row in passing
                    if row.route != "wei-norman" and row.emitted_cx < wn.emitted_cx)
        print("Observed lower-CX alternatives to Wei-Norman: " +
              (", ".join(f"{name}/{route}" for name, route in comparisons) or "none in the selected cases"))
        print("Neither a tie nor a win proves 'always better'; runtime, gates, and accuracy are separate objectives.")
    return int(any(row.status.startswith("FAIL") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
