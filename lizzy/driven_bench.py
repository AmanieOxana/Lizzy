"""Small, reproducible empirical comparison of driven synthesis routes.

Run ``python -m lizzy.driven_bench``. Expansion and midpoint product-formula
steps are searched on doubling grids, not optimized for minimum step counts.
All errors are measured operator norms against independent small dense references;
they are empirical checks, not certified error bounds or scalability claims.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import DrivenHamiltonian, IntegrationFailure, synthesize_driven
from lizzy.expansions import expand_driven, synthesize_expansion
from lizzy.hamiltonian import Circuit, fold_phases
from lizzy.native import ladder_circuit, native_frame_candidate, native_frame_circuit


@dataclass(frozen=True)
class _Case:
    name: str
    hamiltonian: DrivenHamiltonian
    time_span: tuple[float, float]
    target: np.ndarray
    reference: str


def _dense_reference(hamiltonian, time_span):
    """Integrate the full Schrodinger equation independently of Lie coordinates."""
    matrices = np.array([pauli_matrix(w) for w in hamiltonian.paulis])
    dimension = 2**hamiltonian.n_qubits

    def rhs(time, flattened):
        generator = np.einsum("k,kij->ij", hamiltonian.at(time), matrices)
        return (-1j * generator @ flattened.reshape(dimension, dimension)).ravel()

    solution = solve_ivp(
        rhs,
        time_span,
        np.eye(dimension, dtype=complex).ravel(),
        method="DOP853",
        rtol=3e-13,
        atol=1e-14,
        max_step=0.005,
    )
    if not solution.success:
        raise RuntimeError(f"Dense reference failed: {solution.message}")
    return solution.y[:, -1].reshape(dimension, dimension)


def _cases():
    omega, detuning = 1.7, -0.24
    start, end = 0.2, 1.4
    for name, words in (
        ("rotating-spin", ["X", "Y", "Z"]),
        ("encoded-spin", ["XXX", "XXY", "IIZ"]),
    ):
        hamiltonian = DrivenHamiltonian(
            words, lambda t: [np.cos(omega * t), np.sin(omega * t), detuning]
        )
        x_axis, _, z_axis = map(pauli_matrix, words)
        target = (
            expm(-0.5j * omega * end * z_axis)
            @ expm(-1j * (end - start) * (x_axis + (detuning - omega / 2) * z_axis))
            @ expm(0.5j * omega * start * z_axis)
        )
        yield _Case(name, hamiltonian, (start, end), target, "analytic rotating frame")

    hamiltonian = DrivenHamiltonian(
        ["ZZI", "IZZ", "XII", "IXI", "IIX"],
        lambda t: [
            0.7 + 0.12 * np.sin(0.9 * t),
            -0.45 + 0.08 * np.cos(1.1 * t),
            0.35 + 0.11 * np.cos(0.7 * t),
            0.5 + 0.13 * np.sin(1.3 * t),
            -0.4 + 0.09 * np.cos(1.7 * t),
        ],
    )
    span = (0.0, 1.2)
    yield _Case(
        "driven-tfim-3", hamiltonian, span, _dense_reference(hamiltonian, span),
        "dense DOP853 (rtol=3e-13, atol=1e-14, max_step=0.005)",
    )


def _midpoint_formula(hamiltonian, time_span, steps):
    """Midpoint time sampling and a second-order symmetric Pauli splitting."""
    start, end = time_span
    duration = (end - start) / steps
    paulis = [get_pauli_string(word) for word in hamiltonian.paulis]
    circuit = Circuit()
    for step in range(steps):
        angles = duration * hamiltonian.at(start + (step + 0.5) * duration)
        for pauli, angle in zip(paulis[:-1], angles[:-1]):
            circuit.add(pauli, float(angle / 2), "midpoint2")
        circuit.add(paulis[-1], float(angles[-1]), "midpoint2")
        for pauli, angle in reversed(list(zip(paulis[:-1], angles[:-1]))):
            circuit.add(pauli, float(angle / 2), "midpoint2")
    return fold_phases(circuit, tolerance=0)


def _quote(circuit, width, logical_unitary):
    """Same fixed portfolio for all routes; verify native output before quoting."""
    candidates = [("native-ladder", ladder_circuit(circuit, width))]
    if native_frame_candidate(circuit, width):
        candidates.append(("native-frame", native_frame_circuit(circuit, width)))
    for _, emitted in candidates:
        emitted_unitary = emitted.get_unitary()
        discrepancy = np.linalg.norm(emitted_unitary - logical_unitary, ord=2)
        if not np.isfinite(discrepancy) or discrepancy > 1e-9:
            raise RuntimeError(f"Native emission equivalence failed: {discrepancy:.3e}")
    backend, emitted = min(candidates, key=lambda item: (item[1].two_qubit_gates, len(item[1].gates)))
    return emitted.two_qubit_gates, backend, emitted.get_unitary()


def _measure(case, circuit):
    logical_unitary = circuit_matrix(circuit, case.hamiltonian.n_qubits)
    emitted_cost, backend, emitted_unitary = _quote(
        circuit, case.hamiltonian.n_qubits, logical_unitary
    )
    measured = float(np.linalg.norm(emitted_unitary - case.target, ord=2))
    return emitted_cost, backend, measured


def _plan_matrix(plan):
    """Dense ideal exponentials for screening only; final circuits are rechecked."""
    basis = np.array([pauli_matrix(word) for word in plan.basis])
    unitary = np.eye(basis.shape[1], dtype=complex)
    for factor in plan.factors:
        generator = np.einsum("k,kij->ij", factor, basis)
        unitary = expm(-1j * generator) @ unitary
    return unitary


def _search_expansion(case, method, error, max_steps):
    """Screen ideal plans, then verify the compiled output before accepting."""
    steps = 1
    while True:
        plan = expand_driven(case.hamiltonian, case.time_span, method=method, steps=steps)
        plan_error = float(np.linalg.norm(_plan_matrix(plan) - case.target, ord=2))
        at_cap = steps * 2 > max_steps
        if plan_error <= error or at_cap:
            result = synthesize_expansion(
                case.hamiltonian, case.time_span, method=method, steps=steps,
                rtol=1e-10, atol=1e-12,
            )
            measurement = _measure(case, result.circuit)
            if measurement[2] <= error or at_cap:
                return result, plan_error, measurement
        steps *= 2


def _report(
    case, route, circuit, steps, dimension, intervals, error, *,
    exponentials="-", plan_error=None, measurement=None,
):
    emitted_cost, backend, measured = measurement or _measure(case, circuit)
    passed = np.isfinite(measured) and measured <= error
    plan_text = "-" if plan_error is None else f"{plan_error:.3e}"
    print(
        f"{case.name:<15} {route:<10} {dimension:>3} {intervals:>5} {steps:>6} "
        f"{exponentials:>5} {len(circuit):>9} {circuit.two_qubit_gates:>10} "
        f"{emitted_cost:>10} {backend:<12} {plan_text:>10} {measured:>10.3e} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def main() -> int:
    """Print all cases and return nonzero if any route fails the common target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error", type=float, default=1e-6, help="operator-norm threshold")
    parser.add_argument("--max-steps", type=int, default=8192, help="product-formula search cap")
    parser.add_argument(
        "--max-expansion-steps", type=int, default=256,
        help="separate Magnus/Fer doubling-grid search cap",
    )
    args = parser.parse_args()
    if not np.isfinite(args.error) or args.error <= 0:
        parser.error("--error must be finite and positive")
    if args.max_steps < 1 or args.max_steps > 8192:
        parser.error("--max-steps must be between 1 and 8192 for this small benchmark")
    if args.max_expansion_steps < 1 or args.max_expansion_steps > 256:
        parser.error("--max-expansion-steps must be between 1 and 256")

    print(f"Common operator-norm threshold: {args.error:.1e}; empirical, not certified.")
    print("PF searches powers of two; PASS is the first passing grid point, not a minimum.")
    print("Magnus/Fer use dense ideal-plan screening on the same grid; final circuits are verified.")
    print("Fixed portfolio: concrete native-ladder + eligible, dense-verified native-frame.")
    print("Charts count WN compilation segments; plan_error excludes compilation, op_error includes it.")
    print("Wei-Norman chart policy: compact (chart_radius=None), sampled condition-limited restarts.")
    print("case            route        dim charts  steps  exps rotations logical_CX emitted_CX backend      plan_error   op_error status")
    success = True
    for case in _cases():
        print(f"# {case.name}: t={case.time_span}; reference={case.reference}")
        try:
            result = synthesize_driven(
                case.hamiltonian, case.time_span, rtol=1e-10, atol=1e-12, max_step=0.025
            )
            passed = _report(
                case, "Wei-Norman", result.circuit, "-", result.dimension,
                len(result.intervals), args.error,
            )
            success = passed and success
        except IntegrationFailure as exc:
            print(f"{case.name}: Wei-Norman FAIL: {exc}")
            success = False

        for method in ("magnus4", "fer4"):
            try:
                result, plan_error, measurement = _search_expansion(
                    case, method, args.error, args.max_expansion_steps
                )
                passed = _report(
                    case, method, result.circuit, result.steps, result.dimension,
                    result.compilation_segments, args.error,
                    exponentials=result.exponentials, plan_error=plan_error,
                    measurement=measurement,
                )
                success = passed and success
            except IntegrationFailure as exc:
                print(f"{case.name}: {method} FAIL: {exc}")
                success = False

        steps = 1
        while True:
            circuit = _midpoint_formula(case.hamiltonian, case.time_span, steps)
            measured = np.linalg.norm(
                circuit_matrix(circuit, case.hamiltonian.n_qubits) - case.target, ord=2
            )
            if measured <= args.error or steps * 2 > args.max_steps:
                break
            steps *= 2
        passed = _report(case, "midpoint2", circuit, steps, "-", "-", args.error)
        success = passed and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
