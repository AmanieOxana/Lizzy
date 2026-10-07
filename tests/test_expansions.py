"""Independent matrix checks of expansion truncation AND final circuit synthesis."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.synthesis.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    IntegrationFailure,
    _closure,
)
from lizzy.synthesis.expansions import (
    _PauliBracket,
    expand_driven,
    synthesize_expansion,
)

METHODS = ("magnus4", "fer4")


def plan_matrix(plan):
    matrices = np.array([pauli_matrix(w) for w in plan.basis])
    total = np.eye(matrices.shape[1], dtype=complex)
    for factor in plan.factors:
        total = expm(-1j * np.einsum("k,kij->ij", factor, matrices)) @ total
    return total


def rotating_field(words=("X", "Y", "Z")):
    omega, detuning = 1.7, -0.24
    h = DrivenHamiltonian(words, lambda t: [np.cos(omega * t), np.sin(omega * t), detuning])

    def exact(start, end):
        x, _, z = map(pauli_matrix, words)
        return (expm(-0.5j * omega * end * z)
                @ expm(-1j * (end - start) * (x + (detuning - omega / 2) * z))
                @ expm(0.5j * omega * start * z))

    return h, exact


def test_coefficient_bracket_matches_dense_commutator_in_full_su4():
    basis = _closure(["XI", "ZI", "IX", "IZ", "ZZ", "II"], 16)
    assert len(basis) == 16
    rng = np.random.default_rng(61)
    left, right = rng.normal(size=(2, len(basis)))
    matrices = np.array([pauli_matrix(p) for p in basis])
    a = -1j * np.einsum("k,kij->ij", left, matrices)
    b = -1j * np.einsum("k,kij->ij", right, matrices)
    coefficients = _PauliBracket(basis)(left, right)
    actual = -1j * np.einsum("k,kij->ij", coefficients, matrices)
    assert np.linalg.norm(actual - (a @ b - b @ a), 2) < 1e-12


def test_magnus_linear_drive_commutator_sign_and_fer_nested_term():
    # H(t)=X+tY, t in [0,h]. Hand-expanded coefficients in T_j=-iP_j.
    h = 0.4
    drive = DrivenHamiltonian(["X", "Y"], lambda t: [1, t])
    magnus = expand_driven(drive, (0, h), method="magnus4")
    fer = expand_driven(drive, (0, h), method="fer4")
    assert magnus.basis == fer.basis == ("X", "Y", "Z")
    np.testing.assert_allclose(magnus.factors[0], [h, h**2 / 2, -h**3 / 6], atol=1e-15)
    # Correction must be applied FIRST; its nested term is essential for order 4.
    np.testing.assert_allclose(fer.factors[0], [h**5 / 12, -h**4 / 6, -h**3 / 6], atol=1e-15)
    np.testing.assert_allclose(fer.factors[1], [h, h**2 / 2, 0], atol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_global_fourth_order_convergence_of_plans_and_circuits(method):
    drive, exact = rotating_field()
    target = exact(0.2, 1.4)
    plan_errors, circuit_errors = [], []
    for steps in (8, 16):
        result = synthesize_expansion(drive, (0.2, 1.4), method=method, steps=steps)
        ideal = plan_matrix(result.plan)
        actual = circuit_matrix(result.circuit, 1)
        assert np.linalg.norm(actual - ideal, 2) < 1e-9
        assert not result.error_guaranteed and not result.plan.error_guaranteed
        assert result.plan.order == 4
        assert result.plan.control_evaluations == 2 * steps
        plan_errors.append(np.linalg.norm(ideal - target, 2))
        circuit_errors.append(np.linalg.norm(actual - target, 2))
    assert 15 < plan_errors[0] / plan_errors[1] < 17
    assert 15 < circuit_errors[0] / circuit_errors[1] < 17


@pytest.mark.parametrize("method", METHODS)
def test_encoded_drive_absolute_times_and_reverse_time(method):
    span = (1.1, 0.3)
    drive, exact = rotating_field(("XXX", "XXY", "IIZ"))
    result = synthesize_expansion(drive, span, method=method, steps=32)
    assert np.linalg.norm(circuit_matrix(result.circuit, 3) - exact(*span), 2) < 1e-6
    assert result.plan.intervals[0][0] == span[0]
    assert result.plan.intervals[-1][1] == span[1]
    assert all(a[1] == b[0] for a, b in zip(result.plan.intervals, result.plan.intervals[1:]))
    assert all(route.startswith(method + ":") for route in result.circuit.provenance)


@pytest.mark.parametrize("method", METHODS)
def test_general_two_qubit_drive_and_identity_against_independent_dense_ode(method):
    # One full-unitary oracle covers generic noncommuting evolution and the
    # central time-dependent phase, which must not disappear during synthesis.
    drive = DrivenHamiltonian(["XI", "ZI", "IX", "IZ", "ZZ", "II"],
                              lambda t: [0.3 + t / 4, 0.2 * np.cos(t), 0.4,
                                         np.sin(t) / 5, 0.3, 0.17 + t**3])
    matrices = np.array([pauli_matrix(p) for p in drive.paulis])

    def rhs(time, flattened):
        matrix = np.einsum("j,jab->ab", drive.at(time), matrices)
        return (-1j * matrix @ flattened.reshape(4, 4)).ravel()

    reference = solve_ivp(rhs, (0.2, 0.8), np.eye(4, dtype=complex).ravel(),
                          method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.005)
    assert reference.success
    result = synthesize_expansion(drive, (0.2, 0.8), method=method, steps=16)
    assert result.dimension == 16
    assert np.linalg.norm(circuit_matrix(result.circuit, 2)
                          - reference.y[:, -1].reshape(4, 4), 2) < 1e-7


def test_static_noncommuting_hamiltonian_has_no_fer_correction():
    drive = DrivenHamiltonian(["X", "Y", "I"], lambda t: [0.7, -0.2, 0.4])
    result = synthesize_expansion(drive, (0, 1.2), method="fer4", steps=1)
    target = expm(-1.2j * (0.7 * pauli_matrix("X") - 0.2 * pauli_matrix("Y") + 0.4 * np.eye(2)))
    assert result.exponentials == 1  # Fer correction is exactly zero, not a second gate.
    assert np.linalg.norm(plan_matrix(result.plan) - target, 2) < 1e-14
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 2e-9
    assert result.compilation_rhs_evaluations > 0  # exp(-itH) still needs compilation.


def test_commuting_cubic_drive_and_large_identity_phase_emit_directly():
    drive = DrivenHamiltonian(["ZI", "IZ", "II"], lambda t: [t**3, 2 * t, 1000])
    result = synthesize_expansion(drive, (0, 1), method="fer4", steps=1,
                                  max_rhs_evaluations=1, max_compilation_segments=1)
    target = expm(-1j * (0.25 * pauli_matrix("ZI") + pauli_matrix("IZ") + 1000 * np.eye(4)))
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - target, 2) < 1e-10
    assert result.exponentials == result.direct_exponentials == 1
    assert result.compilation_rhs_evaluations == result.compilation_segments == 0


def test_quadrature_error_remains_even_when_all_commutators_vanish():
    drive = DrivenHamiltonian(["Z"], lambda t: [t**4])
    target = expm(-0.2j * pauli_matrix("Z"))
    errors = [np.linalg.norm(plan_matrix(expand_driven(drive, (0, 1), method="magnus4", steps=n))
                             - target, 2) for n in (2, 4)]
    assert errors[1] > 1e-6
    assert 15.9 < errors[0] / errors[1] < 16.1


def test_zero_duration_and_zero_drive_keep_closure_but_emit_no_factors():
    def unused(t):
        raise AssertionError("no control evaluation expected")

    result = synthesize_expansion(DrivenHamiltonian(["X", "Y"], unused), (0.2, 0.2), method="fer4")
    assert result.dimension == 3
    assert result.steps == result.exponentials == len(result.circuit) == 0
    assert result.plan.control_evaluations == result.compilation_rhs_evaluations == 0
    plan = expand_driven(DrivenHamiltonian(["X", "Y"], lambda t: [0, 0]), (0, 1), method="fer4")
    assert plan.dimension == 3
    assert plan.exponentials == 0
    assert plan.steps == 1


def test_large_width_small_algebra_never_builds_dense_matrices(monkeypatch):
    monkeypatch.setattr("lizzy.dense.pauli_matrix", lambda *a: pytest.fail("dense synthesis"))
    drive = DrivenHamiltonian(["X" * 40, "X" * 39 + "Y"], lambda t: [0.1, t / 10])
    result = synthesize_expansion(drive, (0, 0.1), method="fer4")
    assert result.dimension == 3
    assert len(result.circuit) > 0
    assert all(len(p) == 40 for p, _ in result.circuit.rotations)


def test_closure_and_exponential_budgets_precede_control_evaluation():
    def unused(t):
        raise AssertionError("budget must be checked before controls")

    drive = DrivenHamiltonian(["X", "Y"], unused)
    with pytest.raises(AlgebraTooLarge):
        expand_driven(drive, (0, 1), max_dimension=2)
    with pytest.raises(IntegrationFailure, match="max_exponentials"):
        expand_driven(drive, (0, 1), method="fer4", steps=3, max_exponentials=5)


def test_compilation_budgets_are_global_across_factors():
    drive, _ = rotating_field()
    full = synthesize_expansion(drive, (0.2, 1.4), method="magnus4", steps=8)
    with pytest.raises(IntegrationFailure):
        synthesize_expansion(drive, (0.2, 1.4), method="magnus4", steps=8,
                             max_rhs_evaluations=full.compilation_rhs_evaluations // 2)
    with pytest.raises(IntegrationFailure):
        synthesize_expansion(drive, (0.2, 1.4), method="magnus4", steps=8,
                             max_compilation_segments=2)


@pytest.mark.parametrize("kwargs", [
    {"method": "fer2"}, {"steps": 0}, {"steps": True},
    {"max_rhs_evaluations": 0}, {"rtol": 0},
])
def test_invalid_options_are_rejected_even_for_direct_factors(kwargs):
    with pytest.raises(ValueError):
        synthesize_expansion(DrivenHamiltonian(["Z"], lambda t: [1]), (0, 1), **kwargs)


def test_unresolvable_time_grid_fails_instead_of_repeating_nodes():
    with pytest.raises(IntegrationFailure, match="cannot be resolved"):
        expand_driven(DrivenHamiltonian(["Z"], lambda t: [1]),
                      (1e16, np.nextafter(1e16, np.inf)))


def test_arithmetic_overflow_fails_without_nonfinite_factors():
    drive = DrivenHamiltonian(["X", "Y"], lambda t: [1e308 if t < 0.5 else -1e308, 1e308])
    with pytest.raises(IntegrationFailure, match="nonfinite expansion"):
        expand_driven(drive, (0, 1))
