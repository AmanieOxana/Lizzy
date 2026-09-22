"""Wei–Norman circuits checked independently, including their global phase."""

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy import driven
from lizzy.dense import circuit_matrix, evolution, pauli_matrix
from lizzy.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    IntegrationFailure,
    synthesize_driven,
)
from lizzy.hamiltonian import hamiltonian


def rotating_field(words=("X", "Y", "Z")):
    omega, detuning = 1.7, 0.3
    h = DrivenHamiltonian(words, lambda t: [np.cos(omega * t), np.sin(omega * t), detuning])

    def reference(start, end):
        x, _, z = map(pauli_matrix, words)
        return (expm(-1j * omega * end * z / 2)
                @ expm(-1j * (end - start) * (x + (detuning - omega / 2) * z))
                @ expm(1j * omega * start * z / 2))

    return h, reference


@pytest.mark.parametrize("words", [("X", "Y", "Z"), ("XIX", "XIY", "IIZ")])
@pytest.mark.parametrize("span", [(0.0, 1.2), (0.3, 1.8), (1.8, 0.3)])
def test_driven_rotation_matches_analytic_solution(words, span):
    h, reference = rotating_field(words)
    result = synthesize_driven(h, span)
    actual = circuit_matrix(result.circuit, h.n_qubits)
    assert np.linalg.norm(actual - reference(*span), 2) < 2e-8
    assert result.dimension == 3
    assert result.chart_restarts > 0
    assert not result.error_guaranteed
    assert result.intervals[0][0] == span[0]
    assert result.intervals[-1][1] == span[1]
    assert all(a[1] == b[0] for a, b in zip(result.intervals, result.intervals[1:]))
    assert set(result.circuit.provenance) == {"wei-norman"}


def test_general_two_qubit_drive_against_independent_dense_ode():
    h = DrivenHamiltonian(["XI", "ZI", "XX", "IY"],
                          lambda t: [0.3 + t / 7, 0.2 * np.cos(t), 0.4, np.sin(2 * t) / 5])
    matrices = np.array([pauli_matrix(p) for p in h.paulis])
    start, end = 0.2, 0.7

    def rhs(time, vector):
        matrix = np.einsum("j,jab->ab", h.at(time), matrices)
        return (-1j * matrix @ vector.reshape(4, 4)).ravel()

    reference = solve_ivp(rhs, (start, end), np.eye(4, dtype=complex).ravel(),
                          method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.01)
    assert reference.success
    result = synthesize_driven(h, (start, end), rtol=1e-10, atol=1e-12)
    assert result.dimension > 3
    assert np.linalg.norm(circuit_matrix(result.circuit, 2)
                          - reference.y[:, -1].reshape(4, 4), 2) < 2e-9


def test_static_embedding_matches_expm():
    static = hamiltonian({"XX": 0.3, "ZI": -0.5, "IY": 0.2, "II": 0.1})
    result = synthesize_driven(DrivenHamiltonian.from_static(static), (0.4, 1.1))
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - evolution(static, 0.7), 2) < 1e-8


def test_full_su4_closure_matches_expm():
    static = hamiltonian({"XI": 0.2, "ZI": -0.3, "IX": 0.4, "IZ": 0.1, "ZZ": 0.35})
    result = synthesize_driven(DrivenHamiltonian.from_static(static), (0, 0.8))
    assert result.dimension == 15
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - evolution(static, 0.8), 2) < 2e-8


@pytest.mark.parametrize("order", list(permutations(["X", "Y", "Z"])))
def test_every_su2_factor_order(order):
    h, reference = rotating_field()
    result = synthesize_driven(h, (0.1, 0.9), basis_order=order)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - reference(0.1, 0.9), 2) < 2e-8


def test_static_embedding_rejects_nonhermitian_terms():
    with pytest.raises(ValueError, match="Hermitian"):
        DrivenHamiltonian.from_static(hamiltonian({"X": 1j}))


def test_commuting_coefficients_integrate_and_preserve_identity_phase():
    h = DrivenHamiltonian(["ZI", "IZ", "II"], lambda t: [t, 2 * t, 0.7])
    result = synthesize_driven(h, (0.3, 1.1))
    integral = (1.1**2 - 0.3**2) / 2
    matrix = integral * (pauli_matrix("ZI") + 2 * pauli_matrix("IZ")) + 0.56 * np.eye(4)
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - expm(-1j * matrix), 2) < 1e-11
    assert len(result.circuit) == 3
    assert result.chart_restarts == 0


def test_large_central_angles_do_not_force_restarts():
    h = DrivenHamiltonian(["II", "IZ"], lambda t: [1000, -2000])
    result = synthesize_driven(h, (0, 1), max_segments=1)
    target = expm(-1j * (1000 * np.eye(4) - 2000 * pauli_matrix("IZ")))
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - target, 2) < 1e-9
    assert result.chart_restarts == result.rejected_intervals == 0


def test_large_identity_shift_does_not_force_excessive_restarts():
    h, reference = rotating_field()
    shifted = DrivenHamiltonian(["X", "Y", "Z", "I"], lambda t: [*h.at(t), 1000])
    result = synthesize_driven(shifted, (0, 0.7))
    target = np.exp(-700j) * reference(0, 0.7)
    assert len(result.intervals) < 10
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 2e-8


def test_noncommuting_drive_preserves_identity_phase():
    spin, reference = rotating_field()
    h = DrivenHamiltonian(["X", "Y", "Z", "I"], lambda t: [*spin.at(t), t * t])
    result = synthesize_driven(h, (0.2, 0.8))
    target = np.exp(-1j * (0.8**3 - 0.2**3) / 3) * reference(0.2, 0.8)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 2e-8


def test_zero_duration_needs_no_control_evaluation():
    def unused(time):
        raise AssertionError("zero duration must not evaluate controls")

    result = synthesize_driven(DrivenHamiltonian(["X"], unused), (0.4, 0.4))
    assert len(result.circuit) == 0
    assert result.intervals == ()
    assert result.rhs_evaluations == result.chart_restarts == 0


def test_initially_zero_control_is_kept_in_closure():
    result = synthesize_driven(DrivenHamiltonian(["X", "Y"], lambda t: [1, t]), (0, 0.2))
    assert result.basis == ("X", "Y", "Z")


def test_identically_zero_controls_emit_identity():
    result = synthesize_driven(DrivenHamiltonian(["X", "Y"], lambda t: [0, 0]), (0, 5))
    assert result.dimension == 3
    assert len(result.circuit) == 0
    assert result.chart_restarts == 0


def test_restarts_cross_an_euler_chart_singularity():
    # In the X,Y,Z chart a single H=Y solution reaches det M=0 at pi/4.
    h = DrivenHamiltonian(["X", "Y", "Z"], lambda t: [0, 1, 0])
    result = synthesize_driven(h, (0, 2), basis_order=["X", "Y", "Z"])
    assert result.rejected_intervals > 0
    assert result.chart_restarts > 0
    assert np.linalg.norm(circuit_matrix(result.circuit, 1)
                          - expm(-2j * pauli_matrix("Y")), 2) < 1e-11


def test_condition_guard_retries_and_order_can_be_changed():
    h, reference = rotating_field()
    result = synthesize_driven(h, (0.2, 1), condition_limit=1.05,
                               basis_order=["Z", "Y", "X"])
    assert result.rejected_intervals > 0
    assert result.basis == ("Z", "Y", "X")
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - reference(0.2, 1), 2) < 2e-8


def test_adjoint_jacobian_matches_dense_finite_difference():
    basis = driven._closure(["XI", "ZY", "IZ"], 32)
    theta = np.linspace(-0.09, 0.08, len(basis))
    matrix = driven._coordinate_matrix(theta, driven._adjoint_pairs(basis))
    paulis = [pauli_matrix(p) for p in basis]

    def unitary(angles):
        out = np.eye(4, dtype=complex)
        for p, angle in zip(paulis, angles):
            out = out @ expm(-1j * angle * p)
        return out

    base = unitary(theta)
    for j in range(len(basis)):
        offset = np.zeros(len(basis))
        offset[j] = 1e-6
        derivative = (unitary(theta + offset) - unitary(theta - offset)) / 2e-6
        predicted = -1j * sum(c * p for c, p in zip(matrix[:, j], paulis)) @ base
        assert np.linalg.norm(derivative - predicted, 2) < 1e-8


def test_closure_budget_is_checked_before_any_control_evaluation():
    def unused(time):
        raise AssertionError("closure should be declined first")

    with pytest.raises(AlgebraTooLarge, match="max_dimension=2"):
        synthesize_driven(DrivenHamiltonian(["X", "Y"], unused), (0, 1), max_dimension=2)
    with pytest.raises(AlgebraTooLarge):
        synthesize_driven(DrivenHamiltonian(["X", "Y", "Z"], unused), (0, 1), max_dimension=2)


@pytest.mark.parametrize("words", [[], [""], ["A"], ["X", "ZI"], ["X", "X"]])
def test_invalid_words(words):
    with pytest.raises(ValueError):
        DrivenHamiltonian(words, lambda t: [1] * len(words))


@pytest.mark.parametrize("values", [[1j], [np.nan], [np.inf], [1, 2], 1, [[1]]])
def test_invalid_controls(values):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: values), (0, 1))


@pytest.mark.parametrize("kwargs", [
    {"max_dimension": 0}, {"max_dimension": 1.5}, {"max_segments": False},
    {"rtol": 0}, {"atol": np.nan}, {"max_step": 0}, {"max_step": np.nan},
    {"chart_radius": 0}, {"chart_radius": 1}, {"condition_limit": 1},
    {"condition_limit": np.inf}, {"max_rhs_evaluations": -1},
    {"basis_order": ["X", "Z"]}, {"basis_order": ["X", "Y", "Y"]},
])
def test_invalid_numerical_options(kwargs):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X", "Y"], lambda t: [1, 1]), (0, 1), **kwargs)


@pytest.mark.parametrize("span", [(0, np.inf), (0,), (0, 1, 2)])
def test_invalid_time_span(span):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: [1]), span)


def test_work_budgets_raise_instead_of_returning_partial_circuit():
    h, _ = rotating_field()
    with pytest.raises(IntegrationFailure, match="max_segments"):
        synthesize_driven(h, (0, 2), max_segments=1)
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_driven(h, (0, 2), max_rhs_evaluations=1)


def test_solver_failure_is_not_returned_as_success(monkeypatch):
    monkeypatch.setattr(driven, "solve_ivp", lambda *a, **k:
                        SimpleNamespace(success=False, message="failed test integration"))
    with pytest.raises(IntegrationFailure, match="failed test integration"):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: [1]), (0, 1))
