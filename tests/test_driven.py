"""Wei–Norman equations, independent evolution references, and chart safety."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy import driven
from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    IntegrationFailure,
    WeiNormanBasis,
    synthesize_driven,
)


def rotating_field(words=("X", "Y", "Z")):
    omega, detuning = 1.7, 0.3
    h = DrivenHamiltonian(words, lambda t: [np.cos(omega * t), np.sin(omega * t), detuning])

    def reference(start, end):
        x, _, z = map(pauli_matrix, words)
        return (expm(-1j * omega * end * z / 2)
                @ expm(-1j * (end - start) * (x + (detuning - omega / 2) * z))
                @ expm(1j * omega * start * z / 2))

    return h, reference


@pytest.mark.parametrize("words, span, chart_radius", [
    pytest.param(("XIX", "XIY", "IIZ"), (0.3, 1.8), None, id="encoded-nonzero-start"),
    pytest.param(("XIX", "XIY", "IIZ"), (1.8, 0.3), 0.5, id="encoded-legacy-restarts"),
])
def test_driven_rotation_matches_analytic_solution(words, span, chart_radius):
    h, reference = rotating_field(words)
    result = synthesize_driven(h, span, chart_radius=chart_radius, max_step=0.025)
    actual = circuit_matrix(result.circuit, h.n_qubits)
    assert np.linalg.norm(actual - reference(*span), 2) < 2e-8
    assert result.dimension == 3
    if chart_radius is not None:
        assert result.chart_restarts > 0
    assert not result.error_guaranteed
    assert result.intervals[0][0] == span[0]
    assert result.intervals[-1][1] == span[1]
    assert all(a[1] == b[0] for a, b in zip(result.intervals, result.intervals[1:]))
    assert set(result.circuit.provenance) == {"wei-norman"}


def test_general_two_qubit_drive_against_independent_dense_ode():
    h = DrivenHamiltonian(["XI", "ZI", "XX", "IY", "II"],
                          lambda t: [0.3 + t / 7, 0.2 * np.cos(t), 0.4,
                                     np.sin(2 * t) / 5, 17 + t*t])
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


def test_large_commuting_angles_integrate_without_restarts_and_preserve_phase():
    h = DrivenHamiltonian(["ZI", "IZ", "II"], lambda t: [1000*t, -2000*t, 1000])
    result = synthesize_driven(h, (0.3, 1.1), max_segments=1)
    integral = (1.1**2 - 0.3**2) / 2
    matrix = integral * (1000*pauli_matrix("ZI") - 2000*pauli_matrix("IZ")) + 800*np.eye(4)
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - expm(-1j * matrix), 2) < 1e-9
    assert len(result.circuit) == 3
    assert result.chart_restarts == result.rejected_intervals == 0


def test_large_identity_shift_does_not_force_excessive_restarts():
    h, reference = rotating_field()
    shifted = DrivenHamiltonian(["X", "Y", "Z", "I"], lambda t: [*h.at(t), 1000 + t*t])
    result = synthesize_driven(shifted, (0, 0.7))
    target = np.exp(-1j*(700 + 0.7**3/3)) * reference(0, 0.7)
    assert len(result.intervals) < 10
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 2e-8


def test_zero_duration_needs_no_control_evaluation():
    def unused(time):
        raise AssertionError("zero duration must not evaluate controls")

    result = synthesize_driven(DrivenHamiltonian(["X"], unused), (0.4, 0.4))
    assert len(result.circuit) == 0
    assert result.intervals == ()
    assert result.rhs_evaluations == result.chart_restarts == 0


def test_identically_zero_controls_emit_identity():
    result = synthesize_driven(DrivenHamiltonian(["X", "Y"], lambda t: [0, 0]), (0, 5))
    assert result.dimension == 3
    assert len(result.circuit) == 0
    assert result.chart_restarts == 0


def test_prepared_basis_reuses_immutable_structure_not_integration_state(monkeypatch):
    controls = ["X", "Z", "I"]
    prepared = WeiNormanBasis(controls, basis_order=["Z", "Y", "X", "I"])
    controls[0] = "Y"
    assert prepared.controls == ("X", "Z", "I")
    with pytest.raises(ValueError, match="WRITEABLE"):
        prepared._pairs[0][0].setflags(write=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared integration must not rebuild its algebra")

    monkeypatch.setattr(driven, "_closure", forbidden)
    monkeypatch.setattr(driven, "_adjoint_pairs", forbidden)
    h = DrivenHamiltonian(prepared.controls, lambda t: [0.7, -0.2, 0.13])
    options = {"prepared_basis": prepared, "max_step": 0.05}
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_driven(h, (0, 0.25), max_rhs_evaluations=1, **options)
    first = synthesize_driven(h, (0, 0.25), **options)
    again = synthesize_driven(h, (0, 0.25), **options)
    assert first.basis == again.basis == prepared.words
    assert first.rhs_evaluations == again.rhs_evaluations
    assert first.intervals == again.intervals
    assert first.circuit is not again.circuit
    generator = sum(c * pauli_matrix(w) for c, w in zip(h.at(0), h.paulis))
    target = expm(-0.25j * generator)
    assert np.linalg.norm(circuit_matrix(first.circuit, 1) - target, 2) < 1e-9
    assert np.array_equal(circuit_matrix(first.circuit, 1), circuit_matrix(again.circuit, 1))
    with pytest.raises(AlgebraTooLarge, match="max_dimension=3"):
        synthesize_driven(h, (0, 0.25), max_dimension=3, **options)
    with pytest.raises(ValueError, match="ordered Hamiltonian controls"):
        synthesize_driven(DrivenHamiltonian(["Z", "X", "I"], h.coefficients),
                          (0, 0.25), **options)


def test_full_su4_jacobian_matches_paper_prefix_conjugation():
    # An independent dense product rule, at large angles and a shuffled basis:
    # no ODE, finite-difference approximation, or truncated BCH reference.
    words = ["".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")]
    closure = driven._closure(words, 15)
    rng = np.random.default_rng(20260925)
    basis = [closure[index] for index in rng.permutation(len(closure))]
    theta = rng.uniform(-2*np.pi, 2*np.pi, len(basis))
    pairs = driven._adjoint_pairs(basis)
    jacobian = driven._coordinate_matrix(theta, pairs)
    assert np.array_equal(driven._coordinate_matrix(np.zeros(len(basis)), pairs), np.eye(15))
    matrices = np.array([pauli_matrix(word) for word in basis])
    prefix = np.eye(4, dtype=complex)
    for column, (angle, pauli) in enumerate(zip(theta, matrices)):
        expected = prefix @ pauli @ prefix.conj().T
        represented = np.einsum("j,jab->ab", jacobian[:, column], matrices)
        assert np.linalg.norm(represented - expected, 2) < 1e-12
        prefix = prefix @ expm(-1j * angle * pauli)


@pytest.mark.parametrize("angles", [(0.31, -0.27, 0.49), (-1.2, np.pi/4, 0.73)],
                         ids=["regular", "singular"])
def test_xyz_jacobian_and_singular_set_match_closed_form(angles):
    # For exp(-ixX) exp(-iyY) exp(-izZ), det(M)=cos(2y) in Lizzy's -iP units.
    x, y, _ = angles
    cosine_x, sine_x = np.cos(2*x), np.sin(2*x)
    cosine_y, sine_y = np.cos(2*y), np.sin(2*y)
    expected = np.array([
        [1, 0, sine_y],
        [0, cosine_x, -sine_x*cosine_y],
        [0, sine_x, cosine_x*cosine_y],
    ])
    basis = driven._closure(["X", "Y", "Z"], 3)
    actual = driven._coordinate_matrix(np.array(angles), driven._adjoint_pairs(basis))
    assert np.allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.linalg.det(actual) == pytest.approx(cosine_y, abs=1e-14)


def test_closure_budget_is_checked_before_any_control_evaluation():
    def unused(time):
        raise AssertionError("closure should be declined first")

    with pytest.raises(AlgebraTooLarge, match="max_dimension=2"):
        synthesize_driven(DrivenHamiltonian(["X", "Y"], unused), (0, 1), max_dimension=2)
    with pytest.raises(AlgebraTooLarge):
        synthesize_driven(DrivenHamiltonian(["X", "Y", "Z"], unused), (0, 1), max_dimension=2)


@pytest.mark.parametrize("words", [[], ["A"], ["X", "ZI"], ["X", "X"]])
def test_invalid_words(words):
    with pytest.raises(ValueError):
        DrivenHamiltonian(words, lambda t: [1] * len(words))


@pytest.mark.parametrize("values", [[1j], [np.nan], [1, 2]])
def test_invalid_controls(values):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: values), (0, 1))


@pytest.mark.parametrize("kwargs", [
    {"max_dimension": 0}, {"max_dimension": 1.5}, {"max_segments": False},
    {"rtol": 0}, {"atol": np.nan}, {"max_step": 0},
    {"chart_radius": 1}, {"condition_limit": 1},
    {"basis_order": ["X", "Z"]}, {"basis_order": ["X", "Y", "Y"]},
])
def test_invalid_numerical_options(kwargs):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X", "Y"], lambda t: [1, 1]), (0, 1), **kwargs)


@pytest.mark.parametrize("span", [(0, np.inf), (0,)])
def test_invalid_time_span(span):
    with pytest.raises(ValueError):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: [1]), span)


def _singular_drive():
    # Initially inactive X/Z still belong to the declared closure. The XYZ
    # Jacobian is singular at theta_Y=pi/4; max_step resolves the guard region.
    return DrivenHamiltonian(["X", "Y", "Z"], lambda t: [0, 1, 0])


@pytest.mark.parametrize("span", [(0, 2), (2, 0)])
def test_condition_restarts_cross_sampled_singularity(span):
    result = synthesize_driven(_singular_drive(), span, basis_order=["X", "Y", "Z"],
                               max_step=0.005, condition_limit=20)
    assert result.basis == ("X", "Y", "Z")
    assert result.chart_restarts > 0
    assert result.rejected_intervals > 0
    assert all(left[1] == right[0] for left, right in zip(result.intervals, result.intervals[1:]))
    target = expm(-1j*(span[1] - span[0])*pauli_matrix("Y"))
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 1e-11


def test_singularity_refuses_output_when_chart_budget_is_insufficient():
    with pytest.raises(IntegrationFailure, match="max_segments"):
        synthesize_driven(_singular_drive(), (0, 2), max_step=0.005, max_segments=1)


@pytest.mark.parametrize("bad_theta", [[0, np.pi/4, 0], [0, np.nan, 0]],
                         ids=["singular", "nonfinite"])
def test_bad_accepted_mesh_state_rejects_whole_attempt(monkeypatch, bad_theta):
    real_solver = driven.solve_ivp
    attempts = []

    def inject_bad_mesh_once(*args, **kwargs):
        attempts.append(args[1])
        if len(attempts) == 1:
            # Endpoints are healthy; the interior accepted state must be checked.
            return SimpleNamespace(success=True, t=np.array([0, 0.5, 1]),
                                   y=np.array([[0, 0, 0], bad_theta, [0, 0, 0]]).T)
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(driven, "solve_ivp", inject_bad_mesh_once)
    drive = DrivenHamiltonian(["X", "Y", "Z"], lambda t: [1, 0, 0])
    result = synthesize_driven(drive, (0, 1), max_step=0.025)
    assert result.rejected_intervals == 1
    assert result.intervals[0] == (0, 0.5)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - expm(-1j*pauli_matrix("X")), 2) < 1e-11


def test_rhs_and_control_evaluation_budget_is_global_across_retries(monkeypatch):
    attempts, controls = [], []

    def reject_chart(rhs, span, initial, **kwargs):
        attempts.append(span)
        for _ in range(21):
            rhs(span[0], initial)
        rhs(span[0], np.array([0, np.pi/4, 0]))
        raise AssertionError("singular chart must reject RHS")

    def coefficients(time):
        controls.append(time)
        return [0, 1, 0]

    monkeypatch.setattr(driven, "solve_ivp", reject_chart)
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_driven(DrivenHamiltonian(["X", "Y", "Z"], coefficients),
                          (0, 2), max_rhs_evaluations=30)
    assert len(attempts) == 2
    assert len(controls) == 30


def test_nonfinite_coordinate_derivative_is_rejected_before_ode_use(monkeypatch):
    real_solve = driven.np.linalg.solve
    calls = []

    def nonfinite_once(matrix, coefficients):
        calls.append(None)
        if len(calls) == 1:
            return np.full(len(coefficients), np.inf)
        return real_solve(matrix, coefficients)

    monkeypatch.setattr(driven.np.linalg, "solve", nonfinite_once)
    drive = DrivenHamiltonian(["X", "Y", "Z"], lambda t: [1, 0, 0])
    result = synthesize_driven(drive, (0, 1), max_step=0.025)
    assert result.rejected_intervals == 1
    assert result.intervals[0] == (0, 0.5)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - expm(-1j*pauli_matrix("X")), 2) < 1e-11


def test_solver_failure_is_not_returned_as_success(monkeypatch):
    monkeypatch.setattr(driven, "solve_ivp", lambda *a, **k:
                        SimpleNamespace(success=False, message="failed test integration"))
    with pytest.raises(IntegrationFailure, match="failed test integration"):
        synthesize_driven(DrivenHamiltonian(["X"], lambda t: [1]), (0, 1))
