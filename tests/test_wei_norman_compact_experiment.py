"""Independent correctness and failure checks for the isolated chart experiment."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from experiments import wei_norman_compact as compact
from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import AlgebraTooLarge, DrivenHamiltonian, IntegrationFailure
from lizzy.native import ladder_circuit


def _dense_reference(drive, span):
    matrices = np.array([pauli_matrix(word) for word in drive.paulis])
    dimension = 2**drive.n_qubits

    def rhs(time, vector):
        generator = np.einsum("j,jab->ab", drive.at(time), matrices)
        return (-1j * generator @ vector.reshape(dimension, dimension)).ravel()

    solution = solve_ivp(
        rhs, span, np.eye(dimension, dtype=complex).ravel(),
        method="DOP853", rtol=3e-13, atol=1e-14, max_step=0.005,
    )
    assert solution.success
    return solution.y[:, -1].reshape(dimension, dimension)


@pytest.mark.parametrize("driven", [False, True])
def test_tfim_uses_one_full_product_with_strict_native_accuracy(driven):
    words = ["ZZI", "IZZ", "XII", "IXI", "IIX"]
    if driven:
        controls = lambda t: [
            0.7 + 0.12*np.sin(0.9*t), -0.45 + 0.08*np.cos(1.1*t),
            0.35 + 0.11*np.cos(0.7*t), 0.5 + 0.13*np.sin(1.3*t),
            -0.4 + 0.09*np.cos(1.7*t),
        ]
        span = (0, 1.2)
    else:
        controls = lambda t: [0.7, -0.45, 0.35, 0.5, -0.4]
        span = (0, 1.0)
    drive = DrivenHamiltonian(words, controls)
    result = compact.synthesize_compact(drive, span)
    reference = _dense_reference(drive, span)
    native = ladder_circuit(result.circuit, 3)
    assert result.dimension == len(result.circuit) == 15
    assert result.intervals == (span,)
    assert result.chart_restarts == result.rejected_intervals == 0
    assert result.max_observed_condition < 15
    assert native.two_qubit_gates == 32
    assert np.linalg.norm(native.get_unitary() - reference, 2) < 1e-10
    assert not result.error_guaranteed


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("span", [(0.2, 0.9), (0.9, 0.2)])
def test_general_drive_backwards_and_identity_phase(span, restart):
    drive = DrivenHamiltonian(
        ["XI", "ZI", "XX", "IY", "II"],
        lambda t: [0.3 + t/7, 0.2*np.cos(t), 0.4, np.sin(2*t)/5, 17 + t*t],
    )
    result = compact.synthesize_compact(drive, span, restart=restart)
    assert np.linalg.norm(
        circuit_matrix(result.circuit, 2) - _dense_reference(drive, span), 2
    ) < 2e-10
    assert result.intervals[0][0] == span[0]
    assert result.intervals[-1][1] == span[1]


def test_large_central_angles_are_not_limited():
    drive = DrivenHamiltonian(["II", "IZ"], lambda t: [1000, -2000])
    result = compact.synthesize_compact(drive, (0, 1), max_segments=1)
    target = expm(-1j * (1000*np.eye(4) - 2000*pauli_matrix("IZ")))
    assert result.max_observed_condition == 1
    assert len(result.intervals) == 1
    assert np.linalg.norm(circuit_matrix(result.circuit, 2) - target, 2) < 1e-9


def test_zero_duration_does_not_evaluate_callback():
    def unused(time):
        raise AssertionError("zero duration must not evaluate controls")

    result = compact.synthesize_compact(DrivenHamiltonian(["X", "Y"], unused), (0.4, 0.4))
    assert result.basis == ("X", "Y", "Z")
    assert result.intervals == ()
    assert result.rhs_evaluations == len(result.circuit) == 0


def _singular_drive():
    # With XYZ order, theta_Y=t reaches a singular Jacobian at t=pi/4.
    return DrivenHamiltonian(["X", "Y", "Z"], lambda t: [0, 1, 0])


def test_single_chart_refuses_sampled_singularity():
    with pytest.raises(IntegrationFailure, match="single-chart.*condition_limit"):
        compact.synthesize_compact(
            _singular_drive(), (0, 2), basis_order=["X", "Y", "Z"], max_step=0.005,
        )


@pytest.mark.parametrize("span", [(0, 2), (2, 0)])
def test_condition_restarts_cross_euler_singularity(span):
    result = compact.synthesize_compact(
        _singular_drive(), span, basis_order=["X", "Y", "Z"],
        condition_limit=20, restart=True,
    )
    assert result.chart_restarts > 0
    assert result.rejected_intervals > 0
    assert all(left[1] == right[0] for left, right in zip(result.intervals, result.intervals[1:]))
    target = expm(-1j * (span[1] - span[0]) * pauli_matrix("Y"))
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - target, 2) < 1e-11


def test_condition_guard_is_order_dependent_without_changing_physics():
    result = compact.synthesize_compact(
        _singular_drive(), (0, 2), basis_order=["Y", "X", "Z"],
    )
    assert result.intervals == ((0, 2),)
    assert result.basis == ("Y", "X", "Z")
    assert np.linalg.norm(
        circuit_matrix(result.circuit, 1) - expm(-2j*pauli_matrix("Y")), 2
    ) < 1e-11


def test_closure_cap_precedes_controls():
    def unused(time):
        raise AssertionError("closure must be checked first")

    with pytest.raises(AlgebraTooLarge, match="max_dimension=2"):
        compact.synthesize_compact(DrivenHamiltonian(["X", "Y"], unused), (0, 1), max_dimension=2)


@pytest.mark.parametrize("kwargs, message", [
    ({"max_rhs_evaluations": 1}, "max_rhs_evaluations"),
    ({"restart": True, "max_segments": 1}, "max_segments"),
])
def test_work_caps_fail_without_returning_partial_output(kwargs, message):
    with pytest.raises(IntegrationFailure, match=message):
        compact.synthesize_compact(_singular_drive(), (0, 2), **kwargs)


def test_rhs_budget_is_global_across_failed_restarts(monkeypatch):
    attempts = []

    def rejected_solver(rhs, span, initial, **kwargs):
        attempts.append(span)
        for _ in range(21):
            rhs(span[0], initial)
        # Deliberately request a singular chart after spending work.
        rhs(span[0], np.array([0, np.pi/4, 0]))
        raise AssertionError("singular RHS should have raised")

    monkeypatch.setattr(compact, "solve_ivp", rejected_solver)
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        compact.synthesize_compact(
            _singular_drive(), (0, 2), restart=True, max_rhs_evaluations=30,
        )
    assert len(attempts) == 2


@pytest.mark.parametrize("bad_theta", [[0, np.pi/4, 0], [0, np.nan, 0]])
def test_accepted_mesh_is_checked_even_if_rhs_never_saw_bad_state(monkeypatch, bad_theta):
    def unchecked_mesh(*args, **kwargs):
        return SimpleNamespace(
            success=True, t=np.array([0, 0.5, 1]),
            y=np.array([[0, 0, 0], bad_theta, [0, 0, 0]]).T,
        )

    monkeypatch.setattr(compact, "solve_ivp", unchecked_mesh)
    with pytest.raises(IntegrationFailure, match="single-chart integration failed"):
        compact.synthesize_compact(_singular_drive(), (0, 1))


def test_unsuccessful_solver_is_not_returned_as_success(monkeypatch):
    monkeypatch.setattr(
        compact, "solve_ivp", lambda *args, **kwargs: SimpleNamespace(success=False, message="failed"),
    )
    with pytest.raises(IntegrationFailure, match="ODE solver failed"):
        compact.synthesize_compact(_singular_drive(), (0, 1))


@pytest.mark.parametrize("kwargs", [
    {"max_dimension": 0}, {"max_rhs_evaluations": False}, {"max_segments": 1.5},
    {"rtol": 0}, {"atol": np.nan}, {"max_step": 0}, {"max_step": np.nan},
    {"condition_limit": 1}, {"condition_limit": np.inf},
    {"basis_order": ["X", "Y"]}, {"basis_order": ["X", "Y", "Y"]},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        compact.synthesize_compact(_singular_drive(), (0, 1), **kwargs)


def test_restart_requires_boolean():
    with pytest.raises(TypeError, match="restart must be a boolean"):
        compact.synthesize_compact(_singular_drive(), (0, 1), restart="yes")


@pytest.mark.parametrize("span", [(0,), (0, 1, 2), (0, np.inf)])
def test_invalid_time_span(span):
    with pytest.raises(ValueError):
        compact.synthesize_compact(_singular_drive(), span)
