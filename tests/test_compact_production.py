"""Regressions for production condition-managed Wei--Norman coordinates."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy import driven
from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import DrivenHamiltonian, IntegrationFailure, synthesize_driven
from lizzy.native import ladder_circuit
from lizzy.wei_norman import synthesize_wei_norman


def _reference(drive, span):
    """Independent Schrodinger integration, not Lie coordinates."""
    matrices = np.array([pauli_matrix(word) for word in drive.paulis])
    size = 2**drive.n_qubits

    def rhs(time, vector):
        generator = np.einsum("j,jab->ab", drive.at(time), matrices)
        return (-1j * generator @ vector.reshape(size, size)).ravel()

    solution = solve_ivp(
        rhs, span, np.eye(size, dtype=complex).ravel(),
        method="DOP853", rtol=3e-13, atol=1e-14, max_step=0.005,
    )
    assert solution.success
    return solution.y[:, -1].reshape(size, size)


@pytest.mark.parametrize("time_dependent", [False, True])
@pytest.mark.parametrize("entrypoint", [synthesize_driven, synthesize_wei_norman])
def test_default_tfim_uses_one_chart_and_32_concrete_cx(entrypoint, time_dependent):
    words = ["ZZI", "IZZ", "XII", "IXI", "IIX"]
    if time_dependent:
        coefficients = lambda t: [
            0.7 + 0.12*np.sin(0.9*t), -0.45 + 0.08*np.cos(1.1*t),
            0.35 + 0.11*np.cos(0.7*t), 0.5 + 0.13*np.sin(1.3*t),
            -0.4 + 0.09*np.cos(1.7*t),
        ]
        span = (0, 1.2)
    else:
        coefficients = lambda t: [0.7, -0.45, 0.35, 0.5, -0.4]
        span = (0, 1)
    drive = DrivenHamiltonian(words, coefficients)
    options = {"rtol": 1e-10, "atol": 1e-12, "max_step": 0.025}
    compact = entrypoint(drive, span, **options)
    legacy = entrypoint(drive, span, chart_radius=0.5, **options)
    native = ladder_circuit(compact.circuit, 3)
    reference = _reference(drive, span)
    charts = len(compact.intervals) if entrypoint is synthesize_driven else compact.charts
    assert charts == 1
    assert compact.dimension == len(compact.circuit) == 15
    assert compact.chart_restarts == compact.rejected_intervals == 0
    assert compact.max_observed_condition < 15
    assert native.two_qubit_gates == 32
    assert len(legacy.circuit) == 120
    assert ladder_circuit(legacy.circuit, 3).two_qubit_gates == 256
    assert np.linalg.norm(native.get_unitary() - reference, 2) < 1e-10
    assert np.linalg.norm(circuit_matrix(legacy.circuit, 3) - reference, 2) < 1e-10
    assert not compact.error_guaranteed
    if entrypoint is synthesize_wei_norman:
        assert compact.two_qubit_gates == 32
        assert np.linalg.norm(compact.emitted_circuit.get_unitary() - reference, 2) < 1e-10


@pytest.mark.parametrize("entrypoint", [synthesize_driven, synthesize_wei_norman])
@pytest.mark.parametrize("span", [(0.2, 0.9), (0.9, 0.2)])
def test_compact_strict_phase_and_backwards_evolution(entrypoint, span):
    drive = DrivenHamiltonian(
        ["XI", "ZI", "XX", "IY", "II"],
        lambda t: [0.3 + t/7, 0.2*np.cos(t), 0.4, np.sin(2*t)/5, 17 + t*t],
    )
    result = entrypoint(drive, span, max_step=0.025, rtol=1e-10, atol=1e-12)
    actual = ladder_circuit(result.circuit, 2).get_unitary()
    assert np.linalg.norm(actual - _reference(drive, span), 2) < 2e-10


def _singular_drive():
    # Declared XYZ closure includes initially inactive X/Z. The XYZ Jacobian is
    # singular at theta_Y=pi/4; the selected max_step resolves its guard region.
    return DrivenHamiltonian(["X", "Y", "Z"], lambda t: [0, 1, 0])


@pytest.mark.parametrize("span", [(0, 2), (2, 0)])
def test_condition_restarts_cross_sampled_singularity(span):
    result = synthesize_driven(
        _singular_drive(), span, basis_order=["X", "Y", "Z"],
        max_step=0.005, condition_limit=20,
    )
    assert result.chart_restarts > 0
    assert result.rejected_intervals > 0
    assert all(left[1] == right[0] for left, right in zip(result.intervals, result.intervals[1:]))
    assert np.linalg.norm(
        circuit_matrix(result.circuit, 1)
        - expm(-1j*(span[1] - span[0])*pauli_matrix("Y")), 2
    ) < 1e-11


def test_singularity_refuses_output_when_one_chart_budget_is_insufficient():
    with pytest.raises(IntegrationFailure, match="max_segments"):
        synthesize_driven(_singular_drive(), (0, 2), max_step=0.005, max_segments=1)


def test_explicit_none_is_a_valid_condition_only_policy():
    drive = DrivenHamiltonian(["X", "Y", "Z"], lambda t: [1, 0, 0])
    result = synthesize_driven(drive, (0, 2), chart_radius=None, max_step=0.025, max_segments=1)
    assert result.intervals == ((0, 2),)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - expm(-2j*pauli_matrix("X")), 2) < 1e-11


@pytest.mark.parametrize("bad_theta", [[0, np.pi/4, 0], [0, np.nan, 0]])
def test_bad_accepted_mesh_state_rejects_whole_attempt(monkeypatch, bad_theta):
    real_solver = driven.solve_ivp
    attempts = []

    def inject_bad_mesh_once(*args, **kwargs):
        attempts.append(args[1])
        if len(attempts) == 1:
            # Endpoints are healthy; an interior accepted state must be checked.
            return SimpleNamespace(
                success=True, t=np.array([0, 0.5, 1]),
                y=np.array([[0, 0, 0], bad_theta, [0, 0, 0]]).T,
            )
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(driven, "solve_ivp", inject_bad_mesh_once)
    drive = DrivenHamiltonian(["X", "Y", "Z"], lambda t: [1, 0, 0])
    result = synthesize_driven(drive, (0, 1), max_step=0.025)
    assert result.rejected_intervals == 1
    assert result.intervals[0] == (0, 0.5)
    assert np.linalg.norm(circuit_matrix(result.circuit, 1) - expm(-1j*pauli_matrix("X")), 2) < 1e-11


def test_rhs_budget_is_global_across_condition_retries(monkeypatch):
    attempts = []

    def reject_chart(rhs, span, initial, **kwargs):
        attempts.append(span)
        for _ in range(21):
            rhs(span[0], initial)
        rhs(span[0], np.array([0, np.pi/4, 0]))
        raise AssertionError("singular chart must reject RHS")

    monkeypatch.setattr(driven, "solve_ivp", reject_chart)
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_driven(_singular_drive(), (0, 2), max_rhs_evaluations=30)
    assert len(attempts) == 2


def test_no_control_evaluations_exceed_rhs_budget():
    calls = []

    def coefficients(time):
        calls.append(time)
        return [1, 0, 0]

    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_driven(
            DrivenHamiltonian(["X", "Y", "Z"], coefficients), (0, 1), max_rhs_evaluations=1,
        )
    assert len(calls) == 1


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
