"""End-to-end numerical synthesis, including absolute phase and resource contracts."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy.dense import evolution, pauli_matrix
from lizzy.emission.native import NativeCircuit
from lizzy.hamiltonian import hamiltonian
from lizzy.synthesis.driven import (
    AlgebraTooLarge,
    DrivenHamiltonian,
    IntegrationFailure,
)
from lizzy.synthesis.wei_norman import synthesize_wei_norman
from lizzy.synthesize import synthesize


def test_static_su4_produces_a_concrete_phase_preserving_circuit():
    h = hamiltonian({"XI": 0.2, "ZI": -0.3, "IX": 0.4, "IZ": 0.1, "ZZ": 0.35, "II": 0.7})
    result = synthesize_wei_norman(h, 0.8, rtol=1e-10, atol=1e-12)
    assert isinstance(result.emitted_circuit, NativeCircuit)
    assert result.component_dimensions == (15, 1)
    assert result.dimension == 16
    assert result.width == 2
    assert result.time_span == (0.0, 0.8)
    assert result.two_qubit_gates == result.emitted_circuit.n_2qb_gates()
    assert result.logical_two_qubit_gates == result.circuit.two_qubit_gates
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - evolution(h, 0.8), 2) < 1e-8
    assert not result.error_guaranteed


@pytest.mark.parametrize("split", [True, False], ids=["components", "unsplit"])
def test_commuting_static_components_need_no_ode_evaluations(split):
    h = hamiltonian({"ZZZ": 0.2, "ZII": 0.7, "III": -100.0})
    result = synthesize_wei_norman(h, (1.1, -0.3), split_components=split,
                                   max_rhs_evaluations=1)
    assert result.rhs_evaluations == 0
    assert result.charts == (3 if split else 1)
    assert result.chart_restarts == result.rejected_intervals == 0
    assert result.max_observed_condition == 1.0
    assert len(result.circuit) == 3
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - evolution(h, -1.4), 2) < 1e-11


def test_independent_spins_scale_without_dense_hilbert_space(monkeypatch):
    from lizzy import dense

    def forbidden(*args, **kwargs):
        raise AssertionError("synthesis must not use dense Hilbert-space simulation")

    monkeypatch.setattr(dense, "circuit_matrix", forbidden)
    monkeypatch.setattr(dense, "hamiltonian_matrix", forbidden)
    monkeypatch.setattr(dense, "pauli_matrix", forbidden)
    width = 12
    words = ["I" * q + axis + "I" * (width - q - 1) for q in range(width) for axis in "XZ"]
    drive = DrivenHamiltonian(words, lambda t: np.full(len(words), 0.2))
    result = synthesize_wei_norman(drive, 0.1, max_dimension=3)
    assert result.component_dimensions == (3,) * width
    assert result.dimension == 36
    assert result.two_qubit_gates == 0
    with pytest.raises(AlgebraTooLarge):
        synthesize_wei_norman(drive, 0.1, max_dimension=3, split_components=False)


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_pulse_boundaries_use_one_sided_controls(reverse):
    seen = []

    def coefficients(t):
        seen.append(t)
        if t == 0.5:
            raise AssertionError("jump value must not be used by either pulse")
        return [1.0, 0.0] if t < 0.5 else [0.0, 0.7]

    drive = DrivenHamiltonian(["X", "Z"], coefficients)
    result = synthesize_wei_norman(drive, (1, 0) if reverse else (0, 1), breakpoints=[0.5])
    target = expm(-0.35j * pauli_matrix("Z")) @ expm(-0.5j * pauli_matrix("X"))
    if reverse:
        target = target.conj().T
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - target, 2) < 1e-9
    assert len(result.segments) == 2
    assert all(0 < t < 1 for t in seen)


@pytest.mark.parametrize("reverse", [False, True])
def test_split_driven_pulses_and_identity_match_independent_dense_ode(reverse):
    words = ["XI", "ZI", "IX", "IY", "II"]
    matrices = np.array([pauli_matrix(word) for word in words])
    controls = [lambda t: [0.3 + t, -0.2, 0.4, 0.1*t, 0.17],
                lambda t: [-0.1, 0.4*t, -0.3 + t, 0.2, 0.17 + t]]
    drive = DrivenHamiltonian(words, lambda t: controls[int(t >= 0.5)](t))
    span = (0.8, 0.2) if reverse else (0.2, 0.8)
    pieces = [(0.8, 0.5, 1), (0.5, 0.2, 0)] if reverse else [(0.2, 0.5, 0), (0.5, 0.8, 1)]
    target = np.eye(4, dtype=complex).ravel()
    for start, end, index in pieces:
        # Independent Schrodinger integration, using this pulse's smooth formula
        # even at the endpoint (not the WN coordinates or its clipping helper).
        def rhs(t, vector, control=controls[index]):
            generator = np.einsum("k,kij->ij", control(t), matrices)
            return (-1j * generator @ vector.reshape(4, 4)).ravel()

        reference = solve_ivp(rhs, (start, end), target, method="DOP853",
                              rtol=2e-12, atol=2e-14, max_step=0.01)
        assert reference.success
        target = reference.y[:, -1]
    result = synthesize_wei_norman(drive, span, breakpoints=[0.5], max_dimension=3,
                                   rtol=1e-10, atol=1e-12, max_step=0.02)
    assert result.component_dimensions == (3, 3, 1)
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - target.reshape(4, 4), 2) < 1e-9
    assert result.emitted_circuit.global_phase == pytest.approx(0.297 if reverse else -0.297)


def test_condition_managed_chart_keeps_driven_tfim_compact_and_correct():
    # One representative resource regression, not a model/API Cartesian sweep.
    words = ["ZZI", "IZZ", "XII", "IXI", "IIX"]
    drive = DrivenHamiltonian(words, lambda t: [
        0.7 + 0.12*np.sin(0.9*t), -0.45 + 0.08*np.cos(1.1*t),
        0.35 + 0.11*np.cos(0.7*t), 0.5 + 0.13*np.sin(1.3*t),
        -0.4 + 0.09*np.cos(1.7*t),
    ])
    matrices = np.array([pauli_matrix(word) for word in words])

    def rhs(time, vector):
        generator = np.einsum("j,jab->ab", drive.at(time), matrices)
        return (-1j * generator @ vector.reshape(8, 8)).ravel()

    reference = solve_ivp(rhs, (0, 1.2), np.eye(8, dtype=complex).ravel(),
                          method="DOP853", rtol=3e-13, atol=1e-14, max_step=0.005)
    assert reference.success
    target = reference.y[:, -1].reshape(8, 8)
    options = {"rtol": 1e-10, "atol": 1e-12, "max_step": 0.025}
    compact = synthesize_wei_norman(drive, (0, 1.2), **options)
    legacy = synthesize_wei_norman(drive, (0, 1.2), chart_radius=0.5, **options)
    assert compact.charts == 1
    assert compact.dimension == len(compact.circuit) == 15
    assert compact.chart_restarts == compact.rejected_intervals == 0
    assert compact.two_qubit_gates <= 32
    assert compact.two_qubit_gates < legacy.two_qubit_gates
    assert legacy.chart_restarts > 0
    assert np.linalg.norm(compact.emitted_circuit.get_unitary() - target, 2) < 1e-10
    assert np.linalg.norm(legacy.emitted_circuit.get_unitary() - target, 2) < 1e-10


def test_dimension_checks_precede_all_driven_callback_evaluations():
    def forbidden(t):
        raise AssertionError("all closures must be checked before any control evaluation")

    drive = DrivenHamiltonian(["XII", "ZII", "IXI", "IZI", "IIX", "IIZ", "IZZ"], forbidden)
    with pytest.raises(AlgebraTooLarge, match="max_dimension"):
        synthesize_wei_norman(drive, 0.5, max_dimension=3)
    drive = DrivenHamiltonian(["XI", "ZI", "IX", "IZ"], forbidden)
    with pytest.raises(AlgebraTooLarge, match="max_total_dimension"):
        synthesize_wei_norman(drive, 0.5, max_total_dimension=5)
    with pytest.raises(AlgebraTooLarge, match="declared controls"):
        synthesize_wei_norman(drive, 0.5, max_total_dimension=3)


def test_work_limits_apply_across_components_and_pulses():
    drive = DrivenHamiltonian(["XI", "ZI", "IX", "IZ"], lambda t: [0.2, 0.3, 0.4, -0.1])
    enough = synthesize_wei_norman(drive, 0.1)
    assert len(enough.segments) == 2
    first_work = enough.segments[0].rhs_evaluations
    with pytest.raises(IntegrationFailure, match="max_rhs_evaluations"):
        synthesize_wei_norman(drive, 0.1, max_rhs_evaluations=first_work + 1)
    with pytest.raises(IntegrationFailure, match="max_segments"):
        synthesize_wei_norman(drive, 0.1, max_segments=1)
    with pytest.raises(IntegrationFailure, match="max_segments"):
        synthesize_wei_norman(hamiltonian({"X": 0.2}), 1.0, breakpoints=[0.5], max_segments=1)


def test_zero_duration_never_evaluates_a_driven_callback():
    def forbidden(t):
        raise AssertionError("zero duration must not evaluate controls")

    result = synthesize_wei_norman(DrivenHamiltonian(["XI", "ZI"], forbidden), (0.5, 0.5))
    assert not result.circuit.rotations
    assert result.segments == ()
    assert result.charts == result.rhs_evaluations == 0
    assert result.dimension == 3
    assert np.array_equal(result.emitted_circuit.get_unitary(), np.eye(4))


def test_basis_order_is_prepared_once_per_component_across_pulses(monkeypatch):
    from lizzy.synthesis import driven, wei_norman

    calls = {"closure": 0, "adjoints": 0}
    real_closure, real_pairs = driven._closure, driven._adjoint_pairs

    def counted_closure(*args):
        calls["closure"] += 1
        return real_closure(*args)

    def counted_pairs(*args):
        calls["adjoints"] += 1
        return real_pairs(*args)

    monkeypatch.setattr(driven, "_closure", counted_closure)
    monkeypatch.setattr(wei_norman, "_closure", counted_closure)
    monkeypatch.setattr(driven, "_adjoint_pairs", counted_pairs)
    drive = DrivenHamiltonian(["XI", "ZI", "IX", "IZ"], lambda t: [0.2, 0.3, 0.4, -0.1])
    result = synthesize_wei_norman(drive, 0.1, breakpoints=[0.05],
                                   basis_order=["ZI", "IZ", "YI", "IY", "XI", "IX"])
    assert result.component_bases == (("ZI", "YI", "XI"), ("IZ", "IY", "IX"))
    assert len(result.segments) == 4
    assert calls == {"closure": 2, "adjoints": 2}
    target = expm(-0.1j * sum(c * pauli_matrix(w) for c, w in zip(drive.at(0), drive.paulis)))
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - target, 2) < 1e-9
    with pytest.raises(ValueError, match="full Pauli closure"):
        synthesize_wei_norman(drive, 0.1, basis_order=["XI"])


def test_main_static_api_exposes_explicit_numerical_route():
    h = hamiltonian({"X": 0.5, "Z": -0.2})
    result = synthesize(h, 0.3, method="wei-norman", numerical_options={"rtol": 1e-10})
    assert not result.error_guaranteed
    assert result.numerical is not None
    assert result.emission_is_concrete
    assert result.emission is result.numerical.emission
    assert result.emitted_circuit is result.numerical.emitted_circuit
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - evolution(h, 0.3), 2) < 1e-8
    regular = synthesize(h, 0.3)
    assert regular.error_guaranteed
    assert regular.numerical is None


@pytest.mark.parametrize("kwargs", [{"steps": 1}, {"randomized": True}])
def test_main_api_rejects_conflicting_product_formula_controls(kwargs):
    with pytest.raises(ValueError, match="product-formula controls"):
        synthesize(hamiltonian({"X": 1}), 1, method="wei-norman", **kwargs)


def test_main_api_rejects_unknown_or_ignored_numerical_options():
    with pytest.raises(ValueError, match="method"):
        synthesize(hamiltonian({"X": 1}), 1, method="unknown")
    with pytest.raises(ValueError, match="requires"):
        synthesize(hamiltonian({"X": 1}), 1, numerical_options={})


def test_main_api_rejects_falsey_non_dictionary_options():
    with pytest.raises(TypeError, match="dict"):
        synthesize(hamiltonian({"X": 1}), 1, method="wei-norman", numerical_options=[])


def test_main_api_honestly_labels_explicitly_skipped_emission():
    result = synthesize(hamiltonian({"X": 1}), 1, method="wei-norman",
                        numerical_options={"emission": "none"})
    assert result.emission_backend == result.numerical.emission_backend == "none"
    assert not result.emission_is_concrete
    assert result.emission is None
    assert result.emitted_circuit is result.circuit


@pytest.mark.parametrize("kwargs", [
    {"breakpoints": [0]}, {"breakpoints": [0.7, 0.2]},
    {"breakpoints": [np.nan]}, {"breakpoints": [[0.5]]},
    {"max_total_dimension": True}, {"emission": "unknown"},
])
def test_pipeline_options_are_validated_for_direct_static_evolution(kwargs):
    with pytest.raises(ValueError):
        synthesize_wei_norman(hamiltonian({"X": 1}), 1, **kwargs)


def test_finite_endpoints_with_overflowing_duration_are_rejected():
    with pytest.raises(ValueError, match="time_span"):
        synthesize_wei_norman(hamiltonian({"X": 1}), [-1e308, 1e308])


def test_invalid_types_and_nonhermitian_static_inputs():
    with pytest.raises(TypeError, match="hamiltonian"):
        synthesize_wei_norman({"X": 1}, 1)
    with pytest.raises(TypeError, match="boolean"):
        synthesize_wei_norman(hamiltonian({"X": 1}), 1, split_components=1)
    with pytest.raises(ValueError, match="Hermitian"):
        synthesize_wei_norman(hamiltonian({"X": 1j}), 1)


def test_static_overflow_raises_instead_of_emitting_nonfinite_angles():
    with pytest.raises(IntegrationFailure, match="nonfinite"):
        synthesize_wei_norman(hamiltonian({"X": 1e308}), 10)
