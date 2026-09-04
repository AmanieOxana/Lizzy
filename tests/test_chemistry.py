import numpy as np
import pytest
from scipy.linalg import expm

from lizzy._chemistry_extensions import prune_double_factorization
from lizzy.chemistry import (
    MolecularHamiltonian,
    factorize_molecular_ffsim,
    fermion_operator,
    fermion_operator_openfermion,
    interaction_operator_openfermion,
    molecular_from_openfermion_ffsim,
    synthesize_molecular,
    synthesize_molecular_ffsim,
    to_ffsim,
    to_pauli,
    to_pauli_openfermion,
)
from lizzy.dense import hamiltonian_matrix, infidelity
from lizzy.hamiltonian import terms_of
from lizzy.hamlib import load_molecular


def _molecule() -> MolecularHamiltonian:
    one = np.array([[-0.8, 0.12], [0.12, -0.35]])
    factor = np.array([[0.7, 0.09], [0.09, 0.4]])
    two = np.einsum("pq,rs->pqrs", factor, factor)
    return MolecularHamiltonian(one, two, constant=0.23)


def _two_factor_molecule() -> MolecularHamiltonian:
    first = np.array([[0.7, 0.09], [0.09, 0.4]])
    second = np.array([[0.2, -0.12], [-0.12, 0.5]])
    return MolecularHamiltonian(
        _molecule().one_body_tensor,
        np.einsum("pq,rs->pqrs", first, first)
        + np.einsum("pq,rs->pqrs", second, second),
        constant=0.07,
    )


def _manual_fermion_operator(molecular, qubit_order="interleaved"):
    """Independent reference for the ffsim tensor convention."""
    import openfermion

    n = molecular.n_orbitals

    def mode(orbital, spin):
        if qubit_order == "interleaved":
            return 2 * orbital + spin
        return orbital + spin * n

    result = openfermion.FermionOperator((), molecular.constant)
    for p in range(n):
        for q in range(n):
            for spin in (0, 1):
                result += openfermion.FermionOperator(
                    ((mode(p, spin), 1), (mode(q, spin), 0)),
                    molecular.one_body_tensor[p, q],
                )
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    for spin_p in (0, 1):
                        for spin_r in (0, 1):
                            result += openfermion.FermionOperator(
                                (
                                    (mode(p, spin_p), 1),
                                    (mode(r, spin_r), 1),
                                    (mode(s, spin_r), 0),
                                    (mode(q, spin_p), 0),
                                ),
                                0.5 * molecular.two_body_tensor[p, q, r, s],
                            )
    return result


@pytest.mark.parametrize(
    ("one", "two", "constant"),
    [
        (np.array([[np.nan]]), np.zeros((1, 1, 1, 1)), 0.0),
        (np.zeros((1, 1)), np.array([[[[np.inf]]]]), 0.0),
        (np.zeros((1, 1)), np.zeros((1, 1, 1, 1)), 1j),
        (
            np.zeros((2, 2)),
            np.arange(16, dtype=float).reshape(2, 2, 2, 2),
            0.0,
        ),
    ],
)
def test_molecular_ir_rejects_invalid_numeric_data(one, two, constant):
    with pytest.raises(ValueError):
        MolecularHamiltonian(one, two, constant)


def test_openfermion_and_ffsim_adapters_roundtrip_tensors():
    ffsim = pytest.importorskip("ffsim")
    pytest.importorskip("openfermion")
    expected = _molecule()

    ff_molecular = to_ffsim(expected)
    recovered = molecular_from_openfermion_ffsim(
        fermion_operator_openfermion(expected),
        n_orbitals=expected.n_orbitals,
    )

    assert isinstance(ff_molecular, ffsim.MolecularHamiltonian)
    assert to_ffsim(ff_molecular) is ff_molecular
    assert recovered.constant == pytest.approx(expected.constant)
    assert np.allclose(recovered.one_body_tensor, expected.one_body_tensor)
    assert np.allclose(recovered.two_body_tensor, expected.two_body_tensor)


def test_molecular_hamlib_loader_uses_independent_raw_fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    openfermion = pytest.importorskip("openfermion")
    pytest.importorskip("ffsim")
    source = openfermion.FermionOperator((), 0.23)
    source += openfermion.FermionOperator(((0, 1), (0, 0)), -0.8)
    source += openfermion.FermionOperator(((1, 1), (1, 0)), -0.8)
    for term in (
        ((0, 1), (0, 1), (0, 0), (0, 0)),
        ((0, 1), (1, 1), (1, 0), (0, 0)),
        ((1, 1), (0, 1), (0, 0), (1, 0)),
        ((1, 1), (1, 1), (1, 0), (1, 0)),
    ):
        source += openfermion.FermionOperator(term, 0.245)
    path = tmp_path / "molecule.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ham_molec-2", data=str(source))

    loaded = load_molecular(path, "ham_molec-2")

    assert loaded.n_orbitals == 1
    assert loaded.n_qubits == 2
    assert loaded.constant == pytest.approx(0.23)
    assert np.allclose(loaded.one_body_tensor, [[-0.8]])
    assert np.allclose(loaded.two_body_tensor, [[[[0.49]]]])


@pytest.mark.parametrize("qubit_order", ["interleaved", "alpha-then-beta"])
def test_openfermion_conversion_matches_independent_reference(qubit_order):
    openfermion = pytest.importorskip("openfermion")
    molecular = _molecule()
    expected = _manual_fermion_operator(molecular, qubit_order)
    actual = fermion_operator_openfermion(molecular, qubit_order)

    difference = openfermion.normal_ordered(actual - expected)

    assert isinstance(
        interaction_operator_openfermion(molecular),
        openfermion.InteractionOperator,
    )
    assert difference.induced_norm() < 1e-12
    assert fermion_operator(molecular, qubit_order) == actual


def test_pauli_conversion_uses_openfermions_canonical_term_order():
    openfermion = pytest.importorskip("openfermion")
    molecular = _molecule()
    encoded = openfermion.jordan_wigner(_manual_fermion_operator(molecular))
    expected = [term for term in sorted(encoded.terms) if term]

    converted = to_pauli_openfermion(molecular, "jw")
    actual = []
    for _, pauli in terms_of(converted):
        actual.append(
            tuple(
                (qubit, letter)
                for qubit, letter in enumerate(str(pauli))
                if letter != "I"
            )
        )

    assert actual == expected
    assert str(to_pauli(molecular, "jw")) == str(converted)


@pytest.mark.parametrize("encoding", ["jw", "bk"])
def test_molecular_pauli_encoding_preserves_the_operator(encoding):
    openfermion = pytest.importorskip("openfermion")
    molecular = MolecularHamiltonian(
        np.array([[-0.8, 0.12], [0.12, -0.35]]),
        _molecule().two_body_tensor,
    )
    encoded = to_pauli(molecular, encoding)
    source = _manual_fermion_operator(molecular)
    if encoding == "jw":
        reference = openfermion.jordan_wigner(source)
    else:
        reference = openfermion.bravyi_kitaev(source, n_qubits=molecular.n_qubits)
    expected = openfermion.get_sparse_operator(
        reference, n_qubits=molecular.n_qubits
    ).toarray()

    achieved = hamiltonian_matrix(encoded)
    shift = np.trace(expected - achieved) / expected.shape[0]
    assert np.allclose(achieved, expected - shift * np.eye(expected.shape[0]))


def test_factorization_adapter_matches_direct_ffsim():
    ffsim = pytest.importorskip("ffsim")
    source = to_ffsim(_two_factor_molecule())

    direct = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        source,
        z_representation=False,
        tol=1e-12,
        max_vecs=1,
        optimize=False,
        cholesky=True,
    )
    wrapped = factorize_molecular_ffsim(
        source,
        tol=1e-12,
        max_vecs=1,
        optimize=False,
        cholesky=True,
        z_representation=False,
    )

    assert np.allclose(wrapped.one_body_tensor, direct.one_body_tensor)
    assert np.allclose(wrapped.diag_coulomb_mats, direct.diag_coulomb_mats)
    assert np.allclose(wrapped.orbital_rotations, direct.orbital_rotations)
    assert wrapped.constant == pytest.approx(direct.constant)
    assert not wrapped.z_representation


def _direct_ffsim_circuit(molecular, time, *, steps=1, formula_order=2):
    import ffsim
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager,
    )

    factorized = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        to_ffsim(molecular),
        z_representation=True,
        tol=1e-12,
    )
    raw = QuantumCircuit(molecular.n_qubits)
    raw.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            factorized,
            time,
            n_steps=steps,
            order=0 if formula_order == 1 else formula_order // 2,
            tol=1e-10,
        ),
        range(molecular.n_qubits),
    )
    pass_manager = generate_preset_pass_manager(
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=1,
        seed_transpiler=0,
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    return pass_manager.run(raw)


def test_default_synthesis_conforms_to_direct_ffsim_backend():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    molecular = _two_factor_molecule()
    result = synthesize_molecular_ffsim(
        molecular,
        0.23,
        tensor_tolerance=1e-12,
        optimization_level=1,
    )
    direct = _direct_ffsim_circuit(molecular, 0.23)

    assert result.routing_mode == "input"
    assert result.ordering == tuple(range(result.factors))
    assert result.candidate_counts == {"original": result.two_qubit_gates}
    assert result.two_qubit_gates == direct.count_ops().get("cx", 0)
    assert np.allclose(Operator(result.circuit).data, Operator(direct).data)


def test_double_factorized_result_reports_accuracy_and_provenance():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    molecular = MolecularHamiltonian(
        np.array([[-0.7]]), np.array([[[[0.4]]]]), constant=0.11
    )

    result = synthesize_molecular(molecular, 0.3)

    assert result.backend == "ffsim-double-factorized"
    assert result.emitter == "qiskit"
    assert result.two_qubit_gates == result.circuit.count_ops().get("cx", 0)
    assert result.quoted_two_qubit_gates == result.two_qubit_gates
    assert result.two_body_tensor_max_abs_error == result.tensor_error
    assert result.factors == result.retained_factor_count == 1
    assert result.factorization_rank == result.unpruned_factor_count == 1
    assert result.order == 2
    assert result.tensor_tolerance == 1e-8
    assert result.coulomb_cutoff == 0.0
    assert result.optimization_level == 1
    assert result.qubit_order == "alpha-then-beta"
    assert result.routing_mode == "input"
    assert not result.routing_attempted
    assert result.factor_order_preserved
    assert not result.accuracy_certified
    assert result.unitary_error_bound is None
    assert result.trotter_error_bound is None
    assert not result.error_guaranteed
    assert {"finite-step-trotter", "double-factorization"} <= set(
        result.approximation_sources
    )
    assert result.ffsim_version != "unknown"
    assert result.qiskit_version != "unknown"
    assert len(result.factorization_fingerprint) == 64
    assert all(
        instruction.operation.name == "cx"
        for instruction in result.circuit.data
        if len(instruction.qubits) == 2
    )

    from qiskit.quantum_info import Operator

    target = expm(-0.3j * _openfermion_matrix(molecular))
    achieved = Operator(result.circuit.reverse_bits()).data
    assert infidelity(target, achieved) < 1e-10

    previous = result.two_qubit_gates
    result.circuit.cx(0, 1)
    assert result.two_qubit_gates == previous + 1
    assert result.quoted_two_qubit_gates == previous


@pytest.mark.parametrize("qubit_order", ["alpha-then-beta", "interleaved"])
def test_nontrivial_two_body_output_respects_reported_qubit_order(qubit_order):
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    frame = np.array([[-0.73, 0.31], [0.31, 0.17]])
    molecular = MolecularHamiltonian(
        frame,
        np.einsum("pq,rs->pqrs", frame, frame),
        constant=0.11,
    )

    result = synthesize_molecular(
        molecular,
        0.17,
        tensor_tolerance=1e-12,
        qubit_order=qubit_order,
    )

    from qiskit.quantum_info import Operator

    target = expm(-0.17j * _openfermion_matrix(molecular, qubit_order))
    achieved = Operator(result.circuit.reverse_bits()).data
    assert result.qubit_order == qubit_order
    assert infidelity(target, achieved) < 1e-10


def _openfermion_matrix(molecular, qubit_order="interleaved"):
    import openfermion

    return openfermion.get_sparse_operator(
        openfermion.jordan_wigner(_manual_fermion_operator(molecular, qubit_order)),
        n_qubits=molecular.n_qubits,
    ).toarray()


def test_coulomb_cutoff_uses_ffsim_reconstruction_and_reports_h2_error():
    ffsim = pytest.importorskip("ffsim")
    molecular = _molecule()
    source = to_ffsim(molecular)
    number_form = factorize_molecular_ffsim(source, tol=1e-12)

    truncated, reconstructed = prune_double_factorization(
        ffsim, source, number_form, cutoff=0.2
    )
    restored = truncated.to_molecular_hamiltonian()
    result = synthesize_molecular(
        molecular,
        0.2,
        tensor_tolerance=1e-12,
        coulomb_cutoff=0.2,
    )

    expected_error = np.max(
        np.abs(reconstructed.two_body_tensor - molecular.two_body_tensor)
    )
    assert np.allclose(restored.one_body_tensor, molecular.one_body_tensor)
    assert np.allclose(restored.two_body_tensor, reconstructed.two_body_tensor)
    assert result.tensor_error == pytest.approx(expected_error)
    assert "coulomb-cutoff" in result.approximation_sources


def test_coulomb_cutoff_removes_empty_factor_frames():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")

    result = synthesize_molecular(_molecule(), 0.2, coulomb_cutoff=1.0)

    assert result.factorization_rank == 1
    assert result.factors == 0
    assert result.ordering == ()


def test_explicit_frame_permutation_is_accuracy_sensitive_metadata():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    result = synthesize_molecular(
        _two_factor_molecule(),
        0.7,
        tensor_tolerance=1e-12,
        frame_ordering=(1, 0),
    )

    assert result.factors == 2
    assert result.ordering == (1, 0)
    assert result.frame_ordered
    assert result.routing_mode == "explicit-permutation"
    assert not result.factor_order_preserved
    assert not result.accuracy_certified
    assert "fragment-reordering" in result.approximation_sources


def test_factor_permutation_changes_finite_step_approximant():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    molecular = _two_factor_molecule()
    original = synthesize_molecular(
        molecular,
        0.7,
        tensor_tolerance=1e-12,
        frame_ordering="input",
        optimization_level=0,
    )
    reversed_result = synthesize_molecular(
        molecular,
        0.7,
        tensor_tolerance=1e-12,
        frame_ordering=(1, 0),
        optimization_level=0,
    )

    difference = np.linalg.norm(
        Operator(original.circuit).data - Operator(reversed_result.circuit).data
    )
    assert difference > 1e-8
    assert original.tensor_error == pytest.approx(reversed_result.tensor_error)


def test_experimental_frame_ordering_is_explicit_and_monotone_in_cx():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    molecular = _two_factor_molecule()

    original = synthesize_molecular(molecular, 0.2)
    optimized = synthesize_molecular(
        molecular, 0.2, frame_ordering="experimental-givens"
    )

    assert sorted(optimized.ordering) == list(range(optimized.factors))
    assert optimized.two_qubit_gates <= original.two_qubit_gates
    assert optimized.candidate_counts["original"] == original.two_qubit_gates
    assert optimized.routing_mode == "experimental-givens"
    assert optimized.routing_attempted


def test_old_portfolio_name_warns_and_remains_an_alias():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")

    with pytest.warns(DeprecationWarning, match="experimental-givens"):
        old = synthesize_molecular(
            _two_factor_molecule(), 0.2, frame_ordering="portfolio"
        )
    new = synthesize_molecular(
        _two_factor_molecule(), 0.2, frame_ordering="experimental-givens"
    )

    assert old.ordering == new.ordering
    assert old.candidate_counts == new.candidate_counts


@pytest.mark.parametrize("formula_order", [1, 2, 4])
def test_physical_formula_order_is_mapped_to_ffsim(formula_order):
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")
    molecular = MolecularHamiltonian(np.array([[-0.7]]), np.array([[[[0.4]]]]))

    result = synthesize_molecular(
        molecular, 0.1, formula_order=formula_order, optimization_level=0
    )

    assert result.order == formula_order


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 0},
        {"formula_order": 0},
        {"formula_order": 3},
        {"formula_order": 2.0},
        {"tensor_tolerance": True},
        {"tensor_tolerance": float("nan")},
        {"max_vecs": True},
        {"max_vecs": 0},
        {"factorization_optimize": 1},
        {"cholesky": 1},
        {"coulomb_cutoff": 1j},
        {"coulomb_cutoff": float("inf")},
        {"givens_tolerance": -1.0},
        {"optimization_level": True},
        {"frame_ordering": "unknown"},
        {"frame_ordering": ()},
        {"frame_ordering": (0, 0)},
        {"qubit_order": "unknown"},
        {"formula_order": 4, "frame_ordering": "experimental-givens"},
    ],
)
def test_molecular_synthesis_rejects_ambiguous_controls(kwargs):
    with pytest.raises(ValueError):
        synthesize_molecular(_molecule(), 0.2, **kwargs)


@pytest.mark.parametrize("time", [True, 1j, np.array([0.2]), float("nan")])
def test_molecular_synthesis_rejects_nonfinite_time(time):
    with pytest.raises(ValueError):
        synthesize_molecular(_molecule(), time)
