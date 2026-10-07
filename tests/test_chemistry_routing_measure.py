"""Pin the frozen experiment's conventions, artifacts, and failure reporting."""

import numpy as np
import pytest

pytest.importorskip("openfermion")
pytest.importorskip("ffsim")
pytest.importorskip("qiskit")

from experiments import chemistry_routing_measure as measure
from lizzy.chemistry import MolecularHamiltonian, factorize_molecular_ffsim


def _molecule():
    matrix = np.array([[0.7, 0.09], [0.09, 0.4]])
    return MolecularHamiltonian(np.array([[-0.8, 0.12], [0.12, -0.35]]),
                                np.einsum("pq,rs->pqrs", matrix, matrix), constant=0.23)


def test_reference_determinant_and_random_states_use_declared_sector():
    indices, states = measure._probe_states(4, (2, 2))
    assert len(indices) == 36
    assert states[(3 << 4) + 3, 0] == 1
    assert np.allclose(np.linalg.norm(states, axis=0), 1)
    assert np.all(np.count_nonzero(states, axis=1)[np.setdiff1d(np.arange(256), indices)] == 0)
    assert np.array_equal(states, measure._probe_states(4, (2, 2))[1])


def test_bk_encoder_maps_the_same_fermionic_operator():
    import openfermion as of

    from lizzy.chemistry import fermion_operator_openfermion
    source = fermion_operator_openfermion(_molecule(), qubit_order="alpha-then-beta")
    reverse = measure._bit_reverse(4)
    jw = of.get_sparse_operator(of.jordan_wigner(source), n_qubits=4).toarray()[reverse][:, reverse]
    bk = of.get_sparse_operator(of.bravyi_kitaev(source, n_qubits=4), n_qubits=4).toarray()[reverse][:, reverse]
    permutation = measure._bk_permutation(4)
    assert np.max(np.abs(bk[np.ix_(permutation, permutation)]-jw)) < 1e-12


@pytest.mark.parametrize("backend", measure.BACKENDS)
def test_upstream_portfolio_preserves_the_declared_pauli_formula(backend):
    if backend.startswith("pytket"):
        pytest.importorskip("pytket")
    from qiskit.quantum_info import Statevector
    payload = {"words": ["XI", "ZZ", "YX"], "coefficients": [0.3, -0.2, 0.15],
               "width": 2, "time": 0.25, "steps": 2}
    states = np.eye(4, dtype=complex)
    expected = measure._pauli_outputs(payload["words"], payload["coefficients"], .25, 2, states)
    circuit, _ = measure._compile_backend(backend, payload)
    outputs = np.column_stack([Statevector(states[:, j]).evolve(circuit).data for j in range(4)])
    errors, norm_error = measure._infidelities(outputs, expected)
    assert max(errors) < 1e-12 and norm_error < 1e-12
    assert all(len(instruction.qubits) < 2 or instruction.operation.name == "cx" for instruction in circuit.data)


def test_df_gate_uses_retained_factorization_and_same_fock_states():
    from qiskit.quantum_info import Statevector
    factorized = factorize_molecular_ffsim(_molecule()).to_z_representation()
    fingerprint = measure._factorization_fingerprint(factorized)
    _, states = measure._probe_states(2, (1, 1))
    circuit, metadata = measure._compile_backend("ffsim-df", {
        "factorized": factorized, "width": 4, "time": .25, "steps": 2})
    logical = measure._df_outputs(factorized, 2, (1, 1), .25, 2, states)
    outputs = np.column_stack([Statevector(states[:, j]).evolve(circuit).data for j in range(4)])
    errors, norm_error = measure._infidelities(outputs, logical)
    assert max(errors) < 1e-12 and norm_error < 1e-12
    assert metadata["factorization_fingerprint"] == fingerprint
    assert measure._factorization_fingerprint(factorized) == fingerprint


def test_pytket_export_materializes_and_charges_implicit_wire_swaps(monkeypatch):
    pytest.importorskip("pytket")
    from pytket import Circuit
    from qiskit.quantum_info import Operator

    import lizzy.emit
    raw = Circuit(2).SWAP(0, 1)
    raw.replace_SWAPs()
    assert raw.has_implicit_wireswaps
    monkeypatch.setattr(lizzy.emit, "pauli_exp_boxes", lambda *args: raw.copy())
    emitted, _ = measure._compile_backend("pytket-direct", {
        "words": ["ZI"], "coefficients": [1.0], "width": 2, "time": .25, "steps": 1})
    target = np.eye(4)[[0, 2, 1, 3]]
    errors, _ = measure._infidelities(Operator(emitted).data, target)
    assert max(errors) < 1e-12
    assert emitted.count_ops()["cx"] == 3


def test_verification_rejects_an_incorrect_emitter(monkeypatch):
    from qiskit import QuantumCircuit
    wrong = QuantumCircuit(2)
    wrong.x(0)
    monkeypatch.setattr(measure, "_compile_bounded", lambda *args: ("OK", wrong, {}, 0.0, 0.0))
    states = np.eye(4, dtype=complex)
    record = measure._measure_backend("fake", {}, states, states, states, 20)
    assert record["status"] == "FAIL_VERIFICATION"
    assert record["emission_mismatch_infidelity"] > .9


def test_backend_failure_retains_reason_without_fake_cost(monkeypatch):
    monkeypatch.setattr(measure, "_compile_bounded", lambda *args:
        ("TIMEOUT", None, {"reason": "bounded runtime"}, 20.0, 20.1))
    record = measure._measure_backend("fake", {}, None, None, None, 20)
    assert record["status"] == "TIMEOUT"
    assert "two_qubit_gates" not in record


def test_cached_inputs_only(monkeypatch, tmp_path):
    monkeypatch.setattr(measure, "DEFAULT_CACHE", tmp_path)
    with pytest.raises(FileNotFoundError, match="no automatic downloads"):
        measure._load_case("H2-4")
