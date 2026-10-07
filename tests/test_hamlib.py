"""HamLib ingestion contracts exercised with local, independently specified data."""

import zipfile

import numpy as np
import pytest

from lizzy.hamlib import fetch, keys, load, load_molecular


def test_cached_archive_loads_pauli_supports_and_coefficients_without_network(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("openfermion")

    def no_network(*args, **kwargs):
        pytest.fail("a cached archive must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    # Sparse qubit indices must be padded, not compressed; the documented Pauli
    # loader contract omits the scalar identity shift.
    raw = "0.25 [X0 Y2] +\n-0.4 [Z1] +\n0.73 []"
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("ham-test", data=raw)
    with zipfile.ZipFile(tmp_path / "cached.zip", "w") as archive:
        archive.write(source, arcname="operator.hdf5")

    path = fetch("cached.zip", cache=tmp_path)
    assert keys(path) == ["ham-test"]
    assert {str(word): coefficient for coefficient, word in load(path, "ham-test")} == {
        "XIY": 0.25, "IZI": -0.4,
    }
    assert fetch("cached.zip", cache=tmp_path) == path


def test_molecular_loader_uses_independent_raw_fixture(tmp_path):
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
