"""
Loading Hamiltonians from HamLib.

HamLib ships as HDF5 files hosted at NERSC. They are fetched on demand and cached,
so nothing here is needed to run the builtin model families. Tests exercise the
loaders with local fixtures and never download archives.

See https://portal.nersc.gov/cfs/m888/dcamps/hamlib/ and arXiv:2306.13126.
"""

import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.chemistry import (
    MolecularHamiltonian,
    fermion_operator_openfermion,
    molecular_from_openfermion_ffsim,
)
from lizzy.hamiltonian import hamiltonian

BASE_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib"

DEFAULT_CACHE = Path(
    os.environ.get("LIZZY_HAMLIB_CACHE", Path.home() / ".cache" / "lizzy" / "hamlib")
)


def fetch(relative_path: str, cache: Path | None = None) -> Path:
    """
    Download a HamLib file if it is not already cached.

    Args:
        relative_path (str): Path within the HamLib portal, e.g.
            ``"condensedmatter/tfim/tfim.hdf5"``.
        cache (Path, optional): Cache directory. Defaults to
            ``~/.cache/lizzy/hamlib``, overridable with ``LIZZY_HAMLIB_CACHE``.
    Returns:
        Path: Local path to the file.
    """
    cache = cache or DEFAULT_CACHE
    target = cache / relative_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # macOS pythons often lack system CA certificates; certifi carries them.
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(
            f"{BASE_URL}/{relative_path}", context=context
        ) as response:
            target.write_bytes(response.read())

    # The portal serves zipped HDF5; unpack next to the archive and hand back the
    # contained file.
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as archive:
            inner = archive.namelist()[0]
            extracted = target.parent / inner
            if not extracted.exists():
                archive.extractall(target.parent)
        return extracted
    return target


def keys(path: Path) -> list[str]:
    """
    List the Hamiltonians inside a HamLib file.

    Args:
        path (Path): Local path to a HamLib HDF5 file.
    Returns:
        list[str]: The dataset keys.

    Raises:
        ImportError: If h5py is not installed.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def load(path: Path, key: str) -> PauliStringLinear:
    """
    Read one Hamiltonian out of a HamLib file.

    HamLib stores operators in OpenFermion's ``QubitOperator`` text form, which names
    only the qubits a term acts on; they are padded out to full-width Pauli strings
    here.

    Args:
        path (Path): Local path to a HamLib HDF5 file.
        key (str): Dataset key within the file.
    Returns:
        PauliStringLinear: The Hamiltonian.

    Raises:
        ImportError: If h5py or openfermion is not installed.
    """
    h5py = _require_h5py()
    openfermion = _require_openfermion()

    with h5py.File(path, "r") as handle:
        text = handle[key][()]
    operator = openfermion.QubitOperator(
        text.decode("utf-8") if isinstance(text, bytes) else text
    )

    width = 1 + max((qubit for term in operator.terms for qubit, _ in term), default=0)
    terms = []
    for term, coefficient in operator.terms.items():
        if not term:
            continue  # A constant shifts the global phase only.
        letters = ["I"] * width
        for qubit, letter in term:
            letters[qubit] = letter
        terms.append(("".join(letters), float(coefficient.real)))

    return hamiltonian(terms)


def load_molecular(path: Path, key: str) -> MolecularHamiltonian:
    r"""Read a HamLib ``ham_molec-*`` dataset without losing its raw integrals.

    OpenFermion parses and validates the symbolic text.  After translating HamLib's
    interleaved integer modes to explicit spin-labelled actions, ffsim's
    ``MolecularHamiltonian.from_fermion_operator`` recovers the spatial tensors
    without normal ordering.  Calling OpenFermion's ``get_interaction_operator`` here
    would be lossy because it normal-orders away HamLib's redundant raw tensor.

    HamLib orders spin orbitals as interleaved alpha/beta pairs.  The returned object
    stores the corresponding real spin-restricted tensors in spatial-orbital form.

    Args:
        path: Local HamLib HDF5 file.
        key: A molecular dataset such as ``"ham_molec-12"``.
    Returns:
        MolecularHamiltonian: Spatial one- and two-electron tensors.

    Raises:
        ValueError: If the dataset is not a real spin-restricted molecular operator.
        ImportError: If h5py, openfermion, or ffsim is not installed.
    """
    h5py = _require_h5py()
    openfermion = _require_openfermion()
    with h5py.File(path, "r") as handle:
        text = handle[key][()]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    source = openfermion.FermionOperator(text)

    modes = 1 + max((index for term in source.terms for index, _ in term), default=-1)
    suffix = key.rsplit("-", 1)[-1]
    if suffix.isdigit():
        modes = max(modes, int(suffix))
    if modes <= 0 or modes % 2:
        raise ValueError(
            f"molecular dataset must contain an even number of modes, got {modes}"
        )
    molecular = molecular_from_openfermion_ffsim(
        source,
        n_orbitals=modes // 2,
        qubit_order="interleaved",
    )
    # Validate every redundant spin block.  This catches an incompatible orbital
    # ordering instead of silently returning plausible-looking but wrong tensors.
    difference = openfermion.normal_ordered(
        fermion_operator_openfermion(molecular) - source
    )
    if difference.induced_norm() > 1e-8:
        raise ValueError(
            "molecular dataset is not a real spin-restricted operator in "
            "interleaved order"
        )
    return molecular


def _require_h5py():
    """Import h5py, with a message pointing at the extra that provides it."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Reading HamLib files needs h5py: pip install 'lizzy[hamlib]'."
        ) from exc
    return h5py


def _require_openfermion():
    """Import openfermion, with a message pointing at the extra that provides it."""
    try:
        import openfermion
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Reading HamLib files needs openfermion: pip install 'lizzy[hamlib]'."
        ) from exc
    return openfermion
