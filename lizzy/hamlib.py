"""
    Loading Hamiltonians from HamLib.

    HamLib ships as HDF5 files hosted at NERSC. They are fetched on demand and cached,
    so nothing here is needed to run the builtin model families -- the offline path
    stays open, which is what keeps the tests independent of the network.

    See https://portal.nersc.gov/cfs/m888/dcamps/hamlib/ and arXiv:2306.13126.
"""

import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

from paulie.common.pauli_string_linear import PauliStringLinear

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
    operator = openfermion.QubitOperator(text.decode("utf-8") if isinstance(text, bytes) else text)

    width = 1 + max(
        (qubit for term in operator.terms for qubit, _ in term), default=0
    )
    terms = []
    for term, coefficient in operator.terms.items():
        if not term:
            continue  # A constant shifts the global phase only.
        letters = ["I"] * width
        for qubit, letter in term:
            letters[qubit] = letter
        terms.append(("".join(letters), float(coefficient.real)))

    return hamiltonian(terms)


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
