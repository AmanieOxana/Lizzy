"""Algebraic reductions, Clifford frames, and independent dense-reference checks."""

import itertools

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.algebra.classify import classify, is_fast_forwardable, summands
from lizzy.algebra.frame import (
    clifford_to,
    conjugate,
    find_assignment,
    is_symplectic,
    pauli_vectors,
    witt_extend,
)
from lizzy.algebra.gf2 import gram
from lizzy.algebra.symmetry import taper, z2_symmetries
from lizzy.dense import (
    circuit_matrix,
    hamiltonian_matrix,
    monomial_form,
    pauli_matrix,
)
from lizzy.hamiltonian import (
    Circuit,
    anticommutation_matrix,
    hamiltonian,
    model,
    n_qubits,
    symplectic_vectors,
    terms_of,
    weight,
)
from lizzy.synthesize import synthesize


def test_summands_commute_and_are_complete() -> None:
    """Splitting must lose no term and must leave the parts mutually commuting."""
    h = model("xy", 4, seed=2)
    parts = summands(h)
    assert len(parts) == 2

    original = {str(p): c for c, p in terms_of(h)}
    recovered = {str(p): c for part in parts for c, p in terms_of(part)}
    assert recovered == original
    assert sum(len(terms_of(part)) for part in parts) == len(original)

    for i, first in enumerate(parts):
        for second in parts[i + 1 :]:
            assert all(
                a.commutes_with(b)
                for _, a in terms_of(first)
                for _, b in terms_of(second)
            )


def _spectrum(hamiltonian_):
    """Sorted eigenvalues, the dense reference behind every isospectral claim."""
    return np.sort(np.linalg.eigvalsh(hamiltonian_matrix(hamiltonian_)))


@pytest.mark.parametrize("name,n", [("tfim", 4), ("tfxy", 4), ("heisenberg", 3)])
def test_tapering_keeps_the_spectrum_of_its_sector(name: str, n: int) -> None:
    """Every eigenvalue of the tapered Hamiltonian must be one of the original's.

    Tapering claims to remove qubits without changing physics, which is only true if
    the reduced spectrum sits inside the full one -- the claim worth checking, since a
    wrong Clifford or a dropped phase would still produce a plausible-looking operator.
    """
    h = model(name, n, seed=1)
    charges = z2_symmetries(h)
    assert all(charge.commutes_with(p) for charge in charges for _, p in terms_of(h))
    tapered, removed = taper(h)
    assert tapered is not None
    # One qubit per *commuting* charge: anticommuting charges share no eigenbasis, so
    # an odd chain's X^n and Z^n cannot both be fixed and only one is used.
    assert 0 < len(removed) <= len(charges)

    full = _spectrum(h)
    reduced = _spectrum(tapered)
    assert all(np.min(np.abs(full - value)) < 1e-8 for value in reduced)


@pytest.mark.parametrize("sector, indices", [(1, [0, 2]), (-1, [1, 3])])
def test_tapering_removes_weight_and_keeps_the_chosen_sector(sector, indices) -> None:
    """Fixing the spectator Z must produce its exact positive or negative block."""
    h = hamiltonian({"XI": 0.7, "ZI": 0.3, "XZ": 0.2, "ZZ": 0.4})
    tapered, removed = taper(h, sector=[sector])
    assert tapered is not None
    assert removed == [1]
    np.testing.assert_allclose(
        hamiltonian_matrix(tapered),
        hamiltonian_matrix(h)[np.ix_(indices, indices)],
        atol=1e-12,
    )

    before = np.mean([weight(p) for _, p in terms_of(h)])
    after = np.mean([weight(p) for _, p in terms_of(tapered)])
    assert after < before


@pytest.fixture
def equivalent_representations():
    """Local generators encoded by CNOT(0,1), then CNOT(1,2), without a download."""
    local = ("XII", "YII", "ZII", "IXI", "IYI", "IZI", "IIX", "IIY", "IIZ")
    encoded = ("XXX", "YXX", "ZII", "IXX", "ZYX", "ZZI", "IIX", "IZY", "IZZ")
    return tuple(
        hamiltonian([(word, 0.1 * (index + 1)) for index, word in enumerate(words)])
        for words in (encoded, local)
    )


def test_clifford_mapping_preserves_dependencies_and_recovers_local_cost(
    equivalent_representations,
) -> None:
    encoded, local = equivalent_representations
    source, target = pauli_vectors(encoded), pauli_vectors(local)
    found = find_assignment(source, target)
    assert found is not None
    assignment, _ = found
    assert sorted(assignment) == list(range(len(source)))

    matrix = clifford_to(encoded, local)
    assert matrix is not None and is_symplectic(matrix, 3)
    np.testing.assert_array_equal(source @ matrix % 2, target[assignment])
    moved = conjugate(encoded, matrix)
    np.testing.assert_allclose(_spectrum(encoded), _spectrum(moved), atol=1e-12)

    before = synthesize(encoded, time=1.0, steps=2).two_qubit_gates
    after = synthesize(moved, time=1.0, steps=2).two_qubit_gates
    reference = synthesize(local, time=1.0, steps=2).two_qubit_gates
    assert after == reference == 0
    assert before > after


def test_matching_rejects_same_graph_with_different_dependencies() -> None:
    """Three commuting rows can have rank two or three; the graph cannot tell."""
    dependent = hamiltonian(dict.fromkeys(["ZII", "IZI", "ZZI"], 1.0))
    independent = hamiltonian(dict.fromkeys(["ZII", "IZI", "IIZ"], 1.0))
    source, target = pauli_vectors(dependent), pauli_vectors(independent)
    np.testing.assert_array_equal(gram(source), gram(target))
    assert find_assignment(source, target) is None
    assert clifford_to(dependent, independent) is None


def test_clifford_conjugation_keeps_y_phases_against_a_dense_reference() -> None:
    """CNOT sends YY to -XZ: unsigned support matching cannot check this sign."""
    matrix = np.eye(4, dtype=int)
    matrix[0, 1] = matrix[3, 2] = 1
    operator = hamiltonian({"YY": 0.7, "YZ": -0.4, "XY": 0.2, "ZI": 0.3})
    cnot = np.eye(4)[[0, 1, 3, 2]]
    transformed = conjugate(operator, matrix)
    np.testing.assert_allclose(
        hamiltonian_matrix(transformed),
        cnot @ hamiltonian_matrix(operator) @ cnot.T,
        atol=1e-12,
    )


def test_gram_survives_an_empty_operator() -> None:
    """Empty inputs reach the Gram matrix through the free-part split, and used to
    raise from deep inside numpy rather than returning an empty adjacency."""
    assert gram(symplectic_vectors([])).shape == (0, 0)
    assert anticommutation_matrix([]).shape == (0, 0)


def test_witt_extension_scales_past_toy_sizes() -> None:
    """The image search walks the null basis, not its 2^k subsets.

    At twenty qubits the null space starts at dimension forty, so an enumeration over
    subsets would never return; this pins that the construction stays linear in it.
    """
    width = 20
    rng = np.random.default_rng(0)
    source = [rng.integers(0, 2, 2 * width) for _ in range(3)]
    matrix = witt_extend(source, source, width)
    assert is_symplectic(matrix, width)
    assert all(np.array_equal((v @ matrix) % 2, v % 2) for v in source)


def test_witt_extension_refuses_a_non_isometry() -> None:
    """A matching that does not preserve the form has no symplectic extension, and
    saying so beats returning a matrix that quietly is not a Clifford."""
    width = 2
    source = [np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])]  # anticommuting pair
    target = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]  # commuting pair
    with pytest.raises(ValueError):
        witt_extend(source, target, width)


def test_classification_routes_poly_and_exponential_apart() -> None:
    """The routing verdict has to follow the dimension, not the model's name."""
    tfxy = model("tfxy", 6, seed=0)
    heisenberg = model("heisenberg", 6, seed=0)
    assert is_fast_forwardable(classify(tfxy), n_qubits(tfxy))
    assert not is_fast_forwardable(classify(heisenberg), n_qubits(heisenberg))


def test_pauli_matrix_matches_the_string() -> None:
    """The dense reference is only useful if it agrees with the Pauli convention."""
    assert np.allclose(pauli_matrix("IZ"), np.diag([1, -1, 1, -1]))
    assert np.allclose(pauli_matrix("XI"), np.kron(pauli_matrix("X"), np.eye(2)))


def test_monomial_form_is_the_pauli_matrix() -> None:
    """The reference replays rotations through the signed permutation rather than the
    matrix, so that permutation has to be the matrix -- checked on every string."""
    width = 3
    for letters in itertools.product("IXYZ", repeat=width):
        word = "".join(letters)
        rows, phases = monomial_form(word)
        rebuilt = np.zeros((2**width, 2**width), dtype=complex)
        rebuilt[rows, np.arange(2**width)] = phases
        assert np.allclose(rebuilt, pauli_matrix(word)), word


def test_circuit_matrix_agrees_with_a_general_exponential() -> None:
    """The closed form is only a shortcut if it lands where scipy's expm does."""
    from scipy.linalg import expm

    circuit = Circuit()
    for word, angle in (("XY", 0.7), ("ZZ", -1.3), ("IY", 2.9), ("XY", 0.4)):
        circuit.add(get_pauli_string(word), angle, "r")

    expected = np.eye(4, dtype=complex)
    for pauli, angle in circuit.rotations:
        expected = expm(-1j * angle * pauli_matrix(pauli)) @ expected
    assert np.allclose(circuit_matrix(circuit, 2), expected)
