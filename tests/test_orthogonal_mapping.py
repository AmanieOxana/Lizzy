"""Generic mapping distinguishes horizontal input from a general SO element."""

from itertools import combinations, product

import numpy as np
import pytest
from kak_tools import map_irrep_to_matrices, pauli_word_to_string

from lizzy._orthogonal_mapping import map_orthogonal
from lizzy.hamiltonian import model, terms_of


def _assert_lie_mapping(mapping, signs, size):
    basis = map_irrep_to_matrices(mapping, signs, size, "BDI")
    assert len(set(mapping.values())) == len(mapping) == size * (size - 1) // 2
    for first, second in combinations(basis, 2):
        physical_bracket = first.commutator(second)
        expected = sum(
            (1j * coefficient * basis[word] for word, coefficient in physical_bracket.items()),
            start=np.zeros((size, size), dtype=complex),
        )
        actual = basis[first] @ basis[second] - basis[second] @ basis[first]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("words,width,size", [
    (("X", "Y", "Z"), 1, 3),
    # For so(4), graph cliques alone cannot distinguish a star from a triangle.
    (("XX", "XY", "YX", "YY", "ZI", "IZ"), 2, 4),
    (tuple("".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")), 2, 6),
    # CNOT-encoded single-qubit XYZ algebra; no chain/model recognition applies.
    (("XX", "YX", "ZI"), 2, 3),
])
def test_nonhorizontal_algebras_have_verified_general_plane_mapping(words, width, size):
    mapping, signs, actual_size, partition = map_orthogonal(words, width)
    assert actual_size == size
    assert partition is None
    assert set(words) <= {str(pauli_word_to_string(word, width)) for word in mapping.values()}
    _assert_lie_mapping(mapping, signs, size)


def test_horizontal_input_keeps_a_balanced_raw_mapping():
    words = tuple(str(word) for _, word in terms_of(model("tfim", 3, seed=7)))
    mapping, signs, size, partition = map_orthogonal(words, 3)
    assert size == 6 and partition == (3, 3)
    p, _ = partition
    assert all(i < p <= j for (i, j), word in mapping.items()
               if str(pauli_word_to_string(word, 3)) in words)
    _assert_lie_mapping(mapping, signs, size)


def test_unsupported_algebra_is_rejected_before_closure(monkeypatch):
    def no_closure(*args, **kwargs):
        pytest.fail("unsupported classifications must not enter the Lie closure")

    monkeypatch.setattr("lizzy._orthogonal_mapping.dla_pauli_basis", no_closure)
    words = tuple(str(word) for _, word in terms_of(model("heisenberg", 4, seed=7)))
    with pytest.raises(NotImplementedError, match="single so\\(m\\) presentation"):
        map_orthogonal(words, 4)
