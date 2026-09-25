"""Direct matrix checks of the paper's product rule in Lizzy's -iP convention."""

from itertools import product

import numpy as np
from scipy.linalg import expm

from lizzy.dense import pauli_matrix
from lizzy.driven import _adjoint_pairs, _closure, _coordinate_matrix


def test_full_su4_jacobian_is_prefix_conjugation_for_large_angles_and_orders():
    # Independent dense conjugation, not the coordinate ODE or Pauli adjoint
    # implementation. Large angles would expose a truncated BCH approximation.
    words = ["".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")]
    closure = _closure(words, 15)
    rng = np.random.default_rng(20260925)
    for _ in range(10):
        basis = [closure[index] for index in rng.permutation(len(closure))]
        theta = rng.uniform(-2*np.pi, 2*np.pi, len(basis))
        pairs = _adjoint_pairs(basis)
        jacobian = _coordinate_matrix(theta, pairs)
        assert np.array_equal(_coordinate_matrix(np.zeros(len(basis)), pairs), np.eye(15))
        matrices = np.array([pauli_matrix(word) for word in basis])
        prefix = np.eye(4, dtype=complex)
        for column, (angle, pauli) in enumerate(zip(theta, matrices)):
            expected = prefix @ pauli @ prefix.conj().T
            represented = np.einsum("j,jab->ab", jacobian[:, column], matrices)
            assert np.linalg.norm(represented - expected, 2) < 1e-12
            prefix = prefix @ expm(-1j * angle * pauli)


def test_pauli_adjoint_is_trigonometric_not_a_finitely_truncated_bch_series():
    basis = _closure(["X", "Y", "Z"], 3)
    angle = np.pi / 3
    jacobian = _coordinate_matrix(np.array([angle, 0, 0]), _adjoint_pairs(basis))
    # Column Y = Ad(exp(-i angle X))Y. The factor two and sign are physical.
    assert np.allclose(jacobian[:, 1], [0, np.cos(2*angle), np.sin(2*angle)], atol=1e-14)
