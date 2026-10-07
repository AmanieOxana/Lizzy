"""Dense correctness and nonlocal-cost regressions for two-qubit kernels."""

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.dense import (
    circuit_matrix,
    evolution,
    infidelity,
)
from lizzy.hamiltonian import (
    Circuit,
    hamiltonian,
    terms_of,
)


def test_noncommuting_kernels_compile_exactly() -> None:
    """A kernel with fields folded in is not internally commuting, and its KAK
    emission must still equal the 4x4 exponential -- checked here through the dense
    machinery, on top of the self-check every kernel runs at synthesis time."""
    from lizzy.kernels import compile_kernel

    h = hamiltonian({"IXIX": 0.3, "IYIZ": -0.7, "IZIY": 0.2, "IXII": 0.5, "IIIZ": -0.4})
    tau = 0.9

    circuit = compile_kernel(terms_of(h), tau, 4, "t")
    target = evolution(h, tau)
    assert infidelity(target, circuit_matrix(circuit, 4)) < 1e-9
    assert circuit.two_qubit_gates <= 3


def test_a_pair_block_is_charged_its_canonical_class() -> None:
    """Charge every exact 0/1/2/3-CNOT local-equivalence class correctly.

    The high-symmetry CNOT, iSWAP and SWAP points are the important regression: a
    phase-based diagonalization has degenerate eigenspaces there and used to erase the
    SWAP point's three nonlocal coordinates entirely.
    """
    from lizzy.kernels import canonical_cost

    def block(*words, angle=0.3):
        circuit = Circuit()
        for word in words:
            circuit.add(get_pauli_string(word), angle, "k")
        return circuit.two_qubit_gates

    assert block("XX", "YY", "ZZ") == 3
    assert block("XX", "YY") == 2  # an XY bond; the cap said three
    assert block("XX") == 2
    assert block("XX", angle=np.pi / 4) == 1  # CNOT class
    assert block("XX", "YY", angle=np.pi / 4) == 2  # iSWAP class
    assert block("XX", "YY", "ZZ", angle=np.pi / 4) == 3  # SWAP class
    assert block("XX", "YY", "ZZ", angle=np.pi / 8) == 3  # sqrt-SWAP class
    assert block("XX", "YY", "ZZ", angle=1e-4) == 3  # small but still generic
    assert block("ZI", "IX") == 0  # local, whatever the angles
    assert (
        canonical_cost(
            [
                ("ZI", 0.17),
                ("IX", -0.31),
                ("XX", np.pi / 4),
                ("YI", 0.23),
                ("IZ", -0.29),
            ],
            (0, 1),
        )
        == 1
    )  # arbitrary local dressing cannot change the CNOT class
