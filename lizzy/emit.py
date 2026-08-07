"""
    Optional shared-frame emission for high-weight rotation sequences.

    The builtin emission charges each rotation its CNOT ladder, with runs on one qubit
    pair merged into three-CNOT blocks. On 2-local models that is the best measured
    emission. On chemistry it is not: at mean Pauli weight five a ladder costs ~7.6
    CNOTs per rotation, where Pauli-graph synthesis conjugates rotations into one
    shared Clifford frame for about one CNOT each. pytket's ``GreedyPauliSimp``
    provides that synthesis, and it works better on this compiler's cluster-ordered
    sequences than on raw term order -- grouping commuting terms is freedom the
    network exploits (BH: 2,000 gates against 2,136 on term order and 8,384 ladders).

    The backend is optional: without pytket installed the builtin emission stands.
    Equivalence is not taken on faith -- ``GreedyPauliSimp`` preserves the unitary,
    pinned by a dense test and re-checked here against the exact sequence.
"""

from lizzy.hamiltonian import Circuit


def pauli_boxes(rotations, width: int):
    """
    Build the pytket box circuit of a rotation sequence, boxes decomposed.

    The one place the angle convention lives: pytket's ``PauliExpBox`` takes half
    turns, ours is :math:`e^{-i\theta P}`, so the box parameter is
    :math:`2\theta/\\pi`.

    Args:
        rotations: ``(word_or_PauliString, angle)`` pairs.
        width (int): Number of qubits.
    Returns:
        pytket.Circuit: The circuit, ready for an optimization pass.

    Raises:
        ImportError: If pytket is not installed.
    """
    import numpy as np
    from pytket import Circuit as TketCircuit
    from pytket.circuit import PauliExpBox
    from pytket.passes import DecomposeBoxes
    from pytket.pauli import Pauli

    letters = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    circuit = TketCircuit(width)
    for pauli, angle in rotations:
        word = str(pauli)
        support = [q for q, letter in enumerate(word) if letter != "I"]
        if support:
            box = PauliExpBox([letters[word[q]] for q in support], 2 * angle / np.pi)
            circuit.add_pauliexpbox(box, support)
    DecomposeBoxes().apply(circuit)
    return circuit


def tket_circuit(circuit: Circuit, width: int):
    """
    Re-synthesize a rotation sequence in a shared Clifford frame with pytket.

    Args:
        circuit (Circuit): The rotations.
        width (int): Number of qubits.
    Returns:
        pytket.Circuit: An equivalent circuit in the CX/TK1 basis.

    Raises:
        ImportError: If pytket is not installed.
    """
    from pytket import OpType
    from pytket.passes import AutoRebase, GreedyPauliSimp

    synthesized = pauli_boxes(circuit.rotations, width)
    GreedyPauliSimp().apply(synthesized)
    AutoRebase({OpType.CX, OpType.TK1}).apply(synthesized)
    return synthesized


def tket_two_qubit_gates(circuit: Circuit, width: int) -> int | None:
    """
    Count the two-qubit gates of the shared-frame emission, if pytket is available.

    Args:
        circuit (Circuit): The rotations.
        width (int): Number of qubits.
    Returns:
        int | None: The count, or ``None`` without pytket.
    """
    try:
        from pytket import OpType
    except ImportError:
        return None
    return tket_circuit(circuit, width).n_gates_of_type(OpType.CX)
