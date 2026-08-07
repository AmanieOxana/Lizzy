"""
    Dense reference simulation, for checking that a circuit means what it claims.

    Only usable for a handful of qubits, which is the point: the synthesis itself never
    builds a :math:`2^{n}` matrix, so this exists purely to hold it to account at sizes
    where the truth is computable.
"""

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.linalg import expm

from lizzy.hamiltonian import Circuit, n_qubits, terms_of

_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(pauli: PauliString | str) -> np.ndarray:
    """
    Build the dense matrix of a Pauli string.

    Args:
        pauli (PauliString | str): The Pauli string.
    Returns:
        numpy.ndarray: Its matrix.
    """
    matrix = np.array([[1]], dtype=complex)
    for letter in str(pauli):
        matrix = np.kron(matrix, _SINGLE[letter])
    return matrix


def hamiltonian_matrix(hamiltonian: PauliStringLinear) -> np.ndarray:
    """
    Build the dense matrix of a Hamiltonian.

    Args:
        hamiltonian (PauliStringLinear): The Hamiltonian.
    Returns:
        numpy.ndarray: Its matrix.
    """
    width = n_qubits(hamiltonian)
    total = np.zeros((2**width, 2**width), dtype=complex)
    for coefficient, pauli in terms_of(hamiltonian):
        total = total + coefficient.real * pauli_matrix(pauli)
    return total


def evolution(hamiltonian: PauliStringLinear, time: float) -> np.ndarray:
    r"""
    Build :math:`e^{-itH}` exactly.

    Args:
        hamiltonian (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
    Returns:
        numpy.ndarray: The exact evolution operator.
    """
    return expm(-1j * time * hamiltonian_matrix(hamiltonian))


def circuit_matrix(circuit: Circuit, width: int) -> np.ndarray:
    r"""
    Build the unitary a circuit implements.

    Each rotation ``(P, theta)`` contributes :math:`e^{-i\theta P}`, applied left to
    right.

    Args:
        circuit (Circuit): The circuit.
        width (int): Number of qubits.
    Returns:
        numpy.ndarray: The circuit's unitary.
    """
    total = np.eye(2**width, dtype=complex)
    for pauli, angle in circuit.rotations:
        total = expm(-1j * angle * pauli_matrix(pauli)) @ total
    return total


def infidelity(target: np.ndarray, achieved: np.ndarray) -> float:
    r"""
    Measure how far a circuit is from its target, ignoring global phase.

    Uses :math:`1 - |\mathrm{tr}(U^{\dagger}V)| / d`, which is zero exactly when the two
    agree up to a phase no measurement can see.

    Args:
        target (numpy.ndarray): The intended unitary.
        achieved (numpy.ndarray): The circuit's unitary.
    Returns:
        float: The infidelity, in :math:`[0, 1]`.
    """
    dimension = target.shape[0]
    overlap = abs(np.trace(target.conj().T @ achieved)) / dimension
    return float(1.0 - min(overlap, 1.0))
