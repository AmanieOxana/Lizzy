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


def monomial_form(pauli: PauliString | str) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Get a Pauli string as the signed permutation it is.

    Every Pauli string has exactly one non-zero entry per column: ``X`` permutes,
    ``Z`` signs, ``Y`` does both. So the matrix is determined by where each basis
    state goes and with what phase, and applying it is an indexing operation rather
    than a matrix product -- :math:`O(4^{n})` against :math:`O(8^{n})`.

    Args:
        pauli (PauliString | str): The Pauli string.
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(rows, phases)`` with
        ``P[rows[k], k] == phases[k]`` and every other entry zero.
    """
    word = str(pauli)
    states = np.arange(2 ** len(word))
    flip = 0
    phases = np.ones(states.size, dtype=complex)
    for position, letter in enumerate(word):
        bit = 1 << (len(word) - 1 - position)
        if letter in "XY":
            flip |= bit
        if letter == "Y":
            phases = phases * np.where(states & bit, -1j, 1j)
        elif letter == "Z":
            phases = phases * np.where(states & bit, -1, 1)
    return states ^ flip, phases


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

    A Pauli squares to the identity, so the exponential is closed-form,
    :math:`e^{-i\theta P} = \cos\theta - i\sin\theta\,P`, and no general matrix
    exponential is needed. What is left is the product :math:`PU`, and that is an
    indexed read rather than a matrix product because :math:`P` is monomial. The
    reference is the thing every exactness claim rests on, so it stays a literal
    replay of the rotations -- it just stops paying twenty matrix products per
    rotation for an answer that costs one pass over the matrix.

    Args:
        circuit (Circuit): The circuit.
        width (int): Number of qubits.
    Returns:
        numpy.ndarray: The circuit's unitary.
    """
    total = np.eye(2**width, dtype=complex)
    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pauli, angle in circuit.rotations:
        word = str(pauli)
        if word not in forms:
            forms[word] = monomial_form(word)
        rows, phases = forms[word]
        applied = np.empty_like(total)
        applied[rows] = phases[:, None] * total
        total = np.cos(angle) * total - 1j * np.sin(angle) * applied
    return total


def operator_errors(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Strict and trace-phase-aligned spectral norms for dense verification.

    Alignment uses the phase of ``trace(target† actual)``, or one when that
    overlap is zero. It is not a minimization of spectral norm over phases.
    Thresholds and any requirement to retain absolute phase belong to callers.
    """
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite output cannot pass a unitary error check")
    overlap = np.trace(target.conj().T @ actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return {"strict": float(np.linalg.norm(actual - target, ord=2)),
            "phase_aligned": float(np.linalg.norm(actual - phase * target, ord=2))}


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
