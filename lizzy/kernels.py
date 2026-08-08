"""
    Exact compilation of two-qubit kernels.

    A kernel is every term a Hamiltonian places on one qubit pair. Whatever those
    terms are -- commuting or not, fields included -- their joint exponential is one
    element of U(4), and its KAK decomposition writes it as single-qubit rotations
    (free) around one canonical block of three commuting rotations (at most three
    CNOTs). This is Kernpiler's partial Trotterization at the support size where the
    exact synthesis is closed-form (arXiv:2504.07214).

    Every decomposition is verified against its own 4x4 exponential before it is
    returned; at that size the check costs nothing, so no convention is taken on
    faith.
"""

import itertools

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.linalg import expm

from lizzy.hamiltonian import Circuit, terms_of

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# The magic basis, in which SU(2) x SU(2) becomes SO(4) and the canonical
# XX/YY/ZZ rotations become diagonal.
_MAGIC = np.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex
) / np.sqrt(2)

# Diagonals of XX, YY, ZZ in the magic basis, for reading canonical angles off a
# magic-diagonal phase vector by least squares -- solved numerically so no sign
# convention is memorized.
_CANONICAL_DIAGONALS = np.column_stack(
    [
        np.diag(_MAGIC.conj().T @ np.kron(_PAULI[a], _PAULI[a]) @ _MAGIC).real
        for a in "XYZ"
    ]
    + [np.ones(4)]
)


def _euler_rotations(u: np.ndarray, qubit_word: tuple[str, str, str]) -> list:
    """Write a single-qubit unitary as up to three weight-one rotations.

    ``u = phase * Rz(a) Ry(b) Rz(c)`` with ``R_P(t) = exp(-i t/2 P)``; the words to
    rotate about are supplied so the caller can place the qubit in a wider register.
    Returned in application order.
    """
    det = np.linalg.det(u)
    su = u / np.sqrt(det)
    b = 2 * np.arctan2(abs(su[1, 0]), abs(su[0, 0]))
    sum_ac = 2 * np.angle(su[1, 1]) if abs(su[1, 1]) > 1e-12 else 0.0
    diff_ac = 2 * np.angle(su[1, 0]) if abs(su[1, 0]) > 1e-12 else 0.0
    a, c = (sum_ac + diff_ac) / 2, (sum_ac - diff_ac) / 2

    z_word, y_word, _ = qubit_word
    return [
        (z_word, c / 2),
        (y_word, b / 2),
        (z_word, a / 2),
    ]


def _tensor_factors(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a product unitary ``a (x) b`` into its single-qubit factors."""
    reshaped = g.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    left, singulars, right = np.linalg.svd(reshaped)
    a = left[:, 0].reshape(2, 2) * np.sqrt(singulars[0])
    b = right[0].reshape(2, 2) * np.sqrt(singulars[0])
    return a, b


def _orthogonal_diagonalization(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a symmetric unitary as ``O e^{i phi} O^T`` with real ``O``.

    The real and imaginary parts of ``w`` are commuting real symmetric matrices, so
    a generic linear combination has a common real eigenbasis; the combination
    breaks the degeneracies either part alone may have.
    """
    _, basis = np.linalg.eigh(w.real + np.pi * w.imag)
    phases = np.angle(np.diag(basis.T @ w @ basis))
    return basis, phases


def two_qubit_kak(unitary: np.ndarray, qubits: tuple[int, int], width: int) -> Circuit:
    """
    Decompose a two-qubit unitary into Pauli rotations on a wider register.

    Args:
        unitary (numpy.ndarray): The 4x4 unitary, qubit ``qubits[0]`` major.
        qubits (tuple[int, int]): Which register qubits it acts on.
        width (int): Register width.
    Returns:
        Circuit: Weight-one Euler rotations around one canonical XX/YY/ZZ block,
        equal to ``unitary`` up to global phase.
    """
    v = _MAGIC.conj().T @ unitary @ _MAGIC
    v = v / np.linalg.det(v) ** 0.25

    basis, phases = _orthogonal_diagonalization(v.T @ v)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    left = v @ basis @ np.diag(np.exp(-1j * phases / 2))
    if np.linalg.det(left).real < 0:
        left[:, 0] *= -1
        phases[0] += 2 * np.pi

    # Canonical angles from the magic-diagonal phases, least squares against the
    # diagonals of XX, YY, ZZ and the global phase.
    xyz = np.linalg.lstsq(_CANONICAL_DIAGONALS, phases / 2, rcond=None)[0][:3]

    def word(letter: str, qubit: int) -> str:
        return "".join(letter if k == qubit else "I" for k in range(width))

    def qubit_words(qubit: int) -> tuple[str, str, str]:
        return word("Z", qubit), word("Y", qubit), word("X", qubit)

    a_left, b_left = _tensor_factors(_MAGIC @ left @ _MAGIC.conj().T)
    a_right, b_right = _tensor_factors(_MAGIC @ basis.T @ _MAGIC.conj().T)

    i, j = qubits
    circuit = Circuit()
    for factor, qubit in ((a_right, i), (b_right, j)):
        for w, angle in _euler_rotations(factor, qubit_words(qubit)):
            circuit.add(get_pauli_string(w), angle, "kernel")
    for letter, angle in zip("XYZ", xyz):
        pair = "".join(letter if k in (i, j) else "I" for k in range(width))
        circuit.add(get_pauli_string(pair), -angle, "kernel")
    for factor, qubit in ((a_left, i), (b_left, j)):
        for w, angle in _euler_rotations(factor, qubit_words(qubit)):
            circuit.add(get_pauli_string(w), angle, "kernel")
    return circuit


def compile_kernel(
    kernel: list[tuple[complex, object]], time: float, width: int, route: str
) -> Circuit:
    """
    Compile one kernel's exact exponential.

    Internally commuting kernels stay plain rotations. Anything else goes through
    the dense 4x4 exponential and :func:`two_qubit_kak`, and the result is verified
    against that exponential before being returned.

    Args:
        kernel (list): The kernel's ``(coefficient, PauliString)`` terms.
        time (float): Evolution time of this exponential.
        width (int): Register width.
        route (str): Label for rotations of the commuting case.
    Returns:
        Circuit: The rotations.

    Raises:
        ValueError: If the verification fails, which would mean a convention bug
            rather than an input problem.
    """
    support = sorted({q for _, p in kernel for q in p.get_support()})

    commuting = all(
        a.commutes_with(b)
        for (_, a), (_, b) in itertools.combinations(kernel, 2)
    )
    if commuting:
        circuit = Circuit()
        for coefficient, pauli in kernel:
            circuit.add(pauli, coefficient.real * time, route)
        return circuit

    i, j = support
    local = np.zeros((4, 4), dtype=complex)
    for coefficient, pauli in kernel:
        w = str(pauli)
        local += coefficient.real * np.kron(_PAULI[w[i]], _PAULI[w[j]])
    target = expm(-1j * time * local)

    circuit = two_qubit_kak(target, (i, j), width)

    built = np.eye(4, dtype=complex)
    for pauli, angle in circuit.rotations:
        w = str(pauli)
        built = expm(-1j * angle * np.kron(_PAULI[w[i]], _PAULI[w[j]])) @ built
    overlap = abs(np.trace(built.conj().T @ target)) / 4
    if overlap < 1 - 1e-9:
        raise ValueError(f"Kernel decomposition failed verification: overlap {overlap}.")
    return circuit


def compile_layer(layer: PauliStringLinear, time: float, route: str = "trotter2") -> Circuit:
    """
    Compile one layer of disjoint kernels exactly.

    Args:
        layer (PauliStringLinear): Terms grouped on pairwise disjoint supports.
        time (float): Evolution time of the layer.
        route (str): Label for plain-rotation kernels.
    Returns:
        Circuit: The rotations, kernel by kernel.
    """
    terms = terms_of(layer)
    width = max(len(p) for _, p in terms)

    # Pairs first; then each field joins the kernel whose pair covers its qubit.
    # Folding is what keeps the layer exponential exact -- a field on qubit i does
    # not commute with the pair terms touching i, so it must be inside the same
    # exponential, not beside it.
    kernels: dict[tuple[int, ...], list] = {}
    for coefficient, pauli in terms:
        support = tuple(sorted(pauli.get_support()))
        if len(support) == 2:
            kernels.setdefault(support, []).append((coefficient, pauli))
    by_qubit = {q: s for s in kernels for q in s}
    for coefficient, pauli in terms:
        support = tuple(sorted(pauli.get_support()))
        if len(support) == 1:
            home = by_qubit.get(support[0], support)
            kernels.setdefault(home, []).append((coefficient, pauli))

    circuit = Circuit()
    for support in sorted(kernels):
        kernel = kernels[support]
        if len(support) == 1:
            # A lone field exponentiates as itself.
            for coefficient, pauli in kernel:
                circuit.add(pauli, coefficient.real * time, route)
        else:
            circuit.extend(compile_kernel(kernel, time, width, route))
    return circuit
