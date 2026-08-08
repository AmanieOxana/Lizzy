"""
    Clifford frames between representations of the same algebra.

    The Pauli representation of a Hamiltonian is a choice, not a given, and the choice
    decides whether this compiler's exact tools apply at all: the same Fermi-Hubbard
    model is entirely two-local under Jordan-Wigner and only 58% two-local under
    Bravyi-Kitaev, a factor of three in gate count. The algebra is the same either way
    -- PauLie classifies both as ``4*so(6)`` -- so the two are Clifford equivalent and
    the better representation is reachable.

    This module constructs that Clifford. Following
    `Aguilar et al. <https://arxiv.org/abs/2408.00081>`__, two Pauli Lie algebras are
    Clifford equivalent when they share an anticommutation graph *and* the same
    algebraic dependencies, and both conditions are needed here: a matching that only
    preserves anticommutation leaves dependent generators with nowhere to go, and the
    linear map fails on exactly those. With a matching that respects both, Witt's
    extension theorem turns the isometry on the span into a symplectic map of the whole
    space, which is a Clifford.

    What is built here is the symplectic matrix. Using it inside synthesis needs two
    further pieces: a reference representation derived from the classification rather
    than supplied, and phase tracking to turn the matrix into a signed circuit.
"""

import numpy as np
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.hamiltonian import terms_of


def symplectic_product(left: np.ndarray, right: np.ndarray, width: int) -> int:
    r"""
    Get the symplectic form of two Pauli bit vectors: 1 iff they anticommute.

    Args:
        left (numpy.ndarray): Bit vector ``[x | z]``.
        right (numpy.ndarray): Bit vector ``[x | z]``.
        width (int): Number of qubits.
    Returns:
        int: 0 or 1.
    """
    return int(left[:width] @ right[width:] + left[width:] @ right[:width]) % 2


def pauli_vectors(hamiltonian_: PauliStringLinear) -> np.ndarray:
    """
    Get a Hamiltonian's terms as symplectic bit vectors, one row each.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        numpy.ndarray: Integer array of shape ``(terms, 2 * qubits)``.
    """
    return np.array(
        [
            [int(b) for b in pauli.bits[::2]] + [int(b) for b in pauli.bits[1::2]]
            for _, pauli in terms_of(hamiltonian_)
        ],
        dtype=np.int64,
    )


def _rank(matrix: np.ndarray) -> int:
    """Rank over GF(2)."""
    work = matrix.copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivot = next((r for r in range(rank, work.shape[0]) if work[r, column]), None)
        if pivot is None:
            continue
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
    return rank


def _solve(matrix: np.ndarray, target: np.ndarray):
    """Solve ``matrix @ x = target`` over GF(2); returns a solution and a null basis."""
    rows, columns = matrix.shape
    augmented = np.hstack([matrix % 2, target.reshape(-1, 1) % 2])
    pivots, rank = [], 0
    for column in range(columns):
        pivot = next((r for r in range(rank, rows) if augmented[r, column]), None)
        if pivot is None:
            continue
        augmented[[rank, pivot]] = augmented[[pivot, rank]]
        for row in range(rows):
            if row != rank and augmented[row, column]:
                augmented[row] ^= augmented[rank]
        pivots.append(column)
        rank += 1
    if any(
        augmented[r, :columns].sum() == 0 and augmented[r, columns]
        for r in range(rank, rows)
    ):
        return None, None

    particular = np.zeros(columns, dtype=np.int64)
    for index, column in enumerate(pivots):
        particular[column] = augmented[index, columns]

    null = []
    for free in (c for c in range(columns) if c not in pivots):
        vector = np.zeros(columns, dtype=np.int64)
        vector[free] = 1
        for index, column in enumerate(pivots):
            vector[column] = augmented[index, free]
        null.append(vector)
    return particular, null


def _inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a square GF(2) matrix."""
    size = matrix.shape[0]
    work = np.hstack([matrix % 2, np.eye(size, dtype=np.int64)])
    for rank in range(size):
        pivot = next((r for r in range(rank, size) if work[r, rank]), None)
        if pivot is None:
            raise ValueError("The matrix is singular over GF(2).")
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(size):
            if row != rank and work[row, rank]:
                work[row] ^= work[rank]
    return work[:, size:]


def is_symplectic(matrix: np.ndarray, width: int) -> bool:
    """
    Check that a matrix preserves the symplectic form, i.e. is a Clifford.

    Args:
        matrix (numpy.ndarray): Candidate ``2n x 2n`` matrix over GF(2).
        width (int): Number of qubits.
    Returns:
        bool: True if it preserves all commutation relations.
    """
    zero, identity = np.zeros((width, width), int), np.eye(width, dtype=int)
    omega = np.block([[zero, identity], [identity, zero]])
    return np.array_equal((matrix.T @ omega @ matrix) % 2, omega % 2)


def independent_rows(vectors: np.ndarray) -> tuple[list[int], np.ndarray]:
    """
    Pick a maximal independent subset of rows, and express every row in it.

    Args:
        vectors (numpy.ndarray): Bit vectors, one per row.
    Returns:
        tuple[list[int], numpy.ndarray]: Indices of the chosen rows, and the
        coefficient matrix writing every row as a combination of them.
    """
    chosen: list[int] = []
    for index in range(vectors.shape[0]):
        trial = np.vstack([vectors[chosen], vectors[index]]) % 2 if chosen else vectors[index : index + 1] % 2
        if _rank(trial) > len(chosen):
            chosen.append(index)

    basis = vectors[chosen] % 2
    coordinates = np.zeros((vectors.shape[0], len(chosen)), dtype=np.int64)
    for index in range(vectors.shape[0]):
        solution, _ = _solve(basis.T, vectors[index] % 2)
        coordinates[index] = solution
    return chosen, coordinates


def find_assignment(source: np.ndarray, target: np.ndarray, width: int, budget: int = 2_000_000):
    """
    Match source terms to target terms, preserving commutation and dependencies.

    Preserving anticommutation alone is not enough. Terms that are products of other
    terms must map to terms that are the same products, or no linear map -- let alone a
    Clifford -- can send every term where the matching says. This searches for an
    independent subset's images, which fixes the map, and accepts only when every
    remaining term then lands on an actual target term.

    Args:
        source (numpy.ndarray): Source bit vectors, one per row.
        target (numpy.ndarray): Target bit vectors, one per row.
        width (int): Number of qubits.
        budget (int): Maximum candidate placements to try.
    Returns:
        list[int] | None: For each source row, the target row it maps to, or None.
    """
    chosen, coordinates = independent_rows(source)
    source, target = source % 2, target % 2
    lookup = {tuple(row): index for index, row in enumerate(target)}
    gram = [
        [symplectic_product(source[a], source[b], width) for b in chosen] for a in chosen
    ]
    remaining = [budget]

    def extend(position: int, picked: list[int]):
        if remaining[0] <= 0:
            return None
        if position == len(chosen):
            images = (coordinates @ target[picked]) % 2
            landed = {}
            for index, row in enumerate(images):
                key = tuple(row)
                if key not in lookup:
                    return None
                landed[lookup[key]] = index
            return picked if len(landed) == source.shape[0] else None

        for candidate in range(target.shape[0]):
            if candidate in picked:
                continue
            remaining[0] -= 1
            if any(
                symplectic_product(target[candidate], target[picked[j]], width) != gram[position][j]
                for j in range(position)
            ):
                continue
            found = extend(position + 1, picked + [candidate])
            if found is not None:
                return found
        return None

    picked = extend(0, [])
    if picked is None:
        return None
    images = (coordinates @ target[picked]) % 2
    return [lookup[tuple(row)] for row in images]


def witt_extend(source_basis: list[np.ndarray], target_basis: list[np.ndarray], width: int) -> np.ndarray:
    r"""
    Extend an isometry between subspaces to a symplectic map of the whole space.

    Witt's theorem says the extension exists; this constructs it. The source basis is
    grown one vector at a time, and each new image is required to have the same
    symplectic products with everything placed so far -- a linear system over GF(2)
    whose solution space is then searched for an independent choice. Once both bases
    are complete, the map is one matrix inversion.

    Args:
        source_basis (list[numpy.ndarray]): Independent source vectors.
        target_basis (list[numpy.ndarray]): Their images, with matching Gram matrix.
        width (int): Number of qubits.
    Returns:
        numpy.ndarray: Symplectic matrix ``S`` with ``source[i] @ S == target[i]``.

    Raises:
        ValueError: If the inputs are not an isometry, so no extension exists.
    """
    source = [vector % 2 for vector in source_basis]
    target = [vector % 2 for vector in target_basis]

    def candidates():
        for i in range(2 * width):
            vector = np.zeros(2 * width, dtype=np.int64)
            vector[i] = 1
            yield vector
        for i in range(2 * width):
            for j in range(i + 1, 2 * width):
                vector = np.zeros(2 * width, dtype=np.int64)
                vector[i] = vector[j] = 1
                yield vector

    while len(source) < 2 * width:
        fresh = next(
            (c for c in candidates() if _rank(np.array(source + [c])) > len(source)), None
        )
        if fresh is None:
            raise ValueError("The source basis cannot be extended.")

        # The image must reproduce every symplectic product the new vector has.
        wanted = np.array([symplectic_product(fresh, v, width) for v in source], dtype=np.int64)
        products = np.array(
            [np.concatenate([v[width:], v[:width]]) % 2 for v in target], dtype=np.int64
        )
        particular, null = _solve(products, wanted)
        if particular is None:
            raise ValueError("The matching is not an isometry, so it does not extend.")

        image = None
        for mask in range(1 << len(null)):
            trial = particular.copy()
            for bit in range(len(null)):
                if mask >> bit & 1:
                    trial = (trial ^ null[bit]) % 2
            if _rank(np.array(target + [trial])) > len(target):
                image = trial
                break
        if image is None:
            raise ValueError("No independent image satisfies the symplectic constraints.")
        source.append(fresh)
        target.append(image)

    return (_inverse(np.array(source)) @ np.array(target)) % 2


def clifford_to(hamiltonian_: PauliStringLinear, reference: PauliStringLinear):
    """
    Find the Clifford carrying a Hamiltonian onto a Clifford-equivalent reference.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian to transform.
        reference (PauliStringLinear): A representation of the same algebra, in the
            form worth reaching -- typically the more local one.
    Returns:
        numpy.ndarray | None: The symplectic matrix, or None when the two are not
        Clifford equivalent or no matching was found within budget.
    """
    source, target = pauli_vectors(hamiltonian_), pauli_vectors(reference)
    if source.shape != target.shape:
        return None
    width = source.shape[1] // 2

    assignment = find_assignment(source, target, width)
    if assignment is None:
        return None

    images = np.array([target[assignment[i]] for i in range(source.shape[0])])
    chosen, _ = independent_rows(source)
    try:
        return witt_extend([source[i] for i in chosen], [images[i] for i in chosen], width)
    except ValueError:
        return None
