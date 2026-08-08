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

    :func:`clifford_to` builds the symplectic matrix and :func:`conjugate` applies it
    with the signs, so a Hamiltonian handed over in the worse representation can be
    moved to the better one and compiled there. What is still supplied rather than
    derived is the reference: the classification names the canonical graph, so
    constructing a target representation from it alone is reachable but not done here.
"""

import numpy as np
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.hamiltonian import gram, symplectic_vectors, terms_of


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
    return symplectic_vectors([pauli for _, pauli in terms_of(hamiltonian_)])


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
    # The Gram matrix of the chosen rows, from the same matrix product the rest of the
    # compiler uses -- the matching needs exactly the anticommutation structure that is
    # already computed everywhere else.
    wanted = gram(source)[np.ix_(chosen, chosen)]
    candidates = gram(target)
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
                candidates[candidate, picked[j]] != wanted[position, j]
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


def _generator_images(matrix: np.ndarray, width: int):
    """The signed Paulis that X_i and Z_i become under the Clifford."""
    from paulie.common.pauli_string_factory import get_pauli_string

    images = []
    for row in matrix:
        letters = []
        for qubit in range(width):
            x, z = int(row[qubit]), int(row[width + qubit])
            letters.append({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(x, z)])
        images.append(get_pauli_string("".join(letters)))
    return images


def _decompose_phase(pauli, width: int):
    r"""Get :math:`\mu` and the generator list with :math:`P = \mu \prod_i g_i`.

    A Pauli is the product of the ``X`` and ``Z`` generators its bit vector names, but
    only up to a power of ``i`` -- ``Y`` is ``XZ`` times a phase. Conjugation acts on
    the generators, so that phase has to be divided out first and multiplied back
    afterwards, or every ``Y`` in the Hamiltonian acquires a wrong sign.
    """
    from paulie.common.pauli_string_factory import get_pauli_string

    bits = pauli.bits
    generators = []
    for qubit in range(width):
        if int(bits[2 * qubit]):
            generators.append(("x", qubit))
    for qubit in range(width):
        if int(bits[2 * qubit + 1]):
            generators.append(("z", qubit))

    accumulated, phase = None, 1 + 0j
    for kind, qubit in generators:
        letter = "X" if kind == "x" else "Z"
        single = get_pauli_string("".join(letter if k == qubit else "I" for k in range(width)))
        if accumulated is None:
            accumulated = single
        else:
            phase *= accumulated.sign(single)
            accumulated = accumulated @ single
    return phase, generators


def conjugate(hamiltonian_: PauliStringLinear, matrix: np.ndarray) -> PauliStringLinear:
    """
    Apply the Clifford given by a symplectic matrix to a Hamiltonian.

    The matrix fixes where the generators go; the signs are the free part, and taking
    them all positive picks one Clifford out of the Pauli coset, which is enough since
    the coset members differ by signs that a synthesis can absorb. Every term's phase
    is then bookkeeping: divide out the phase relating it to its generator product,
    map each generator, and multiply the images back together.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        matrix (numpy.ndarray): Symplectic matrix from :func:`clifford_to`.
    Returns:
        PauliStringLinear: The conjugated Hamiltonian, same spectrum.
    """
    from lizzy.hamiltonian import hamiltonian

    width = matrix.shape[0] // 2
    images = _generator_images(matrix, width)

    transformed: dict[str, complex] = {}
    for coefficient, pauli in terms_of(hamiltonian_):
        phase, generators = _decompose_phase(pauli, width)
        accumulated, built = None, 1 + 0j
        for kind, qubit in generators:
            image = images[qubit if kind == "x" else width + qubit]
            if accumulated is None:
                accumulated = image
            else:
                built *= accumulated.sign(image)
                accumulated = accumulated @ image
        if accumulated is None:
            continue
        value = coefficient * built / phase
        word = str(accumulated)
        transformed[word] = transformed.get(word, 0) + value
    return hamiltonian(
        [(word, value) for word, value in transformed.items() if abs(value) > 1e-12]
    )
