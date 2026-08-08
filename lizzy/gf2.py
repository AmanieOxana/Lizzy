"""
    Linear algebra over GF(2).

    Pauli strings are bit vectors and commutation is a bilinear form over the two
    element field, so most of the structure this compiler exploits -- summand
    splitting, conserved charges, clustering, Clifford frames -- reduces to Gaussian
    elimination mod two. It lives here once rather than in each module that needs it.
"""

import numpy as np


def gram(vectors: np.ndarray) -> np.ndarray:
    r"""
    Get the symplectic Gram matrix of bit vectors: 1 where they anticommute.

    One integer matrix product rather than :math:`L^{2}` pairwise tests -- the
    difference between milliseconds and minutes once dense models reach thousands of
    terms.

    Args:
        vectors (numpy.ndarray): Bit vectors ``[x | z]``, one per row.
    Returns:
        numpy.ndarray: Symmetric 0/1 matrix with a zero diagonal.
    """
    if vectors.size == 0:
        return np.zeros((vectors.shape[0], vectors.shape[0]), dtype=np.int64)
    width = vectors.shape[1] // 2
    x, z = vectors[:, :width], vectors[:, width:]
    adjacency = (x @ z.T + z @ x.T) % 2
    np.fill_diagonal(adjacency, 0)
    return adjacency


def _eliminate(matrix: np.ndarray, columns: int):
    """Row-reduce in place; returns the pivot columns found."""
    pivots, rank = [], 0
    for column in range(columns):
        pivot = next((r for r in range(rank, matrix.shape[0]) if matrix[r, column]), None)
        if pivot is None:
            continue
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(matrix.shape[0]):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        pivots.append(column)
        rank += 1
    return pivots


def rank(matrix: np.ndarray) -> int:
    """
    Get the rank of a 0/1 matrix over GF(2).

    Args:
        matrix (numpy.ndarray): The matrix.
    Returns:
        int: Its rank.
    """
    if matrix.size == 0:
        return 0
    work = np.atleast_2d(matrix).copy() % 2
    return len(_eliminate(work, work.shape[1]))


def inverse(matrix: np.ndarray):
    """
    Invert a square 0/1 matrix over GF(2).

    Args:
        matrix (numpy.ndarray): The matrix.
    Returns:
        numpy.ndarray | None: The inverse, or None if singular.
    """
    size = matrix.shape[0]
    work = np.hstack([matrix % 2, np.eye(size, dtype=np.int64)])
    if len(_eliminate(work, size)) < size:
        return None
    return work[:, size:]


def solve(matrix: np.ndarray, target: np.ndarray):
    """
    Solve ``matrix @ x = target`` over GF(2).

    Args:
        matrix (numpy.ndarray): Coefficient matrix.
        target (numpy.ndarray): Right-hand side.
    Returns:
        tuple: A particular solution and a basis of the null space, or
        ``(None, None)`` when the system is inconsistent.
    """
    rows, columns = matrix.shape
    augmented = np.hstack([matrix % 2, target.reshape(-1, 1) % 2])
    pivots = _eliminate(augmented, columns)
    if any(
        augmented[r, :columns].sum() == 0 and augmented[r, columns]
        for r in range(len(pivots), rows)
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
