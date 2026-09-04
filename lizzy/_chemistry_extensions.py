"""Lizzy-specific extensions around ffsim's molecular circuit path.

Everything in this module is deliberately outside the backend adapter: ffsim owns
the molecular/double-factorized representations and their circuit decomposition.
The helpers below are optional Lizzy policies for post-factorization pruning,
experimental fragment routing, and the external fermionic mode layout.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import expm


def prune_double_factorization(ffsim, source, factorized, cutoff: float):
    """Apply Lizzy's optional Coulomb cutoff using ffsim for reconstruction.

    ``factorized`` must be in ffsim's number representation.  The one-body tensor
    is corrected after pruning so that this optional approximation changes only the
    reconstructed two-body tensor.  ffsim remains the tensor-convention authority:
    no local reconstruction einsum or normal-ordering formula is used here.
    """
    matrices = factorized.diag_coulomb_mats.copy()
    matrices[np.abs(matrices) <= cutoff] = 0.0
    active = np.any(matrices != 0.0, axis=(1, 2))
    pruned = replace(
        factorized,
        diag_coulomb_mats=matrices[active],
        orbital_rotations=factorized.orbital_rotations[active],
    )

    reconstructed = pruned.to_molecular_hamiltonian()
    pruned = replace(
        pruned,
        one_body_tensor=(
            pruned.one_body_tensor
            + source.one_body_tensor
            - reconstructed.one_body_tensor
        ),
    )
    reconstructed = pruned.to_molecular_hamiltonian()
    return pruned.to_z_representation(), reconstructed


def reorder_double_factorization(factorized, ordering: tuple[int, ...]):
    """Return an ffsim factorization with the Coulomb fragments permuted."""
    indices = list(ordering)
    return replace(
        factorized,
        diag_coulomb_mats=factorized.diag_coulomb_mats[indices],
        orbital_rotations=factorized.orbital_rotations[indices],
    )


def optimized_frame_order(
    ffsim,
    factorized,
    step_time: float,
    tolerance: float,
    steps: int,
) -> tuple[int, ...]:
    """Find a short S2 orbital-frame path with a Lizzy-specific heuristic.

    This nearest-neighbour/2-opt policy is not part of ffsim and changes the
    finite-step product formula.  ffsim's Givens decomposition supplies the local
    edge cost; callers must keep the policy explicit and accuracy-uncertified.
    """
    rotations = factorized.orbital_rotations
    count = len(rotations)

    def givens(matrix) -> int:
        decomposition, _ = ffsim.linalg.givens_decomposition(matrix, tol=tolerance)
        return len(decomposition)

    half_pass = expm(-0.5j * step_time * factorized.one_body_tensor)
    from_start = [givens(rotation.T.conj() @ half_pass) for rotation in rotations]
    between_steps = [
        givens(rotation.T.conj() @ half_pass @ half_pass @ rotation)
        for rotation in rotations
    ]
    to_finish = [givens(half_pass @ rotation) for rotation in rotations]
    distances = np.array(
        [
            [givens(rotations[j].T.conj() @ rotations[i]) for j in range(count)]
            for i in range(count)
        ],
        dtype=int,
    )
    density = np.array(
        [density_interactions(matrix) for matrix in factorized.diag_coulomb_mats]
    )

    def score(path: list[int] | tuple[int, ...]) -> int:
        edges = sum(
            distances[path[index], path[index + 1]] for index in range(count - 1)
        )
        boundaries = (
            from_start[path[0]]
            + (steps - 1) * between_steps[path[0]]
            + to_finish[path[0]]
        )
        return 4 * boundaries + 8 * steps * edges - 2 * steps * int(density[path[-1]])

    candidates: list[list[int]] = [list(range(count))]
    for first in range(count):
        path = [first]
        remaining = set(range(count)) - {first}
        while remaining:
            next_index = min(
                remaining,
                key=lambda item: (
                    distances[path[-1], item],
                    -density[item],
                    item,
                ),
            )
            path.append(next_index)
            remaining.remove(next_index)
        candidates.append(path)

    best = min(candidates, key=score)
    improved = True
    while improved:
        improved = False
        best_score = score(best)
        for left in range(count):
            for right in range(left + 1, count):
                candidate = (
                    best[:left]
                    + list(reversed(best[left : right + 1]))
                    + best[right + 1 :]
                )
                candidate_score = score(candidate)
                if candidate_score < best_score:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return tuple(best)


def density_interactions(matrix: np.ndarray) -> int:
    """Count RZZ interactions in one spin-restricted Coulomb fragment."""
    size = len(matrix)
    return sum(
        1 if left == right else 4
        for left in range(size)
        for right in range(left, size)
        if matrix[left, right] != 0.0
    )


def to_interleaved_jw(circuit, n_orbitals: int, QuantumCircuit):
    """Conjugate a block-spin JW circuit into interleaved mode order.

    For every inversion in the block-to-interleaved permutation, a CZ supplies the
    occupation-basis sign on both sides of the evolution.  This is a fermionic mode
    permutation, not a free relabeling of wires.
    """
    remapped = QuantumCircuit(2 * n_orbitals)
    for beta in range(n_orbitals):
        for alpha in range(beta + 1, n_orbitals):
            remapped.cz(2 * beta + 1, 2 * alpha)

    internal_to_external = [
        *(2 * orbital for orbital in range(n_orbitals)),
        *(2 * orbital + 1 for orbital in range(n_orbitals)),
    ]
    remapped.compose(circuit, qubits=internal_to_external, inplace=True)

    for beta in range(n_orbitals):
        for alpha in range(beta + 1, n_orbitals):
            remapped.cz(2 * beta + 1, 2 * alpha)
    return remapped
