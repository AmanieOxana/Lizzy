"""Generic Pauli-to-so(m) mappings for the two algorithms of paper Section VI.

Prefer the horizontal mapping of Appendix F.6. If the Hamiltonian does not fit
any horizontal space, recover a representation from its full Pauli Lie basis;
Section VI.1 can then decompose the general group element. No model names,
Jordan--Wigner templates, or gate-cost reorderings enter either construction.
"""

from kak_tools import (
    as_pauli_collection,
    dla_pauli_basis,
    map_dla_to_irrep,
    map_simple_to_irrep,
    pauli_word_to_string,
)
from kak_tools.map_to_irrep import HorizontalEmbeddingError


def _horizontal_partition(mapping, words, width, size):
    """Find the most balanced canonical cut crossed by every input plane."""
    inverse = {
        str(pauli_word_to_string(word, width)): pair
        for pair, word in mapping.items()
    }
    pairs = [inverse[word] for word in words]
    lower = max(i for i, _ in pairs) + 1
    upper = min(j for _, j in pairs)
    if lower > upper:
        return None
    p = min(range(lower, upper + 1), key=lambda cut: (abs(size - 2 * cut), cut))
    return p, size - p


def _star_generators(basis, size):
    """Recover the m-1 planes incident to one axis, without a clique search.

    In the anticommutation graph of a complete so(m) Pauli-plane basis, choose
    adjacent vertices P_ab and P_ac. Their common neighbours are P_ad for all
    other d, plus the triangle-closing word P_bc proportional to their
    commutator. Removing that word leaves the star at a. The explicit m=2
    case is abelian. Upstream subsequently verifies the entire Lie mapping,
    so the graph argument is never accepted as a substitute for a Lie check.
    """
    if size == 2:
        return basis
    first = basis[0]
    second = next((word for word in basis if not first.commutes_with(word)), None)
    if second is None:
        raise NotImplementedError("The Pauli basis has no so(m) rotation-plane star.")
    closing_word = next(iter(first.commutator(second)))
    star = [first, second] + [
        word for word in basis
        if word != closing_word
        and not first.commutes_with(word)
        and not second.commutes_with(word)
    ]
    if len(star) != size - 1:
        raise NotImplementedError("The Pauli basis has no complete so(m) rotation-plane star.")
    return star


def map_orthogonal(words: tuple[str, ...], width: int):
    """Return ``(mapping, signs, m, horizontal_partition_or_None)``.

    The returned mapping always labels the full so(m) Pauli basis. A partition
    ``(p,q)`` means every Hamiltonian term is horizontal in those coordinates;
    ``None`` selects the general recursive group decomposition instead.

    Dimension and term budgets match Lizzy's existing exact-route search
    limits. They are practical cutoffs, not representation-theoretic claims.
    A failed horizontal embedding triggers the general mapping; other errors
    from the upstream mapping remain visible.
    """
    if not words:
        raise ValueError("An orthogonal mapping requires at least one Pauli word.")
    if len(words) > max(64, 8 * width):
        raise NotImplementedError("The generator count exceeds the exact-route mapping budget.")
    classification = as_pauli_collection(words, n_qubits=width).get_class()
    size = classification.get_orthogonal_size()
    dimension = classification.get_dla_dim()
    if size is None or dimension != size * (size - 1) // 2:
        raise NotImplementedError("The Pauli algebra has no single so(m) presentation.")
    if dimension > 8 * width**2:
        raise NotImplementedError("The algebra dimension exceeds the exact-route mapping budget.")

    try:
        mapping, signs, _ = map_dla_to_irrep(words, n_qubits=width)
    except ValueError as exc:
        if not isinstance(exc.__cause__, HorizontalEmbeddingError):
            raise
        basis = sorted(
            dla_pauli_basis(words, n_qubits=width),
            key=lambda word: str(pauli_word_to_string(word, width)),
        )
        if len(basis) != dimension:
            raise ValueError("The Pauli closure disagrees with the classified so(m) dimension.") from exc
        star = _star_generators(basis, size)
        mapping, signs = map_simple_to_irrep(
            basis,
            horizontal_ops={(0, j): word for j, word in enumerate(star, 1)},
            n=size,
            invol_type="BDI",
            invol_kwargs={"p": 1, "q": size - 1},
        )
    partition = _horizontal_partition(mapping, words, width, size)
    return mapping, signs, size, partition
