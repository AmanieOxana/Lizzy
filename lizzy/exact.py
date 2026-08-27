"""
    The exact branch: fixed-depth synthesis for algebras small enough to decompose.

    A DLA of dimension d decomposes into d Pauli rotations whose count does not depend
    on the evolution time, where a product formula needs a step count that grows with
    both time and precision. When the DLA is polynomial, this is the whole game.

    :func:`free_part` extends the branch to Hamiltonians whose full DLA is exponential,
    by finding the largest subset of terms that still closes into a small algebra. That
    subset leaves the product formula entirely, taking its commutators out of the error
    bound with it.
"""

import numpy as np
from kak_tools import labelled_matrix_basis, map_dla_to_irrep, pauli_word_to_string
from paulie.classifier.classification import TypeGraph
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.linalg import expm

from lizzy.classify import classify, is_fast_forwardable, summands
from lizzy.hamiltonian import (
    Circuit,
    anticommutation_matrix,
    hamiltonian,
    n_qubits,
    terms_of,
)

# Irrep data by generator set; the split-step route decomposes the same free part once
# per Trotter step, and the mapping (with its Lie closure) is the expensive part.
_irrep_cache: dict = {}


def is_decomposable(hamiltonian_: PauliStringLinear) -> bool:
    """
    Check whether the exact branch can handle a Hamiltonian in one piece.

    Two conditions: the algebra must be small enough to be worth decomposing, and it
    must have an ``so(m)`` presentation, which is what the Pauli-word pipeline of
    ``kak_tools`` implements.

    Parts with more than ``max(64, 8n)`` terms are declined without classifying them, on the
    same budget reasoning as :func:`free_part`: classification at dense sizes costs
    minutes, and every polynomial DLA of a 2-local model lives on a chain or circle
    with O(n) terms (`Wiersema et al. <https://doi.org/10.1038/s41534-024-00900-2>`__;
    all-to-all XX+YY and all-to-all TFIM both classify exponential when checked).

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        bool: True if :func:`decompose` will succeed on it directly.
    """
    width = n_qubits(hamiltonian_)
    if len(terms_of(hamiltonian_)) > max(64, 8 * width):
        return False
    classification = classify(hamiltonian_)
    if not (
        is_fast_forwardable(classification, width)
        and classification.get_orthogonal_size() is not None
    ):
        return False
    # Only canonical graphs of type A (a genuine so line) or NONE (a lone u(1))
    # are accepted. The low-rank coincidences -- su(2) as so(3), sp(2) as so(5) --
    # also present an orthogonal size, but their irrep matching is not the line-graph
    # embedding, and the subgraph search behind it can run for hours on the junk
    # sets the free-part greedy otherwise assembles out of them. Every model the
    # exact branch is for classifies as type A.
    return all(
        morph.get_type() in (TypeGraph.A, TypeGraph.NONE)
        for morph in classification.get_morphs()
    )


def decomposes_by_summand(hamiltonian_: PauliStringLinear) -> bool:
    """
    Check whether the exact branch can handle a Hamiltonian after splitting it.

    Summands commute, so a Hamiltonian is exactly compilable as soon as each of its
    summands is -- which is a strictly weaker condition than :func:`is_decomposable`.
    The XY model is the standard example: as a whole it is ``2*so(n)``, which has no
    ``so(m)`` presentation, while each summand is ``so(n)`` and decomposes.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        bool: True if every summand decomposes.
    """
    return all(is_decomposable(part) for part in summands(hamiltonian_))


def _irrep(words: tuple[str, ...], width: int) -> tuple:
    """Map a generator set into its so(m) irrep, cached.

    Returns the labelled basis for the generators (to build the irrep Hamiltonian), a
    per-plane table of qubit words and spinor scales, and an index ordering that makes
    adjacent planes carry the cheapest words available.
    """
    if words in _irrep_cache:
        return _irrep_cache[words]

    mapping, signs, info = map_dla_to_irrep(list(words))
    m = info.orthogonal_size
    basis = labelled_matrix_basis(mapping, signs, info)

    generators = {
        str(pauli_word_to_string(word, width)): basis[word] for word in basis
    }
    planes = {
        pair: (pauli_word_to_string(word, width), 2.0 * signs[pair])
        for pair, word in mapping.items()
    }

    # An adjacent-plane rotation costs what its Pauli word weighs, and the mapping
    # scatters the cheap words over arbitrary index pairs. Relabelling the basis is
    # free, so order the indices so that adjacent ones are joined by low-weight
    # planes wherever possible. The weight-two planes are the Majorana line of the
    # model -- for a chain they form exactly a path over all m indices -- so the
    # ordering walks the connected paths of that subgraph and concatenates them,
    # paying an expensive plane only at the seams between components.
    cheap: dict[int, list[int]] = {index: [] for index in range(m)}
    for (i, j), (word, _) in planes.items():
        if sum(1 for letter in str(word) if letter != "I") <= 2:
            cheap[i].append(j)
            cheap[j].append(i)

    order = []
    visited: set[int] = set()
    for start in sorted(range(m), key=lambda k: (len(cheap[k]), k)):
        if start in visited:
            continue
        node = start
        while node is not None:
            order.append(node)
            visited.add(node)
            node = next((k for k in cheap[node] if k not in visited), None)

    _irrep_cache[words] = (generators, planes, order, m)
    return _irrep_cache[words]


def decompose(hamiltonian_: PauliStringLinear, time: float, route: str = "exact") -> Circuit:
    r"""
    Synthesize :math:`e^{-itH}` exactly, at a depth independent of the time.

    The algebra's irrep is so(m), so :math:`e^{tH_{\text{irrep}}}` is one orthogonal
    m-by-m matrix, and reducing it to the identity with Givens rotations between
    *adjacent* indices writes it as at most :math:`m(m-1)/2` plane rotations. Each
    adjacent plane maps back to one Pauli rotation, and with the index ordering chosen
    by :func:`_irrep` those words stay low-weight -- where mapping arbitrary-plane
    rotations back yields Jordan-Wigner strings of weight up to m and inflates the
    two-qubit count by a factor of the system size.

    A plane rotation by :math:`\varphi` about a word with spinor scale :math:`a`
    (``2 * sign``, the doubling of the covering map) is the gate rotation by
    :math:`\varphi / a`; the leftover ambiguity of the cover is a global phase, which
    the dense tests confirm.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        route (str): Label recorded against the rotations produced.
    Returns:
        Circuit: The rotations; tests/test_lizzy.py checks them against a dense
        :math:`e^{-itH}`.

    Raises:
        NotImplementedError: If the algebra has no ``so(m)`` Pauli-word mapping.
    """
    terms = terms_of(hamiltonian_)
    words = tuple(str(p) for _, p in terms)
    width = n_qubits(hamiltonian_)
    generators, planes, order, m = _irrep(words, width)

    irrep_h = np.zeros((m, m))
    for coefficient, pauli in terms:
        irrep_h = irrep_h + coefficient.real * generators[str(pauli)]

    # Reduce the permuted orthogonal matrix to the identity column by column with
    # adjacent-row Givens rotations; the applied rotations, transposed and in reverse,
    # are the circuit.
    matrix = expm(time * irrep_h)[np.ix_(order, order)]
    applied: list[tuple[int, float]] = []
    for column in range(m - 1):
        for row in range(m - 2, column - 1, -1):
            a, b = matrix[row, column], matrix[row + 1, column]
            if abs(b) < 1e-14:
                continue
            angle = float(np.arctan2(b, a))
            cosine, sine = np.cos(angle), np.sin(angle)
            upper = cosine * matrix[row] + sine * matrix[row + 1]
            matrix[row + 1] = -sine * matrix[row] + cosine * matrix[row + 1]
            matrix[row] = upper
            applied.append((row, angle))

    # The applied rotations satisfy R_N ... R_1 V = I, so V = R_1^T ... R_N^T. The map
    # from plane generators to qubit rotations sends B to -iP, which negates brackets,
    # so it reverses products: the qubit circuit applies M(R_1^T) first. Transposing a
    # plane rotation negates its angle, and a plane whose indices come out reversed
    # under the ordering negates it once more.
    circuit = Circuit()
    for row, angle in applied:
        i, j = order[row], order[row + 1]
        angle = -angle
        if i > j:
            i, j = j, i
            angle = -angle
        word, scale = planes[(i, j)]
        circuit.add(word, angle / scale, route)
    return circuit


def free_part(
    hamiltonian_: PauliStringLinear,
) -> tuple[PauliStringLinear | None, PauliStringLinear]:
    r"""
    Split a Hamiltonian into a decomposable part and the rest.

    Even when the full DLA is exponential, a subset of the terms often closes into a
    small one -- the transverse-field part of a Heisenberg model, say. That subset can
    be compiled exactly at fixed depth, and because it is no longer inside the product
    formula, none of its internal commutators appear in the Trotter error bound.

    The subsets worth finding are structured rather than arbitrary -- the free part of a
    Heisenberg chain is *all* of its XX and YY terms, not some mixture -- so terms are
    offered in whole interaction families, ordered by the total weight they carry. A
    second pass then tops the set up with individual terms. Greedy growth cannot
    backtrack, so the ordering is what decides the answer.

    Ordering by weight alone decides it badly on chemistry. The heaviest families there
    are the diagonal ones, which commute with each other; the greedy takes them first,
    and the set they build is large enough that every family which does *not* commute
    then fails the classification. What comes out is a set of mutually commuting terms,
    which is discarded at the end -- extracting it would remove no commutator from the
    bound -- so the answer is no free part at all, after several hundred classifications
    spent reaching it.

    Only that outcome is worth a second attempt, and it gets one: the families that
    leave the set commuting are held back and offered again once something
    non-commuting has been found. Taking them early is not wrong in general -- it is
    how the set grows largest, and their commutators with the rest of the free part do
    leave the bound -- so the second ordering is used where the first has already
    failed rather than in place of it. On LiH that is the difference between no free
    part and a forty-four term one.

    Candidates are capped at ``max(64, 8n)`` terms before any classification runs. This is a
    search budget, not a theorem: what it protects against is spending minutes
    classifying a dense model's n^2-term families -- all-to-all XX+YY looks like
    hopping but is not, since beyond nearest neighbours the Jordan-Wigner strings make
    the terms more than quadratic and the algebra exponential, so those classifications
    were expensive ways of hearing no. Every quadratic family of a local model has O(n)
    terms and fits comfortably.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        tuple[PauliStringLinear | None, PauliStringLinear]: The decomposable part, or
        None if no subset of at least two terms qualifies, and the remainder. The two
        together always contain every original term.
    """
    terms = [(c, str(p)) for c, p in terms_of(hamiltonian_)]
    budget = max(64, 8 * n_qubits(hamiltonian_))

    families: dict[str, list[tuple[complex, str]]] = {}
    for coefficient, pauli in terms:
        families.setdefault(_family(pauli), []).append((coefficient, pauli))
    ordered = sorted(
        families.values(), key=lambda f: -sum(abs(c) for c, _ in f)
    )

    free = _grow(ordered, terms, budget, defer_abelian=False)
    if not _is_non_abelian(free):
        free = _grow(ordered, terms, budget, defer_abelian=True)

    if len(free) < 2 or not _is_non_abelian(free):
        # A set of mutually commuting terms contributes nothing to the Trotter error in
        # the first place, so extracting it buys no accuracy and only adds a branch.
        return None, hamiltonian_
    taken = {pauli for _, pauli in free}
    rest = [(c, p) for c, p in terms if p not in taken]
    return _build(free), _build(rest)


def _grow(
    ordered: list[list[tuple[complex, str]]],
    terms: list[tuple[complex, str]],
    budget: int,
    defer_abelian: bool,
) -> list[tuple[complex, str]]:
    """Grow a decomposable set greedily: whole families first, then single terms.

    With ``defer_abelian`` a family that would leave the set mutually commuting is held
    back and offered again at the end of the family pass, once something non-commuting
    has been found for it to join. See :func:`free_part` for why that ordering is the
    second one tried rather than the first.
    """
    free: list[tuple[complex, str]] = []
    deferred: list[list[tuple[complex, str]]] = []

    for family in ordered:
        candidate = free + family
        if not 2 <= len(candidate) <= budget:
            continue
        if defer_abelian and not _is_non_abelian(candidate):
            deferred.append(family)
            continue
        if decomposes_by_summand(_build(candidate)):
            free = candidate

    for family in deferred:
        candidate = free + family
        if 2 <= len(candidate) <= budget and decomposes_by_summand(_build(candidate)):
            free = candidate

    # Top up with any individual term the family pass could not take wholesale.
    taken = {pauli for _, pauli in free}
    for coefficient, pauli in sorted(terms, key=lambda t: -abs(t[0])):
        if pauli in taken:
            continue
        candidate = free + [(coefficient, pauli)]
        if 2 <= len(candidate) <= budget and decomposes_by_summand(_build(candidate)):
            free = candidate
            taken.add(pauli)
    return free


def _is_non_abelian(terms: list[tuple[complex, str]]) -> bool:
    """Check that at least one pair of terms fails to commute."""
    paulis = [p for _, p in terms_of(_build(terms))]
    return bool(anticommutation_matrix(paulis).any())


def _family(pauli: str) -> str:
    """Name the interaction family of a Pauli string, ignoring where it sits.

    ``"IXXI"`` and ``"XXII"`` are both ``"XX"``, so a chain's bond terms form one
    family and its field terms another.
    """
    return "".join(letter for letter in pauli if letter != "I")


def _build(terms: list[tuple[complex, str]]) -> PauliStringLinear:
    """Rebuild a Hamiltonian from coefficient/string pairs."""
    return hamiltonian([(pauli, coefficient) for coefficient, pauli in terms])
