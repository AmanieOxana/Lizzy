"""
    Symmetry reductions that run before any synthesis.

    The conserved charges come from the commutant PauLie computes as a GF(2) null
    space; each independent one would let a qubit be removed isospectrally, and the
    benchmark reports that count. Clustering groups the terms into mutually commuting
    sets, which is what sizes the second-order formula and its error constant.
"""

import networkx as nx
import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.algebra.gf2 import inverse, rank
from lizzy.hamiltonian import anticommutation_matrix, hamiltonian, terms_of


def z2_symmetries(hamiltonian_: PauliStringLinear) -> list[PauliString]:
    r"""
    Find the independent Pauli symmetries of a Hamiltonian.

    These are the elements of the commutant: Pauli strings commuting with every term,
    so each one is a conserved :math:`\mathbb{Z}_{2}` charge. Every charge in a
    mutually commuting subset lets one qubit be removed isospectrally.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        list[PauliString]: Independent symmetries, excluding the identity.
    """
    collection = PauliStringCollection([p for _, p in terms_of(hamiltonian_)])
    return [s for s in collection.get_commutant_basis() if not s.is_identity()]


def commuting_clusters(
    hamiltonian_: PauliStringLinear, strategy: str = "largest_first"
) -> list[PauliStringLinear]:
    r"""
    Partition a Hamiltonian into mutually commuting groups of terms.

    Each group is exponentiated behind a single basis change, so the number of groups
    -- not the number of terms -- sets how many Clifford syntheses a Trotter step needs.
    That number is remarkably stable: two for a transverse-field Ising chain and three
    for Heisenberg, whether the model has fifty terms or eight hundred.

    Finding the fewest groups is graph colouring and is NP-hard, so the default is
    NetworkX's largest-first heuristic on the anticommutation graph. Other NetworkX
    greedy-colouring strategies can be selected when they find a better partition for
    a particular Hamiltonian; useful deterministic choices include ``independent_set``
    and ``saturation_largest_first`` (also available as ``DSATUR``).

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        strategy (str): A strategy accepted by :func:`networkx.greedy_color`.
            Defaults to ``largest_first``.
    Returns:
        list[PauliStringLinear]: The groups, largest first. Terms within a group
        commute pairwise; terms in different groups need not. Equal-sized groups and
        terms within each group follow their original Hamiltonian order.
    """
    terms = terms_of(hamiltonian_)
    paulis = [p for _, p in terms]

    adjacency = anticommutation_matrix(paulis)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(paulis)))
    graph.add_edges_from(zip(*np.nonzero(np.triu(adjacency))))

    colours = nx.coloring.greedy_color(graph, strategy=strategy)
    grouped: dict[int, list[int]] = {}
    for index in range(len(terms)):
        grouped.setdefault(colours[index], []).append(index)

    # Colour numbers and the order in which a strategy visits nodes are implementation
    # details. Canonicalizing with original term indices makes the public result stable
    # while retaining the documented largest-cluster-first ordering.
    groups = sorted(grouped.values(), key=lambda indices: (-len(indices), indices))
    return [
        hamiltonian([(str(terms[index][1]), terms[index][0]) for index in indices])
        for indices in groups
    ]


def pair_clusters(hamiltonian_: PauliStringLinear) -> list[PauliStringLinear] | None:
    r"""
    Partition a weight-<=2 Hamiltonian into layers of disjoint two-qubit kernels.

    A kernel is every term on one qubit pair, fields on its qubits folded in --
    folding is what keeps each kernel's exponential exact, since a field does not
    commute with the pair terms touching its qubit. Kernels on disjoint supports
    commute, so an edge colouring of the interaction graph yields layers that serve
    directly as the formula's summands; :func:`lizzy.emission.kernels.compile_layer`
    emits each kernel as one canonical two-qubit block of at most three CNOTs,
    however many terms it holds. Kernpiler's partial Trotterization
    (arXiv:2504.07214) at the support size where exact synthesis is closed-form.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        list[PauliStringLinear] | None: The layers, or ``None`` when a term has
        weight above two, in which case this clustering does not apply.
    """
    pairs: dict[tuple[int, ...], list[tuple[complex, PauliString]]] = {}
    fields: list[tuple[complex, PauliString]] = []
    for coefficient, pauli in terms_of(hamiltonian_):
        support = tuple(sorted(pauli.get_support()))
        if len(support) == 2:
            pairs.setdefault(support, []).append((coefficient, pauli))
        elif len(support) == 1:
            fields.append((coefficient, pauli))
        else:
            return None

    # Each field joins the heaviest pair covering its qubit; a qubit no pair
    # touches keeps its field as a free-standing kernel of its own.
    weight_of = {s: sum(abs(c) for c, _ in group) for s, group in pairs.items()}
    for coefficient, pauli in fields:
        qubit = next(iter(pauli.get_support()))
        # Only original two-qubit supports have pair weights. A preceding field
        # may have created a singleton kernel; another axis on that same qubit
        # should join it, not look up a nonexistent pair weight.
        homes = [s for s in weight_of if qubit in s]
        home = max(homes, key=lambda s: weight_of[s]) if homes else (qubit,)
        pairs.setdefault(home, []).append((coefficient, pauli))

    # Greedy edge colouring: each layer holds kernels on pairwise disjoint supports.
    colours: dict[tuple[int, ...], int] = {}
    used: dict[int, set[int]] = {}
    for support in sorted(pairs):
        taken = set().union(*(used.setdefault(q, set()) for q in support))
        colour = next(c for c in range(len(pairs)) if c not in taken)
        colours[support] = colour
        for q in support:
            used[q].add(colour)

    layers: dict[int, list[tuple[complex, PauliString]]] = {}
    for support, colour in colours.items():
        layers.setdefault(colour, []).extend(pairs[support])
    return [
        hamiltonian([(str(p), c) for c, p in layers[colour]])
        for colour in sorted(layers)
    ]


def _single_qubit_partners(symmetries, width):
    """Pick one single-qubit Pauli per charge, anticommuting with that charge alone.

    A charge can only be moved onto a qubit by a Pauli that anticommutes with it, and
    the qubits must be distinct or two charges would land on the same one. The
    anticommutation pattern of a candidate against the charge basis is a GF(2) vector;
    ``k`` candidates on distinct qubits whose patterns are independent can be turned
    into the identity by changing the charge basis, which is legitimate because any
    product of charges is again a charge.

    Returns ``(partners, qubits, basis)`` with ``basis`` the GF(2) change of charge
    basis to apply, or ``None`` if no such selection exists.
    """
    count = len(symmetries)
    candidates = []
    for qubit in range(width):
        for letter in "XYZ":
            word = "".join(letter if k == qubit else "I" for k in range(width))
            pauli = get_pauli_string(word)
            pattern = np.array(
                [int(not pauli.commutes_with(s)) for s in symmetries], dtype=np.int64
            )
            if pattern.any():
                candidates.append((qubit, pauli, pattern))

    chosen: list[tuple[int, PauliString]] = []
    matrix = np.zeros((0, count), dtype=np.int64)
    used: set[int] = set()
    for qubit, pauli, pattern in candidates:
        if qubit in used or len(chosen) == count:
            continue
        trial = np.vstack([matrix, pattern])
        if rank(trial) > len(chosen):
            matrix, chosen = trial, chosen + [(qubit, pauli)]
            used.add(qubit)
    if len(chosen) < count:
        return None

    change = inverse(matrix)
    if change is None:
        return None
    return [p for _, p in chosen], [q for q, _ in chosen], change


def _conjugate(coefficient, pauli, charge, partner):
    """Conjugate one term by ``(charge + partner)/sqrt(2)``.

    Only two cases arise here, because every Hamiltonian term commutes with a charge
    by definition: the term is left alone when it also commutes with the partner, and
    otherwise picks up the product with both. The phases are those of the underlying
    Pauli products, pinned against dense matrices in the tests.
    """
    if pauli.commutes_with(partner):
        return coefficient, pauli
    phase = pauli.sign(charge) * (pauli @ charge).sign(partner)
    return coefficient * phase, pauli @ charge @ partner


def taper(hamiltonian_: PauliStringLinear, sector: list[int] | None = None):
    r"""
    Remove one qubit per commuting charge, isospectrally.

    Each :math:`\mathbb{Z}_{2}` charge is rotated onto a single-qubit Pauli by a
    Clifford :math:`(\tau + \sigma)/\sqrt{2}`, after which every term either acts as
    the identity or as :math:`\sigma` on that qubit. The qubit then carries no
    dynamics: it can be replaced by its eigenvalue and dropped
    (`Bravyi et al. <https://arxiv.org/abs/1701.08213>`__). The spectrum of the result
    is the part of the original spectrum lying in the chosen sector.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        sector (list[int], optional): Eigenvalue ``+1`` or ``-1`` per charge. Defaults
            to all ``+1``.
    Returns:
        tuple: ``(tapered, qubits)`` -- the Hamiltonian on the remaining qubits and the
        removed qubit indices -- or ``(None, [])`` when no charge admits a partner.
    """
    # Tapering needs an abelian charge group: two anticommuting charges are not
    # simultaneously diagonalizable, so no common eigenbasis exists to fix. The
    # centralizer basis need not be abelian -- a spin chain of odd length has both
    # X^n and Z^n, which anticommute -- so a mutually commuting subset is taken.
    symmetries: list[PauliString] = []
    for candidate in z2_symmetries(hamiltonian_):
        if all(candidate.commutes_with(kept) for kept in symmetries):
            symmetries.append(candidate)
    if not symmetries:
        return None, []

    width = max(len(p) for _, p in terms_of(hamiltonian_))
    selection = _single_qubit_partners(symmetries, width)
    if selection is None:
        return None, []
    partners, qubits, basis = selection

    # Re-express the charges so that charge i is the only one the partner i sees.
    charges = []
    for index in range(len(symmetries)):
        product = None
        for source, use in enumerate(basis[:, index]):
            if use:
                product = symmetries[source] if product is None else product @ symmetries[source]
        charges.append(product)

    terms = [(c, p) for c, p in terms_of(hamiltonian_)]
    for charge, partner in zip(charges, partners):
        terms = [_conjugate(c, p, charge, partner) for c, p in terms]

    signs = sector or [1] * len(qubits)
    keep = [q for q in range(width) if q not in set(qubits)]
    reduced: dict[str, complex] = {}
    for coefficient, pauli in terms:
        word = str(pauli)
        for qubit, sign in zip(qubits, signs):
            if word[qubit] != "I":
                coefficient = coefficient * sign
        short = "".join(word[q] for q in keep)
        reduced[short] = reduced.get(short, 0) + coefficient
    return hamiltonian(
        [(w, c) for w, c in reduced.items() if abs(c) > 1e-12 and set(w) != {"I"}]
    ), qubits
