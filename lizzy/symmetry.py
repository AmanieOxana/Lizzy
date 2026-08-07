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
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.hamiltonian import anticommutation_matrix, hamiltonian, terms_of


def z2_symmetries(hamiltonian_: PauliStringLinear) -> list[PauliString]:
    r"""
    Find the independent Pauli symmetries of a Hamiltonian.

    These are the elements of the commutant: Pauli strings commuting with every term,
    so each one is a conserved :math:`\mathbb{Z}_{2}` charge. Every independent charge
    lets one qubit be removed isospectrally.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        list[PauliString]: Independent symmetries, excluding the identity.
    """
    collection = PauliStringCollection([p for _, p in terms_of(hamiltonian_)])
    return [s for s in collection.get_commutant_basis() if not s.is_identity()]


def commuting_clusters(hamiltonian_: PauliStringLinear) -> list[PauliStringLinear]:
    r"""
    Partition a Hamiltonian into mutually commuting groups of terms.

    Each group is exponentiated behind a single basis change, so the number of groups
    -- not the number of terms -- sets how many Clifford syntheses a Trotter step needs.
    That number is remarkably stable: two for a transverse-field Ising chain and three
    for Heisenberg, whether the model has fifty terms or eight hundred.

    Finding the fewest groups is graph colouring and is NP-hard, so this uses the
    largest-first heuristic on the anticommutation graph.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        list[PauliStringLinear]: The groups, largest first. Terms within a group
        commute pairwise; terms in different groups need not.
    """
    terms = terms_of(hamiltonian_)
    paulis = [p for _, p in terms]

    adjacency = anticommutation_matrix(paulis)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(paulis)))
    graph.add_edges_from(zip(*np.nonzero(np.triu(adjacency))))

    colours = nx.coloring.greedy_color(graph, strategy="largest_first")
    grouped: dict[int, list[tuple[complex, PauliString]]] = {}
    for index, colour in colours.items():
        grouped.setdefault(colour, []).append(terms[index])

    clusters = [
        hamiltonian([(str(p), c) for c, p in group]) for group in grouped.values()
    ]
    return sorted(clusters, key=len, reverse=True)


def pair_clusters(hamiltonian_: PauliStringLinear) -> list[PauliStringLinear] | None:
    r"""
    Partition a weight-<=2 Hamiltonian into layers of disjoint two-qubit kernels.

    A kernel is every term on one qubit pair, fields on its qubits folded in --
    folding is what keeps each kernel's exponential exact, since a field does not
    commute with the pair terms touching its qubit. Kernels on disjoint supports
    commute, so an edge colouring of the interaction graph yields layers that serve
    directly as the formula's summands; :func:`lizzy.kernels.compile_layer`
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
        homes = [s for s in pairs if qubit in s]
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
