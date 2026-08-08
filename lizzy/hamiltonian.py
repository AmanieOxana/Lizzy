"""
    Pauli Hamiltonians, and the circuits they compile to.

    A Hamiltonian is a :class:`paulie.common.pauli_string_linear.PauliStringLinear`, a
    linear combination of Pauli strings. This module holds the few things PauLie has no
    reason to provide: constructors for the model families used as benchmarks, and the
    gate-counted circuit that synthesis produces.
"""

from dataclasses import dataclass, field

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy import gf2

# Rotating about a Pauli string of weight w costs 2(w-1) two-qubit gates: a CNOT ladder
# onto one qubit, the rotation, and the ladder back.
TWO_QUBIT_COST_PER_WEIGHT = 2


def symplectic_vectors(paulis: list[PauliString]) -> np.ndarray:
    """
    Get Pauli strings as symplectic bit vectors ``[x | z]``, one row each.

    Args:
        paulis (list[PauliString]): The Pauli strings.
    Returns:
        numpy.ndarray: Integer array of shape ``(len(paulis), 2 * qubits)``.
    """
    if not paulis:
        return np.zeros((0, 0), dtype=np.int64)
    x = np.array([[int(b) for b in p.bits[::2]] for p in paulis], dtype=np.int64)
    z = np.array([[int(b) for b in p.bits[1::2]] for p in paulis], dtype=np.int64)
    return np.hstack([x, z])


def anticommutation_matrix(paulis: list[PauliString]) -> np.ndarray:
    """
    Get the anticommutation adjacency of a list of Pauli strings.

    Args:
        paulis (list[PauliString]): The Pauli strings.
    Returns:
        numpy.ndarray: Symmetric 0/1 matrix; entry ``(a, b)`` is one iff they
        anticommute.
    """
    return gf2.gram(symplectic_vectors(paulis))


def terms_of(hamiltonian: PauliStringLinear) -> list[tuple[complex, PauliString]]:
    """
    Get a Hamiltonian's terms as coefficient/Pauli pairs.

    Args:
        hamiltonian (PauliStringLinear): The Hamiltonian.
    Returns:
        list[tuple[complex, PauliString]]: The terms.
    """
    return list(hamiltonian)


def n_qubits(hamiltonian: PauliStringLinear) -> int:
    """
    Get the number of qubits a Hamiltonian acts on.

    Args:
        hamiltonian (PauliStringLinear): The Hamiltonian.
    Returns:
        int: The qubit count.
    """
    return max(len(p) for _, p in terms_of(hamiltonian))


def weight(pauli: PauliString) -> int:
    """
    Get the number of non-identity factors in a Pauli string.

    Args:
        pauli (PauliString): The Pauli string.
    Returns:
        int: Its weight.
    """
    return sum(1 for letter in str(pauli) if letter != "I")


def rotation_cost(pauli: PauliString) -> int:
    """
    Get the two-qubit gate cost of one rotation about a Pauli string.

    Args:
        pauli (PauliString): The Pauli string being rotated about.
    Returns:
        int: Two-qubit gate count, zero for weight-one and identity strings.
    """
    return TWO_QUBIT_COST_PER_WEIGHT * max(weight(pauli) - 1, 0)


def _block_charge(support: frozenset, cost: int, run: list) -> int:
    """Charge one run of rotations: a run held by a single qubit pair compiles as one
    canonical block, priced by its canonical class; a wider run pays its ladders.

    The import is local because the canonical form belongs to the two-qubit synthesis,
    which is built on this module.
    """
    if len(support) > 2:
        return cost
    if len(support) < 2 or cost == 0:
        return 0
    from lizzy.kernels import canonical_cost

    pair = tuple(sorted(support))
    return min(cost, canonical_cost([(str(p), a) for p, a in run], pair))


@dataclass
class Circuit:
    r"""
    A synthesized circuit, as the sequence of Pauli rotations it applies.

    A rotation ``(P, theta)`` means :math:`e^{-i\theta P}`, and the rotations are
    applied left to right, so the unitary is
    :math:`e^{-i\theta_k P_k} \cdots e^{-i\theta_1 P_1}`.

    Attributes:
        rotations (list): ``(PauliString, angle)`` pairs, applied left to right.
        provenance (list[str]): One label per rotation naming the route that produced
            it, so a benchmark can attribute cost to the branch that incurred it.
    """

    rotations: list[tuple[PauliString, float]] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    def add(self, pauli: PauliString, angle: float, route: str) -> None:
        """
        Append one Pauli rotation.

        Args:
            pauli (PauliString): The Pauli string to rotate about.
            angle (float): The rotation angle.
            route (str): Label of the branch that produced this rotation.
        """
        self.rotations.append((pauli, angle))
        self.provenance.append(route)

    def extend(self, other: "Circuit") -> None:
        """
        Append another circuit's rotations.

        Args:
            other (Circuit): The circuit to append.
        """
        self.rotations.extend(other.rotations)
        self.provenance.extend(other.provenance)

    def blocks(self):
        r"""
        Walk the circuit as the blocks its gate count is charged in.

        A maximal run of consecutive rotations whose joint support fits on one qubit
        pair compiles as a single canonical two-qubit block, and is charged what that
        block's canonical class costs -- at most three CNOTs however many rotations it
        holds (Kernpiler's partial-Trotterization observation, arXiv:2504.07214), and
        two where a canonical parameter is trivial. Runs that a pair cannot hold are
        charged their CNOT ladders. A run spanning routes is attributed to the route
        that started it, which is what makes the per-route costs sum to the total.

        Yields:
            tuple[str, int]: Route label and two-qubit gate count, one per block.
        """
        run: list[tuple[PauliString, float]] = []
        run_cost = 0
        run_route: str | None = None
        run_support: frozenset[int] = frozenset()
        for rotation, route in zip(self.rotations, self.provenance):
            pauli = rotation[0]
            support = frozenset(pauli.get_support())
            joined = run_support | support
            if len(joined) <= 2 and run_route is not None:
                run_support = joined
                run_cost += rotation_cost(pauli)
                run.append(rotation)
            else:
                if run_route is not None:
                    yield run_route, _block_charge(run_support, run_cost, run)
                run_support, run_cost, run_route = support, rotation_cost(pauli), route
                run = [rotation]
        if run_route is not None:
            yield run_route, _block_charge(run_support, run_cost, run)

    @property
    def two_qubit_gates(self) -> int:
        """int: Total two-qubit gate count, the quantity being minimized.

        Counted block-aware; see :meth:`blocks`.
        """
        return sum(charge for _, charge in self.blocks())

    def cost_by_route(self) -> dict[str, int]:
        """
        Get the two-qubit gate count attributed to each route.

        Returns:
            dict[str, int]: Route label to two-qubit gate count, summing to
            :attr:`two_qubit_gates`.
        """
        costs: dict[str, int] = {}
        for route, charge in self.blocks():
            costs[route] = costs.get(route, 0) + charge
        return costs

    def __len__(self) -> int:
        return len(self.rotations)


def fold_phases(circuit: "Circuit", tolerance: float = 1e-12) -> "Circuit":
    r"""
    Merge rotations about the same Pauli, and drop the ones that cancel.

    Two rotations about :math:`P` can be brought together whenever every rotation
    between them commutes with :math:`P`, and once adjacent they are one rotation of
    the summed angle. A symmetric step repeats every term twice by construction, so
    this is not a rare coincidence: on a fully commuting Hamiltonian it halves the
    sequence, and it merges across steps as well.

    A merged angle that is a multiple of :math:`2\pi` leaves the identity behind and
    the rotation is dropped; multiples of :math:`\pi` are kept, since those differ by
    a global phase this representation does not carry.

    Written as a backward scan this is quadratic, and on a Trotter circuit of tens of
    thousands of rotations it costs more than the synthesis that produced them. Two
    facts make it near-linear instead. Only a rotation sharing a qubit can fail to
    commute, so the barrier can be looked for among those alone; and the rotation to
    merge with is the *last* one about the same Pauli, so the search can stop as soon
    as it falls behind that one -- whatever lies further back cannot decide anything.
    Both are bookkeeping over the same scan, and the result is the one the naive
    version produces, rotation for rotation.

    Args:
        circuit (Circuit): The rotations.
        tolerance (float): Angle below which a rotation counts as the identity.
    Returns:
        Circuit: An equivalent circuit, never longer.
    """
    kept: list[tuple[PauliString, float, str]] = []
    last_about: dict[str, int] = {}
    touching: dict[int, list[int]] = {}

    for index, (pauli, angle) in enumerate(circuit.rotations):
        route = circuit.provenance[index] if index < len(circuit.provenance) else ""
        word = str(pauli)
        support = pauli.get_support()
        target = last_about.get(word, -1)

        # Nothing to merge with, or something between here and it fails to commute:
        # either way this rotation stands on its own.
        blocked = target < 0
        if not blocked:
            checked: set[int] = set()
            for qubit in support:
                for other in reversed(touching.get(qubit, ())):
                    if other <= target:
                        break
                    if other in checked:
                        continue
                    checked.add(other)
                    if not kept[other][0].commutes_with(pauli):
                        blocked = True
                        break
                if blocked:
                    break

        if not blocked:
            other, other_angle, other_route = kept[target]
            kept[target] = (other, other_angle + angle, other_route)
            continue

        kept.append((pauli, angle, route))
        last_about[word] = len(kept) - 1
        for qubit in support:
            touching.setdefault(qubit, []).append(len(kept) - 1)

    folded = Circuit()
    for pauli, angle, route in kept:
        if abs(angle % (2 * np.pi)) < tolerance:
            continue
        folded.add(pauli, angle, route)
    return folded


def hamiltonian(terms: dict[str, float] | list[tuple[str, float]]) -> PauliStringLinear:
    """
    Build a Hamiltonian from Pauli strings and coefficients.

    Args:
        terms: Pauli string to coefficient, as a mapping or a list of pairs.
    Returns:
        PauliStringLinear: The Hamiltonian.
    """
    pairs = terms.items() if isinstance(terms, dict) else terms
    return PauliStringLinear([(complex(c), s) for s, c in pairs])


def _chain(pattern: str, n: int, sites: int | None = None) -> list[str]:
    """Place ``pattern`` at every position along an open chain of ``n`` qubits."""
    width = len(pattern)
    stop = (n - width + 1) if sites is None else sites
    return ["I" * w + pattern + "I" * (n - w - width) for w in range(stop)]


def model(name: str, n: int, seed: int | None = None) -> PauliStringLinear:
    """
    Build one of the benchmark spin models on an open chain.

    Args:
        name (str): One of ``"tfim"``, ``"tfxy"``, ``"heisenberg"``, ``"xy"`` or
            ``"heisenberg_all_to_all"``.
        n (int): Number of qubits.
        seed (int, optional): Seed for random coefficients. Uniform coefficients when
            omitted.
    Returns:
        PauliStringLinear: The Hamiltonian.

    Raises:
        ValueError: If the model name is unknown.
    """
    if name == "tfim":
        strings = _chain("XX", n) + _chain("Z", n)
    elif name == "tfxy":
        strings = _chain("XX", n) + _chain("YY", n) + _chain("Z", n)
    elif name == "xy":
        strings = _chain("XX", n) + _chain("YY", n)
    elif name == "heisenberg":
        strings = _chain("XX", n) + _chain("YY", n) + _chain("ZZ", n)
    elif name == "heisenberg_all_to_all":
        strings = [
            "I" * i + a + "I" * (j - i - 1) + a + "I" * (n - j - 1)
            for a in "XYZ"
            for i in range(n)
            for j in range(i + 1, n)
        ]
    else:
        raise ValueError(f"Unknown model {name!r}.")

    if seed is None:
        coefficients = np.ones(len(strings))
    else:
        coefficients = np.random.default_rng(seed).normal(size=len(strings))
    return hamiltonian(list(zip(strings, coefficients)))
