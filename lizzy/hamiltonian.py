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

# Rotating about a Pauli string of weight w costs 2(w-1) two-qubit gates: a CNOT ladder
# onto one qubit, the rotation, and the ladder back.
TWO_QUBIT_COST_PER_WEIGHT = 2


def anticommutation_matrix(paulis: list[PauliString]) -> np.ndarray:
    r"""
    Get the anticommutation adjacency of a list of Pauli strings.

    This is the symplectic Gram matrix over GF(2), so it is one integer matrix product
    rather than :math:`L^{2}` pairwise tests -- the difference between milliseconds
    and minutes once dense models reach thousands of terms. Everything graph-shaped
    downstream (summand splitting, clustering, the pairwise commutator sum) is built
    on it.

    Args:
        paulis (list[PauliString]): The Pauli strings.
    Returns:
        numpy.ndarray: Symmetric 0/1 matrix; entry ``(a, b)`` is one iff they
        anticommute.
    """
    x = np.array([[int(b) for b in p.bits[::2]] for p in paulis], dtype=np.int64)
    z = np.array([[int(b) for b in p.bits[1::2]] for p in paulis], dtype=np.int64)
    adjacency = (x @ z.T + z @ x.T) % 2
    np.fill_diagonal(adjacency, 0)
    return adjacency


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

    @property
    def two_qubit_gates(self) -> int:
        r"""int: Total two-qubit gate count, the quantity being minimized.

        Counted block-aware: a maximal run of consecutive rotations whose joint
        support fits on one qubit pair compiles as a single canonical two-qubit
        block, which costs at most three CNOTs however many rotations it holds
        (Kernpiler's partial-Trotterization observation, arXiv:2504.07214). Runs
        that a pair cannot hold are charged their CNOT ladders.
        """
        def charge(support: frozenset[int], cost: int) -> int:
            # The three-CNOT cap is what a canonical two-qubit block costs, so it
            # only applies to a run that fits on one pair. A wider rotation pays
            # its ladder; capping it too would report a block that nothing emits.
            return min(cost, 3) if len(support) <= 2 else cost

        total = 0
        run_cost = 0
        run_support: frozenset[int] = frozenset()
        for pauli, _ in self.rotations:
            support = frozenset(pauli.get_support())
            joined = run_support | support
            if len(joined) <= 2:
                run_support = joined
                run_cost += rotation_cost(pauli)
            else:
                total += charge(run_support, run_cost)
                run_support, run_cost = support, rotation_cost(pauli)
        return total + charge(run_support, run_cost)

    def cost_by_route(self) -> dict[str, int]:
        """
        Get the two-qubit gate count attributed to each route.

        Uses the same block-aware count as :attr:`two_qubit_gates`; a run spanning
        routes is attributed to the route that started it, so the values sum to the
        total.

        Returns:
            dict[str, int]: Route label to two-qubit gate count.
        """
        costs: dict[str, int] = {}
        run_cost = 0
        run_route: str | None = None
        run_support: frozenset[int] = frozenset()
        for (pauli, _), route in zip(self.rotations, self.provenance):
            support = frozenset(pauli.get_support())
            joined = run_support | support
            if len(joined) <= 2 and run_route is not None:
                run_support = joined
                run_cost += rotation_cost(pauli)
            else:
                if run_route is not None:
                    capped = min(run_cost, 3) if len(run_support) <= 2 else run_cost
                    costs[run_route] = costs.get(run_route, 0) + capped
                run_support, run_cost, run_route = support, rotation_cost(pauli), route
        if run_route is not None:
            capped = min(run_cost, 3) if len(run_support) <= 2 else run_cost
            costs[run_route] = costs.get(run_route, 0) + capped
        return costs

    def __len__(self) -> int:
        return len(self.rotations)


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
