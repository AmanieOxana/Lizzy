"""
    The top-level route: classify, reduce, and take the cheapest exact way out.

    Nothing here decides anything the classification cannot justify. Each step either
    removes work exactly -- summand splitting, the free part -- or sizes a product
    formula from a bound it can defend. The sampled branch is the exception and is off
    by default; see :func:`synthesize`.
"""

from dataclasses import dataclass, field

from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy import exact, trotter
from lizzy.classify import classify, summands
from lizzy.emit import tket_two_qubit_gates
from lizzy.hamiltonian import (
    Circuit,
    fold_phases,
    hamiltonian,
    n_qubits,
    rotation_cost,
    terms_of,
    weight,
)
from lizzy.kernels import compile_layer
from lizzy.symmetry import commuting_clusters, pair_clusters, z2_symmetries

# What an exact-branch attempt can raise when the classification admits an algebra but
# the upstream irrep machinery cannot realize this particular generator set in it.
_EXACT_FAILURES = (StopIteration, NotImplementedError, ValueError)


@dataclass
class Result:
    """
    What a synthesis run produced, and how it got there.

    Attributes:
        circuit (Circuit): The rotations implementing the evolution.
        algebra (str): PauLie's name for the DLA of the whole Hamiltonian.
        summands (int): Number of parts the Hamiltonian split into.
        symmetries (int): Independent Z2 symmetries found, i.e. qubits tapering could
            remove.
        clusters (int): Commuting groups in the largest Trotterized part, zero if
            nothing was Trotterized.
        routes (list[str]): Which branches were used.
        randomized (bool): True if any part was sampled, in which case the error is a
            bound in expectation rather than on this one circuit.
    """

    circuit: Circuit
    algebra: str = ""
    summands: int = 0
    symmetries: int = 0
    clusters: int = 0
    routes: list[str] = field(default_factory=list)
    randomized: bool = False

    @property
    def two_qubit_gates(self) -> int:
        """int: Two-qubit gate count of the whole circuit."""
        return self.circuit.two_qubit_gates


def synthesize(
    hamiltonian_: PauliStringLinear,
    time: float,
    error: float = 1e-3,
    order: int = 4,
    seed: int | None = None,
    randomized: bool = False,
    calibration: float = 1.0,
    steps: int | None = None,
) -> Result:
    r"""
    Synthesize :math:`e^{-itH}` for a Pauli Hamiltonian.

    The route is chosen per part:

    * a part whose algebra is small and orthogonal is decomposed exactly, at a depth
      that does not grow with ``time``;
    * otherwise the largest exactly-compilable subset of its terms is taken out and
      decomposed, and only the remainder goes through a product formula -- either the
      requested order sized by the chain bound, or the second-order formula over
      commuting clusters sized by the collected cluster bound (route ``trotter2``),
      whichever costs fewer gates.

    With ``randomized=True`` the remainder is additionally split by coefficient
    magnitude and its small terms are sampled rather than stepped through.
    **The sampled branch does not keep the error budget, so it is off by default**: its
    gate count is sized from a bound on the averaged channel that the count does not in
    fact deliver, and the shortfall is in the gate count rather than in sampling noise.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        error (float): Total error budget, divided over the parts.
        order (int): Product-formula order for whatever cannot be done exactly.
        seed (int, optional): Seed for the sampled branch.
        randomized (bool): Allow the sampled branch. See the warning above.
        calibration (float): Divides the estimated step count. One keeps the bound; a
            measured factor trades it for a smaller circuit. See
            :func:`lizzy.bench.calibrate`.
        steps (int, optional): Fixed step count for the inexact branches, like the
            ``reps`` of a Qiskit ``PauliEvolutionGate``. Sizing, bounds and their cost
            are skipped entirely, and **no error statement is made**: the circuit is
            the second-order cluster formula with exactly this many steps, however
            accurate that turns out to be. Exact branches still run where the routing
            allows them.
    Returns:
        Result: The circuit and an account of the route taken. ``Result.randomized``
        records whether anything was actually sampled.
    """
    parts = summands(hamiltonian_)
    terms_count = len(terms_of(hamiltonian_))
    # Naming the algebra is reporting, not routing, and classification at dense sizes
    # costs minutes; the same budget that gates the exact branches gates the name.
    small = terms_count <= max(64, 8 * n_qubits(hamiltonian_))
    result = Result(
        circuit=Circuit(),
        algebra=classify(hamiltonian_).get_algebra()
        if small
        else f"(unclassified, {terms_count} terms)",
        summands=len(parts),
        symmetries=len(z2_symmetries(hamiltonian_)),
    )

    for part in parts:
        _synthesize_part(
            part,
            time,
            error / len(parts),
            order,
            seed,
            randomized,
            calibration,
            steps,
            result,
        )

    result.routes = sorted(set(result.circuit.provenance))
    return result


def emission_cost(circuit: Circuit, width: int) -> int:
    """
    Price a rotation sequence under the cheapest available emission.

    The builtin count charges CNOT ladders with single-pair runs merged into
    canonical blocks. With pytket installed, :mod:`lizzy.emit` offers a second
    emission that conjugates rotations into a shared Clifford frame, which wins on
    high-weight sequences. Whichever is cheaper is the price, so the emission tier
    takes part in the routing decision rather than being applied after it -- an
    ordering that emits badly under the builtin count can still be the right one.

    Args:
        circuit (Circuit): The rotations.
        width (int): Number of qubits.
    Returns:
        int: The two-qubit gate count of the cheaper emission.
    """
    builtin = circuit.two_qubit_gates
    if not circuit.rotations:
        return builtin
    # Frame conjugation pays for itself only on high-weight sequences: below weight
    # four a ladder is already near-optimal, and the synthesis costs more to run than
    # it saves. Gating on that keeps 2-local routing free of the call entirely.
    mean_weight = sum(weight(p) for p, _ in circuit.rotations) / len(circuit.rotations)
    if mean_weight < 4.0:
        return builtin
    shared = tket_two_qubit_gates(circuit, width)
    return builtin if shared is None else min(builtin, shared)


def _synthesize_part(
    part: PauliStringLinear,
    time: float,
    error: float,
    order: int,
    seed: int | None,
    randomized: bool,
    calibration: float,
    steps: int | None,
    result: Result,
) -> None:
    """Route one summand: every branch is a candidate, the fewest gates win.

    Exactness is not a priority order. The exact route pays its full fixed depth
    however close the evolution is to the identity, and the hybrid pays its free
    networks every step, so at short times a plain formula can beat both; which one
    wins is arithmetic on this instance, not a preference. Exact attempts can also
    fail structurally -- the classification says so(m), yet the irrep mapping cannot
    embed this generator set -- and a failed candidate simply drops out.
    """
    candidates: list[tuple[int, object]] = []

    if exact.is_decomposable(part):
        try:
            fixed = exact.decompose(part, time, route="exact")
            candidates.append((emission_cost(fixed, n_qubits(part)), fixed))
        except _EXACT_FAILURES:
            pass

    free, rest = exact.free_part(part)
    # An empty remainder means the free part is the whole summand, so the hybrid is
    # the exact route wearing a different label; it is not a split and not a candidate.
    if free is not None and terms_of(rest):
        try:
            hybrid = _hybrid_plan(free, rest, time, error, calibration, steps)
        except _EXACT_FAILURES:
            hybrid = None
        if hybrid is not None:
            candidates.append(hybrid)

    formula = _formula_plan(part, time, error, calibration, steps)
    if formula is not None:
        candidates.append(formula)
    if steps is None:
        candidates.append(_chain_plan(part, time, error, order, calibration))

    cost, plan = min(candidates, key=lambda c: c[0])

    if randomized and steps is None:
        large, small = trotter.split_by_magnitude(part)
        if terms_of(small) and terms_of(large):
            sampled = trotter.product_formula_cost(
                large, time, error / 2, order, calibration
            ) + trotter.qdrift_cost(small, time, error / 2)
            if sampled < cost:
                inner = trotter.steps_for(large, time, error / 2, order, calibration)
                result.circuit.extend(
                    trotter.product_formula(large, time, inner, order, route="trotter")
                )
                result.circuit.extend(
                    trotter.qdrift(small, time, error / 2, seed, route="qdrift")
                )
                result.randomized = True
                return

    # Folding is applied to what is emitted, not to the one step a plan is priced
    # from, so a plan's price is an upper bound on the circuit it produces.
    result.circuit.extend(fold_phases(plan() if callable(plan) else plan))
    result.clusters = max(result.clusters, len(commuting_clusters(part)))


def _hybrid_plan(free, rest, time, error, calibration, steps):
    r"""Price the free-part-as-summand formula, deferring the full build.

    The cluster bound never asked its summands to be commuting clusters -- only that
    each :math:`e^{\tau H_\gamma}` is implemented exactly, which the free part
    satisfies through its Givens network. It enters as the last summand, applied once
    per step at full step time, decomposed once and replayed. One built step prices
    the whole formula.
    """
    clusters = commuting_clusters(rest) + [free]
    if steps is None:
        steps = trotter.steps_for_clusters(clusters, time, error, calibration)
    if steps is None:
        return None

    free_step = Circuit()
    for piece in summands(free):
        free_step.extend(exact.decompose(piece, time / steps, route="exact-in-step"))
    one_step = trotter.cluster_formula(clusters, time / steps, 1, middle=free_step)
    fixed = steps
    width = n_qubits(rest) if terms_of(rest) else n_qubits(free)
    return (
        steps * emission_cost(one_step, width),
        lambda: trotter.cluster_formula(clusters, time, fixed, middle=free_step),
    )


def _formula_plan(part, time, error, calibration, steps):
    """Price the second-order formula under each applicable ordering.

    Letter clusters emit plain rotations; pair-kernel layers emit through the exact
    two-qubit KAK, fields folded in; term order is the ungrouped formula, which wins
    where an instance has no structure to group and a downstream Pauli-network
    emission would rather see the terms as given. Each is priced from one built step,
    with its own error constant, and the cheapest stands.
    """
    term_order = [hamiltonian([(str(pauli), c)]) for c, pauli in terms_of(part)]
    plans = []
    for clusters, builder, regroup in (
        (commuting_clusters(part), None, True),
        (pair_clusters(part), compile_layer, True),
        (term_order, None, False),
    ):
        if clusters is None:
            continue
        if regroup:
            # The formula merges the last summand's half-passes into one full pass, so
            # the merge goes to the summand whose rotations cost the most. Term order
            # is exempt: its whole point is to leave the sequence as given.
            clusters = sorted(
                clusters,
                key=lambda cluster: sum(rotation_cost(p) for _, p in terms_of(cluster)),
            )
        count = steps
        if count is None:
            count = trotter.steps_for_clusters(clusters, time, error, calibration)
        if count is None:
            continue
        one_step = trotter.cluster_formula(
            clusters, time / count, 1, compile_cluster=builder
        )
        fixed = count
        plans.append(
            (
                count * emission_cost(one_step, n_qubits(part)),
                lambda c=clusters, k=fixed, b=builder: trotter.cluster_formula(
                    c, time, k, compile_cluster=b
                ),
            )
        )
    return min(plans, key=lambda x: x[0]) if plans else None


def _chain_plan(part, time, error, order, calibration):
    """Price the requested-order formula sized by the chain bound.

    Priced from one built step like the others, so the emission tier sees it too.
    """
    steps = trotter.steps_for(part, time, error, order, calibration)
    one_step = trotter.product_formula(part, time / steps, 1, order, route="trotter")
    return (
        steps * emission_cost(one_step, n_qubits(part)),
        lambda: trotter.product_formula(part, time, steps, order, route="trotter"),
    )


"""
    The top-level route: classify, reduce, and take the cheapest exact way out.

    Nothing here decides anything the classification cannot justify. Each step either
    removes work exactly -- summand splitting, the free part -- or sizes a product
    formula from a bound it can defend. The sampled branch is the exception and is off
    by default; see :func:`synthesize`.
"""

from dataclasses import dataclass, field

from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.hamiltonian import Circuit

# What an exact-branch attempt can raise when the classification admits an algebra but
# the upstream irrep machinery cannot realize this particular generator set in it.
_EXACT_FAILURES = (StopIteration, NotImplementedError, ValueError)
