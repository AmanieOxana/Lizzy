"""
    The top-level route: classify, reduce, and price every feasible way out.

    Nothing here decides anything the classification cannot justify. Each step either
    removes work exactly -- summand splitting, the free part -- or sizes a product
    formula from a bound it can defend. The sampled branch is the exception and is off
    by default; see :func:`synthesize`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy import exact, trotter
from lizzy.classify import classify, summands
from lizzy.emit import (
    EmissionQuote,
    best_emission,
    emission_candidates,
    greedy_emission_quote,
    shared_frame_candidate,
)
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
        clusters (int): Commuting groups in the part that had the most of them,
            counted whichever route that part ended up taking.
        routes (list[str]): Which branches were used.
        randomized (bool): True if any part was sampled, in which case the requested
            error is not guaranteed for the returned circuit or channel.
        error_guaranteed (bool): Whether the selected deterministic route retains the
            requested error-budget contract. Fixed-step, calibrated and qDRIFT
            formulas are false; exact routes remain true.
        routing_estimated (bool): True when at least one oversized route was compared
            using the 1/2/4-repetition cost model instead of full materialization.
        emission (EmissionQuote, optional): The concrete backend artifact selected for
            the complete folded circuit. The logical Pauli sequence remains in
            ``circuit`` for provenance and verification.
    """

    circuit: Circuit
    algebra: str = ""
    summands: int = 0
    symmetries: int = 0
    clusters: int = 0
    routes: list[str] = field(default_factory=list)
    randomized: bool = False
    error_guaranteed: bool = True
    routing_estimated: bool = False
    emission: EmissionQuote | None = None

    @property
    def logical_two_qubit_gates(self) -> int:
        """int: Builtin block/ladder count of the logical Pauli sequence."""
        return self.circuit.two_qubit_gates

    @property
    def two_qubit_gates(self) -> int:
        """int: Actual two-qubit count of the selected complete emission."""
        if self.emission is None:
            return self.logical_two_qubit_gates
        return self.emission.two_qubit_gates

    @property
    def emission_backend(self) -> str:
        """str: Name of the backend that won the complete-circuit portfolio."""
        return "builtin" if self.emission is None else self.emission.backend

    @property
    def emitted_circuit(self) -> object:
        """The selected backend artifact; ``circuit`` remains the logical sequence."""
        return self.circuit if self.emission is None else self.emission.circuit


@dataclass
class _Plan:
    """A full candidate, or a bounded-size repetition estimate awaiting selection."""

    circuit: Circuit | None
    emission: EmissionQuote | None
    clusters: int
    emissions: list[EmissionQuote] = field(default_factory=list)
    estimate: int | None = None
    builder: Callable[[], Circuit] | None = None
    width: int | None = None
    used_estimates: bool = False

    @property
    def cost(self) -> int:
        if self.emission is not None:
            return self.emission.two_qubit_gates
        if self.estimate is None:
            raise RuntimeError("unpriced synthesis plan")
        return self.estimate

    def materialize(self) -> "_Plan":
        """Build and quote a deferred full sequence once it wins the shortlist."""
        if self.circuit is not None:
            return self
        if self.builder is None or self.width is None:
            raise RuntimeError("deferred synthesis plan has no builder")
        plan = _priced_plan(self.builder(), self.width, self.clusters)
        plan.used_estimates = True
        return plan

    def exhaust(self) -> "_Plan":
        """Run the slower emission backend on a materialized shortlisted route."""
        plan = self.materialize()
        if plan.circuit is None or plan.width is None:
            raise RuntimeError("materialized synthesis plan lost its circuit")
        direct_was_eligible = any(
            quote.backend == "pytket-direct" for quote in plan.emissions
        )
        if direct_was_eligible or shared_frame_candidate(plan.circuit):
            shared = greedy_emission_quote(plan.circuit, plan.width)
            if shared is not None:
                plan.emissions.append(shared)
        plan.emission = min(
            plan.emissions,
            key=lambda quote: quote.two_qubit_gates,
        )
        return plan


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

    Every manageable candidate the classification allows is priced per part and the
    cheapest circuit wins -- exactness is not a priority order:

    * an exact decomposition, where the algebra is small and orthogonal, at a depth
      that does not grow with ``time``;
    * a hybrid, taking the largest exactly-compilable subset out and carrying it as
      one more summand inside each step;
    * a product formula -- either the requested order sized by the chain bound, or
      the second-order formula over commuting clusters sized by the collected
      cluster bound (route ``trotter2``).

    Complete folded sequences are emission-priced whenever their projected size is at
    most 20,000 rotations. Larger losing candidates are shortlisted from one, two and
    four repetitions; the selected route is always fully materialized and quoted.

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
        records whether anything was actually sampled; ``error_guaranteed`` and
        ``routing_estimated`` distinguish accuracy from cost-model status.
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

    # Parts commute, but their emitted Clifford frames need not compose additively.
    # Multiple parts therefore need one complete-circuit quote. A single selected
    # plan already retains exactly that artifact, so do not compile it a third time.
    if len(parts) > 1:
        result.circuit = fold_phases(result.circuit)
        result.emission = best_emission(result.circuit, n_qubits(hamiltonian_))
    elif result.emission is not None and result.emission.backend == "builtin":
        result.emission = EmissionQuote(
            "builtin", result.circuit.two_qubit_gates, result.circuit
        )
    result.routes = sorted(set(result.circuit.provenance))
    exact_only = set(result.routes).issubset({"exact"})
    result.error_guaranteed = not result.randomized and (
        exact_only or (steps is None and calibration <= 1.0)
    )
    return result


def _priced_plan(circuit: Circuit, width: int, clusters: int) -> _Plan:
    """Fold a complete candidate, then keep the cheapest exact emission as its price.

    Whichever emission is cheaper is what the candidate costs, so the emission tier
    takes part in routing rather than being applied only after it.
    """
    logical = fold_phases(circuit)
    quotes = emission_candidates(logical, width, exhaustive=False)
    return _Plan(
        logical,
        min(quotes, key=lambda quote: quote.two_qubit_gates),
        clusters,
        emissions=quotes,
        width=width,
    )


_FULL_PRICE_ROTATIONS = 20_000


def _repeated_plan(
    build: Callable[[int], Circuit],
    repetitions: int,
    width: int,
    clusters: int,
) -> _Plan:
    """Price a repeated formula exactly when affordable, otherwise shortlist it.

    Building every losing million-rotation formula would make tight error budgets
    unusable. Up to four repetitions capture setup, steady-state frame cost and
    boundary folding. The extrapolation is used only for routing; whichever plan wins
    is materialized and quoted on its complete folded sequence before it is returned.
    """
    if repetitions <= 4:
        return _priced_plan(build(repetitions), width, clusters)

    raw_four = build(4)
    sample_four = _priced_plan(raw_four, width, clusters)
    projected_rotations = len(raw_four.rotations) * repetitions / 4
    if projected_rotations <= _FULL_PRICE_ROTATIONS:
        return _priced_plan(build(repetitions), width, clusters)

    # An affine fit over 1/2/4 repetitions models setup plus steady-state frame cost.
    # It is deliberately exposed as an estimate on Result: heuristic backend choices
    # need not scale monotonically enough for this to be a mathematical bound.
    samples = [
        (1, _priced_plan(build(1), width, clusters).cost),
        (2, _priced_plan(build(2), width, clusters).cost),
        (4, sample_four.cost),
    ]
    mean_x = sum(count for count, _ in samples) / len(samples)
    mean_y = sum(cost for _, cost in samples) / len(samples)
    denominator = sum((count - mean_x) ** 2 for count, _ in samples)
    slope = max(
        sum((count - mean_x) * (cost - mean_y) for count, cost in samples)
        / denominator,
        0.0,
    )
    intercept = max(mean_y - slope * mean_x, 0.0)
    estimate = round(intercept + repetitions * slope)
    return _Plan(
        circuit=None,
        emission=None,
        clusters=clusters,
        estimate=estimate,
        builder=lambda: build(repetitions),
        width=width,
        used_estimates=True,
    )


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
    candidates: list[_Plan] = []
    clusters = commuting_clusters(part)
    width = n_qubits(part)

    if exact.is_decomposable(part):
        try:
            fixed = exact.decompose(part, time, route="exact")
            candidates.append(_priced_plan(fixed, width, len(clusters)))
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

    formula = _formula_plan(part, clusters, time, error, calibration, steps)
    if formula is not None:
        candidates.append(formula)
    if steps is None:
        candidates.append(_chain_plan(part, len(clusters), time, error, order, calibration))

    result.routing_estimated = result.routing_estimated or any(
        candidate.used_estimates for candidate in candidates
    )
    # Direct-box quotes cheaply shortlist routes. Only the winner receives the slower
    # decomposed-box pass: materializing even the second-best deferred candidate can
    # mean millions of rotations at a tight error budget.
    plan = min(candidates, key=lambda candidate: candidate.cost).exhaust()

    if randomized and steps is None:
        large, small = trotter.split_by_magnitude(part)
        if terms_of(small) and terms_of(large):
            sampled = trotter.product_formula_cost(
                large, time, error / 2, order, calibration
            ) + trotter.qdrift_cost(small, time, error / 2)
            if sampled < plan.cost:
                inner = trotter.steps_for(large, time, error / 2, order, calibration)
                sampled_circuit = trotter.product_formula(
                    large, time, inner, order, route="trotter"
                )
                sampled_circuit.extend(
                    trotter.qdrift(small, time, error / 2, seed, route="qdrift")
                )
                sampled_plan = _priced_plan(
                    sampled_circuit, width, len(clusters)
                ).exhaust()
                if sampled_plan.cost < plan.cost:
                    plan = sampled_plan
                    result.randomized = True

    assert plan.circuit is not None
    result.circuit.extend(plan.circuit)
    result.emission = plan.emission
    result.clusters = max(result.clusters, plan.clusters)


def _hybrid_plan(free, rest, time, error, calibration, steps):
    r"""Build and price the complete free-part-as-summand formula.

    The cluster bound never asked its summands to be commuting clusters -- only that
    each :math:`e^{\tau H_\gamma}` is implemented exactly, which the free part
    satisfies through its Givens network. It enters as the last summand, applied once
    per step at full step time, decomposed once and replayed. The complete folded
    sequence is priced so cancellation and shared frames across steps count.
    """
    clusters = commuting_clusters(rest) + [free]
    if steps is None:
        steps = trotter.steps_for_clusters(clusters, time, error, calibration)
    if steps is None:
        return None

    free_step = Circuit()
    for piece in summands(free):
        free_step.extend(exact.decompose(piece, time / steps, route="exact-in-step"))
    width = n_qubits(rest) if terms_of(rest) else n_qubits(free)
    step_time = time / steps
    return _repeated_plan(
        lambda repetitions: trotter.cluster_formula(
            clusters,
            step_time * repetitions,
            repetitions,
            middle=free_step,
        ),
        steps,
        width,
        len(clusters),
    )


def _formula_plan(part, commuting, time, error, calibration, steps):
    """Price the second-order formula under each applicable ordering.

    Letter clusters emit plain rotations; pair-kernel layers emit through the exact
    two-qubit KAK, fields folded in; term order is the ungrouped formula, which wins
    where an instance has no structure to group and a downstream Pauli-network
    emission would rather see the terms as given. Each complete folded formula is
    emitted, with its own error constant, and the cheapest stands.

    The commuting clusters are passed in rather than found here: the caller reports
    their count either way, and the colouring is not free on a Hamiltonian with
    hundreds of terms.
    """
    terms = terms_of(part)
    term_order = [hamiltonian([(str(pauli), c)]) for c, pauli in terms]
    commuting_options = [commuting]
    # In error-budget mode each partition needs its own expensive, order-dependent
    # bound. The alternative therefore starts as a fixed-depth high-weight candidate;
    # a term cap keeps independent-set colouring out of quadratic-size stress cases.
    mean_weight = sum(weight(pauli) for _, pauli in terms) / len(terms)
    if steps is not None and mean_weight >= 4.0 and len(terms) <= 768:
        alternative = commuting_clusters(part, strategy="independent_set")
        if _cluster_signature(alternative) != _cluster_signature(commuting):
            commuting_options.append(alternative)

    families = [(clusters, None, True) for clusters in commuting_options] + [
        (pair_clusters(part), compile_layer, True),
        (term_order, None, False),
    ]
    plans: list[_Plan] = []
    for clusters, builder, regroup in families:
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
        step_time = time / count
        build = lambda repetitions, c=clusters, b=builder, tau=step_time: (
            trotter.cluster_formula(
                c,
                tau * repetitions,
                repetitions,
                compile_cluster=b,
            )
        )
        plans.append(_repeated_plan(build, count, n_qubits(part), len(clusters)))
    if not plans:
        return None
    used_estimates = any(plan.used_estimates for plan in plans)
    winner = min(plans, key=lambda plan: plan.cost)
    winner.used_estimates = winner.used_estimates or used_estimates
    return winner


def _cluster_signature(clusters) -> tuple[tuple[str, ...], ...]:
    """Canonical term-partition key used to avoid pricing duplicate colourings."""
    groups = [
        tuple(sorted(str(pauli) for _, pauli in terms_of(cluster)))
        for cluster in clusters
    ]
    return tuple(sorted(groups))


def _chain_plan(part, clusters, time, error, order, calibration):
    """Price the requested-order formula sized by the chain bound.

    The complete folded sequence is priced so the emission tier sees cancellation
    across steps and the actual terminal frame.
    """
    steps = trotter.steps_for(part, time, error, order, calibration)
    step_time = time / steps
    return _repeated_plan(
        lambda repetitions: trotter.product_formula(
            part,
            step_time * repetitions,
            repetitions,
            order,
            route="trotter",
        ),
        steps,
        n_qubits(part),
        clusters,
    )
