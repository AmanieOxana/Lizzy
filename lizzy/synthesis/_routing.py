"""Candidate construction and cost selection, independent of public result assembly.

Deferred CX plans carry estimates explicitly; CompiledCandidate always retains
a complete emitted artifact. Selection never mutates a caller's Result.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from numpy.linalg import LinAlgError
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.algebra.classify import summands
from lizzy.algebra.symmetry import commuting_clusters, pair_clusters
from lizzy.emission.emit import (
    EmissionQuote,
    emission_candidates,
    greedy_emission_quote,
    shared_frame_candidate,
)
from lizzy.emission.kernels import compile_layer
from lizzy.hamiltonian import (
    Circuit,
    fold_phases,
    hamiltonian,
    n_qubits,
    rotation_cost,
    terms_of,
    weight,
)
from lizzy.synthesis import exact, trotter

# Structurally eligible decompositions may still fail their numerical mapping.
_EXACT_FAILURES = (StopIteration, NotImplementedError, ValueError, LinAlgError)


@dataclass(frozen=True)
class CompiledCandidate:
    """A complete logical/emitted pair measured at one rotation-error budget.

    The budget covers rotation approximation, not floating-point decomposition
    error or controlled global phase. Estimated plans are deliberately separate.
    """

    label: str
    logical: Circuit
    emission: EmissionQuote
    route: str
    rotation_error: float

    @property
    def t_count(self) -> int:
        return self.emission.circuit.t_count

    @property
    def cx_count(self) -> int:
        return self.emission.two_qubit_gates


@dataclass(frozen=True)
class PartSelection:
    """Selected complete CX plan and diagnostics from all attempted plans."""

    circuit: Circuit
    emission: EmissionQuote
    clusters: int
    used_estimates: bool
    randomized: bool


@dataclass(frozen=True)
class TCountSelection:
    """Actual full-circuit costs at one common rotation-approximation budget.

    BDI's gauge search uses an estimate, but the final choice includes the
    unchanged reference compiled with the same backend and error policy.
    Shared frames may replace the previous T-only winner only without increasing
    either T or CX. The chosen output minimizes (T, CX) within that constraint,
    not over all reported candidates or all possible decompositions.
    ``rotation_error`` does not certify floating-point algebraic decomposition
    error or implement a controlled circuit's global phase.
    """

    candidate_t_counts: tuple[tuple[str, int], ...]
    selected: str
    rotation_error: float
    bdi_optimizations: tuple = ()
    rejected_candidates: tuple[tuple[str, str], ...] = ()
    candidate_cx_counts: tuple[tuple[str, int], ...] = ()
    no_regression_reference: str | None = None


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
    exhausted: bool = False

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
        if plan.exhausted:
            return plan
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
        plan.exhausted = True
        return plan


def select_exact_t(
    parts, time, width, error, method, *, gaussian_circuit=None,
    gaussian_failure=None, compare_orthogonal=True,
) -> tuple[CompiledCandidate, TCountSelection]:
    """Compare complete exact candidates at one total rotation-error budget.

    Components are never priced independently: allocation and cancellations can
    change their costs when joined. Auto compares uniform BDI/Givens circuits,
    the BDI gauge variant and an eligible whole-input Gaussian circuit, not every
    mixed component assignment.
    """
    import math

    from lizzy.emission.clifford_t import compile_clifford_t, compile_frame_candidates

    if isinstance(error, bool) or not math.isfinite(error) or not 0 < error < 1:
        raise ValueError("T synthesis error must be finite and between zero and one")
    methods = (("bdi", "givens") if compare_orthogonal else ()) if method == "auto" else (method,)
    if method == "auto" and gaussian_circuit is not None:
        methods += ("gaussian",)
    candidates = []
    diagnostics = []
    rejected = []
    failures = []

    def reject(label, exc):
        rejected.append((label, f"{type(exc).__name__}: {exc}"))
        failures.append(exc)

    if gaussian_failure is not None:
        reject("gaussian-reference", gaussian_failure)
    if method == "auto" and not compare_orthogonal:
        rejected.extend((f"{algorithm}-reference", "Skipped: fixed Gaussian comparison resource cap")
                        for algorithm in ("bdi", "givens"))

    def compile_candidate(label, logical, route, *, optional):
        # No uncharged tolerance-based deletion in the T path. Approximation
        # of every surviving rotation, including tiny angles, is accounted for.
        logical = fold_phases(logical, tolerance=0.0)
        try:
            emitted = compile_clifford_t(logical, width, error=error)
        except (ValueError, ImportError, LinAlgError) as exc:
            if not optional:
                raise
            reject(label, exc)
        else:
            candidates.append(CompiledCandidate(
                label, logical, EmissionQuote("clifford-t", emitted.n_2qb_gates(), emitted),
                route, float(error),
            ))

    for algorithm in methods:
        route = f"exact-{algorithm}"
        label = f"{algorithm}-reference" if method == "auto" else "reference"
        reference = Circuit()
        try:
            if algorithm == "gaussian":
                if gaussian_circuit is None:
                    raise NotImplementedError("No Gaussian candidate was constructed")
                reference = gaussian_circuit
            elif algorithm == "bdi":
                plans = [exact.prepare_bdi(part) for part in parts]
                for plan in plans:
                    reference.extend(plan.circuit(time, route=route))
            else:
                for index, part in enumerate(parts):
                    if not exact.is_decomposable(part):
                        raise NotImplementedError(
                            "method='givens' requires supported so(m) summands; "
                            f"summand {index + 1} is not eligible for exact synthesis"
                        )
                    reference.extend(exact.decompose(part, time, route=route, method=algorithm))
        except _EXACT_FAILURES as exc:
            if method != "auto":
                raise
            reject(label, exc)
            continue
        compile_candidate(label, reference, route, optional=method == "auto")
        if algorithm != "bdi":
            continue

        # Fixed per-entry precision scores gauges consistently. Actual final
        # costs instead use the same TOTAL budget for every complete candidate.
        precision = error / max(1, sum(plan.parameter_bound for plan in plans))
        label = "bdi-nullspace-optimized" if method == "auto" else "nullspace-optimized"
        optimized, has_alternative = Circuit(), False
        try:
            for part in parts:
                candidate = exact.prepare_bdi(part, optimize="t", rotation_error=precision)
                optimized.extend(candidate.circuit(time, route=route))
                diagnostics.append(candidate.optimization)
                has_alternative |= candidate.optimization.optimized
        except _EXACT_FAILURES as exc:
            # An optional gauge-search failure must not suppress its reference.
            reject(label, exc)
        else:
            if has_alternative:
                compile_candidate(label, optimized, route, optional=True)

    if not candidates:
        detail = "; ".join(f"{label}: {reason}" for label, reason in rejected)
        unavailable = next((exc for exc in failures if isinstance(exc, ImportError)), None)
        if unavailable is not None:
            raise ImportError(f"No exact T candidate could be compiled. {detail}") from unavailable
        if failures and all(isinstance(exc, NotImplementedError) for exc in failures):
            raise NotImplementedError(f"No supported exact T synthesis route. {detail}")
        raise ValueError(f"No exact T candidate passed numerical compilation. {detail}")

    # Freeze the former T-only winner before expanding the emission portfolio.
    # A local per-decomposition CX guard would not protect against a global
    # route switch with a higher CX count. Both constraints apply here instead.
    baseline = min(candidates, key=lambda candidate: candidate.t_count)
    frame_cache = {}
    for candidate in tuple(candidates):
        key = tuple((str(word), float(angle)) for word, angle in candidate.logical.rotations)
        if key not in frame_cache:
            frame_cache[key] = compile_frame_candidates(candidate.logical, width, error=error)
        alternatives, frame_rejections = frame_cache[key]
        candidates.extend(
            CompiledCandidate(
                f"{candidate.label}/{strategy}", candidate.logical,
                EmissionQuote("clifford-t", output.n_2qb_gates(), output),
                candidate.route, float(error),
            )
            for strategy, output in alternatives
        )
        rejected.extend((f"{candidate.label}/{strategy}", reason)
                        for strategy, reason in frame_rejections)
    eligible = [candidate for candidate in candidates
                if candidate.t_count <= baseline.t_count
                and candidate.cx_count <= baseline.cx_count]
    winner = min(eligible, key=lambda candidate: (candidate.t_count, candidate.cx_count))
    selection = TCountSelection(
        candidate_t_counts=tuple((candidate.label, candidate.t_count) for candidate in candidates),
        selected=winner.label,
        rotation_error=winner.rotation_error,
        bdi_optimizations=tuple(diagnostics),
        rejected_candidates=tuple(rejected),
        candidate_cx_counts=tuple((candidate.label, candidate.cx_count) for candidate in candidates),
        no_regression_reference=baseline.label,
    )
    return winner, selection


def _priced_plan(circuit: Circuit, width: int, clusters: int) -> _Plan:
    """Fold a complete candidate and retain its cheapest logical/concrete quote.

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


def select_part(
    part: PauliStringLinear,
    time: float,
    error: float,
    order: int,
    seed: int | None,
    randomized: bool,
    calibration: float,
    steps: int | None,
) -> PartSelection:
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

    # BDI owns its broader verified mapping and resource checks; the Givens
    # legacy eligibility filter is not a reason to suppress a BDI candidate.
    # The outer term cap also keeps dense parts out of expensive classification.
    algorithms = ["bdi"] if len(terms_of(part)) <= max(64, 8 * width) else []
    if exact.is_decomposable(part):
        algorithms.append("givens")
    for algorithm in algorithms:
        try:
            fixed = exact.decompose(part, time, route=f"exact-{algorithm}", method=algorithm)
            # Both bounded exact candidates receive identical emission effort;
            # a cheap preliminary quote must not decide BDI versus Givens.
            candidates.append(_priced_plan(fixed, width, len(clusters)).exhaust())
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

    used_estimates = any(
        candidate.used_estimates for candidate in candidates
    )
    # Direct-box quotes cheaply shortlist routes. Only the winner receives the slower
    # decomposed-box pass: materializing even the second-best deferred candidate can
    # mean millions of rotations at a tight error budget.
    plan = min(candidates, key=lambda candidate: candidate.cost).exhaust()

    selected_randomized = False
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
                    selected_randomized = True

    assert plan.circuit is not None
    assert plan.emission is not None
    return PartSelection(
        plan.circuit, plan.emission, plan.clusters, used_estimates, selected_randomized,
    )


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
        def build(repetitions, c=clusters, b=builder, tau=step_time):
            return trotter.cluster_formula(
                c,
                tau * repetitions,
                repetitions,
                compile_cluster=b,
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
