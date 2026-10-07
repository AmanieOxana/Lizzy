"""
    The top-level route: classify, reduce, and price every feasible way out.

    Algebraic eligibility, bounded free-part search, and emission-cost heuristics
    select among exact and product-formula candidates. Deterministic bound-sized
    formulas retain an error-budget contract; fixed steps, calibration, sampling,
    and the explicit numerical route do not. See :func:`synthesize`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from numpy.linalg import LinAlgError
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy import exact, trotter
from lizzy.classify import classify, summands
from lizzy.emit import (
    EmissionQuote,
    best_emission,
    emission_candidates,
    greedy_emission_quote,
    native_emission_candidates,
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

if TYPE_CHECKING:
    from lizzy.wei_norman import WeiNormanResult

# What an exact-branch attempt can raise when the classification admits an algebra but
# the upstream irrep machinery cannot realize this particular generator set in it.
_EXACT_FAILURES = (StopIteration, NotImplementedError, ValueError, LinAlgError)
# Gaussian decomposition itself stays polynomial. Only the optional comparison
# with the existing DLA/formula portfolio is bounded by this fixed resource cap.
_GAUSSIAN_COMPARE_MAX_MODES = 8


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
class Result:
    """
    What a synthesis run produced, and how it got there.

    Attributes:
        circuit (Circuit): The rotations implementing the evolution.
        algebra (str): PauLie's DLA name, or an explicit structural/unclassified
            label when the route avoids DLA classification.
        summands (int): Number of parts compiled by the route. Direct Gaussian
            delegation treats the whole input as one part without splitting.
        symmetries (int): Independent Z2 symmetries found, i.e. qubits tapering could
            remove; ``None`` when a delegated route did not calculate them.
        clusters (int): Commuting groups in the part that had the most of them,
            counted whichever route that part ended up taking; ``None`` when not calculated.
        routes (list[str]): Which branches were used.
        randomized (bool): True if any part was sampled, in which case the requested
            error is not guaranteed for the returned circuit or channel.
        error_guaranteed (bool): Whether the selected deterministic route retains the
            requested error-budget contract. Fixed-step, calibrated and qDRIFT
            formulas are false; BDI/Givens exact routes remain true. Gaussian and
            Clifford+T results are false because numerical reconstruction is not
            a certified total error bound. This is not an absolute global-phase
            guarantee: the static orthogonal routes can change that phase.
        routing_estimated (bool): True when at least one oversized route was compared
            using the 1/2/4-repetition cost model instead of full materialization.
        emission (EmissionQuote, optional): The selected quote for the complete
            folded circuit. The builtin quote retains logical rotations and an
            analytical block cost; native/pytket quotes retain concrete gates.
            ``emission_is_concrete`` distinguishes these cases. The logical Pauli
            sequence remains in ``circuit`` for provenance and verification.
        numerical (WeiNormanResult, optional): Diagnostics for the explicitly
            requested numerical Wei–Norman route, otherwise ``None``.
        t_selection (TCountSelection, optional): Actual reference/candidate T/CX
            counts, no-regression reference and gauge diagnostics for ``objective='t'``. Its
            rotation budget is not an end-to-end numerical error certificate.
        routing_notes (tuple[str, ...]): Skipped optional comparisons or dependency
            diagnostics. A delegated Gaussian result does not calculate DLA,
            symmetry or clustering diagnostics; their counts are ``None``.
    """

    circuit: Circuit
    algebra: str = ""
    summands: int = 0
    symmetries: int | None = 0
    clusters: int | None = 0
    routes: list[str] = field(default_factory=list)
    randomized: bool = False
    error_guaranteed: bool = True
    routing_estimated: bool = False
    emission: EmissionQuote | None = None
    numerical: "WeiNormanResult | None" = None
    t_selection: TCountSelection | None = None
    routing_notes: tuple[str, ...] = ()

    @property
    def t_count(self) -> int | None:
        """Actual T + T-dagger count, or None for an arbitrary-rotation output."""
        if self.emission is None:
            return None
        return getattr(self.emission.circuit, "t_count", None)

    @property
    def logical_two_qubit_gates(self) -> int:
        """int: Builtin block/ladder count of the logical Pauli sequence."""
        return self.circuit.two_qubit_gates

    @property
    def two_qubit_gates(self) -> int:
        """int: Selected gate count, or analytical block cost without concrete gates."""
        if self.emission is None:
            return self.logical_two_qubit_gates
        return self.emission.two_qubit_gates

    @property
    def emission_is_concrete(self) -> bool:
        """Whether ``two_qubit_gates`` counts a retained emitted gate artifact."""
        return self.emission is not None and self.emission.is_concrete

    @property
    def emission_backend(self) -> str:
        """str: Name of the backend that won the complete-circuit portfolio."""
        if self.numerical is not None:
            return self.numerical.emission_backend
        return "builtin" if self.emission is None else self.emission.backend

    @property
    def emitted_circuit(self) -> object:
        """Selected artifact, possibly logical; inspect ``emission_is_concrete``."""
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


def synthesize(
    hamiltonian_: PauliStringLinear,
    time: float,
    error: float = 1e-3,
    order: int = 4,
    seed: int | None = None,
    randomized: bool = False,
    calibration: float = 1.0,
    steps: int | None = None,
    *,
    method: str = "auto",
    numerical_options: dict | None = None,
    objective: str = "cx",
) -> Result:
    r"""
    Synthesize :math:`e^{-itH}` for a Pauli Hamiltonian.

    ``method='auto'`` compares BDI and Givens within the static router while
    retaining its error contract. Exact choices are reported in ``routes``.
    A genuine Jordan--Wigner quadratic input also receives an optional whole-input
    OpenFermion Gaussian candidate, including pairing. Above eight fermionic modes,
    an available Gaussian candidate is delegated directly before DLA work;
    ``routing_notes`` records the skipped cost comparison, not a cheapest claim.
    ``method='wei-norman'`` explicitly requests numerical Lie-coordinate synthesis
    and concrete phase-preserving native gate emission. That route does NOT promise
    the ``error`` budget: ``numerical_options`` passes local tolerances and resource
    limits to :func:`lizzy.wei_norman.synthesize_wei_norman`. Its diagnostics are in
    ``result.numerical`` and ``error_guaranteed`` is always false. Product-formula
    controls cannot be combined with the explicit numerical route.

    ``method='givens'`` and ``method='bdi'`` explicitly select the supported
    small-orthogonal-algebra exact synthesis methods. They split commuting
    summands and reject unsupported parts instead of falling back to another
    route or competing with product formulas. Both guarantee at least equivalence
    up to global phase; a horizontal BDI plan additionally preserves the logical
    circuit's phase (see ``prepare_bdi``). Neither accepts product-formula controls
    or ``numerical_options``. Their
    emitted gate counts retain the same logical/concrete distinction as auto.

    ``method='gaussian'`` explicitly delegates full-unitary JW-quadratic evolution
    to OpenFermion, without DLA classification or a fixed-particle-number/state
    restriction. It rejects interacting terms and needs the optional Gaussian
    dependency. No terms are silently discarded to make an input quadratic.
    Its checked floating-point residual is not a certified ``error`` bound, so
    ``error_guaranteed`` is false for a selected Gaussian route under either objective.

    ``objective='t'`` is an opt-in fault-tolerant path. With ``method='auto'``
    it compares complete BDI, Givens and eligible Gaussian circuits at one total
    rotation budget;
    it does not fall back to product formulas or numerical Wei--Norman. BDI
    searches legal nullspace gauges using the same paper
    recursion, then compares the optimized and unchanged complete circuits by
    actual Clifford+T count. Givens provides a reference without a gauge search.
    The optional ``ft`` extra supplies arbitrary-angle synthesis. Here ``error``
    budgets rotation approximation only; ``error_guaranteed`` is false because
    floating-point decomposition errors are not interval-certified. The emitted
    artifact retains phase metadata, which is not a synthesized controlled phase.
    The original stable T-only ladder winner remains a reference. Bounded shared
    Clifford frames are then compared by actual (T, CX), allowing no increase
    in either count relative to that reference. Equal pairs retain stable route
    order. Explicit methods stay within the requested algorithm. Numerical
    Wei--Norman is not T-optimized.

    In auto mode, every manageable candidate the classification allows is priced
    per part and the cheapest quote wins -- exactness is not a priority order:

    * BDI and Givens exact decompositions, where supported, at a depth
      that does not grow with ``time``;
    * a hybrid, greedily taking an exactly-compilable subset out and carrying it as
      one more summand inside each step;
    * a product formula -- either the requested order sized by the chain bound, or
      the second-order formula over commuting clusters sized by the collected
      cluster bound (route ``trotter2``).

    Complete folded sequences are priced whenever their projected size is at most
    20,000 rotations. Larger losing candidates are shortlisted from one, two and
    four repetitions; the selected logical sequence is always fully materialized
    and quoted. A builtin quote uses analytical block/ladder costs, while native
    and pytket quotes count concrete gates (see ``result.emission_is_concrete``).

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
    if method not in {"auto", "wei-norman", "givens", "bdi", "gaussian"}:
        raise ValueError("method must be 'auto', 'wei-norman', 'givens', 'bdi', or 'gaussian'")
    if objective not in {"cx", "t"}:
        raise ValueError("objective must be 'cx' or 't'")
    if objective == "t" and method == "wei-norman":
        raise ValueError("objective='t' requires method='auto', 'bdi', 'givens', or 'gaussian'")
    if numerical_options is not None and not isinstance(numerical_options, dict):
        raise TypeError("numerical_options must be a dict or None")
    if (method != "auto" or objective == "t") and (
        steps is not None or randomized or calibration != 1.0 or order != 4 or seed is not None
    ):
        raise ValueError(
            f"product-formula controls cannot be used with method='{method}', "
            f"objective='{objective}'"
        )
    if method == "wei-norman":
        from lizzy.wei_norman import synthesize_wei_norman

        compiled = synthesize_wei_norman(hamiltonian_, time, **(numerical_options or {}))
        return Result(
            circuit=compiled.circuit,
            algebra=f"Pauli closure (dimension {compiled.dimension})",
            summands=len(compiled.component_bases),
            symmetries=len(z2_symmetries(hamiltonian_)),
            routes=sorted(set(compiled.circuit.provenance)),
            error_guaranteed=False,
            emission=compiled.emission,
            numerical=compiled,
        )
    if numerical_options is not None:
        raise ValueError("numerical_options requires method='wei-norman'")
    width = n_qubits(hamiltonian_)
    gaussian_circuit, gaussian_failure = None, None
    if method in {"auto", "gaussian"}:
        from lizzy import gaussian

        if method == "gaussian" or gaussian.is_gaussian(hamiltonian_):
            try:
                # OpenFermion has numerical near-zero decisions. Reject any
                # resulting algebra residual that is material at this requested
                # precision; this screening is not an interval certificate.
                gaussian_circuit = gaussian.decompose(
                    hamiltonian_, time, error_tolerance=min(1e-8, 0.1 * error),
                )
            except (*_EXACT_FAILURES, ImportError) as exc:
                if method == "gaussian":
                    raise
                if width > _GAUSSIAN_COMPARE_MAX_MODES and not isinstance(exc, ImportError):
                    raise ValueError(
                        f"Gaussian synthesis failed for {width} modes: {exc}. "
                        "Automatic DLA fallback was not attempted above the fixed "
                        f"{_GAUSSIAN_COMPARE_MAX_MODES}-mode comparison cap; "
                        "choose an explicit alternative method to attempt it."
                    ) from exc
                gaussian_failure = exc

    if gaussian_circuit is not None and (
        method == "gaussian" or width > _GAUSSIAN_COMPARE_MAX_MODES
    ):
        result = Result(
            circuit=gaussian_circuit, algebra="JW quadratic (Gaussian)",
            summands=1, symmetries=None, clusters=None, routes=["exact-gaussian"],
            error_guaranteed=False,
        )
        if method == "auto":
            result.routing_notes = (
                f"Gaussian delegation: {width} modes exceeds the fixed "
                f"{_GAUSSIAN_COMPARE_MAX_MODES}-mode DLA/formula comparison cap; "
                "alternative costs were not compared.",
            )
        if objective == "t":
            return _t_exact_result(
                result, [], time, width, error, method,
                gaussian_circuit=gaussian_circuit, compare_orthogonal=False,
            )
        result.circuit = fold_phases(gaussian_circuit, tolerance=0.0)
        result.emission = _gaussian_emission(result.circuit, width)
        return result

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
        routing_notes=(
            (f"Gaussian candidate unavailable: {type(gaussian_failure).__name__}: {gaussian_failure}",)
            if gaussian_failure is not None else ()
        ),
    )

    if objective == "t":
        return _t_exact_result(
            result, parts, time, width, error, method,
            gaussian_circuit=gaussian_circuit, gaussian_failure=gaussian_failure,
        )

    if method in {"givens", "bdi"}:
        for index, part in enumerate(parts):
            # BDI validates its graph-derived embedding itself, including
            # nonhorizontal inputs. Givens retains its narrower legacy filter.
            if method == "givens" and not exact.is_decomposable(part):
                raise NotImplementedError(
                    f"method='{method}' requires supported so(m) summands; "
                    f"summand {index + 1} is not eligible for exact synthesis"
                )
        route = f"exact-{method}"
        for part in parts:
            result.circuit.extend(exact.decompose(part, time, route=route, method=method))
            result.clusters = max(result.clusters, len(commuting_clusters(part)))
        result.circuit = fold_phases(result.circuit)
        result.emission = best_emission(result.circuit, n_qubits(hamiltonian_))
        result.routes = [route]
        return result

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
    selected_gaussian = False
    if gaussian_circuit is not None:
        # Compare complete artifacts: separate Gaussian eigensystems per
        # commuting component would miss whole-input simplifications.
        try:
            logical = fold_phases(gaussian_circuit, tolerance=0.0)
            emitted = _gaussian_emission(logical, width)
        except _EXACT_FAILURES as exc:
            result.routing_notes += (
                f"Gaussian emission rejected: {type(exc).__name__}: {exc}",
            )
        else:
            if emitted.two_qubit_gates < result.two_qubit_gates:
                result.circuit, result.emission = logical, emitted
                result.randomized = False
                selected_gaussian = True
    result.routes = sorted(set(result.circuit.provenance))
    exact_only = set(result.routes).issubset({"exact-bdi", "exact-givens", "exact-gaussian"})
    result.error_guaranteed = not selected_gaussian and not result.randomized and (
        exact_only or (steps is None and calibration <= 1.0)
    )
    return result


def _gaussian_emission(circuit, width) -> EmissionQuote:
    """Retain a concrete phase-preserving Gaussian artifact without an SDK.

    The optional pytket Pauli-box adapter drops scalar phases. Gaussian evolution
    retains them, so use the same native ladder/frame choices as the numerical
    phase-preserving route, rather than a logical-only quote or a phase-losing SDK.
    """
    # Shared frames remain optional; a valid ladder survives known frame errors.
    candidates = native_emission_candidates(circuit, width, frame_failures=_EXACT_FAILURES)
    backend, emitted = min(candidates, key=lambda item: (item[1].two_qubit_gates, len(item[1].gates)))
    return EmissionQuote(backend, emitted.two_qubit_gates, emitted)


def _t_exact_result(
    result, parts, time, width, error, method, *, gaussian_circuit=None,
    gaussian_failure=None, compare_orthogonal=True,
) -> Result:
    """Compare complete exact candidates at one total rotation-error budget.

    Components are never priced independently: allocation and cancellations can
    change their costs when joined. Auto compares uniform BDI/Givens circuits,
    the BDI gauge variant and an eligible whole-input Gaussian circuit, not every
    mixed component assignment.
    """
    import math

    from lizzy.clifford_t import compile_clifford_t, compile_frame_candidates

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
            candidates.append((label, logical, emitted, route))

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
    baseline = min(candidates, key=lambda candidate: candidate[2].t_count)
    frame_cache = {}
    for label, logical, _, route in tuple(candidates):
        key = tuple((str(word), float(angle)) for word, angle in logical.rotations)
        if key not in frame_cache:
            frame_cache[key] = compile_frame_candidates(logical, width, error=error)
        alternatives, frame_rejections = frame_cache[key]
        candidates.extend((f"{label}/{strategy}", logical, output, route)
                          for strategy, output in alternatives)
        rejected.extend((f"{label}/{strategy}", reason) for strategy, reason in frame_rejections)
    eligible = [candidate for candidate in candidates
                if candidate[2].t_count <= baseline[2].t_count
                and candidate[2].n_2qb_gates() <= baseline[2].n_2qb_gates()]
    name, logical, emitted, route = min(
        eligible, key=lambda candidate: (candidate[2].t_count, candidate[2].n_2qb_gates()),
    )
    result.circuit = logical
    result.emission = EmissionQuote("clifford-t", emitted.n_2qb_gates(), emitted)
    result.routes = [route]
    if parts:
        result.clusters = max(len(commuting_clusters(part)) for part in parts)
    result.error_guaranteed = False
    result.t_selection = TCountSelection(
        candidate_t_counts=tuple((label, output.t_count) for label, _, output, _ in candidates),
        selected=name,
        rotation_error=float(error),
        bdi_optimizations=tuple(diagnostics),
        rejected_candidates=tuple(rejected),
        candidate_cx_counts=tuple((label, output.n_2qb_gates()) for label, _, output, _ in candidates),
        no_regression_reference=baseline[0],
    )
    return result


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
