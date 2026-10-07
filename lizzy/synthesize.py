"""Public compiler configuration, synthesis entry point, and result reporting.

Algorithm/cost candidates live in _routing; the compiler owns validation and
assembly. Optional numerical, chemistry and emission SDKs remain lazily loaded.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.algebra.classify import classify, summands
from lizzy.algebra.symmetry import commuting_clusters, z2_symmetries
from lizzy.emission.emit import EmissionQuote, best_emission, native_emission_candidates
from lizzy.hamiltonian import Circuit, fold_phases, n_qubits, terms_of
from lizzy.synthesis import exact
from lizzy.synthesis._routing import (
    _EXACT_FAILURES,
    select_exact_t,
    select_part,
)
from lizzy.synthesis._routing import (
    TCountSelection as TCountSelection,
)

if TYPE_CHECKING:
    from lizzy.synthesis.wei_norman import WeiNormanResult

# The Gaussian decomposition is polynomial; only the optional DLA comparison is capped.
_GAUSSIAN_COMPARE_MAX_MODES = 8


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


@dataclass(frozen=True)
class Compiler:
    """Reusable synthesis policy, with fresh candidate/result state on each call.

    Use Compiler(objective="t", error=1e-6).compile(H, time) to apply one policy
    to multiple Hamiltonians. Arguments and accuracy contracts are identical to
    synthesize(). Reusing a compiler does not cache a Hamiltonian's decomposition;
    use ``lizzy.synthesis.exact.prepare_bdi`` or
    ``lizzy.synthesis.driven.WeiNormanBasis`` for algebraic preparation.
    """

    error: float = 1e-3
    order: int = 4
    seed: int | None = None
    randomized: bool = False
    calibration: float = 1.0
    steps: int | None = None
    method: str = "auto"
    numerical_options: dict | None = field(default=None, hash=False)
    objective: str = "cx"

    def __post_init__(self):
        method, objective = self.method, self.objective
        numerical_options = self.numerical_options
        steps, randomized, calibration = self.steps, self.randomized, self.calibration
        order, seed = self.order, self.seed
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
        if numerical_options is not None and method != "wei-norman":
            raise ValueError("numerical_options requires method='wei-norman'")
        if numerical_options is not None:
            # A caller changing its input dict must not change a prepared policy.
            options = dict(numerical_options)
            for name in ("basis_order", "breakpoints"):
                if options.get(name) is not None:
                    options[name] = tuple(options[name])
            object.__setattr__(self, "numerical_options", MappingProxyType(options))

    def compile(self, hamiltonian_: PauliStringLinear, time: float) -> Result:
        """Compile one evolution; see synthesize() for route/error contracts."""
        method, objective, error = self.method, self.objective, self.error
        order, seed = self.order, self.seed
        randomized, calibration, steps = self.randomized, self.calibration, self.steps
        numerical_options = self.numerical_options
        if method == "wei-norman":
            from lizzy.synthesis.wei_norman import synthesize_wei_norman

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
        width = n_qubits(hamiltonian_)
        gaussian_circuit, gaussian_failure = None, None
        if method in {"auto", "gaussian"}:
            from lizzy.fermions import gaussian

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
                return self._compile_t(
                    result, [], time, width,
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
            return self._compile_t(
                result, parts, time, width,
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
            selection = select_part(
                part,
                time,
                error / len(parts),
                order,
                seed,
                randomized,
                calibration,
                steps,
            )
            result.circuit.extend(selection.circuit)
            result.emission = selection.emission
            result.clusters = max(result.clusters, selection.clusters)
            result.randomized |= selection.randomized
            result.routing_estimated |= selection.used_estimates

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

    def _compile_t(self, result, parts, time, width, **options) -> Result:
        candidate, selection = select_exact_t(
            parts, time, width, self.error, self.method, **options,
        )
        result.circuit = candidate.logical
        result.emission = candidate.emission
        result.routes = [candidate.route]
        if parts:
            result.clusters = max(len(commuting_clusters(part)) for part in parts)
        result.error_guaranteed = False
        result.t_selection = selection
        return result


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
    limits to :func:`lizzy.synthesis.wei_norman.synthesize_wei_norman`. Its diagnostics are in
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
            measured factor trades it for a smaller circuit.
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
    return Compiler(
        error=error, order=order, seed=seed, randomized=randomized,
        calibration=calibration, steps=steps, method=method,
        numerical_options=numerical_options, objective=objective,
    ).compile(hamiltonian_, time)


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
