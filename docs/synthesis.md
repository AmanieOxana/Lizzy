# Synthesis guide

[Overview](../README.md) · [Get started](getting_started.md) · [Architecture](architecture.md) · [Chemistry](chemistry.md) · [Benchmarks](benchmarks.md)

Lizzy compiles Pauli-Hamiltonian evolution. Its default static router, numerical
driven compiler, and molecular adapter solve different problems and have different
accuracy contracts. It is not an arbitrary-unitary matrix compiler.

## Choose an interface

| Task | Interface | Accuracy contract |
|---|---|---|
| Static Pauli Hamiltonian | `synthesize(H, time=t, error=eps)` | Algebraic exact routes or bound-sized deterministic product formulas; inspect `error_guaranteed` |
| Fixed-step static comparison | `synthesize(H, time=t, steps=k)` | No error guarantee for product-formula routes |
| Explicit exact factorization | `synthesize(H, time=t, method="bdi" / "givens")` | Supported orthogonal summands only; horizontal BDI preserves phase, endpoint methods allow global phase; no cross-method fallback |
| Quadratic fermionic evolution | `synthesize(H, time=t, method="gaussian")` | Full Jordan–Wigner unitary from OpenFermion; includes pairing and scalar phase; numerical, not an interval certificate |
| Automatic T-aware exact selection | `synthesize(H, time=t, objective="t", error=eps)` | Whole-H BDI/Givens/Gaussian actual T comparison; numerically checked rotation budget, no formal total error certificate |
| Static or driven small Pauli closure | `synthesize_wei_norman(H_or_drive, time_span)` | Numerical; local tolerances do not bound final operator error |
| Fourth-order driven time stepping | `synthesize_expansion(drive, span, method=..., steps=k)` | Numerical expansion and factor synthesis; no global error certificate |
| Molecular orbital tensors | `synthesize_molecular_ffsim(...)` | Explicit upstream DF-Trotter baseline; see [chemistry](chemistry.md) |

## Static synthesis

```python
from lizzy.hamiltonian import hamiltonian, model
from lizzy.synthesize import synthesize

H = model("tfim", 8)
result = synthesize(H, time=1.0, error=1e-3)

print(result.routes, result.algebra)
print(result.two_qubit_gates, result.emission_backend)
print(result.emission_is_concrete)
print(result.error_guaranteed, result.routing_estimated)

# Custom real coefficients, with a common Pauli-word width:
custom = hamiltonian({"XX": 0.7, "ZI": 0.2, "IZ": -0.1})
fixed = synthesize(custom, time=1.0, steps=2)
```

After the quadratic-input recognition/delegation check described below, the
general router splits anticommutation-connected components. Terms in different
components commute, so this split itself adds no approximation error. Each
manageable component is classified by PauLie; classification is skipped for
oversized candidates rather than expanding every potentially exponential DLA.

The router prices eligible alternatives, rather than always preferring an exact
route:

- **Exact:** supported BDI and Givens circuits compete under the same emission
  cost policy. Provenance reports `exact-bdi` or `exact-givens`. BDI uses the
  paper recursion; Givens uses adjacent-plane elimination.
- **Hybrid:** a supported exactly compilable subset is treated as one summand
  inside a second-order formula for the full component.
- **Product formulas:** commuting clusters, two-qubit kernels, ungrouped terms,
  or the requested Suzuki order provide alternative sequences. Pair kernels use
  a verified 4-by-4 decomposition, not a new many-qubit algorithm.
- **Randomized:** qDRIFT is opt-in through `randomized=True`. It does not certify
  the returned realization and is not part of the default accuracy contract.

Complete folded candidates are priced up to a projected 20,000 rotations. Larger
ones are shortlisted using one, two and four repetitions; the winner is then
fully built and quoted. `routing_estimated=True` records use of that heuristic.
It concerns route cost, not the independently computed step-size error bound.
Selection is per commuting component; final emission may share a frame across
components, but does not reconsider every discarded combination of routes.
An eligible OpenFermion Gaussian circuit challenges the assembled whole-H
result, rather than diagonalizing each commuting component independently.

The default `order=4` applies to the chain-bound Suzuki candidate, not to every
candidate in the portfolio. Bound-sized second-order candidates can still win.
Supplying `steps` selects fixed-step formulas and bypasses accuracy sizing.
`calibration != 1` also relinquishes the product-formula error guarantee. Always
inspect the result rather than infer its contract from the input arguments alone.

### OpenFermion quadratic evolution

OpenFermion is an integrated backend, not a separate competitor. With the
existing `chemistry` dependency installed, auto routing recognizes Pauli inputs
that are quadratic under the **given Jordan–Wigner ordering**. Alternatively,
install only `lizzy[gaussian]`. Imports remain optional: without OpenFermion,
auto keeps its other routes, while explicit `method="gaussian"` reports the
missing dependency.

The adapter calls upstream quadratic diagonalization and
[`bogoliubov_transform`](https://quantumai.google/reference/python/openfermion/circuits/bogoliubov_transform)
with `initial_state=None`. It includes both basis transformations, mode phases
and the scalar phase, so the circuit acts on **arbitrary input states**, not
just the vacuum or a fixed particle-number sector. Complex hopping and pairing
are supported. No dense `2**n` matrix is used to construct it.

For an existing upstream `QuadraticHamiltonian`, use
`lizzy.gaussian.from_quadratic(quadratic, time=t)` directly. It returns the
same logical `Circuit` used by Lizzy's native and Clifford+T emitters, preserving
the chemical potential and constant. There is no need to convert orbital tensors
to a dense matrix or reconstruct a Pauli Hamiltonian first.

Eligibility is structural, not a named-model whitelist. A nonzero interacting
term is not dropped to make an input fit. Quadratic recognition in the supplied
JW basis does not search arbitrary Clifford frames, fermion orderings or BK
encodings; a rejected input may still be supported by BDI/Givens or formulas.
Interacting molecular tensors continue through the [ffsim interface](chemistry.md).

For up to eight modes, auto compares the whole Gaussian circuit with the
existing portfolio under the requested objective. Above eight modes, a
recognized quadratic input uses the available Gaussian route before expensive
DLA analysis. This fixed search cap preserves scalable synthesis; it is **not
evidence that Gaussian always uses fewer gates**. Explicit BDI/Givens remain
available when that additional search is wanted. The selected Gaussian route
is reported as `exact-gaussian`; `result.routing_notes` records a capped search.
Uncomputed symmetry/cluster counts are `None`, not a claim that no symmetries
or clusters exist.

The logical circuit retains scalar phase. Native/Clifford+T emission preserves
it as metadata, not a synthesized controlled-phase gate. Floating-point
diagonalization and circuit decomposition do not certify a total unitary-error
bound; Gaussian results report `error_guaranteed=False`. A polynomial-size
reconstruction check guards the actual emitted basis, including time-amplified
errors from upstream near-zero cutoffs. Its numerical threshold is at most
`min(1e-8, 0.1 * error)` in the high-level router, without changing T mode's
rotation-only budget contract. If the original upstream basis fails, one
deterministic Fourier basis change is tried, with its compensating circuit;
both circuits still come from OpenFermion and the same check must pass. This
stability fallback neither relaxes the threshold nor modifies library globals.
A remaining failure rejects the candidate. Small-input auto can retain another
valid route; above the comparison cap it reports the numerical failure rather
than silently starting an unbounded DLA search. An explicitly requested
alternative is still available.

### What "exact" and "Cartan" mean here

The exact backends use PauLie and kak-tools' representation machinery.
`synthesize(H, time=t, method="bdi")` selects structure-derived recursive BDI;
`method="givens"` selects adjacent elimination. Both compile every commuting
summand with the chosen method, and reject unsupported inputs rather than
silently switching methods. Product-formula controls and `numerical_options`
do not apply to these explicit routes. The default `method="auto"` compares both
exact methods. The free-part kernel inside a hybrid formula still uses Givens;
this does not substitute Givens inside the explicitly requested paper BDI route.

Givens no longer requires the supplied terms to fit a horizontal BDI subspace:
if its original horizontal mapping fails for that reason, it uses the verified
full-algebra embedding already available to BDI. Its elimination and low-weight
ordering are unchanged, as are the router's eligibility filters and budgets.

BDI derives its mapping from the input generators, without model-name dispatch,
predefined chain templates or Givens' low-weight index reordering. If the supplied
terms admit a horizontal mapping, a generator-level SVD gives a time-independent
`K` and unwrapped central rates. The `K` blocks are recursively BDI-factorized,
and the inverse wing is exactly reversed and negated. Only the center changes
with evolution time. Otherwise a supported general `so(m)` embedding is used
to factor the time-dependent endpoint recursively. These are the paper's
Sections VI.2 and VI.1 respectively, with Appendix F.4 recursion.
Neither is arbitrary `SU(2**n)` synthesis: supported embeddings and finite
search budgets still matter. See the [paper-to-code review](cartan_review.md).

For repeated evolution of one supported component, preparation is explicit:

```python
from lizzy.exact import prepare_bdi
from lizzy.hamiltonian import model

plan = prepare_bdi(model("tfim", 4, seed=7))
print(plan.mapping_kind, plan.partition, plan.phase_preserving)
short = plan.circuit(0.1)
long = plan.circuit(5.0)
```

`mapping_kind` is `"horizontal-graph"` or `"general-graph"`. A horizontal plan
reuses both outer wings; a general plan reuses the mapping but recomputes the
endpoint decomposition. `cache=False` on `prepare_bdi` requests fresh preparation.
It bypasses Lizzy's plan cache, not unrelated dependency/classification caches.
The plan exposes `irrep_size` and `parameter_bound`; the latter bounds rotation
entries, not CX gates or independent free parameters. This low-level API prepares
one component; use `synthesize` to split commuting summands automatically.

An opt-in `prepare_bdi(H, optimize="t", rotation_error=delta)` searches legal
nullspace bases before compiling the same BDI recursion. `delta` is a per-rotation
precision for its heuristic cost, not a final-error certificate. Diagnostics in
`plan.optimization` distinguish the estimate from actual T counts. The high-level
`objective="t"` route additionally compiles complete candidates at one common
total rotation budget and retains an actual emitted artifact. With default
`method="auto"`, Givens and an eligible OpenFermion Gaussian circuit join the
original and optimized BDI candidates. Its original stable T-only ladder winner
is retained as a reference. Shared frames may replace it only without increasing
either T or CX; eligible candidates are selected by `(T, CX)` with stable ties.
This is whole-H selection, not per-component T pricing.
It does not use the approximate CX portfolio or automatically invoke Wei–Norman;
unsupported inputs are reported explicitly. `result.t_selection` records actual
counts, the no-regression reference and rejections. Frame search is bounded to
eight qubits and 256 rotations. See the [fault-tolerant guide](fault_tolerant.md).

Givens and balanced general BDI use at most `d=m(m-1)/2` rotation entries.
A horizontal plan with partition `(p,q)` uses at most
`2*(p*(p-1)/2 + q*(q-1)/2) + min(p,q)`: this equals `d` for
`abs(p-q)<=1`, but can exceed it for an unbalanced initial partition. Do not
apply an unconditional `4**n-1` entry bound to every horizontal plan. Repeated
product formulas, hybrid steps and multiple numerical charts can also exceed
that bound. Rotation count is not a hardware gate count.

All these are algebraic factorizations evaluated in floating-point arithmetic.
**The horizontal BDI logical circuit preserves global phase** through its
generator rates and exact inverse wings. Givens and general endpoint BDI have a
spin-cover ambiguity and
promise only equivalence up to global phase. Check `plan.phase_preserving` when
phase is required, for example before constructing a controlled evolution;
`error_guaranteed` alone does not distinguish these contracts. Benchmarks report
strict and phase-aligned errors separately.

The plan's phase flag describes its logical sequence, not every optional SDK's
optimization contract. Explicit native-ladder emission preserves that sequence's
phase and is verified strictly in the benchmark. Do not infer phase-sensitive
equivalence of an arbitrary selected SDK artifact from `plan.phase_preserving`.

## Logical sequences versus emitted gates

`result.circuit` is a `lizzy.hamiltonian.Circuit`: rotations in application order,
each meaning `exp(-i * angle * Pauli)`, with per-rotation provenance. Its
`two_qubit_gates` property is a **block-aware analytical charge**: consecutive
rotations supported on one pair are priced by the canonical 0/1/2/3-CNOT class,
and wider rotations use CNOT-ladder charges.

The static portfolio retains an `EmissionQuote` with a backend-dependent artifact:

| Backend | `result.emitted_circuit` | Meaning of the quote |
|---|---|---|
| `builtin` | Logical `Circuit` | Analytical pair-block/ladder charge; not an expanded native gate list |
| `native-ladder` / `native-frame` | `NativeCircuit` | Actual emitted CX count |
| `pytket-direct` / `pytket-greedy` | pytket circuit in CX/TK1 | Actual emitted two-qubit count |
| `clifford-t` | Immutable `CliffordTCircuit` | Actual T/T† and CX counts; opt-in T objective |

`result.logical_two_qubit_gates` always exposes the logical charge;
`result.two_qubit_gates` exposes the selected quote. Check the backend when
comparing counts. `result.emission_is_concrete` (or `quote.is_concrete`) makes this
distinction explicit; it is false for the builtin logical artifact. To obtain a
concrete dependency-free fallback explicitly:

```python
from lizzy.native import ladder_circuit

native = ladder_circuit(result.circuit, width=8)
print(native.two_qubit_gates)
qasm = native.to_qasm3()
```

This ladder expansion need not attain the analytical pair-block charge. Native
circuits use `H/S/Sdg/CX/Rz` and record global phase. Expanding a logical circuit
preserves its phase; it cannot restore a phase already lost by Givens or general
endpoint BDI.

Native shared frames are considered for abelian spans and certified encoded
single-qubit algebras. Signed GF(2) dependencies, not merely the DLA name, establish
eligibility. Optional pytket passes handle broader high-weight/shared-parity
sequences. Their absence or failure leaves fallback candidates available. These
counts assume unrestricted connectivity; hardware routing is not included.

## Numerical Wei–Norman synthesis

```python
import numpy as np
from lizzy.driven import DrivenHamiltonian
from lizzy.wei_norman import synthesize_wei_norman

drive = DrivenHamiltonian(
    ["XXX", "XXY", "IIZ"],
    lambda t: [np.cos(1.7 * t), np.sin(1.7 * t), 0.3],
)
driven = synthesize_wei_norman(drive, (0.0, 1.2), max_step=0.02)

print(driven.component_dimensions, driven.charts)
print(driven.two_qubit_gates, driven.rhs_evaluations)
qasm = driven.emitted_circuit.to_qasm3()
```

For a declared Pauli closure of dimension `d`, a local Wei–Norman chart uses
`U = product_j exp(-i theta_j P_j)` and integrates the coordinate equations.
One chart has at most `d` rotation angles; concatenating charts can repeat those
factors and exceed `4**n`. Compact charts are not a globally nonsingular
parameterization or a minimal-CX construction. See the
[paper review](wei_norman_review.md) for conventions, derivation and tests.

The high-level API handles commuting control components separately, since they
commute at every pair of times. It also handles static commuting components
directly, pulse boundaries and phase-preserving native emission. Its controls are:

| Option | Default | Purpose |
|---|---:|---|
| `max_dimension` | `32` | Closure cap per component |
| `max_total_dimension` | `1024` | Cap on the union of component closures |
| `rtol`, `atol` | `1e-9`, `1e-11` | Local ODE tolerances, not operator-error bounds |
| `max_step` | infinity | Set explicitly to resolve the fastest control timescale |
| `chart_radius` | `None` | Compact, condition-limited charts; `0.5` restores the old angle cap |
| `condition_limit` | `100` | Limit on sampled coordinate-Jacobian conditioning |
| `max_segments` | `1024` | Shared chart/direct-segment budget across components and pulses |
| `max_rhs_evaluations` | `100000` | Shared work limit, including rejected attempts |
| `emission` | `"native"` | Concrete ladder/shared-frame portfolio; also `"ladder"` or `"none"` |

Declare **every potentially active control**, including terms initially at zero.
Callbacks must be deterministic and use absolute time. Pass known pulse jumps as
`breakpoints=[...]`, strictly ordered inside the interval; reverse-time calls need
descending breakpoints. One-sided endpoint sampling prevents evaluating across
those declared jumps. Unknown narrow pulses can still be missed.

The solver samples conditioning at RHS evaluations and accepted mesh states,
bisects rejected intervals, and raises on failure or exhausted work limits. It
does not certify continuous nonsingularity or total integration error. The
Pauli closure can still be exponential even though synthesis avoids `2**n` dense
matrices. `split_components=False` and `basis_order` enable explicit comparisons;
changing a basis order can change conditioning and circuit cost.

The static entry point can request the same numerical compiler explicitly:

```python
numerical = synthesize(
    model("tfim", 3), time=1.0, method="wei-norman",
    numerical_options={"rtol": 1e-10, "atol": 1e-12},
)
assert numerical.error_guaranteed is False
```

`method="auto"` never silently selects Wei–Norman. Product-formula controls such
as `steps` cannot be combined with `method="wei-norman"`; `error` is not certified
by this route. For just a single closure's logical sequence and solver diagnostics,
use the lower-level `lizzy.driven.synthesize_driven`.

## Magnus4 and Fer4

```python
from lizzy.expansions import expand_driven, synthesize_expansion

plan = expand_driven(drive, (0.0, 1.2), method="magnus4", steps=32)
expanded = synthesize_expansion(drive, (0.0, 1.2), method="fer4", steps=32)
print(plan.exponentials, expanded.compilation_segments)
native_expansion = ladder_circuit(expanded.circuit, width=3)
```

These are the published fourth-order Gauss–Legendre Magnus and two-factor Fer
schemes, implemented in coefficient space. Each step samples two control vectors.
Magnus4 produces one Pauli-**sum** exponential per step; Fer4 produces up to two,
including its fourth-order commutator correction. These are not individual gates.
Noncommuting factors are themselves synthesized with Wei–Norman; commuting
factors emit directly. `ExpansionResult` contains a logical circuit, not an
automatically selected hardware artifact.

Time-step quadrature/truncation and internal numerical gate synthesis are separate
error sources. `rtol`/`atol` affect only the latter. Step count, closure size,
factor count and aggregate work are capped explicitly. Resolve controls on the
time-step mesh and split discontinuous pulses into separate calls. Fer4 is not
exactly time-reversal symmetric at finite step count. Static Hamiltonians have no
time-ordering correction, but their effective exponential still needs synthesis.

These routes are opt-in numerical methods, not new decomposition theorems or a
demonstration that Wei–Norman, Magnus or Fer always improves circuit cost. The
[benchmarks](benchmarks.md) document both useful cases and counterexamples.
