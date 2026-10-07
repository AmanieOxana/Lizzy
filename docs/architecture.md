# Architecture and development

[Overview](../README.md) · [Synthesis](synthesis.md) · [Chemistry](chemistry.md) · [Benchmarks](benchmarks.md)

The package has separate entry paths with deliberately different contracts:

```text
PauliStringLinear ── synthesize(method="auto") ── static route portfolio ─┐
Pauli controls ──── synthesize_wei_norman ─────── coordinate solver ─────┤
Pauli controls ──── synthesize_expansion ──────── factors + solver ──────┤
                                                                      ├─ Circuit IR
                                                                      └─ emission
Molecular tensors ─ synthesize_molecular_ffsim ── ffsim + Qiskit artifact
```

The molecular path deliberately keeps fermionic tensors and does not pass through
the Pauli IR unless the caller explicitly requests an encoding. ODE-based
Wei–Norman and expansion methods are not automatically mixed into the static
portfolio. Gaussian factorization is numerical and explicitly reports its
uncertified error contract.

## Module map

| Layer | Modules | Responsibility |
|---|---|---|
| Input and logical IR | `hamiltonian.py` | PauLie-based constructors, model families, rotation sequence, folding, analytical gate charge |
| Shared structural operations | `gf2.py`, `classify.py`, `symmetry.py` | Binary linear algebra, cached PauLie classification, commuting components/clusters, conserved charges and explicit tapering |
| Static routing | `synthesize.py` | Candidate construction, bounded cost shortlisting, per-component selection and final quote |
| Static synthesis | `exact.py`, `kernels.py`, `trotter.py` | Supported `so(m)` Givens / structure-derived recursive BDI, reusable horizontal plans, pair KAK kernels, product formulas and error sizing |
| Gaussian adapter | `gaussian.py` | Recognize JW-quadratic inputs, delegate full Bogoliubov synthesis to OpenFermion, preserve phases in the common IR |
| Orthogonal representation adapter | `_orthogonal_mapping.py` | Generic horizontal-map selection, bounded closure-derived general embedding and upstream Lie-map validation; no model templates |
| Representation changes | `frame.py` | Explicit Clifford mapping to a supplied reference with matching commutation and dependencies |
| Emission | `emit.py`, `native.py` | Backend quotes; dependency-free native gates and signed Clifford frames; optional pytket passes |
| Fault-tolerant emission | `clifford_t.py` | Optional pygridsynth rotation compilation, immutable Clifford+T artifacts, numerical local error checks and an explicitly heuristic search estimate |
| Numerical synthesis | `driven.py`, `wei_norman.py`, `expansions.py` | Low-level Lie-coordinate solver; component/pulse orchestration; Magnus4/Fer4 factors and compilation |
| Numerical validation helpers | `_numerical.py` | Shared solver-option checks; private, not a public compilation API |
| Molecular adapters | `chemistry.py`, `_chemistry_extensions.py` | Public OpenFermion/ffsim boundaries; private layout bridge and opt-in pruning/reordering policies |
| Data loading | `hamlib.py` | Cached HamLib downloads, Pauli loading and molecular tensor loading |
| Verification and measurements | `dense.py`, `bench.py`, `compare.py`, `driven_bench.py`, `synthesis_bench.py` | Small dense references and explicit benchmark entry points, not production routing oracles |

`lizzy.__init__` is intentionally minimal. Import public entry points from their
own modules; underscore-prefixed modules and helpers are implementation details.
Public entry points stay in these modules. The unused
`emit.direct_tket_two_qubit_gates` convenience wrapper was removed: retain the
artifact from `direct_tket_circuit` and call its `n_2qb_gates()` method instead.

## Core boundaries

### Algebra versus representation

PauLie owns Pauli operations and DLA classification. kak-tools supplies
representation machinery, horizontal generator factorization and balanced BDI
recursion. Lizzy owns eligibility, the closure-derived general embedding adapter,
circuit conventions, plan reuse and emission. BDI's mapping is derived from the
supplied generators and retains
the horizontal partition when one exists; it does not reuse Givens' cheap-plane
reordering or dispatch on a named spin-chain model. Horizontal plans cache the
time-independent outer transformation and unwrapped central rates. The automatic
router compares eligible BDI and Givens candidates, plus OpenFermion's Gaussian
construction for quadratic inputs. Explicit methods remain available. The
[Cartan/orthogonal review](cartan_review.md) identifies the implemented paper
algorithm and distinguishes it from general Cartan and flag-based synthesis.

An algebra label alone does not determine a circuit. `frame.py` needs a supplied
reference and preserves both anticommutation relations and GF(2) dependencies.
`native.py` tracks signed tableau phases when sharing a frame. `symmetry.taper`
is explicit because choosing a symmetry sector changes the Hilbert space; it is
not an automatic unitary-preserving compilation pass.

### Logical IR versus artifacts

`Circuit.rotations` stores `(PauliString, angle)` in application order, with the
convention `exp(-i angle P)`. Provenance has one route label per rotation.
`fold_phases` reduces the sequence; phase-sensitive numerical paths use zero
dropping tolerance.

`EmissionQuote` retains both a cost and the artifact to which it refers. The
`builtin` artifact remains the logical IR with an analytical block/ladder charge;
`native-ladder`, `native-frame` and pytket quotes retain concrete gates. `NativeCircuit` supports
global phase and OpenQASM 3 export. See the [counting contract](synthesis.md#logical-sequences-versus-emitted-gates)
before comparing backend numbers. `EmissionQuote.is_concrete` and
`Result.emission_is_concrete` expose that distinction without changing route costs.

### Automatic routing versus numerical synthesis

`synthesize(method="auto")` considers exact and approximate static routes with
explicit error-budget semantics. Candidate search is bounded; unsupported exact
embeddings can fall back to formulas under the CX objective. BDI and Givens
receive equal emission effort per eligible component; provenance identifies the
selected exact method. A shortlisting estimate is recorded as
`routing_estimated`, separately from `error_guaranteed`.

The `objective="t"` path automatically compares BDI, Givens and an eligible
whole-H Gaussian circuit, or respects an explicit method restriction. BDI searches
legal structural-nullspace gauges without changing its recursive order. Complete
circuits use one common rotation budget, not independently priced components.
The lowest-T ladder circuit fixes the reference; bounded shared-frame alternatives
can replace it only without increasing either T or CX. Eligible candidates are
ordered by actual `(T, CX)`, with stable ties and the reference retained on failure.
No product-formula or Wei–Norman route is silently added. These
diagnostics are in `t_selection`, not the CX router's `routing_estimated` flag.
The rotation-error budget does not certify floating-point factorization error,
so `error_guaranteed` is false. See [fault-tolerant synthesis](fault_tolerant.md).

`driven.py` operates in a bounded explicit Pauli closure and integrates one local
coordinate chart at a time. `wei_norman.py` owns component splitting, pulse
boundaries, aggregate work limits and native emission. `expansions.py` builds
Magnus/Fer coefficient vectors and sends noncommuting effective factors to the
same numerical solver. None uses a dense qubit matrix during synthesis; explicit
closure and coordinate matrices can nevertheless become prohibitively large.

The mathematics and local/global limitations are documented in the
[Wei–Norman review](wei_norman_review.md). New numerical alternatives should not
silently acquire a certified static error contract.

### Upstream algorithms versus Lizzy policy

Chemistry delegates algebra and encodings to OpenFermion, and factorization and
DF simulation gates to ffsim. Its private extension module contains the layout
conversion and explicit optional policies, not replacement implementations of
those upstream algorithms. The Gaussian adapter likewise delegates diagonalization
and full-unitary Bogoliubov circuits to OpenFermion, without a dense Hilbert-space
matrix or state-preparation shortcut. For recognized quadratic inputs above
eight modes, automatic routing uses that available route before DLA analysis;
this disclosed search cap does not assert gate optimality. Explicit BDI/Givens
remain available. pytket imports are optional emission candidates;
chemistry/HamLib dependencies are loaded at their respective boundaries.

GULPS is not called by the production router or many-qubit exact solver. Direct
upstream hardware-native two-qubit synthesis is a separate task; no benchmarked
CX improvement or internal integration should be inferred from past experiments.

## Verification workflow

From an installed project environment:

```bash
python -m pytest -q
python -m lizzy.synthesis_bench
python -m lizzy.driven_bench
```

The test suite does not download data. HamLib and encoding checks use local
deterministic fixtures; optional SDK integrations skip when the SDK is absent.
Benchmark commands are separate from the test suite and may perform substantial
reference calculations. [Benchmarks](benchmarks.md) lists scope and reproduction
details.

Tests cover dense logical-sequence reconstruction, supported exact models,
4-by-4 pair-kernel self-verification, native signed-frame equivalence, optional
emission fallback, closure/work limits and numerical conventions. Driven tests
also compare analytic rotating fields and independent dense Schrödinger ODEs;
Magnus/Fer tests separate expansion convergence from factor synthesis.

Each retained test should protect a mathematical invariant, a public contract or
a concrete regression. Choose a small set of named cases that exercise distinct
branches (for example odd/even BDI blocks, an encoded algebra, a singular chart,
and negative time). Do not multiply model, register size and time into a grid
when the cases exercise the same behavior, or hide that grid inside a test loop.
Use the benchmark harnesses for broader numerical sweeps. Shared validators need
boundary coverage once plus representative checks that each API calls them.
Prefer independent dense or analytic references to tests that merely compare
two wrappers around the same implementation.

Keep correctness and selection regressions, not snapshots of private caches,
CLI formatting, or a heuristic's gate-count win on one fixture. Archived
experiment tests retain accuracy/coverage safeguards and reproducibility checks;
they do not need an exhaustive input-validation matrix. Wider model/size/time
sweeps belong in the benchmark commands above, not duplicated across API layers.

The test modules follow these boundaries:

| Area | Focused suites |
|---|---|
| Algebra and exact synthesis | `test_algebra`, `test_orthogonal_mapping`, `test_exact`, `test_two_qubit_kernels` |
| Upstream Gaussian synthesis | `test_gaussian`: strict phase, pairing/hopping, structural rejection and polynomial-size reconstruction |
| Static formulas and routing | `test_product_formulas`, `test_coloring_strategies`, `test_core_pipeline` |
| Numerical synthesis | `test_driven` (paper equations and chart guards), `test_wei_norman` (public pipeline), `test_expansions`, `test_numerical_validation` |
| Emission | `test_emission` (SDK-independent contracts), `test_native_frame`, `test_pytket_emission`, `test_synthesis_emission` |
| Clifford+T | `test_clifford_t` (phase, inversion, actual gate counts and local error accounting); T selection contracts in `test_synthesis_emission` |
| Molecular adapters and loading | `test_chemistry`, `test_hamlib` |
| Reproducibility | Benchmark-harness tests check accuracy gates, resource caps and reported artifacts; `test_documentation` checks runnable examples and links |

Keep three questions separate when extending the package:

1. **Does the logical sequence represent the intended evolution?** Use the
   route-appropriate reference and report whether global phase is aligned.
2. **Does the emitted artifact implement that sequence?** Compare artifacts
   independently of the evolution approximation; phase-sensitive routes require
   strict agreement.
3. **Is its cost comparable?** Use the same gate basis, connectivity assumptions
   and achieved accuracy. Report analytical charges separately from emitted gates.

New optional emitters should fail closed without suppressing fallback, retain the
artifact that was counted, and be checked on small dense examples. A hardware
weighted objective needs its own cost model; it is not interchangeable with CX.

## Experiments and records

`experiments/` holds bounded research and reproduction work, not production
dependencies. See the [experiment index](../experiments/README.md) for active
reproduction commands, archived evidence and superseded prototypes. Keep negative
results, raw failures and provenance when retiring experimental code.

`docs/` contains guides and recorded benchmark artifacts. Counts there are dated
measurements with stated settings, not a claim of universal superiority or an
automatic selection rule. Public API docstrings remain the source for complete
argument validation details.
