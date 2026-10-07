# Recursive BDI versus Givens

The first comparison imposed Givens' representation ordering on endpoint BDI.
That does not test the horizontal implementation used in the paper's example.
The current compiler derives BDI's mapping from the supplied Pauli algebra and
uses its horizontal `K A(t) K†` form when possible; it does not dispatch by
model name. See the [paper-to-code review](cartan_review.md).

## Corrected results: 2026-10-07

All **82 logical outputs and 164 concrete artifacts passed** across 41 paired
targets. Every BDI component in this corpus used the generic `horizontal-graph`
mapping and passed the strict phase-preserving target check. The largest aligned
target error across both emitters was `8.09e-14`; BDI's largest strict target
error was `3.55e-14`. The largest strict emission/logical discrepancy was
`2.54e-14`. General nonhorizontal BDI has separate correctness tests; this corpus
does not measure its performance.

The ranking depends on emission:

| Concrete emission policy | BDI CX wins | Givens CX wins | Ties |
| --- | ---: | ---: | ---: |
| Independent native ladders | 0 | 41 | 0 |
| Shared native-frame emitter | 13 | 28 | 0 |
| Best of those two artifacts | 9 | 28 | 4 |

For seeded chains, each row below covers the three evolution times `0.1, 1, 5`;
the counts were identical across those times. All values count actual CX gates.

| Target | Givens ladder | BDI ladder | Givens best-of-two | BDI best-of-two |
| --- | ---: | ---: | ---: | ---: |
| TFIM3 | 28 | 44 | 26 | 24 |
| TFIM4 | 60 | 92 | 54 | 58 |
| TFIM5 | 104 | 196 | 88 | 92 |
| TFIM6 | 160 | 300 | 132 | 150 |
| TFXY3 | 24 | 40 | 24 | 24 |
| TFXY4 | 60 | 68 | 58 | 60 |
| TFXY5 | 104 | 132 | 92 | 86 |
| TFXY6 | 156 | 212 | 136 | 118 |
| XY3 | 12 | 20 | 8 | 10 |
| XY4 | 24 | 48 | 20 | 26 |
| XY5 | 40 | 104 | 38 | 44 |
| XY6 | 60 | 172 | 60 | 66 |

The nine best-of-two BDI wins are TFIM3, TFXY5 and TFXY6 at each of the three
times. The ties are the three TFXY3 cases and the boundary-field `so(7)` case
(32 CX each). Givens wins the remaining four extra cases. Full per-case counts,
input coefficients, partitions, accuracy checks and timing samples are in the
[corrected raw record](../experiments/bdi_paper_results.json).

Prepared BDI evaluation was faster in all 41 pairs: `0.0057–0.0317 ms` versus
Givens' `0.0455–0.2238 ms`; the median paired BDI/Givens ratio was `0.139`
(about 7.2 times faster). This is the benefit of reusing the horizontal wings,
not a measurement that CS factorization is faster than Givens elimination.
Including fresh method preparation and first evaluation reverses that picture:
BDI was faster in only one pair, with median paired ratio `1.43`.
Preparation-plus-first-evaluation ranges were `0.893–3.992 ms` for BDI and
`0.496–2.903 ms` for Givens. These short environment-sensitive measurements are
not large-system scaling results or end-to-end emission timings.

The run used Python 3.13.7, NumPy 2.4.1, SciPy 1.17.1 and PauLie 0.2.3, with a
clean kak-tools checkout at the README's tested revision
[`3980728596a060a7db2cf5f541a4aaf9011a9b1d`](https://github.com/QPauLie/kak-tools/tree/3980728596a060a7db2cf5f541a4aaf9011a9b1d).
Lizzy's implementation was local and uncommitted, not a released version; source
hashes and the base commit are retained in the JSON.

## Current comparison protocol

The [benchmark](../experiments/bdi_benchmark.py) calls the production methods on
the same 41 paired targets described below. Each method keeps its own mapping
and order: Givens favors cheap adjacent planes; BDI retains a structure-derived
horizontal or general embedding. Identical ordering is no longer imposed.

- Every output has a concrete native-ladder artifact, the primary CX comparison.
  Both methods also use the same existing native-frame emitter with
  `lookahead=8, discount_rate=0.9`. A small portfolio reports the cheaper of these
  two concrete artifacts only after both have passed verification.
- Logical circuits receive the same `fold_phases` cleanup. Every emitted
  artifact must strictly agree with its logical sequence. Target checks require
  strict operator-norm accuracy for phase-preserving horizontal BDI; Givens
  and general endpoint BDI are checked up to global phase. Both strict and
  aligned target errors remain in the record; threshold `1e-8`.
- Fresh method preparation is timed independently for each target. BDI uses
  `prepare_bdi(..., cache=False)`; Givens' representation-cache keys are evicted
  before setup. First evaluation, preparation-plus-first-evaluation, and three
  warmed prepared evaluations are reported separately. Timed method order
  alternates. Folding, emission, dense references and external verification
  stay outside the synthesis clock. Ordinary dependency/classification caches
  remain available: these are method-cache-cold, not process-cold measurements.
- A horizontal BDI prepared evaluation reuses fixed outer wings and central
  rates. Givens reuses the mapping but recomputes its matrix exponential and
  elimination. These measure the actual workflows, not isolated CS-versus-Givens
  arithmetic. General endpoint BDI also recomputes its time-dependent factors.

The native ladders and native-frame emitter are controlled emission baselines,
**not the paper's specialized CX construction**. Paired `XY/YX` gates and
cross-layer cancellations from Section VI.3.1 require their own implementation
and verification. Small-chain counts alone do not settle performance on arbitrary
Hamiltonians; generic encoded and nonhorizontal coverage is exercised separately
by the correctness tests.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.bdi_benchmark --json

# One paired case; repeat --case to select more.
python -m experiments.bdi_benchmark --case tfim3-t1
```

The CLI exits nonzero if either backend or either emitted artifact fails.
The automatic router remains unchanged; an explicit BDI comparison is not an
automatic method-selection rule.

## Historical shared-order endpoint experiment

Recorded on 2026-10-07 **before the horizontal correction**. In that implementation,
BDI used fewer ladder CX gates in 4 of 41 paired cases and Givens in 37, with no
ties. These are historical measurements, not results of the current BDI route.

The immutable [full record](../experiments/bdi_results.json) retains all inputs,
component sizes, index orderings, timing samples, errors and dependency versions.

### Historical protocol and target inventory

- Seeded `tfim`, `tfxy` and `xy` chains on 3, 4, 5 and 6 qubits, each at
  times 0.1, 1 and 5; coefficient seed 20261007. The three times reuse the
  same Hamiltonian and are not independent random samples.
- Five additional cases: uniform TFIM4, TFXY4 and XY5, negative-time TFIM4,
  and an explicit odd-dimensional `so(7)` example (TFIM3 with a boundary
  `XII` field of strength 0.41). Lizzy's TFIM constructor uses `XX` bonds
  and `Z` fields.
- Both methods receive the same commuting summands, upstream irrep mapping
  and low-weight index ordering. Both get the same `fold_phases` cleanup
  with its default `1e-12` tolerance, then concrete native Pauli ladders.
  No optional SDK or method-specific emission portfolio is used.
- Every output is compared to an independent dense `expm` reference using
  trace-phase-aligned operator norm, with threshold `1e-8`. Both methods
  promise synthesis only up to global phase. Emission must independently
  agree with the logical circuit in strict, phase-sensitive norm.
- Compile times are medians of three runs after warming both methods;
  the timed method order alternates. They exclude mapping setup, folding,
  native emission, dense reference construction and external dense verification. These
  other timings are recorded separately. Shared mapping setup uses normal
  process caches, with each component's cache-hit state recorded.
  The as-shipped BDI backend calls `recursive_bdi(validate=True)`, so its
  internal factor reconstruction and determinant checks remain inside the clock.

### Historical results

All **82 outputs passed**, with no omitted setup, compilation or accuracy
failures. The largest aligned target error was `4.94e-13`; the largest strict
emission/logical discrepancy was `2.42e-14`. A strict target error near 2
in some rows is the documented spin-cover global sign, not an emission error.

These are **counts from emitted CX gates**, not analytical block estimates.
For the seeded families, each row covers all three evolution times; a range
means the count varied with time.

| Model | Qubits | Givens CX | BDI CX |
| --- | ---: | ---: | ---: |
| TFIM | 3 | 28 | 36 |
| TFIM | 4 | 60 | 88 |
| TFIM | 5 | 104 | 156–164 |
| TFIM | 6 | 160 | 218–244 |
| TFXY | 3 | 24 | 40 |
| TFXY | 4 | 60 | 56 |
| TFXY | 5 | 104 | 124 |
| TFXY | 6 | 156 | 196 |
| XY | 3 | 12 | 16 |
| XY | 4 | 24 | 32 |
| XY | 5 | 40 | 64 |
| XY | 6 | 60 | 100 |
| Uniform TFIM | 4 | 60 | 88 |
| Uniform TFXY | 4 | 60 | 56 |
| Uniform XY | 5 | 40 | 64 |
| Negative-time TFIM | 4 | 60 | 88 |
| Boundary-field `so(7)` | 3 | 32 | 36 |

BDI's four CX wins are the three seeded TFXY4 times and uniform TFXY4:
56 versus 60 CX. Total native-gate counts have the same 4-versus-37 ranking.
Folded logical rotation counts tie in 38 cases; BDI uses fewer in two,
and Givens fewer in one. Equal parameter counts therefore do not imply equal
physical gate costs: the Pauli supports differ.

In this run, Givens' warmed compile medians ranged from 0.048 to 0.220 ms;
BDI's ranged from 0.247 to 2.281 ms. BDI was slower in every pair, with a
median paired ratio of 8.19. These are as-shipped backend timings, including
BDI's internal validation, not a comparison of unchecked CS versus Givens
arithmetic. These short timings are environment-sensitive; they are not a
scaling result or an end-to-end latency comparison.

### Historical scope and provenance

The common index order was designed for cheap adjacent-plane Givens rotations.
It is intentionally held fixed to isolate the decomposition change, but does
not optimize BDI's partition or pairing for Pauli support. This experiment
does not test a BDI-specific ordering search, the paper's separate horizontal
variant, optimized shared Clifford-frame emission, or hardware routing.
The limited 3–6-qubit chain corpus says nothing conclusive about larger
systems, arbitrary mapped algebras, or chemistry.

The run used Python 3.13.7, NumPy 2.4.1, SciPy 1.17.1 and PauLie 0.2.3.
`kak-tools` was an editable but clean checkout at the tested revision for that run
[`3980728596a060a7db2cf5f541a4aaf9011a9b1d`](https://github.com/QPauLie/kak-tools/tree/3980728596a060a7db2cf5f541a4aaf9011a9b1d),
not a locally modified optimizer. Lizzy contained the local, not-yet-committed
BDI implementation. The JSON records that provenance rather than attributing
these results to the older Lizzy base commit.

The current benchmark command above no longer reproduces this superseded
implementation. The preserved JSON is the evidence for these historical numbers;
do not overwrite it with output from the corrected horizontal route. In
particular, the old four-versus-37 ranking cannot justify a claim that the
paper-aligned BDI method is generally inferior to Givens.
