# Chemistry routing: bounded experiment (2026-09-21)

## Outcome: stop this selector hypothesis; no demonstrated new benefit

The formal primary result is **INCONCLUSIVE_COVERAGE**, not a successful feature
test: only **2 of 8 tasks** have all three routes passing every implementation
guard. The fixed gate required at least six, all four training tasks, and both
holdout instances. No selector or ablation was fitted.

A separately labelled **post-hoc sensitivity**, accepting the emitted circuits
that meet the original reference-state target despite emission-mismatch failures,
has 8/8 coverage. Even its hindsight-perfect route oracle saves only **2.33%**
geometric-mean CX against the best fixed route, BK. This is below the protocol's
10% headroom criterion, but does **not** retrospectively convert the primary
verdict into a passed gate or a formal NO_GO_HEADROOM result.

Recommendation: do not spend more implementation effort on this particular
chemistry route-selector hypothesis without an independently motivated new task.
This experiment establishes neither a chemistry advantage nor a novel contribution
for Lizzy; it also does not prove that all structural approaches are unhelpful.

### Every task, without dropping failed routes

The table shows minimum CX across the fixed step grid and emission portfolio.
All displayed minima use one step. JW/BK entries pass the strict primary checks.
DF entries marked * fail the primary emission guard and appear **only in the
post-hoc target-accuracy sensitivity**.

| Instance | Time | JW CX | BK CX | DF CX (target-only if *) | Primary common coverage |
|---|---:|---:|---:|---:|---|
| H2-4 | 0.25 | 27 | 21 | 58 | yes |
| H2-4 | 1.00 | 27 | 21 | 58 | yes |
| LiH-8 | 0.25 | 294 | 313 | 828* | no |
| LiH-8 | 1.00 | 294 | 313 | 828* | no |
| BH-10 | 0.25 | 873 | 901 | 1,856* | no |
| BH-10 | 1.00 | 873 | 901 | 1,934* | no |
| LiH-12 | 0.25 | 1,883 | 1,789 | 3,618* | no |
| LiH-12 | 1.00 | 1,883 | 1,789 | 4,034* | no |

In that sensitivity, geometric means are JW **337.98**, BK **320.83**, DF
**767.84**, and oracle **313.36** CX: saving = 1 - 313.36 / 320.83 = **2.33%**.
BK wins H2 and LiH-12; JW wins LiH-8 and BH-10. DF is never cheapest here.
No zero-cost floor was needed. These are upstream route costs, not savings from
a newly implemented Lizzy synthesis algorithm.

There were **264 backend attempts**: 180 PASS, 52 FAIL_VERIFICATION, and 32
TIMEOUT (28 pytket peephole, 4 Qiskit Rustiq). All 52 verification failures have
finite four-probe target error below 1e-3 and acceptable normalization; their
emission mismatch exceeds 1e-9. The largest observed mismatch is 4.92e-6.
The sensitivity admits those 52 artifacts, not timeouts. All failures remain in
the raw data, and the primary labels are unchanged.

## What was tested

Four cached HamLib instances (H2-4, LiH-8, BH-10, LiH-12), two times
(`0.25, 1.0`), and steps `1, 2, 3`: **72 formula candidates**.
JW and BK each use their canonical-order symmetric second-order formula and a
five-backend Qiskit/pytket emission portfolio. DF uses upstream ffsim with its
original factor order and Qiskit optimization level 3. All counts are emitted
CX gates on unrestricted connectivity; no logical ladder estimate substitutes
for an artifact. State preparation and encoding conversion are excluded.

Four identical physical-state probes per instance are compared against independent
sparse Schrödinger evolution: one reference determinant and three seeded random
states in an assumed electron sector. The requested target is maximum sampled
state infidelity `1e-3`. This is not an operator-norm certificate, a ground-state
chemistry accuracy claim, or an energy-error benchmark.

The [internal protocol](chemistry_routing_protocol.md) was fixed before full
collection, with H2 adapter smoke checks beforehand; it was **not externally
preregistered**. The implementation additionally enforced norm error `1e-8`
and logical/emitted infidelity `1e-9` from the start of collection. Those numeric
guards were not specified in the original protocol prose, which said to inspect
normalization and emission mismatch separately. We retain their original
classifications rather than changing them after inspecting costs.

No production route was changed, dependency installed, or selector fitted.

## Why the emission guard matters

Larger-case DF artifacts can meet the requested evolution target yet fail the much
tighter logical/emitted equivalence guard. This is **not evidence that DF fails
the evolution accuracy target**.

A bounded, separate [LiH-8 diagnostic](chemistry_routing_diagnostic.json) uses
one retained factorization at `t=0.25`, one step, and compares Qiskit levels 0–3:

| Optimization level | CX | Max sampled mismatch to logical DF |
|---|---:|---:|
| 0 | 1,020 | below numerical resolution |
| 1 | 904 | below numerical resolution |
| 2 | 836 | 4.24e-9 |
| 3 | 836 | 4.24e-9 |

The level-3 artifact's maximum reference infidelity is `4.20e-9`, well below
`1e-3`. Pass-boundary replay first observes the discrepancy immediately after
`TwoQubitPeepholeOptimization`, even with `approximation_degree=1.0`.
This identifies the pass boundary, not the precise internal numerical mechanism.

The diagnostic is a **new factorization run**, with its own fingerprint and
slightly different CX counts from the primary LiH-8 run. Numerical orbital gauges
can vary. Each run holds its factorization fixed internally. Diagnostic costs are
not substituted into the primary corpus or sensitivity analysis; levels 0/1 were
not frozen DF candidates. No claim of mathematical exactness follows from the
sampled level-0/1 agreement.

## What the structural measures do—and do not—show

JW and BK have matching GF(2) span ranks, Gram ranks, and anticommutation statistics
in these records. Their term sets are related by Clifford conjugation, which also
preserves the generated Lie algebra. These invariants alone cannot distinguish
the two encodings' different CX costs. This does not rule out DLA information for
recognizing special solvable Hamiltonian families.

The proposed additional feature was normalized pairwise Givens count between DF
orbital frames. It is representation- and gauge-dependent, unlike those invariants.
Its measured values are H2-4 **1.0000**, LiH-8 **0.8148**, BH-10 **0.8762**, and
LiH-12 **0.9062**; no threshold or replacement feature was selected from them.
Because the primary gate did not authorize fitting, **its incremental predictive
value was not tested**. No trained threshold, held-out predictor improvement, or
feature novelty is claimed.

Double-factorized chemistry simulation is established
[Motta et al. prior art](https://arxiv.org/abs/1808.02625).
[Necaise et al.](https://arxiv.org/abs/2606.20805) already optimize DF-frame ordering
using pairwise **Choi distribution cost** and 2-opt for distributed-QPU objectives;
that is not specifically a Givens-count/local-CX selector. This distinction does
not itself establish a novel Lizzy contribution.

## Limits and interpretation

- Only three molecular families, with two active spaces of LiH; the times are
  correlated. Electron sectors are benchmark assumptions, not verified ground-state
  metadata. JW/BK means encoding **plus canonical term ordering**, not a pure
  encoding-only experiment.
- Short times, a three-step grid, four state probes, unrestricted connectivity,
  and a 20-second per-backend budget. Larger systems, longer times, stricter
  accuracy, other orderings and hardware constraints are untested.
- The best backend and step count are chosen with measured accuracy/cost labels.
  Any route oracle is an upper bound on selection benefit **within these measured
  candidates**, not a deployable cheap policy or a bound on all possible compilers.
- Compilation, factorization and verification times are retained but are
  single-run observations, not a controlled classical-runtime comparison.
- A post-hoc sensitivity calculation cannot restore confirmatory holdout status.
  It must not be presented as passing the original gate or proving a feature works.

## Reproduction and provenance

[Raw measurements](chemistry_routing_results.json) preserve every backend result,
including failures/timeouts, sampled errors, timings, fingerprints and versions.
[Analysis](chemistry_routing_analysis.json) separates the original gate from the
explicitly post-hoc, target-accuracy-only sensitivity. That sensitivity admits
only artifacts whose sole failing guard is logical/emitted mismatch; the original
four-probe target, normalization, fixed factorization and concrete CX requirements
remain. It does not add diagnostic candidates or fit a model.

From the project root, with the existing cached HamLib molecular files:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m experiments.chemistry_routing_measure
python3 -m experiments.chemistry_routing_gate experiments/chemistry_routing_results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m experiments.chemistry_routing_diagnostic
```

Measurement and diagnostic commands print JSON to stdout; progress goes to stderr.
The measurement script never downloads missing inputs. Partial-case runs are
supported for diagnostics, but the final analysis rejects an incomplete 72-row
grid or a modified timeout/step budget.

Captured versions: Python 3.13.7, NumPy 2.4.1, SciPy 1.17.1, OpenFermion 1.8.1,
ffsim 0.0.84, Qiskit 2.5.1, pytket 2.18.1. All measurement commands set the three
thread limits above to 1. H2/LiH-8 started before a metadata-only change added thread
environment reporting and the scalar constant to the tensor digest; their original
configurations and digest scopes are preserved explicitly in the combined JSON.
The numerical candidates and acceptance checks did not change.
