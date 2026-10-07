# Experiments and archived evidence

These scripts validate hypotheses; they are not production routing policies and
are not installed as part of the `lizzy` package. Run them from the repository
root. Failed cases and accuracy qualifications are part of the result.

## Current cross-compiler Clifford+T comparison

```bash
python -m pip install -e '.[ft,compare,flags,bqskit,gaussian,plot]'
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.compiler_comparison --output experiments/compiler_comparison_results.json
python docs/figures/generate.py --only compiler
```

The [report and diagram](../docs/compiler_comparison.md) compare public automatic
Lizzy exact synthesis with locally patched FlagSynth SDM, Qiskit QSD, pytket,
BQSKit and Qiskit product formulas (best of ordinary/Rustiq lowering).
Separate BDI, Givens and Wei–Norman rows remain internal diagnostics. Twelve fixed
targets and two accuracy thresholds use a common Clifford+T backend and final
operator-norm checks. The [protocol](compiler_comparison_protocol.md) fixes
candidate portfolios and includes unsupported cases, solver caps and failures.
The [raw record](compiler_comparison_results.json) includes all candidate costs,
errors, versions and source fingerprints. The [per-target matrix](../docs/figures/compiler-comparison-detail.svg)
shows the detailed counts. Unchanged external measurements and the later
Gaussian-enabled and shared-frame auto reruns have separate source/version snapshots
in the record. The [pre-shared-frame snapshot](compiler_comparison_before_shared_frames.json)
preserves the previous comparison; only automatic Lizzy outputs were refreshed.
FlagSynth uses two documented API
bridges plus explicit numerical/zero-rotation robustness repairs around the
upstream implementation; it is labelled **patched SDM**, not unmodified upstream.
Givens now falls back to the verified full-algebra mapping where a horizontal
embedding is impossible. The [pre-fix record](compiler_comparison_before_fixes.json)
preserves the original failures; inputs and accuracy settings have not changed.
The [repaired six-method snapshot](compiler_comparison_separate_methods.json)
also remains available from before adding the automatic selector and new tracks.
The [scope review](../docs/compiler_scope.md) explains the earlier Rustiq and
PHOENIX comparisons and which methods need a different experimental contract.

Run one target with `--case tfim3 --epsilon 1e-6`. Exit status is nonzero if any
compiler fails, **even when the full report was saved**; check its `complete`
field and row statuses. A failing baseline is not a win for another compiler.
This is an ancilla-free small-target resource comparison, not a runtime ranking,
all-compiler survey, exhaustive product-formula search or phase-gradient comparison.

## Joint T and CX experiment

**Shared Clifford frames improve both-cost outcomes; extra gauge search has not
shown an additional benefit on this corpus.** The saved ablation predates
production integration. Shared frames are now used by the public T selector;
the broader gauge search remains experimental.

```bash
python -m experiments.joint_t_cx --output experiments/joint_t_cx_results.json
```

The [saved pre-integration run](joint_t_cx_results.json) uses the same twelve targets at final
operator error `1e-6`, up to global phase. Nine are supported; three remain
unsupported. Six supported targets improve at least one count without increasing
the other, and three are unchanged. For example:

| Target | Before frames (T, CX) | With frames (T, CX) |
|---|---:|---:|
| Encoded SU(2) | 240, 6 | 240, 2 |
| Heisenberg · 3 qubits | 1306, 54 | 1304, 18 |

Four nested stages separate the effects: delivered output → shared frames for
that same decomposition → existing route alternatives with shared frames →
additional legal BDI gauges. All six no-compromise gains arise in the second
stage. The third adds a TFIM3 trade-off: `(1298 T, 26 CX)` or `(1300 T, 24 CX)`.
The fourth does not improve the frontier. Only the anticommuting-star target
has additional structural-nullspace gauges here, so this is a narrow gauge test.

Every frontier retains its actual QASM, both gate counts and checked errors.
All delivered reference hashes match the published benchmark. The run validates
81 distinct emitted artifacts; maximum final error is below `2.1e-8`. Frames are
checked in strict phase-sensitive norm, including their closing Clifford gates.
All candidates share a total rotation budget of `5e-7`; none uses ancillas.
Frame lookahead is bounded to 0 and 8; whole-circuit gauge combinations are capped
at 32, without component-wise cost pruning. No cap was reached in this run.
Rerunning the script uses the current public baseline, which already includes
frames; it does not recreate the archived pre-integration baseline.

## Recursive BDI versus Givens

```bash
python -m experiments.bdi_benchmark --case tfim3-t1
python -m experiments.bdi_benchmark --json
```

The [comparison report](../docs/bdi_benchmark.md) distinguishes the current
structure-derived horizontal/general BDI implementation from the original
shared-order endpoint comparison. The current harness keeps each method's own
representation and reports preparation separately from evaluation. Primary costs
come from concrete native ladders; the existing native-frame emitter is an
independent second baseline. Neither is the paper's specialized CX optimization.
The benchmark is not an automatic route-selection policy.

The [corrected raw record](bdi_paper_results.json) has 82 passing outputs and
164 verified ladder/frame artifacts. Ladder-only CX favors Givens in all 41
pairs; the same best-of-two concrete portfolio gives BDI nine wins, Givens 28
and four ties. Every BDI component in this corpus is horizontal and passed
strict phase-sensitive checks. Nonhorizontal correctness is tested separately.

[bdi_results.json](bdi_results.json) is the immutable historical shared-order
record: 82 passing outputs across 41 pairs, four BDI ladder-CX wins and 37 Givens
wins. Its numbers must not be presented as results of the corrected implementation.

## T-aware BDI nullspace optimization

```bash
python -m experiments.bdi_t_benchmark --output /tmp/lizzy-bdi-t-current.json
```

Requires the `ft` extra. The [raw record](bdi_t_results.json) compares original
BDI with legal nullspace-basis choices under the same recursion, Clifford+T
compiler and total rotation-approximation budget. Five targeted outputs pass
dense phase-appropriate checks: three unbalanced rows improve in actual T count;
balanced and general controls remain unchanged. Two rows share one Hamiltonian.
This historical run predates shared-frame integration; a new run includes that
additional optimization, so its gains cannot be attributed to gauge choice alone.
These are not FlagSynth comparisons or formal total-error certificates. See the
[guide and chart](../docs/fault_tolerant.md) for API, exact counts and limitations.

## Current Wei–Norman production validation

```bash
python -m experiments.wei_norman_validation --case static-tfim3
python -m experiments.wei_norman_validation --json
```

This uses the public Wei–Norman compiler on 27 fixed targets, with a resolved
`max_step=0.025` and the adaptive default. It retains reverse evolution, identity
phases, pulse breakpoints, coordinate singularities, seeded drives and an explicit
dimension-cap case. Every output is checked in strict operator norm against an
independent small dense/analytic reference. A CAP is accepted only on an explicitly
declared cap case; unexpected caps and failures make the command exit nonzero.

The [2026-09-25 validation snapshot](../docs/wei_norman_validation.json) predates
this harness consolidation: its 52 passing rows and two expected caps are retained
without rewriting the historical measurements. The two production step policies
and 27-target corpus are unchanged. Conditioning remains sampled, not certified.

## Historical compact-chart experiment

The [report](wei_norman_compact_report.md) and [raw results](wei_norman_compact_results.json)
document why the production chart policy changed. They compare the old radius cap
with two experimental policies, not three current production methods.

The duplicate `wei_norman_compact.py` solver, monkeypatch-based benchmark, and
prototype-only tests were removed after integration. They remain recoverable in
Git commit `b7f71adde811be6ca91f7266291b11565c0fbc55`. Current mathematical and
failure regressions exercise production code directly.

## Chemistry routing: completed, inconclusive

Read the [protocol](chemistry_routing_protocol.md), [report](chemistry_routing_report.md),
[raw measurements](chemistry_routing_results.json), and [analysis](chemistry_routing_analysis.json).
The primary comparison had common coverage on only 2/8 tasks. A separately labelled
post-hoc sensitivity found only 2.33% geometric-mean CX headroom even with perfect
route selection. No selector was fitted; no new chemistry advantage was established.

The measurement, gate-analysis and diagnostic scripts are retained for provenance.
They require the chemistry/comparison extras and existing cached HamLib inputs;
they do not download missing inputs. Reproduction commands and numerical caveats
are in the report. Do not relabel the sensitivity as the primary result.

## Cleanup boundary

The GULPS dependency/example had no production compiler caller and no demonstrated
end-to-end gain here, so it is no longer a base dependency. The independent
two-qubit canonical-cost correctness fix is retained. Neither this decision nor
the chemistry experiment claims those upstream methods cannot help other tasks.
