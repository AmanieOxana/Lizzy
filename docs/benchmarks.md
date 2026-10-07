# Benchmarks and evidence

[README](../README.md) · [Synthesis](synthesis.md) · [Experiments](../experiments/README.md)

Benchmark records are observations under their recorded settings, not a universal
method ranking. The cleanup does not turn old measurements into fresh results.

## Cost and accuracy conventions

- `logical_two_qubit_gates` / `logical_CX` analytically prices the Pauli IR using
  pair-block KAK classes and wider CNOT ladders. It is not a physical gate list.
- Numerical benchmarks use a fixed concrete native-ladder/eligible native-frame
  portfolio. Their emitted CX counts are read from actual artifacts; no optional
  SDK silently changes the portfolio.
- The current [cross-compiler T benchmark](compiler_comparison.md) uses the same
  phase-aligned spectral-norm rule for **all** methods and reports strict errors
  separately. All outputs are ancilla-free, with the same rotation backend.
- In the earlier method-specific validation records, Givens and nonhorizontal endpoint-BDI allow global phase;
  horizontal BDI and numerical Wei–Norman additionally require strict
  operator-norm accuracy including phase. Emission must preserve the logical
  circuit. See the two paper reviews for scope.
- A state-infidelity target, an operator-norm target, and a logical/emitted
  equivalence check are different tests. Do not exchange their thresholds.

## Current cross-compiler Clifford+T comparison

The [new report](compiler_comparison.md) supersedes the old CX-only cross-compiler
ranking. It compares public automatic Lizzy exact synthesis against
locally patched FlagSynth SDM, Qiskit QSD, pytket, BQSKit and the existing
Qiskit product-formula portfolio (default/Rustiq lowering, now scored with T
gates and full-unitary accuracy). Separate BDI, Givens and Wei–Norman rows remain
internal diagnostics. All use twelve fixed targets at two
end-to-end error thresholds. Actual T/T† gates are counted only after the same
Clifford+T compilation and dense target check. Unsupported targets, caps and
failures remain in the plot and raw data; pairwise scores use common passing
coverage only. The [protocol](../experiments/compiler_comparison_protocol.md) explains FlagSynth's two API bridges and additional
numerical/zero-rotation repairs, Givens' repaired general mapping, the different
input formats, and why no universal ranking follows. The
[pre-fix measurements](../experiments/compiler_comparison_before_fixes.json)
retain the earlier failures rather than presenting patched results as upstream behavior.

```bash
python -m experiments.compiler_comparison --output experiments/compiler_comparison_results.json
```

Needs `.[ft,compare,flags,bqskit,gaussian]`. Inputs are offline, including the fixed H₂ coefficient
snapshot; the benchmark does not download HamLib data. The five-case BDI gauge
experiment below remains a separate internal ablation.

## Recursive BDI versus Givens

```bash
python -m experiments.bdi_benchmark --case tfim3-t1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m experiments.bdi_benchmark --json
```

The [comparison report](bdi_benchmark.md) separates the corrected paper-aligned
implementation from the first shared-order experiment. The current harness
retains BDI's structure-derived mapping and horizontal `K A(t) K†` plan, instead
of imposing Givens' cheap-adjacent ordering. It records cold preparation, first
evaluation and repeated prepared evaluations separately. Common folding and
concrete native ladders provide the primary comparison; the existing native-frame
emitter provides a separately validated second baseline. Neither emitter claims
the paper's specialized paired-gate and layer-cancellation optimizations.

The [corrected record](../experiments/bdi_paper_results.json) contains 82 passing
outputs and 164 verified concrete artifacts. With independent ladders, Givens
wins all 41 pairs. With the same best-of-ladder/frame portfolio, BDI wins nine,
Givens 28, and four tie. Reusing BDI's horizontal plan makes repeated evaluation
faster in this corpus, but fresh preparation plus first evaluation usually costs
more. See the report for exact counts, phase contracts and separate timing scopes.

The [old raw record](../experiments/bdi_results.json) remains historical: its
41 paired targets used endpoint BDI with the Givens ordering, yielding four
BDI CX wins and 37 Givens wins. Those numbers do **not** measure the corrected
horizontal implementation. The automatic router remains unchanged.

## T-aware BDI

The [2026-10-07 targeted comparison](fault_tolerant.md#measured-results--2026-10-07)
uses actual T/T† gates from the same optional pygridsynth backend and a common
`1e-6` rotation budget. Legal nullspace-basis optimization reduces T count in
three rows; two controls are unchanged. All five outputs pass dense numerical
checks, with global-phase alignment permitted only for general endpoint BDI.
This is neither a broad benchmark nor a comparison against FlagSynth.

Run `python -m experiments.bdi_t_benchmark`; raw coefficients, versions, counts
and phase-sensitive errors are in
[bdi_t_results.json](../experiments/bdi_t_results.json). The chosen artifact never
has more T gates than the unchanged reference compiled in the same run. Search
estimates, rotation-only numerical bounds and actual counts remain separate.

## Driven methods

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m lizzy.driven_bench
```

The [2026-09-25 snapshot](driven_benchmark.txt) compares analytic/dense references
at operator-error threshold `1e-6`. The following are actual emitted CX counts:

| Case | Wei–Norman | Magnus4 | Fer4 | Midpoint-S2 |
| --- | ---: | ---: | ---: | ---: |
| Rotating spin | 0 | 0 | 0 | 0 |
| Encoded spin | 4 | 4 | 4 | 4 |
| Driven TFIM3 | 32 | 512 | 1,024 | 4,100 |

Wei–Norman uses one chart on these examples. The TFIM result has 15 rotations and
strict error about `9.3e-13`. Its historical radius-limited result was 256 CX;
the [archived experiment](../experiments/wei_norman_compact_report.md) explains the
policy change. Earlier analytical values must not be substituted for emitted CX.

The expansion/PF step search uses dense-reference screening on a power-of-two
grid. A first passing grid point is not the minimum possible step count. This
is a small-instance validation oracle, not deployable error-based routing.

## Static/driven comparison

```bash
python -m lizzy.synthesis_bench --json
python -m lizzy.synthesis_bench --case su8-closure-cap --max-dimension 64 --json
```

The [14-case snapshot](wei_norman_benchmark.json) records 33 passing comparisons,
eight unsupported static-exact cases and one explicit dimension cap. It compares
Wei–Norman, midpoint-S2 and the supported orthogonal/summand exact decomposition,
**not the complete automatic static portfolio** or all competing solvers.

| Case | Wei–Norman CX | Midpoint-S2 CX | Static exact CX |
| --- | ---: | ---: | ---: |
| TFIM3, `t=0.003` | 32 | 8 | 28 |
| TFIM3, `t=1` | 32 | 2,052 | 28 |
| TFIM3, `t=5` | 32 | 16,388 | 28 |
| Driven TFIM3 | 32 | 4,100 | unsupported |

The exact route still wins the displayed static CX comparison. Some exact outputs
differ from the reference by a global sign, which their phase-aligned acceptance
contract permits; Wei–Norman must pass the strict phase-preserving norm.

The [separate raised-cap run](wei_norman_su8_benchmark.json) allows dimension 63:
162 Wei–Norman CX versus 512 midpoint-S2 CX. Raising this cap is explicit, not
evidence that closure is generically cheap. Failures and unsupported routes remain
in the records, and are not counted as wins. Timings are single-run observations.

For robustness beyond the 14 cases, run
`python -m experiments.wei_norman_validation --json`. The
[integration snapshot](wei_norman_validation.json) contains 52 passing runs and
two expected caps across 27 targets and two step policies. The current harness
calls production directly; it no longer substitutes an experimental solver.

## Chemistry and optional compiler comparisons

The completed [chemistry selector experiment](../experiments/chemistry_routing_report.md)
compares JW/BK/ffsim DF on four molecular instances. Strict common coverage was
only 2/8 tasks. A post-hoc sensitivity found 2.33% geometric-mean CX headroom for
perfect selection over always using BK. No selector was fitted. The
[chemistry guide](chemistry.md) explains why this is neither a new chemistry
advantage nor evidence that all structural information is useless.

`python -m lizzy.compare` runs the separate fixed-step HamLib comparison and
small dense-oracle comparisons. It needs `.[hamlib,compare,plot]`, may download
HamLib inputs and writes `docs/comparison.png` by default. It is not part of
the minimal smoke test. The retained figure is a historical 2026-09-04 snapshot,
not a fresh accuracy-matched comparison or a claim about current upstream releases:

![Historical fixed-step HamLib comparison](comparison.png)

To inspect a small builtin-model route/error run without HamLib:

```bash
python -m lizzy.bench --family tfim --n 4 --time 1 --seeds 1
```

The unused CLI `--calibrate` switch was removed: it previously did nothing.
The tested Python helper `lizzy.bench.calibrate` remains available; its measured
scaling is not a certified error bound.
