# Reproduce the current results

These runners verify actual circuit artifacts against independent small-instance
references. They are not installed production compilers. Run from the repository
root; missing dependencies, unsupported inputs, caps and failures remain visible.

## Compiler comparison

```bash
python -m pip install -e '.[ft,compare,flags,bqskit,gaussian,plot]'
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.compiler_comparison --output /tmp/lizzy-compiler-current.json
python docs/figures/generate.py
```

The [reader-facing comparison](compiler_comparison.md) uses the unchanged
{download}`current record <../experiments/compiler_comparison_results.json>`. The generator reads that
record, not the new temporary output. Replace a published record only after
reviewing a genuinely new run; do not rewrite historical measurements.

The {download}`protocol <../experiments/compiler_comparison_protocol.md>` fixes twelve targets, two error
thresholds, candidate portfolios and common Clifford+T lowering. Lizzy auto and
Qiskit formulas receive H,t; FlagSynth SDM, Qiskit QSD, pytket and BQSKit receive
dense U. All outputs are ancilla-free and checked against the same full unitary.
This compares circuit resources, not equal classical workloads or scaling.
BDI/Givens/Wei–Norman rows are internal diagnostics, not extra compiler brands.

FlagSynth is explicitly **locally patched SDM**; Qiskit formulas include ordinary
and Rustiq lowering in one result. The merged record retains separate source/run
provenance for unchanged external measurements and later automatic-route refreshes.
A failing compiler is not a win. The command exits nonzero on failures even when
the report was saved: inspect `complete` and individual row statuses.
Use `--case tfim3 --epsilon 1e-6` for a bounded rerun.

## BDI versus Givens

```bash
python -m experiments.bdi_benchmark --case tfim3-t1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.bdi_benchmark --json
```

The {download}`corrected record <../experiments/bdi_paper_results.json>` retains each method's own mapping,
separates preparation from reusable evaluation, and checks logical outputs and
both native ladder/frame artifacts. Its 41 pairs favor Givens in every ladder-only
CX comparison; with the same best-of-two emitter portfolio, BDI wins nine, Givens
28, and four tie. These are scoped observations, not a universal ordering or the
paper's specialized CX construction. All recorded BDI components are horizontal;
general encoded/nonhorizontal correctness is covered by production tests.

## Wei–Norman validation

```bash
python -m experiments.wei_norman_validation --case static-tfim3
python -m experiments.wei_norman_validation --json
```

This calls the production API on 27 fixed targets under resolved-step and
adaptive-default policies. Independent analytic/dense references check strict
phase-sensitive operator error. Reverse time, scalar phases, pulse boundaries,
singular charts and explicit closure caps are retained. Only declared cap cases
may return CAP without failing the run. This is validation, not a guaranteed
error bound for unseen controls. Shared reference helpers live in `experiments/_validation.py`;
there is no duplicate numerical solver.

## Earlier research

Completed chemistry routing, compact-chart, BDI gauge and joint T/CX ablations,
older compiler snapshots and their reproduction scripts remain together at
[the pinned research revision](https://github.com/AmanieOxana/Lizzy/tree/f021a9a0f2a8c8f3f97674ae96fb0d8729496830).
Use that revision's code **and** records to reproduce old results; today's
integrated baseline need not reproduce a pre-integration ablation.
No duplicate archive directory is maintained.

The chemistry selector study was inconclusive (strict common coverage 2/8);
its separately labelled post-hoc headroom was only 2.33%. No chemistry advantage
or trained selector follows. Broader compiler research is retained in that
revision's `docs/compiler_scope.md`. F3C++ remains a relevant unmeasured scaling
comparison; exact compression of a product-formula circuit does not remove its
time-discretization error.
