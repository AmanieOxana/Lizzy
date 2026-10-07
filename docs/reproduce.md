# Run the checks

Use these commands from the repository root after [installation](getting_started.md#install).

## Test the implementation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Optional integrations are skipped if their dependencies are absent. Tests do
not download data.

## Compiler comparison

Install the comparison backends, then start with one target:

```bash
python -m pip install -e '.[ft,compare,flags,bqskit,gaussian]'
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.compiler_comparison \
  --case tfim3 --epsilon 1e-6 --output /tmp/lizzy-compiler-current.json
```

Omit `--case tfim3 --epsilon 1e-6` for the full comparison. The report checks
compiled circuits against the target unitary and records both T and CX counts.
Inspect each row's status: a completed run can still contain unavailable,
unsupported or capped methods. Failures are not wins.

The [Results plot](compiler_comparison.md) uses the
{download}`saved measurements <../experiments/compiler_comparison_results.json>`;
a new run does not update it automatically. The
{download}`comparison protocol <../experiments/compiler_comparison_protocol.md>`
records backend settings and workload differences.

To redraw that plot from its saved measurements, without running benchmarks:

```bash
python -m pip install -e '.[plot]'
python docs/figures/generate.py
```

## Check individual methods

These runners compare production circuits with independent small-system references:

```bash
python -m experiments.bdi_benchmark --case tfim3-t1
python -m experiments.wei_norman_validation --case static-tfim3
```

Replace either `--case …` with `--json` to run its full set and print a report.
These checks validate selected instances; they do not establish a universal
winner or an error guarantee for new inputs.
