# Compact Wei–Norman experiment — 2026-09-25

Historical pre-integration snapshot: the compact policy has since been added to
`lizzy.driven` and `lizzy.wei_norman`. The harness pins its historical
`production` comparator to `chart_radius=0.5`; the numbers saved below are not
claims about the new default. See the current README and
[paper review](../docs/wei_norman_review.md) for the integrated interface.

## Result

The hard coordinate-angle cap substantially inflated the previous Wei–Norman
circuits. Removing that cap while retaining sampled conditioning/work checks
improves the prototype on these tests. It does not establish a universally safe
single-chart solver or superiority over Lizzy's specialized exact routes.

Across **27 fixed cases**, the condition-restart experiment passed **26**, with
one expected refusal at the default closure-dimension cap. The worst passing
strict operator error was **1.63e-11**, below the common **1e-6** target. Of the
25 cases where both it and production succeeded, emitted CX decreased in **13**,
tied in **12**, and increased in **none**. One additional case succeeded under
the shared work budget where production exhausted that budget.

### Representative actual native CX counts

| Case | Production WN | Single-product WN | Condition-restart WN | Existing specialized candidate |
|---|---:|---:|---:|---:|
| Short TFIM3, t=0.003 | 32 | 32 | 32 | Orthogonal: 28* |
| Static TFIM3, t=1 | 256 | 32 | 32 | Orthogonal: 28* |
| Static TFIM3, t=5 | 1,024 | 32 | 32 | Orthogonal: 28* |
| Static TFIM3, t=20 | Work cap | 32 | 32 | Orthogonal: 28 |
| Driven TFIM3 | 256 | 32 | 32 | Not tested as a static candidate |
| Static SU4 | 72 | 18 | 18 | Pair kernel: 6 |
| Driven SU4 | 72 | 18 | 18 | Not tested as a static candidate |
| SU8, explicitly raised dimension cap | 324 | 162 | 162 | Orthogonal unsupported |
| Encoded driven spin | 4 | 4 | 4 | Not tested as a static candidate |

All numbers count gates in a concrete native-ladder or eligible native-frame
artifact, selected by the same policy for every row. They are not analytical
block quotes. No pytket results or dense-reference phase corrections enter these
counts. These results are **not a comparison against the complete automatic
router or every available driven/free-fermion solver**.

\* The existing orthogonal solver returns these targets up to global sign:
strict error is approximately 2, while phase-aligned error is approximately
1e-15. Its PASS uses the established phase-insensitive contract. Every numerical
WN row must pass the **strict** phase-preserving norm. Both norms and the
contract are retained in the raw results.

The t=20 production failure is at the experiment's shared **20,000 RHS-call
budget**, not its usual 100,000 default. It is not evidence that production can
never compile that case. Its compact result has strict error 1.63e-11.

## What changed, and what did not

Three policies were tested:

1. **Production:** existing L1 noncentral-coordinate radius of 0.5, plus sampled
   condition checks and bisection/restarts.
2. **Single-product:** no fixed angle cap; one coordinate product per component
   and pulse interval, with failure if sampled condition exceeds 100.
3. **Condition-restart:** no fixed angle cap; retain bisection and append new
   factors when the sampled condition guard rejects an interval.

The experimental helper reuses the exact same Pauli closure and adjoint Jacobian.
A scoped in-process substitution into the existing high-level API preserves
commuting-component splitting, static commuting shortcuts, explicit pulse
boundaries, phase handling, full-closure factor order and global work budgets.
Production source files are unchanged. Unlike production, the helper also checks
the accepted ODE mesh; thus the comparison removes the radius guard while adding
these sampled checks, rather than weakening every safeguard.

Each factor order remains the deterministic production weight/lexicographic
order. There is no order search, dense target fitting, endpoint recompression,
atlas switching, or reference-dependent route selection in this experiment.

Both static t=1 and driven TFIM3 reduce from 8 coordinate products / 120 Pauli
rotations to 1 product / 15 rotations. Static t=5 reduces from 32 products to 1.
That is the source of the large gate reduction, not altered target accuracy.

## Coverage and failures remain visible

The test set contains the original 14 synthesis cases; the previously refused
63-dimensional SU8 case with an explicit separate cap of 64; reverse-time TFIM;
t=20 TFIM; a driven TFIM with a large identity/global-phase term; two Euler-chart
singularity crossings; a discontinuous pulse with an explicit breakpoint; and
six seeded static/driven two- and three-qubit cases (seed 20260925).

| Policy | PASS | Closure cap | Compilation refusal |
|---|---:|---:|---:|
| Production | 25 | 1 | 1 (shared work budget at t=20) |
| Single-product | 25 | 1 | 1 (sampled Euler singularity) |
| Condition-restart | 26 | 1 | 0 |

The ordinary XYZ-coordinate pure-Y evolution intentionally reaches a coordinate
singularity. Single-product mode refuses; condition-restart mode succeeds with
four products. The Hamiltonian is easy to synthesize directly; this is a test of
coordinate robustness with all three controls deliberately declared, not a
useful application benchmark.

The fast pure-Y version also crosses an exact singularity but its sampled
condition checks miss that crossing. Its emitted endpoint is nevertheless
correct for this special path. Therefore **PASS does not certify that a chart
was nonsingular continuously**. Removing the L1 guard is not, by itself, a
production-quality coordinate atlas. Local solver tolerances likewise are not
a global operator-error certificate.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m experiments.wei_norman_compact_bench
```

Use `--json` for all settings, cases, refusals, errors, timings and version
information, or repeat `--case NAME` to select cases. The saved
[results](wei_norman_compact_results.json) contain 99 rows, including the named
existing exact candidates. Timings are single runs with ordinary caches, not
controlled speed benchmarks. Dense references are exclusively small-instance
validation oracles (at most three qubits); synthesis never uses them.

Production revision: `f87685cbdccc0e3168c1f8c38be0324d7455c1ed`. Pre-existing
unrelated working-tree changes were retained. The isolated solver and harness
are in [wei_norman_compact.py](wei_norman_compact.py) and
[wei_norman_compact_bench.py](wei_norman_compact_bench.py).

Validation: **486 tests passed** on the working tree, including 39 new solver
and comparison-harness tests. Ruff passed for all four new Python files. The
full suite produced two existing SciPy sparse-efficiency warnings; no failures.

## Implication

The earlier poor WN counts were not a fair assessment of compact Wei–Norman
synthesis. There is a real, reproducible improvement to pursue in chart policy.
Nevertheless, the existing orthogonal TFIM and static pair-kernel candidates
remain cheaper under this common native emitter. No molecular-chemistry or
large-system advantage has been demonstrated by these tests.
