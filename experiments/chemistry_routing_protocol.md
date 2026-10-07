# Chemistry routing: bounded go/no-go protocol

Frozen before collecting route-cost labels on 2026-09-21. This is an exploratory
falsification experiment, not a claim of novelty, a confirmatory statistical study,
or a new synthesis method. Production compiler behavior is unchanged.

## Question

Can an already available structural quantity improve a cheap choice among existing
chemical evolution representations, beyond simple problem-size/Pauli statistics?
The one additional feature is the normalized mean pairwise Givens count between
the orbital frames of an upstream double factorization:

`mean_{i != j} givens(U_i† U_j) / [norb * (norb - 1) / 2]`.

Precisely, use `len(ffsim.linalg.givens_decomposition(matrix, tol=1e-10)[0])`,
averaged over ordered distinct frame pairs. Factorize each molecule once with
upstream Cholesky double factorization at tolerance `1e-8`, without optimization
or factor truncation beyond that tolerance. Use that identical retained
factorization for the feature and every DF candidate; record a fingerprint of
its matrices and orbital rotations. A constant/saturated training feature is a
feature no-go, not permission to substitute an angle norm after seeing outcomes.

This measures frame-change sparsity, not dynamical Lie algebra dimension. JW and
BK are Clifford-related encodings: DLA, anticommutation, symplectic span and Gram
ranks alone cannot distinguish their circuit costs. No novelty is claimed for
these invariants, Givens decompositions, or pairwise frame-distance heuristics.

## Fixed corpus and accuracy task

Use existing cached HamLib molecular tensors only; no new molecules, geometry
search, coefficient fitting, downloaded datasets or synthetic favorable examples.

| Instance | Qubits | Assumed benchmark `(n_alpha,n_beta)` | Role |
|---|---:|---|---|
| H2-4 | 4 | (1,1) | training |
| LiH-8 | 8 | (2,2) | training |
| BH-10 | 10 | (3,3) | molecule-family holdout |
| LiH-12 | 12 | (2,2) | size holdout, **not** independent molecular family |

HamLib dataset attributes do not establish these electron sectors. They are
explicit benchmark choices, not verified physical ground-state sectors.

Times: `0.25` and `1.0`. Step grid: exactly `1, 2, 3`. No relaxation if it fails.
For each task, use one occupation-basis reference determinant (the lowest orbitals
occupied per spin, conventionally called HF here) and three fixed seeded random
states in the declared electron sector. They are validation probes, not a proof
for all states. The reference is independent sparse Schrödinger evolution.
The seeded states are generated once per molecule and reused across times,
representations and step counts; their seed is recorded in the raw data.

Acceptance: maximum `1 - |<reference|output>|^2` over those four states <= `1e-3`.
Inspect normalization and any emission mismatch separately. No global accuracy
certificate follows from these samples. Operator norms and state infidelity must
not be interchanged in the report.

All molecular modes use alpha-then-beta ordering. JW and BK encode the **same Fock
states**, not identical raw qubit vectors. Encoded-state preparation/conversion is
excluded explicitly: this experiment chooses an encoding before simulation, not
a free mid-circuit basis change. DF and JW use the same occupation convention.

## Existing routes, strong emission baselines

- JW symmetric second-order Pauli formula.
- BK symmetric second-order Pauli formula.
- Unmodified ffsim double-factorized symmetric second-order formula, original
  upstream factor order, no optional reordering or Coulomb pruning.

The Pauli routes retain each existing adapter's deterministic canonical term order.
They therefore mean **JW plus canonical ordering** and **BK plus canonical ordering**,
not an isolated comparison of encoding effects: different orders can change the
finite-step Trotter error. Reference Hamiltonian actions must agree under the
explicit JW/BK state-encoding permutation. This check does not assume their
finite-step formulas agree. Check the ffsim molecular reference convention too.

For each Pauli formula, evaluate concrete available upstream artifacts: Qiskit
level-3 default and Rustiq (preserving order), and pytket direct/decomposed
GreedyPauliSimp and FullPeepholeOptimise. Do not silently substitute logical block
estimates for actual CX counts. Unsupported backends/failures stay visible. Use
ffsim's recommended PRE_INIT pipeline and Qiskit level 3 for DF. Record factor
fingerprints so numerical gauge variation is visible.
Each backend has a hard 20-second budget including worker startup. A timeout is
reported and not retried with a looser budget after inspecting route costs.

This is at most 4 instances × 2 times × 3 routes × 3 step counts = 72 formula
candidates, with a bounded emission portfolio per candidate. No new synthesis
algorithm is introduced. Each route's label is the minimum actual emitted CX among
its passing grid candidates. Step choice is a validation oracle for **all** routes,
not a deployable step-count predictor and not a minimal-step claim beyond this grid.

## Stop gate before fitting any selector

Report every task and route, including missing/capped/failed routes. Primary
comparison uses only tasks on which all three routes pass within the grid;
report that common coverage prominently, never silently drop failures.

At least 6 of 8 tasks must have common feasible coverage, with all four training
tasks and both holdout instances represented. Otherwise stop as **inconclusive under the fixed budget**; do not
increase steps, change states or relax the threshold in this experiment.

On common coverage, let the best single fixed route minimize geometric mean CX,
and let the per-task oracle take the lowest of all three route costs. Oracle
saving is `1 - geometric_mean(oracle_CX / best_fixed_CX)`. If any CX is zero,
use `max(1,CX)` consistently for ratios and disclose it.

If oracle saving <= 10%, or if fewer than two routes are strictly cheapest on any
task, stop as **no-go for a meaningful CX-reducing selector on this corpus/grid**.
No cheap feature can exceed that oracle headroom on the common-coverage subset. This does not rule out reductions
in classical compilation time or benefits outside the tested domain.

## One feature ablation, only if the gate passes

Training instances: H2-4 and LiH-8. Held-out instances are fixed above. No label
from BH-10 or LiH-12 may determine a threshold, feature, tree depth or tie-break.

Use a deterministic depth-one decision stump. Candidate split thresholds are
midpoints of distinct training feature values. Each leaf chooses the route with
minimum training mean log-CX; constant trees are allowed. Minimize training mean
log regret against the oracle. Prefer the constant tree on ties, then the earlier
listed feature/threshold/route. Routes are ordered `jw`, `bk`, `df`.

Baseline features, in fixed order: evolution time, qubit count, JW term count,
JW mean Pauli weight, DF factor count. The augmented selector adds **only** the
normalized pairwise Givens feature last. The ablation is the baseline selector
refitted on exactly the same training rows with that one feature removed.

Report held-out predictions, actual CX, regret to the oracle, and the best
training-selected constant-route baseline. Go only if the augmented rule saves
at least 10% geometric-mean CX against **both** the ablated and constant rules,
introduces no additional accuracy/coverage failures, and improves at least one
BH family-heldout task. Otherwise no-go for this particular predictive hypothesis.
Any held-out task outside common coverage is explicitly reported as missing
coverage, not silently removed from the held-out result.

Even a go is only a small feasibility signal: there are three molecular families,
one genuinely held-out family, correlated times and active spaces, sampled errors,
and no independent replication. A publishable novelty claim would still require
stronger evidence and a literature distinction.

## Deliverables and boundary

An experiment script, raw JSON including versions/settings/timings/failure reasons,
and a short honest report. Do not change production routing, add a synthesis
method, install dependencies, push a commit, or move the goalposts after results.

## Post-collection audit note (not a protocol replacement)

The implementation enforced normalization error <= 1e-8 and logical/emission
infidelity <= 1e-9 from the start; these numerical values were not written in the
original prose above. The primary analysis preserves those checks and returns
INCONCLUSIVE_COVERAGE. The accompanying report explicitly labels a subsequently
chosen target-accuracy-only sensitivity; it does not authorize fitting a selector,
add candidates, or override that primary verdict. This was an internal predeclared
protocol, not an externally preregistered confirmatory study.
