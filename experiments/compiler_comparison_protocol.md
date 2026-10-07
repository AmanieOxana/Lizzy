# Matched Clifford+T compiler comparison

Corpus and accuracy protocol fixed on 2026-10-07 before the comparative run. This replaces stale
CX-only rankings as the current cross-compiler experiment. Earlier records remain
historical; the five-case BDI gauge experiment remains an internal ablation.

The expanded run retains the targets, times, tolerances and existing methods'
settings. It adds the public automatic exact selector, BQSKit and a separate
Hamiltonian-aware product-formula track with the bounded policies declared below.
The repaired six-method measurements are preserved in
[compiler_comparison_separate_methods.json](compiler_comparison_separate_methods.json).
The original pre-repair measurements are preserved in
[compiler_comparison_before_fixes.json](compiler_comparison_before_fixes.json).
This is a disclosed post-diagnosis repair, not a retrospectively successful
unmodified-upstream run.

The shared-frame refresh changes only Lizzy's public automatic selector after
the joint T/CX ablation; it is a disclosed implementation update, not a new
preregistered comparison. Its previous complete 216-row record is preserved in
[compiler_comparison_before_shared_frames.json](compiler_comparison_before_shared_frames.json).
Only the 24 auto rows (12 targets, two tolerances) are remeasured. The other
192 external and diagnostic rows retain their measurements, settings and
per-run provenance. The merged report identifies each row's measurement run.

## Scope and corpus

Twelve fixed Pauli Hamiltonians, two to four qubits, all evolved for t=0.7.
Seeded inputs use NumPy default_rng/model seed 20261007; exact coefficients are
saved in the report. Cases are:

1. Commuting Z, encoded SU(2), an anticommuting star, generic SU(4): two qubits.
2. TFIM, TFXY, Heisenberg, generic SU(8): three qubits.
3. TFIM, TFXY, all-to-all Heisenberg, cached genuine H2/JW: four qubits.

Generic controls contain every nonidentity Pauli, with seeded Gaussian
coefficients divided by the square root of the number of terms. These are
generic Hamiltonian-evolution targets, **not Haar-random unitaries**. The H2
snapshot comes from HamLib `chemistry/electronic/standard/H2.hdf5`, dataset
`ham_molec-4`, via OpenFermion JW in alpha-then-beta order. The scalar energy
shift is removed for every method. Geometry metadata is unavailable. Store its
coefficients and source fingerprint so reproduction needs no download/cache.

This is a bounded, deliberately heterogeneous corpus, not a representative
application distribution or a scaling benchmark. No case is removed because a
compiler fails or is unsupported.

## Methods and resources

- Lizzy automatic exact selection: public `synthesize(method="auto", objective="t")`.
  It compares whole-H BDI reference, legal-nullspace BDI, Givens and eligible
  OpenFermion Gaussian circuits at the same total rotation budget epsilon/2.
  First retain the previous selector's global lowest-T ladder artifact, with
  its original stable BDI-first ties. Then compile bounded native shared-frame
  alternatives for the complete logical candidates. Select by actual `(T, CX)`
  among artifacts that increase neither count relative to that retained
  reference; ties in both counts remain stable. This is a no-regression policy
  over the explored candidates, not a globally optimal or weighted tradeoff.
  Frame expansion is skipped above eight qubits or 256 logical rotations;
  the existing ladder candidates remain available. All emitted alternatives
  use the same total rotation budget; no component is priced independently.
  The wider BDI gauge enumeration remains experimental and is not added to
  this production portfolio. The existing estimated-T nullspace candidate is
  unchanged, as are the paper BDI decomposition and recursion order.
  The benchmark checks the **delivered artifact**, without reselecting an
  alternative after seeing the dense target. Separate BDI/Givens/Wei--Norman
  columns remain internal diagnostics, not three independent compiler brands.
  OpenFermion is an integrated dependency, not an external compiler column.
  The production eight-mode Gaussian delegation cap does not activate on this
  two-to-four-qubit corpus. Gaussian input recognition uses the supplied JW
  ordering, not model labels or benchmark target matrices.
- Lizzy BDI: current reference plus legal-nullspace optimized candidate.
- Lizzy Givens: current supported exact route. A failed horizontal embedding
  now uses the existing verified full-algebra mapper; Givens elimination and its
  low-weight ordering are unchanged. Eligibility and resource limits are retained.
- Lizzy Wei--Norman: existing numerical logical synthesis, per-component
  dimension cap 32, segment cap 128, RHS cap 20000, rtol 1e-10, atol 1e-12,
  max_step 0.05.
- FlagSynth SDM: official implementation at commit
  `f117326f7bf9c847da495993caadd3bcd767bd9d`. Two scoped compatibility
  bridges fix mismatched API calls: `selective_demux` becomes `use_sdm`, and
  `zyz_rotation_angles` requests `return_global_phase=True`. Neither changes the
  decomposition algorithm; all module bindings are restored after the call.
  Unmodified upstream HEAD does not run on these installed dependencies.
  The current baseline additionally has **local robustness repairs**: use a
  verified complex Schur basis when the original demultiplexer's eigenvectors
  are not unitary, and retain zero rotations through PennyLane's trainable-angle
  path so the upstream symmetrization receives its assumed gate layout. Both
  demultiplexed blocks must reconstruct; inputs are never perturbed or projected.
  Candidate metadata records these repairs and their activation. Plots label
  the result **patched SDM**, not unmodified FlagSynth. Any remaining failures
  stay visible.
- Qiskit QSD: upstream `qs_decomposition` (block-ZXZ in the measured Qiskit 2.5),
  followed by exact transpilation at levels 0 and 3, seed 0.
- pytket: upstream exact two-/three-qubit synthesis; unsupported larger widths
  remain explicitly unsupported, not silently replaced with another compiler.
  Both the base synthesis and full-peephole variant are tested, without wire swaps.
- BQSKit 1.2.1: independent numerical unitary synthesis, all-to-all CNOT+U3,
  optimization level 1, seed 0, fixed internal `synthesis_epsilon=1e-14`,
  one worker and at most 120 seconds per attempt. This experiment caps dense
  search at three qubits: larger widths are **CAP**, not a claim of unsupported
  mathematical inputs. Internal Hilbert--Schmidt convergence is not our spectral
  norm contract; the same independent checks still decide acceptance.
- Qiskit product formulas: an **H,t-aware method family**, not another
  compiler brand. Suzuki orders 2 and 4 each try repetitions 1,2,4,8,16,32,64,128,
  with a cap of 4096 expanded Pauli rotations. Select the first default-lowered
  formula per order passing the decomposition check at epsilon/2; compare
  ordinary Qiskit lowering and Rustiq lowering of that identical formula, each
  with exact optimization levels 0 and 3, seed 0. This retains the previous
  Qiskit baseline's best-of-default/Rustiq policy; Rustiq is not a new column.
  Rustiq preserves order, includes the final Clifford and preserves phase.
  The first-passing step check uses level zero; all final variants are checked
  independently. Record all search
  attempts, errors and caps. No later-step or per-target precision tuning occurs;
  first passing is a bounded heuristic, not a T-optimal repetition search.

Every artifact is ancilla-free and has unrestricted connectivity. Matrix
compilers receive the same dense U=exp(-itH); Lizzy and the product-formula track
receive Pauli H,t. This
deliberately tests output resources on identical targets, **not equal classical
input workloads**. Target construction, decomposition, T compilation and dense
verification timings are recorded separately and not ranked as equivalent work.
No QROM, phase-gradient resource state, chemistry-specific fermionic compiler,
or controlled-U synthesis is included. Such comparisons require separate tracks.

## Accuracy, costs and selection

Primary final operator-norm threshold is 1e-6; 1e-4 is a predeclared sensitivity
run. One common phase convention applies to EVERY method: align using the phase
of trace(U_target† U_output), then take spectral norm. Strict error is reported
separately. Thus no route receives a special phase exemption when scoring wins.

Each arbitrary-angle candidate first passes a decomposition check at epsilon/2.
All candidates then use the SAME compile_native_clifford_t backend, pygridsynth
settings and total rotation budget epsilon/2. The final artifact must again
pass the actual target-unitary check at epsilon. Local numeric error estimates
do not replace that check and are not formal interval certificates.
For public Lizzy auto, compilation happens inside the selector via the same
backend; its already selected output is checked without recompilation. Its
recorded rotation budget must match epsilon/2 exactly.

The common backend first recognizes affordable Clifford/T-angle rotations,
charges their snapping error, then allocates its remaining half-budget only to
generic rotations (the other half is reserved for phase bookkeeping). Zero or
Clifford rotations therefore do not impose artificial precision slots on a
compiler that happens to represent fixed gates as Rz. All pygridsynth calls use
seed 0. The recorded run uses PYTHONHASHSEED=0 and one BLAS/OpenMP thread; runtime
environment and source fingerprints are saved in the report.

No dense target is used to tune Lizzy's algebraic decomposition. The common
dense checks only accept/reject outputs. No method gets a compiler-specific
rotation-precision search. Within an explicitly listed upstream candidate
portfolio, select lowest actual T count among passing artifacts; retain all
candidate costs and failures. Public Lizzy auto additionally applies the
no-regression T/CX policy declared above. An invalid cheaper result cannot win.
Reported T and CX counts always describe the same selected artifact, not
independently selected T-minimal and CX-minimal circuits.

Report actual T+Tdagger count, actual CX count, input Rz occurrences, ancillas,
versions, settings, errors, provenance and an emitted-QASM digest. T-depth, when
reported, means the dependency-constrained T-layer count of the emitted sequence
with Clifford gates assigned zero depth, not a globally optimized depth.

Missing dependencies, unsupported inputs, resource caps and failures remain
distinct. Pairwise wins/ties and ratios use only explicitly reported common
passing coverage. Do not pool incomplete rows into a global ranking, replace
missing counts with zero, or count internal BDI fallback as an external victory.
