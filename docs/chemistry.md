# Molecular chemistry

[Overview](../README.md) · [Synthesis](synthesis.md) · [Architecture](architecture.md) · [Benchmarks](benchmarks.md)

Lizzy's molecular path is an explicit adapter to established software, not a
replacement chemistry simulator or a demonstrated DLA-based speedup. Keep orbital
tensors when using this path: converting to Paulis discards the representation
that double factorization needs.

## Responsibility boundary

| Component | Owner |
|---|---|
| Fermionic operator algebra and Jordan–Wigner/Bravyi–Kitaev encodings | OpenFermion |
| Full quadratic/Gaussian evolution, including pairing | OpenFermion, integrated through `lizzy.gaussian` and static routing |
| Double factorization, number/Z conversion, orbital Givens decomposition and adjacent-frame merging | ffsim |
| DF-Trotter semantic gate | ffsim |
| Native circuit optimization and emitted CX count | Qiskit, initialized with `ffsim.qiskit.PRE_INIT` |
| Validated real spatial tensors, HamLib/layout bridges, provenance and opt-in policies | Lizzy |

Install `python -m pip install -e '.[chemistry]'` after the base installation in
the [installation guide](getting_started.md#install). Chemistry imports are lazy; the dependency-light
`MolecularHamiltonian` tensor container can be inspected without the simulation
stack. Its convention is real, spin-restricted spatial orbitals with a symmetric
one-body tensor and chemist-symmetric two-body tensor; see its class docstring for
the exact index convention.

OpenFermion is also used beyond tensor conversion: the static router can delegate
genuinely Jordan–Wigner quadratic Hamiltonians to its Bogoliubov circuit
construction. This handles free fermions, including pairing, not general
interacting molecular tensors. It uses the dependency already installed by
`chemistry`; it is not an extra compiler to benchmark against Lizzy. See
[quadratic evolution](synthesis.md#openfermion-quadratic-evolution). The ffsim
DF-Trotter path below remains the interacting molecular interface.

## Use the upstream baseline

```python
from lizzy.chemistry import synthesize_molecular_ffsim, to_pauli_openfermion
from lizzy.hamlib import fetch, load_molecular
from lizzy.synthesize import synthesize

path = fetch("chemistry/electronic/standard/LiH.zip")
molecule = load_molecular(path, "ham_molec-12")

df = synthesize_molecular_ffsim(molecule, time=1.0, steps=1)
print(df.two_qubit_gates, df.factorization_fingerprint)
assert df.routing_mode == "input"
assert df.factor_order_preserved
assert not df.accuracy_certified

# An explicit alternative representation, not an automatic DF-vs-Pauli selector:
paulis = to_pauli_openfermion(molecule, "jw", qubit_order="alpha-then-beta")
generic = synthesize(paulis, time=1.0, steps=2)
```

`factorize_molecular_ffsim` also exposes the upstream factorization without
building a circuit. `to_ffsim` converts the local tensor container or passes
through an existing upstream object. The short names `to_pauli`,
`fermion_operator` and `synthesize_molecular` are compatibility aliases; explicit
upstream-named APIs make the ownership clearer in new code.

The baseline preserves ffsim's factor order. Its default `formula_order=2` means
physical second-order Suzuki, mapped to ffsim's `order=1`; physical first order
maps to `order=0`, and positive even orders map to half that integer.

## Layout, phase and accuracy

ffsim uses **all alpha modes followed by all beta modes**. HamLib/OpenFermion
helpers default to **interleaved** alpha/beta modes. The APIs therefore have
different defaults:

| API | Default `qubit_order` |
|---|---|
| `synthesize_molecular_ffsim` | `"alpha-then-beta"` |
| `to_pauli_openfermion`, fermionic conversion helpers | `"interleaved"` |

Set the layout explicitly when comparing outputs. Requesting interleaved DF
output inserts a fermionic parity-conjugation network; it is not a free wire
relabeling. Counts include that network. Encoding conversion and state preparation
are separate tasks unless a benchmark explicitly includes them.

The current `to_pauli_openfermion` helper omits the scalar identity term in the
encoded operator. Its evolution therefore matches the full molecular Hamiltonian
only up to the corresponding global phase, even before other synthesis effects.
The ffsim tensor conversion retains the constant. Use phase-aligned comparisons
where appropriate; do not treat the Pauli conversion as phase-preserving input
for a controlled evolution.

`DoubleFactorizedResult.circuit` is the retained Qiskit artifact.
`two_qubit_gates` reads its current CX count, while `quoted_two_qubit_gates`
preserves the construction-time snapshot. `candidate_counts`, package versions,
`ordering` and `factorization_fingerprint` record how it was produced. Numerically
equivalent orbital gauges can change gate counts between factorization runs;
retain the fingerprint rather than treating one count as an invariant of a
molecule.

`two_body_tensor_max_abs_error` is a tensor-reconstruction diagnostic, **not** a
unitary-error bound. Finite-step Trotter error, factorization/truncation tolerances,
optional reordering and compiler numerics all matter. Consequently
`accuracy_certified` and `error_guaranteed` are false; no molecular path silently
replaces the generic router's bound-sized simulation.

## Optional policies are experiments

```python
trial = synthesize_molecular_ffsim(
    molecule, time=1.0, steps=1,
    frame_ordering="experimental-givens",
)
print(trial.candidate_counts, trial.factor_order_preserved)
```

This S2-only option proposes a nearest-neighbour/2-opt order using pairwise
orbital-frame Givens counts. It compiles both that order and the upstream order,
keeps the lower-CX artifact, and preserves the upstream order on ties. Reordering
noncommuting fragments changes the finite-step approximant. A lower count is
therefore **not** evidence of lower cost at equal accuracy. `"portfolio"` remains
a deprecated alias; explicit permutations are also accepted.

Other opt-ins include nonzero `coulomb_cutoff`, which prunes diagonal-Coulomb
coefficients, and upstream factorization controls such as `max_vecs` and
`factorization_optimize`. Their approximation sources are recorded. The default
does not apply Lizzy's Coulomb-pruning policy.

## What the structure experiment found

The bounded [chemistry routing report](../experiments/chemistry_routing_report.md)
compared JW, BK and ffsim DF for four instances at two times. It measured emitted
CX counts and shared physical-state accuracy probes, not just logical costs.

- The strict primary result was **inconclusive coverage: 2 of 8 tasks** passed
  every guard for every route. Larger DF artifacts failed an additional tight
  logical/emitted-equivalence check even when meeting the evolution target.
- A separately labelled post-hoc sensitivity admitted artifacts meeting the
  original sampled target. Even a hindsight-perfect route choice then saved only
  **2.33%** geometric-mean CX over always using BK within that measured portfolio.
- No structure-based selector or feature ablation was fitted. The additional
  Givens-distance feature's predictive value remains untested.

The probe target was sampled state infidelity, not a full operator-norm or
chemically accurate energy certificate. These results motivate stopping that
specific selector hypothesis, not a claim that structure can never help chemistry.
Full failures, timeouts, scope and reproduction commands remain in the report.

DLA recognition can identify supported special cases. It does not by itself
choose between representations with the same algebra but different Pauli weights,
shared parity networks or emitted costs. The production molecular API accordingly
remains an explicit upstream baseline, not a claimed Lizzy chemistry improvement.
