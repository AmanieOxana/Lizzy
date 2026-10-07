# Which compiler comparisons are meaningful?

[README](../README.md) · [Measured comparison](compiler_comparison.md) ·
[Frozen protocol](../experiments/compiler_comparison_protocol.md)

Reviewed 2026-10-07 against official implementations. The question is not how
many compiler names fit in a plot. It is whether they produce a circuit for the
**same full unitary**, at the same accuracy, with comparable resources.

## Included, with distinct input contracts

| Track | Implementations | What is held equal? |
| --- | --- | --- |
| Hamiltonian-aware synthesis | Lizzy automatic exact selection; Qiskit Suzuki formulas with ordinary/Rustiq lowering | Pauli H, time, full-unitary error and common Clifford+T backend |
| Dense-unitary synthesis | FlagSynth SDM, Qiskit QSD/block-ZXZ, pytket unitary boxes, BQSKit numerical search | Target U, full-unitary error and common Clifford+T backend; not equivalent classical input workloads |
| Internal ablations | BDI, Givens, static Wei–Norman separately | Explain Lizzy's choices; not independent competing compiler brands |

All measured tracks are ancilla-free and all-to-all. Numerical search caps are
reported as caps, not losses or claims that the method cannot express a target.
The public automatic result is measured as delivered, not chosen afterwards
from whichever internal method happens to pass a dense benchmark check.

## The main missing scaling comparison

**BQSKit's dense-unitary mode is a small-target control, not the main scalable
competitor for Lizzy.** Its addition does not resolve the scaling question.

[F3C](https://github.com/QuantumComputingLab/f3c) and
[F3C++](https://github.com/QuantumComputingLab/f3cpp) are more directly relevant
to the free-fermion spin examples: they compile TFIM/TFXY/XY and related
time-evolution circuits using algebraic compression, without requiring the
full Hilbert-space matrix. F3C++ supplies OpenQASM output through QCLAB++.
The inspected repository and Git history contain no evidence that Lizzy had
already benchmarked or excluded them.

The crucial contract is **exact circuit compression, not automatically exact
continuous-time evolution**. If the input is a finite-step product formula,
its Hamiltonian simulation error remains after compression. A fair comparison
must independently match that error and then lower both outputs to the same
gate set. This is a priority future scaling track on the overlapping spin-model
families, not a generic Pauli-Hamiltonian or molecular-chemistry claim.

**OpenFermion is an existing dependency to integrate, not a new competitor
column.** Its Gaussian-unitary synthesis is distinct from the interacting-molecule
ffsim interface already in Lizzy.
[`QuadraticHamiltonian.diagonalizing_bogoliubov_transform`](https://quantumai.google/reference/python/openfermion/ops/QuadraticHamiltonian)
provides mode energies and the number-preserving or pairing transformation.
[`bogoliubov_transform`](https://quantumai.google/reference/python/openfermion/circuits/bogoliubov_transform)
emits its full quantum circuit when called with `initial_state=None`; it must
not be replaced with the cheaper state-preparation specialization. Basis
circuits, diagonal mode phases and inverse basis circuits provide quadratic
static evolution, as illustrated by the
[official tutorial](https://quantumai.google/openfermion/tutorials/circuits_1_basis_change).
These routines use single-particle matrices rather than dense Hilbert-space
unitaries, and include non-number-conserving pairing needed by TFIM/TFXY.
Lizzy delegates this construction to OpenFermion through its Gaussian route;
it does not claim that upstream algorithm as a Lizzy invention. Any comparison
with BDI/Givens is an internal routing check, not another compiler brand.

The external scaling priority is therefore F3C++ for circuit compression/static
and driven evolution. Conversion, phase, discretization and emitted T counts
still need independent validation; this scope review does not claim that
comparison has been performed.

PHOENIX belongs to a different comparison: general Pauli-network compilation.
The current [official repository](https://github.com/iqubit-org/phoenix) also
provides the newer Symphony/PHOENIX++ implementation alongside the older
support-grouped compiler. The historical Lizzy result below should not be
transferred to that new implementation. A rerun must pin the algorithm, check
ordering and include the terminal Clifford; observable-only absorption is not
the full-unitary contract. None of these new comparisons has been measured here.

### Why BQSKit?

[BQSKit's standard compiler](https://github.com/BQSKit/bqskit/blob/main/bqskit/compiler/compile.py)
accepts a unitary matrix and uses numerical circuit search, adding an independent
strategy beyond another QSD implementation. This experiment fixes version 1.2.1,
seed, gate set, precision and resource limits. Its internal Hilbert–Schmidt
objective is not the acceptance metric: we independently check spectral-norm
error before and after T compilation.

### Why product formulas and Rustiq?

**This is a refreshed Qiskit baseline, not a newly discovered compiler.** The
earlier `lizzy.compare.qiskit_best` already takes the better of default and
Rustiq lowering. The README at commit `282cc8b` records Rustiq as worse on the
spin examples but useful on chemistry, with 57,620 versus 2,151 CX on the
Heisenberg grid. The later [chemistry record](../experiments/chemistry_routing_results.json)
has 22 common passing default/Rustiq pairs, all with lower CX for Rustiq, plus
failed or timed-out pairs that must not be scored as wins. Neither observation
is a universal ranking, nor an actual T-count comparison.

The current matrix-only FT record omitted this existing H-aware Qiskit baseline.
We keep ordinary/Rustiq choices inside **one Qiskit product-formula result** and
refresh the accuracy/cost contract; we do not add another Rustiq column.
[Qiskit SuzukiTrotter](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.synthesis.SuzukiTrotter)
constructs approximate evolution circuits. Its
[Rustiq synthesis integration](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler_synthesis_plugins)
can lower their Pauli networks. Rustiq is a lowering option, **not a second
compiler brand or an exact replacement for Trotterization**.

The declared search tests fixed orders and repetitions, checks formula error
against the original evolution, and compares ordinary/Rustiq lowering of the
same accepted formulas. Noncommuting order and the final Clifford are retained.
This bounded first-passing search is not a global T-count optimum.

## Useful software that needs another track

| Candidate | Why it is not another column here |
| --- | --- |
| [ffsim](https://qiskit-community.github.io/ffsim/api/stubs/ffsim.qiskit.SimulateTrotterDoubleFactorizedJW.html) | Relevant chemistry baseline, but needs matched orbital tensors, encoding, factorization error and full-space evolution; not a general Pauli compiler. |
| [PyZX](https://pyzx.readthedocs.io/en/stable/notebooks/simplify.html) | Optimizes an existing circuit. A fair follow-up applies the same post-processing to every compiler's output. |
| [BQSKit-FT](https://github.com/BQSKit/bqskit-ft) | Relevant end-to-end fault-tolerant compiler; allowing its own T synthesis changes this controlled common-backend question. |
| [Qualtran GQSP/qubitization](https://qualtran.readthedocs.io/en/latest/bloqs/hamiltonian_simulation/hamiltonian_simulation_by_gqsp.html) | Requires block encoding, selection/signal ancillas and preparation accounting; compare under an explicitly ancilla-aware contract. |
| State/MPS preparation | State fidelity does not verify action on arbitrary input states. |
| Classical simulation | Computes dynamics but does not necessarily produce the required quantum gate artifact. |

[Cirq QSD](https://quantumai.google/reference/python/cirq/quantum_shannon_decomposition)
is a valid additional matrix baseline, but lower priority than numerical search
and an H-aware competitor. [QFAST](https://github.com/BQSKit/qfast) is archived
and superseded by BQSKit; counting both would overstate independent coverage.
[Kernpiler](https://arxiv.org/abs/2504.07214) is conceptually relevant, but this
review did not establish a maintained official runnable distribution. A local
reimplementation should not be labelled the authors' software.

The earlier search also considered Paulihedral, Tetris, PauliOpt and PHOENIX.
The archived README records noncommuting reordering in the tested PauliOpt
configuration, unverified Paulihedral/Tetris outputs, and a separate
matched-accuracy PHOENIX run that was costlier than Lizzy on its tested spin
examples. These are different outcomes, not a blanket exclusion. Reordered
formulas can be compared after checking their own evolution error; unverified
adapters need repair before gate counts are evidence. The old PHOENIX numbers
are historical CX evidence, not a new FT result. The current work neither
reimplements these tools nor silently reuses fixed-step counts under a new
accuracy contract.

Consequently, neither success on selected spin targets nor the single H₂ row
establishes superiority over all Hamiltonian simulation or chemistry compilers.
