# Methods and upstream references

These methods predate Lizzy. Integration, routing and validation are not claims
to have invented a new decomposition. Consult the paper-to-code reviews for the
implemented scope rather than treating every cited paper as a fully implemented API.

## Algebraic and numerical synthesis

- Wierichs et al., [Recursive Cartan decompositions for unitary synthesis](https://arxiv.org/abs/2503.19014).
  Lizzy implements supported `so(m)` simulation with structure-derived mappings
  and recursive BDI via kak-tools, including reusable horizontal evolution when
  the input admits it. Givens remains an alternative. [Review](cartan_review.md).
- Kökcü et al., [Fixed depth Hamiltonian simulation via Cartan decomposition](https://doi.org/10.1103/PhysRevLett.129.070501).
  Background for fixed-depth Hamiltonian synthesis, not Lizzy's implemented variational algorithm.
- Qvarfort & Pikovski, [Solving quantum dynamics with a Lie-algebra decoupling method](https://doi.org/10.1103/PRXQuantum.6.010201).
  Wei–Norman product equations. [Review](wei_norman_review.md).
- Altafini, [On the generation of sequential unitary gates from continuous time Schrödinger equations driven by external fields](https://arxiv.org/abs/quant-ph/0203005).
  Coordinate Jacobians, gate parameters and local singularities.
- Martínez-Tibaduiza et al., [Symdyn](https://doi.org/10.1103/24r3-j9zy).
  Existing symbolic and numerical Wei–Norman automation.
- Blanes et al., [The Fer and Magnus expansions](https://personales.upv.es/serblaza/2011EncyclopediaFerMagnus.pdf).
  Equations 22–24 are the fourth-order schemes implemented here; see also
  [the convergence paper](https://doi.org/10.1088/0305-4470/31/1/023).
- Shende, Bullock & Markov, [Recognizing small-circuit structure in two-qubit operators](https://arxiv.org/abs/quant-ph/0308045).
  Canonical 0/1/2/3-CNOT class test used in analytical pair-block pricing.
- Childs et al., [Theory of Trotter error with commutator scaling](https://doi.org/10.1103/PhysRevX.11.011020).
  Product-formula error sizing.
- Decker et al., [Kernpiler](https://arxiv.org/abs/2504.07214).
  Partial Trotterization and small exact kernels.

## Representations and chemistry

- Aguilar et al., [arXiv:2408.00081](https://arxiv.org/abs/2408.00081): Pauli-algebra structure and Clifford equivalence.
- Bravyi et al., [Tapering off qubits to simulate fermionic Hamiltonians](https://arxiv.org/abs/1701.08213): explicit symmetry-sector reduction.
- Motta et al., [Low-rank representations for quantum simulation of electronic structure](https://arxiv.org/abs/1808.02625): double-factorized chemistry simulation.
- McClean et al., [OpenFermion](https://doi.org/10.1088/2058-9565/ab8ebc): fermionic algebra and encodings.
- [ffsim](https://github.com/qiskit-community/ffsim): molecular factorization and circuit construction.
- [PauLie](https://github.com/QPauLie/PauLie) and [kak-tools](https://github.com/QPauLie/kak-tools): classification and representation machinery.
- Sawaya et al., [HamLib](https://arxiv.org/abs/2306.13126): benchmark inputs.
- Necaise et al., [Distribution complexity of electronic-structure simulations](https://arxiv.org/abs/2606.20805): related DF-frame ordering work, distinct from Lizzy's inconclusive local-CX selector experiment.

## Comparison boundaries

[Parameter-optimal unitary synthesis with flag decompositions](https://arxiv.org/abs/2603.20376)
addresses generic-unitary and MPS synthesis. Lizzy does not implement its SDM or
phase-gradient circuits. A small parameter count in a supported Lie subgroup is
not equivalent to those generic-unitary capabilities or resource guarantees.

The SDM benchmark uses the official dependency with disclosed adapter repairs.
The [compiler scope review](compiler_scope.md) distinguishes measured small-unitary
controls, previous exclusions and the missing scalable specialist comparisons:

- [OpenFermion Bogoliubov circuits](https://quantumai.google/reference/python/openfermion/circuits/bogoliubov_transform): full Gaussian-unitary synthesis, distinct from state preparation and interacting chemistry Trotterization.
- [F3C++](https://github.com/QuantumComputingLab/f3cpp): algebraic free-fermion circuit compression; finite-step simulation error still matters.
- [PHOENIX/Symphony](https://github.com/iqubit-org/phoenix): Pauli-network compilation; earlier PHOENIX CX results do not establish performance of the newer implementation.
