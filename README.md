# Lizzy

Unitary synthesis for Pauli Hamiltonians, routed by what the dynamical Lie algebra
actually is. Classify the DLA with [PauLie](https://github.com/QPauLie/PauLie), price
every branch the classification allows, and let the cheapest circuit win.

Named in honor of [Elizabeth Meckes](https://en.wikipedia.org/wiki/Elizabeth_Meckes)
(1980–2020), mathematician of the classical compact groups — the exact branch of this
compiler lives on them.

```python
from lizzy.hamiltonian import model
from lizzy.synthesize import synthesize

result = synthesize(model("tfim", 100), time=10.0, error=1e-3)
result.routes           # ['exact'] — fixed depth, whatever the time
result.two_qubit_gates  # 6,616 at n=100, in seconds
```

## The route

```
H, t, budget (or a fixed step count)
 ├─ classify the DLA (PauLie), split into commuting summands    [exact]
 └─ per summand, price every candidate and keep the cheapest:
      exact      so(m) → one orthogonal matrix, reduced to adjacent Givens
                 rotations along the Majorana line         [depth flat in t]
      hybrid     free subalgebra compiled exactly inside each second-order
                 step, as one more summand
      clusters   S2 over commuting clusters
      kernels    S2 over two-qubit kernels, fields folded in (3 CNOTs each)
      terms      S2 over the terms as given, ungrouped
      chain      the requested Suzuki order, sized by the chain bound
      (sampled)  qDRIFT on the small terms, only with randomized=True
```

Each candidate is priced from one built step, under **every available emission**, and
the cheapest circuit wins — exactness is not a priority order. The builtin emission
charges CNOT ladders, merging runs that fit on one qubit pair into canonical
three-CNOT blocks. With pytket installed, `lizzy.emit` adds a second: conjugating
rotations into a shared Clifford frame, which wins once mean Pauli weight passes four
(below that the call is skipped, since a ladder is already near-optimal there).

Pricing emission inside the routing decision is what makes the `terms` candidate worth
having: ungrouped it emits badly as ladders, but a shared frame would rather see the
sequence as given, and that combination takes the largest chemistry instances.

`symmetry.taper` removes one qubit per commuting conserved charge, rotating each onto
a single-qubit Pauli and fixing its eigenvalue
([Bravyi et al.](https://arxiv.org/abs/1701.08213)). It is a problem reduction rather
than a compilation choice — the result acts on fewer qubits and holds only the chosen
sector — so it is explicit, not automatic. On chemistry it pays twice: BH loses four of
ten qubits and mean Pauli weight drops from 4.8 to 3.8, halving the circuit (1 890 →
1 087 gates); LiH loses four of twelve (4 340 → 3 162). Every compiler benefits from
it, so the comparison below is untapered throughout.

Step counts come from the collected commutator bound
([Childs et al.](https://doi.org/10.1103/PhysRevX.11.011020), tight second-order
constants with cancellation between chains kept), so sizing needs no oracle;
`bench.calibrate` measures the remaining overshoot per instance where a dense
reference exists. Fully commuting Hamiltonians compile in one exact step. Every
two-qubit kernel decomposition is verified against its own 4×4 exponential at
synthesis time.

The sampled candidate uses qDRIFT
([Campbell](https://doi.org/10.1103/PhysRevLett.123.070503)), whose cost depends on
the coefficients rather than the term count. It is off by default and stays off:
its guarantee is on the averaged channel, not on the circuit you get, so a route
that wins on gate count can still miss the budget it was sized for.

## Measured

Fifteen HamLib instances, the same fixed-depth task for every compiler (two Suzuki-2
steps at t=1), two-qubit gates after each compiler's best effort — Qiskit at
`optimization_level=3`, pytket at the better of `GreedyPauliSimp` and
`FullPeepholeOptimise`. Reproduce with `python -m lizzy.compare`:

| HamLib instance | n | terms | Lizzy | Qiskit | +Rustiq | pytket |
|---|---|---|---|---|---|---|
| tfim 1D chain | 100 | 199 | **396** | 786 | 1 032 | 687 |
| tfim 1D ring | 100 | 200 | **400** | 794 | 1 326 | 771 |
| tfim 2D grid | 100 | 280 | **720** | 1 434 | 2 573 | 1 316 |
| tfim hex lattice | 48 | 111 | **284** | 496 | 865 | 467 |
| tfim 3D grid | 27 | 81 | **216** | 426 | 740 | 386 |
| heisenberg 1D chain | 100 | 397 | **888** | 1 179 | 4 117 | 1 179 |
| heisenberg 2D grid | 100 | 640 | **1 860** | 2 151 | 57 620 | 2 151 |
| heisenberg 2D torus | 100 | 700 | **2 100** | 2 391 | 70 884 | 2 391 |
| fermi-hubbard 1D, JW | 100 | 346 | **1 063** | 1 178 | 24 953 | 1 178 |
| fermi-hubbard 1D, BK | 100 | 346 | **1 714** | 4 594 | 43 104 | 2 866 |
| maxcut circulant | 100 | 200 | **323** | 1 594 | 588 | **323** |
| H2 molecule | 4 | 14 | **26** | 160 | 47 | 28 |
| BH molecule | 10 | 275 | **1 890** | 7 484 | 5 788 | 2 136 |
| LiH molecule, BK | 12 | 630 | **4 340** | 21 664 | 16 982 | 4 405 |
| LiH molecule, JW | 12 | 630 | **4 099** | 24 842 | 16 133 | 4 486 |

Fifteen of fifteen, by margins from a few percent on the largest chemistry instances
to 4.9x on maxcut.

**Mean Pauli weight, not algebra dimension, decides how much of that is this
compiler's own work.** With the pytket backend removed the score is 9/15, and the
split is sharp: every instance of weight ≤1.9 is won on the builtin emission alone
(tfim on all five lattices, heisenberg on all three, fermi-hubbard under
Jordan-Wigner), every instance of weight ≥2.0 is lost (maxcut 2.0, H2 2.6,
fermi-hubbard under Bravyi-Kitaev 2.8, BH 4.8, LiH 5.6-6.3). Algebra dimension does
not predict it — seven of the nine wins have exponential DLAs, and two of the six
losses are polynomial.

Weight two is where the exact tools end. The Givens route needs the whole algebra to
be `so(m)`; the kernels need every term to fit one qubit pair. Above that there is no
structural tool left and the builtin emission falls back to a ladder per rotation,
which is where a shared Clifford frame takes over.

At matched accuracy — eight qubits, every compiler given the fewest steps that reach
1e-3 against a dense reference — Lizzy wins all six model/time combinations measured,
1.4× to 4.1×, and the margin grows with evolution time on fast-forwardable families:
at t=8 the exact branch holds 143 gates against Trotter's 1 148.

**Not every compiler can be scored this way.** Checked against a dense reference on a
duplicate-free Trotter step, only pytket, Qiskit's `PauliEvolutionGate` and Lizzy
reproduce the sequence they are given (infidelity ≤ 2e-16). Paulihedral, Tetris,
PauliOpt and [PHOENIX](https://github.com/iqubit-org/phoenix) reorder non-commuting
terms — legitimate for a Trotter approximation, but the circuit then implements a
different unitary and its gate count is not comparable. Such compilers have to be
scored on accuracy instead. Scored that way, PHOENIX needs 238 gates where Lizzy needs
42 (tfim, t=1), 420 against 132 (heisenberg, t=1) and 3 213 against 1 881 (heisenberg,
t=8): it buys cheaper steps with more of them, because reordering costs Trotter
accuracy. Paulihedral and Tetris are absent because their output could not be verified
against a reference under any convention tried.

The compiler itself stays small: ~2 700 lines of Python across ten modules — for
scale, pytket's `GreedyPauliSimp` pass alone is ~2 500 lines of C++ — because the
algebra lives upstream in PauLie and kak-tools, and the routing is arithmetic over
priced candidates rather than machinery.

## Verification

`lizzy.dense` rebuilds circuits as 2^n matrices and compares them to `expm(-itH)`,
so every exactness claim is checked at the qubit level wherever size allows; the
test suite is built on it. Kernel decompositions self-verify at any width.

The reported gate count is block-aware: a run of consecutive rotations that fits on a
single qubit pair compiles as one canonical block of at most three CNOTs, and a wider
rotation pays its CNOT ladder. Emitting the circuits explicitly confirms the count
(1 440 reported, 1 440 emitted at tfim 2D).

```bash
pytest                   # dense checks, offline
python -m lizzy.bench    # per-instance route, cost, achieved error
python -m lizzy.compare  # the tables above (needs the compare extra)
```

## Install

```bash
pip install -e .
pip install -e '.[hamlib]'    # HamLib loading (h5py, openfermion)
pip install -e '.[compare]'   # benchmark against qiskit and pytket
pip install -e '.[test]'
```

Needs [kak-tools](https://github.com/QPauLie/kak-tools) with
[PR #1](https://github.com/QPauLie/kak-tools/pull/1) and PauLie with
[PR #232](https://github.com/QPauLie/PauLie/pull/232).

## Outlook

**Price more backends rather than write more emissions.** The last two rows of the
table were won this way: the routing tier had no structure to find in a 630-term
molecule, so the term-order candidate and the shared-frame emission decided them
between themselves. Emission is a tier with two entries today; nothing in the design
is limited to two. Rustiq (shipped inside Qiskit) is a third and beats the ladder on
high-weight sequences; a native frame-conjugated emission would be a fourth, and would
remove the pytket dependency for the rows that currently need it. Each backend is a
function from a rotation sequence to a gate count, each is verifiable against a dense
reference at small width, and `emission_cost` already takes a minimum — so adding one
is strictly monotone: it can only be chosen where it wins.

Optional imports are therefore not a weakness to engineer away. They are how a small
compiler stays best-in-class on instance families it was never specialised for. The
work is keeping the interface narrow — sequence in, verified circuit out — so a new
backend costs a function rather than an architecture.

**Close the gap at weight two.** The measurement above localises where this compiler
stops having its own answer, and two techniques sit squarely in it. A *parity network*
would synthesize a commuting diagonal Hamiltonian as one shared CNOT tree rather than a
ladder per term — maxcut is fully commuting, weight two, and still costs 400 gates here
against pytket's 323, which is exactly that gap. *Kernels on wider supports* would
extend the exact two-qubit KAK to three or four qubits; whether it pays is arithmetic,
since a general three-qubit unitary already costs around twenty CNOTs and only wins
where enough terms share the triple.

**Widen what counts as exactly compilable.** The exact route fires when the whole
algebra is `so(m)`, and `free_part` finds one exactly-compilable subset when it does
not. The chemistry literature generalizes this: split a Hamiltonian into
*fast-forwardable fragments*, each implementable for any time at a cost independent of
precision. Two structures are within reach and both belong to the routing tier, where
this compiler's knowledge already lives:

- **Double factorization** — affordable, and the more promising of the two. The
  two-electron tensor has low rank, so a molecular Hamiltonian decomposes into O(N)
  fragments that are each free-fermionic: each one a target for the Givens route
  rather than for a product formula. The decomposition is an O(N^6) classical
  eigendecomposition, routine in quantum chemistry at the sizes here, and implemented
  in [ffsim](https://github.com/qiskit-community/ffsim) and Qiskit with
  [symmetry-compressed variants](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00352).
  The obstacle is not cost but input: it needs the fermionic one- and two-body tensors,
  which a Pauli Hamiltonian only yields if its encoding is known. HamLib names the
  encoding in the instance key; in general it has to be supplied.
- **Free fermions in disguise** — cheap to rule out, infeasible to confirm.
  [Elman, Chapman and Flammia](https://arxiv.org/abs/2012.07857) show that a
  Hamiltonian whose frustration graph is (even-hole, claw)-free with a simplicial
  clique has a free-fermion solution even when no Jordan-Wigner transformation finds
  one, and the frustration graph is `anticommutation_matrix`, already built on every
  route. But recognizing even-hole-free graphs is O(n^9) at best after a long line of
  improvements from O(n^40); at 630 terms that is not a computation anyone runs.
  Claw-freeness alone costs O(n·d^3) and is a sound *necessary* screen — instant on
  these instances, and correctly separating tfim (claw-free, and indeed
  free-fermionic) from heisenberg and H2 (both clawed). It can therefore rule the
  structure out, never in.

Beyond that, in measured order of value:

- **Cluster count on dense instances.** Chemistry gives 17 commuting clusters where
  spin models give three, so a step pays 17 basis changes. Better colouring, or kernels
  on larger supports, attacks the term the formula actually spends.
- **Symmetry protection.** A no-op for models whose terms conserve the charges
  individually; untested where terms violate them — gauge theories, chemistry.
- **Dense bounds past ~20 qubits.** The collected commutator walk exhausts its budget
  on all-to-all models around n=24. Vectorizing it extends the certified regime, though
  the circuits out there run to 10^8 gates.

## References

- Kökcü et al., [Fixed depth Hamiltonian simulation via Cartan decomposition](https://doi.org/10.1103/PhysRevLett.129.070501) — the exact branch
- Wierichs et al., [Recursive Cartan decompositions for unitary synthesis](https://arxiv.org/abs/2503.19014) — what kak-tools implements
- Wiersema et al., [Classification of dynamical Lie algebras of 2-local spin systems](https://doi.org/10.1038/s41534-024-00900-2) — why polynomial DLAs are a chain phenomenon
- Childs et al., [Theory of Trotter error with commutator scaling](https://doi.org/10.1103/PhysRevX.11.011020) — the step counts
- Decker et al., [Kernpiler](https://arxiv.org/abs/2504.07214) — partial Trotterization, the kernels
- Kivlichan et al., [Quantum simulation with linear depth](https://doi.org/10.1103/PhysRevLett.120.110501) — the Givens emission
- Chen et al., [PHOENIX](https://arxiv.org/abs/2504.03529) — global Pauli-IR optimization, compared against
- Goubault de Brugière & Martiel, [Rustiq](https://arxiv.org/abs/2404.03280) — the Pauli-network synthesis Qiskit ships
- Elman, Chapman & Flammia, [Free fermions behind the disguise](https://arxiv.org/abs/2012.07857) — solvability from the frustration graph
- Sawaya et al., [HamLib](https://arxiv.org/abs/2306.13126) — the instances
- Meckes, [The Random Matrix Theory of the Classical Compact Groups](https://doi.org/10.1017/9781108303453) — the namesake
