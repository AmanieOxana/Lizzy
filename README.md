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
result.two_qubit_gates  # 27,208 at n=100, in seconds
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
      kernels    S2 over two-qubit kernels, fields folded in (≤3 CNOTs each)
      terms      S2 over the terms as given, ungrouped
      chain      the requested Suzuki order, sized by the chain bound
      (sampled)  qDRIFT on the small terms, only with randomized=True
```

Each candidate is priced from one built step, under **every available emission**, and
the cheapest circuit wins — exactness is not a priority order. The builtin emission
charges CNOT ladders, merging runs that fit on one qubit pair into a single canonical
block priced by its KAK class: three CNOTs, or two where a canonical parameter is
trivial, which is what an XY bond costs. With pytket installed, `lizzy.emit` adds a second: conjugating
rotations into a shared Clifford frame, which wins once mean Pauli weight passes four
(below that the call is skipped, since a ladder is already near-optimal there).

Pricing emission inside the routing decision is what makes the `terms` candidate worth
having: ungrouped it emits badly as ladders, but a shared frame would rather see the
sequence as given, and that combination takes the largest chemistry instances.

`lizzy.frame` moves a Hamiltonian to a better Pauli representation before compiling.
The representation is a choice and it decides whether the exact tools apply at all:
the same Fermi-Hubbard model is fully two-local under Jordan-Wigner and 58% two-local
under Bravyi-Kitaev. PauLie classifies both as `4*so(6)`, one algebra, so the better
form is Clifford-reachable rather than lost. Following
[Aguilar et al.](https://arxiv.org/abs/2408.00081), Clifford equivalence needs a shared
anticommutation graph *and* shared algebraic dependencies — matching on the graph alone
leaves dependent generators unplaceable — and with both respected Witt's theorem turns
the isometry into a symplectic map. Handed the Bravyi-Kitaev form, the frame recovers
the Jordan-Wigner cost exactly: 20, 40 and 56 gates at 4, 6 and 8 qubits against 42,
124 and 156, spectrum preserved to 1e-14. The published alternative
[anneals on Pauli weight](https://arxiv.org/abs/2502.11933) for 15–40%.

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
`optimization_level=3` and the better of its default and Rustiq synthesis, pytket at
the better of `GreedyPauliSimp` and `FullPeepholeOptimise`. Reproduce with
`python -m lizzy.compare`:

| HamLib instance | n | terms | Lizzy | Qiskit | pytket |
|---|---|---|---|---|---|
| tfim 1D chain | 100 | 199 | **396** | 786 | 687 |
| tfim 1D ring | 100 | 200 | **400** | 794 | 771 |
| tfim 2D grid | 100 | 280 | **720** | 1 434 | 1 316 |
| tfim hex lattice | 48 | 111 | **268** | 496 | 467 |
| tfim 3D grid | 27 | 81 | **216** | 426 | 386 |
| heisenberg 1D chain | 100 | 397 | **888** | 1 179 | 1 179 |
| heisenberg 2D grid | 100 | 640 | **1 860** | 2 151 | 2 151 |
| heisenberg 2D torus | 100 | 700 | **1 989** | 2 391 | 2 391 |
| fermi-hubbard 1D, JW | 100 | 346 | **884** | 1 178 | 1 178 |
| fermi-hubbard 1D, BK | 100 | 346 | **1 714** | 4 594 | 2 866 |
| maxcut circulant | 100 | 200 | **323** | 588 | **323** |
| H2 molecule | 4 | 14 | **26** | 47 | 28 |
| BH molecule | 10 | 275 | **1 890** | 5 788 | 2 136 |
| LiH molecule, BK | 12 | 630 | **4 340** | 16 982 | 4 405 |
| LiH molecule, JW | 12 | 630 | **4 099** | 16 133 | 4 486 |

Rustiq is Qiskit's Pauli-network plugin and folded into that column. It is the better
of the two on chemistry, where sharing a Clifford frame pays, and much the worse
everywhere else — 57 620 gates against 2 151 on the 2D Heisenberg grid, since
preserving rotation order leaves a network with nothing to share.

Fourteen wins and one tie, by margins from 1% on the largest chemistry instance to
1.9x on the tfim ring; maxcut is the tie, where pytket reaches the same 323.

**Mean Pauli weight, not algebra dimension, decides how much of that is this
compiler's own work.** With the pytket backend removed the score is 9/15, and the
split is sharp: every instance of weight ≤1.9 is won on the builtin emission alone
(tfim on all five lattices, heisenberg on all three, fermi-hubbard under
Jordan-Wigner), every instance of weight ≥2.0 is lost (maxcut 2.0, H2 2.6,
fermi-hubbard under Bravyi-Kitaev 2.8, BH 4.8, LiH 5.6-6.3). Algebra dimension does
not predict it — seven of the nine wins have exponential DLAs, and two of the six
losses are polynomial.

Weight two is where the exact tools end. The Givens route needs the whole algebra to
be `so(m)`; the kernels need every term to fit one qubit pair. Above that the builtin
emission falls back to a ladder per rotation, at 2(w-1) gates each, where a shared
frame stays near two regardless of weight — so the gap is the weight: 10.6 gates per
rotation against 1.9 on LiH, 2.0 against 1.6 on maxcut.

`lizzy.frame` closes that gap only where a lighter representation exists, and
sometimes none does. Anticommutation degree is a Clifford invariant, and a weight-two
Pauli can anticommute with at most the weight-two Paulis its two qubits touch — 195 of
them on twelve qubits. LiH's terms reach degree 264, so no Clifford makes it two-local
and its weight is a property of the algebra, not of the encoding. The same bound
leaves Bravyi-Kitaev Fermi-Hubbard wide open at degree 8 of 1779, which is why the
frame recovers the Jordan-Wigner cost there in full.

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
single qubit pair compiles as one canonical block, charged what its KAK class costs —
three CNOTs when all three canonical parameters are non-trivial and two otherwise —
and a wider rotation pays its CNOT ladder. Pricing the class rather than capping at
three is what an XY-type bond is worth: the flat cap overcharged 61% of pair blocks in
a random sweep and never undercharged, so the counts it reported were reachable but
pessimistic. Emitting the circuits explicitly confirms the count on both sides of that
change (1 440 reported, 1 440 emitted at tfim 2D; 884 and 884 on Fermi-Hubbard under
Jordan-Wigner, where the cap said 1 063).

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

PauLie also needs [pauliebits](https://github.com/QPauLie/pauliebits), which is not on
PyPI. Its location lives in PauLie's `[tool.uv.sources]`, which `uv` reads and `pip`
does not, so a pip environment installs it explicitly or the import fails:

```bash
pip install 'git+https://github.com/QPauLie/pauliebits.git@master'
```

## Outlook

Two questions decide whether the remaining gap can be closed at all, and they are
different questions because the two instances fail differently.

**BH: is there a frame, and how would it be found?** The degree bound permits a
two-local representation here — 124 against 159 — but nothing reaches one. No HamLib
encoding does, greedy descent over `H`/`S`/`CNOT` improves mean weight by 11% and
stalls in five moves, and the published annealing ceiling of 15–40% would still leave
4.8 far above 2.0. Either a finer Clifford invariant forbids it, in which case the
degree bound wants sharpening — the support-class structure of the anticommutation
graph is the natural candidate — or a target exists that local search cannot see, in
which case it has to be constructed from the classification rather than searched for.
Both are answerable; neither is answered here.

**LiH: is there a physically motivated route?** Not through the representation — that
is settled, degree 264 against 195. But high weight is a property of the *fermionic
basis*, not of the physics: molecular orbitals are a choice like any other, and the
Coulomb interaction is only all-to-all in the basis one happens to write it in. Double
factorization already exploits this, rewriting the Hamiltonian as O(N) fragments that
are each quadratic and therefore each two-local in their own frame. The open question
is whether the fragments' frames can be reconciled — a shared basis in which several
fragments are simultaneously local would let one Clifford serve many, where today each
fragment would pay its own. That is a question about the geometry of the low-rank
decomposition, and it is where this compiler's classification could contribute
something the chemistry literature does not currently ask for.

**A reference representation derived from the classification.** `lizzy.frame` needs a
target to move towards, and today that target is supplied. The classification already
names the canonical graph — PauLie's types A/B1/B2/B3 *are* the canonical forms of
[Aguilar et al.](https://arxiv.org/abs/2408.00081), and its tracked canonicalizer
records the contractions — so constructing the target from the classification alone is
reachable, and is what would make the frame apply to a Hamiltonian arriving without a
better twin.

**More emission backends, priced rather than written.** Emission is a tier with two
entries and nothing limits it to two: Rustiq (inside Qiskit) is a third, a native
frame-conjugated emission a fourth. Each is a function from a rotation sequence to a
gate count, verifiable densely at small width, and `emission_cost` already takes a
minimum — so adding one is strictly monotone, chosen only where it wins. Optional
imports are how a small compiler stays competitive on families it was never
specialised for; the work is keeping the interface narrow.

**Fast-forwardable fragments.** The exact route fires when the whole algebra is
`so(m)`; the chemistry literature generalizes this to splitting a Hamiltonian into
fragments each implementable for any time at precision-independent cost. Double
factorization is the affordable one: the two-electron tensor has low rank, so a
molecular Hamiltonian decomposes into O(N) free-fermionic fragments, each a target for
the Givens route. The decomposition is an O(N^6) eigendecomposition, routine at these
sizes and implemented in [ffsim](https://github.com/qiskit-community/ffsim); the
obstacle is input rather than cost, since it needs fermionic tensors and a Pauli
Hamiltonian yields those only with its encoding known.

Also open, in measured order of value: applying tapering automatically once a sector is
specified; the 17 commuting clusters chemistry produces against three for spin models;
symmetry protection, a no-op where terms conserve the charges individually and untested
where they do not; and vectorizing the commutator walk, which exhausts its budget on
all-to-all models around n=24.

**Measured and rejected.** Parity networks for diagonal clusters, which looked like
the answer to maxcut: instead of a ladder per term, carry a linear map and pay only
what separates one parity from the next. Built and dense-verified, the network needs
294 CNOTs on maxcut against the ladder's 400 — but the map drifts far from the
identity and restoring it costs 2 403, a cost the naive Gaussian elimination cannot
avoid for a full-rank drift on 100 qubits. Constraining the drift gives the ladder back
exactly. The restore is paid once and the network per step, so the break-even is 23
Trotter steps; a fully commuting Hamiltonian needs one, which is precisely the worst
case for amortizing it. Worth revisiting only for diagonal parts inside deep circuits.

Widening the kernels past two qubits: a general three-qubit
unitary costs about twenty CNOTs, so such a kernel pays only where five or more terms
share a triple, while on LiH 509 of 630 terms are too wide to fit one at all and those
that fit share a triple 2.9 times. Recognizing free-fermions-in-disguise: the
frustration graph is already built, but even-hole-free recognition is O(n^9) at best,
so the condition can be ruled out by the cheap claw-free half and never ruled in.

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
