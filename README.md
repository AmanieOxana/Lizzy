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
 ├─ classify the DLA (PauLie), split into commuting summands   [exact]
 ├─ per summand, price every candidate:
 │    exact       so(m) → one orthogonal matrix, reduced to adjacent
 │                Givens rotations along the Majorana line       [fixed depth]
 │    hybrid      free subalgebra compiled exactly inside each
 │                second-order step, as one more summand
 │    formula     S2 over commuting clusters, or over two-qubit
 │                kernels with fields folded in (3 CNOTs each)
 │  → the fewest two-qubit gates win; exactness is not a priority order
 └─ concatenate
```

Step counts come from the collected commutator bound
([Childs et al.](https://doi.org/10.1103/PhysRevX.11.011020), tight second-order
constants with cancellation between chains kept), so sizing needs no oracle;
`bench.calibrate` measures the remaining overshoot per instance where a dense
reference exists. Fully commuting Hamiltonians compile in one exact step. Every
two-qubit kernel decomposition is verified against its own 4×4 exponential at
synthesis time.

`randomized=True` adds a fifth candidate, sampling the small terms with qDRIFT
([Campbell](https://doi.org/10.1103/PhysRevLett.123.070503)), whose cost depends on
the coefficients rather than the term count. It is off by default and stays off:
its guarantee is on the averaged channel, not on the circuit you get, so a route
that wins on gate count can still miss the budget it was sized for.

## Measured

HamLib instances, the same fixed-depth task for every compiler (two Suzuki-2 steps),
two-qubit gates after each compiler's best effort — Qiskit at `optimization_level=3`,
pytket at the better of `GreedyPauliSimp` and `FullPeepholeOptimise`
(`python -m lizzy.compare`):

| HamLib instance | n | Lizzy | Qiskit | Qiskit+Rustiq | pytket |
|---|---|---|---|---|---|
| tfim 1D chain | 100 | **396** | 786 | 1 032 | 687 |
| tfim 2D grid | 100 | 1 440 | 1 434 | 2 573 | **1 316** |
| heisenberg 1D chain | 100 | **894** | 1 179 | 4 117 | 1 179 |
| heisenberg 2D grid | 100 | **1 968** | 2 151 | 57 620 | 2 151 |
| BH molecule (chemistry) | 10 | **2 000** | 7 484 | 5 788 | 2 136 |

Giving both competitors their strongest setting costs this compiler most of its
reported margin and is the only honest comparison: Qiskit at level 1 needs 4 320 gates
on the 2D grid where level 3 needs 2 151, and `GreedyPauliSimp` is not pytket's best
pass on grids despite being the Pauli-aware one. On grids the two frameworks then
converge to the same count, which the kernels undercut by about 9%.

At matched accuracy — eight qubits, every compiler given the fewest steps that reach
1e-3 against a dense reference — Lizzy wins all six model/time combinations measured,
1.4× to 4.1×, with the margin growing in evolution time on the fast-forwardable
families: at t=8 the exact branch holds 143 gates against Trotter's 1 148.

**Which compilers can be compared at all** was the hard part. Checked against a dense
reference on a duplicate-free Trotter step, only pytket, Qiskit's `PauliEvolutionGate`
and Lizzy reproduce the sequence they are given (infidelity ≤ 2e-16). Paulihedral,
Tetris, PauliOpt and [PHOENIX](https://github.com/iqubit-org/phoenix) reorder
non-commuting terms — legitimate for a Trotter approximation, but it makes gate counts
incomparable, since the circuit implements a different unitary. Reordering compilers
therefore have to be scored on accuracy, not on depth.

Scored that way, PHOENIX needs 238 gates where Lizzy needs 42 (tfim, t=1), 420 against
132 (heisenberg, t=1) and 3 213 against 1 881 (heisenberg, t=8): it buys cheaper steps
with more of them, because the reordering costs Trotter accuracy. Paulihedral and
Tetris are absent because their output could not be verified — even a single Pauli
rotation came back 8.4e-3 from the reference under every convention tried, so any
number reported for them would be unfounded rather than unfavourable.

The compiler itself stays small: the core is ~2 600 lines of Python across nine
modules — for scale, pytket's `GreedyPauliSimp` pass alone is ~2 500 lines of C++ —
because the algebra lives upstream in PauLie and kak-tools, and the routing is
arithmetic over priced candidates rather than machinery.

## Verification

`lizzy.dense` rebuilds circuits as 2^n matrices and compares them to `expm(-itH)`,
so every exactness claim is checked at the qubit level wherever size allows; the
test suite is built on it. Kernel decompositions self-verify at any width.

The reported gate count is the block-aware one: a run of consecutive rotations that
fits on a single qubit pair compiles as one canonical block of at most three CNOTs,
and a wider rotation pays its ladder. Emitting the circuits explicitly confirms the
count on spin models (1 440 reported, 1 440 emitted at tfim 2D). An earlier version
capped every run at three CNOTs regardless of its support, which understated
chemistry by a factor of 2.7 and produced the retracted claim above.

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

## Possibilities to improve

In measured order of value:

- **A native shared Clifford frame.** The chemistry win currently rides on pytket as
  an optional backend; a native implementation of frame-conjugated emission would
  remove the dependency and could exploit the cluster structure directly instead of
  rediscovering it.
- **Whatever pytket finds on 2D grids.** Not cancellation: running Qiskit's level-3
  optimizer over the emitted circuit recovers exactly zero gates (1 440 → 1 440), so
  the 9% must come from a different rotation order or Clifford conjugation, not from
  peepholing what is emitted.
- **Qubit tapering.** Worth far more on chemistry than on spin models: BH at 10 qubits
  carries four Z2 charges, so 40% of the register is removable, against 2% for a spin
  chain. The charges are found and reported but never applied. The open question is
  whether the tapering Clifford's weight growth eats the saving — untested.
- **Cluster count on dense instances.** Chemistry gives 17 commuting clusters where
  spin models give three, so a step pays 17 basis changes. Better colouring, or
  kernels on larger supports, would attack the term the formula is actually spending.
- **Symmetry protection.** Provably a no-op for models whose terms conserve the
  charges individually, untested where terms violate them — gauge theories, chemistry.
- **Dense bounds past ~20 qubits.** The collected commutator walk exhausts its budget
  on all-to-all models around n=24. Vectorizing it would extend the certified regime,
  but the circuits out there run to 10^8 gates, so it certifies what cannot be run.

The per-rotation ladder is the right emission on low-weight instances and the wrong
one on chemistry, where mean Pauli weight is 4.8 and a ladder costs ~7.6 CNOTs per
rotation. There the optional shared-frame backend (`lizzy.emit`, pytket's
`GreedyPauliSimp`) takes over — and it works better on this compiler's cluster-ordered
sequences than on raw term order: 2 000 gates against 2 136, first place on the row,
dense-verified equivalent at the full ten qubits. The routing layer and the emission
layer compose. An earlier version of this file claimed the ladder won on chemistry
outright; that rested on a cost-model bug (see below) and is retracted.

## References

- Kökcü et al., [Fixed depth Hamiltonian simulation via Cartan decomposition](https://doi.org/10.1103/PhysRevLett.129.070501) — the exact branch
- Wierichs et al., [Recursive Cartan decompositions for unitary synthesis](https://arxiv.org/abs/2503.19014) — what kak-tools implements
- Wiersema et al., [Classification of dynamical Lie algebras of 2-local spin systems](https://doi.org/10.1038/s41534-024-00900-2) — why polynomial DLAs are a chain phenomenon
- Childs et al., [Theory of Trotter error with commutator scaling](https://doi.org/10.1103/PhysRevX.11.011020) — the step counts
- Decker et al., [Kernpiler](https://arxiv.org/abs/2504.07214) — partial Trotterization, the kernels
- Kivlichan et al., [Quantum simulation with linear depth](https://doi.org/10.1103/PhysRevLett.120.110501) — the Givens emission
- Chen et al., [PHOENIX](https://arxiv.org/abs/2504.03529) — global Pauli-IR optimization, compared against
- Goubault de Brugière & Martiel, [Rustiq](https://arxiv.org/abs/2404.03280) — the Pauli-network synthesis Qiskit ships
- Sawaya et al., [HamLib](https://arxiv.org/abs/2306.13126) — the instances
- Meckes, [The Random Matrix Theory of the Classical Compact Groups](https://doi.org/10.1017/9781108303453) — the namesake
