# Methods

[Get started](getting_started.md) · [Results](compiler_comparison.md)

Lizzy turns a Hamiltonian into an evolution circuit. Its central question is:
**can the system's structure make that circuit cheaper?**

The dynamical Lie algebra (DLA) describes which operators the evolution can
explore. A small DLA can make a compact construction possible, but does not
guarantee cheap gates. Lizzy therefore separates **finding a construction**
from **comparing its gate cost**.

## Where to start

| Your input | Use |
| --- | --- |
| A static Pauli Hamiltonian | `synthesize`: automatically compare supported constructions. |
| Time-dependent Pauli controls | `synthesize_wei_norman`: solve for changing rotation angles. |
| An OpenFermion quadratic Hamiltonian | `from_quadratic`; static auto also recognizes its Jordan–Wigner Pauli form. |
| Interacting molecular orbital tensors | `synthesize_molecular_ffsim`: use the dedicated chemistry adapter. |

See [Getting started](getting_started.md) for imports and examples. You do not
need to choose BDI versus Givens yourself.

## Cartan BDI and Givens

For supported algebras, Lizzy represents the problem using a smaller real
rotation matrix. It offers two ways to turn that representation into a circuit.

**Cartan BDI recursively splits the rotation problem into simpler pieces.**
In its reusable, *horizontal* case, the idea is a basis change around simple
commuting evolution:

$$
U(t)=K\,e^{-itA}\,K^\dagger.
$$

The circuit changes basis, applies the commuting rotations in $A$, then changes
back. The basis change stays fixed; only the central angles change with time.
General, nonhorizontal BDI instead factors each requested evolution separately.

**Givens directly factors the evolution into plane rotations.** It uses the
same kind of small-matrix representation, but a different factorization and
ordering. Neither construction always produces fewer gates.

Static auto compares these supported exact routes with product formulas
(Trotter steps) and combinations of the two. The search is bounded and may use
cost estimates; it is not a proof of the cheapest circuit.

BDI follows [Wierichs et al., Sections VI.1–VI.2](https://arxiv.org/html/2503.19014v2)
through [kak-tools](https://github.com/QPauLie/kak-tools). Lizzy supports mapped
orthogonal algebras, not every Cartan type, and does not implement that paper's
model-specific XY/YX gate cancellations.

## Wei–Norman

**Keep the rotation axes; solve for their angles as the controls change.**
Wei–Norman writes evolution as a product of rotations along a DLA basis and
integrates differential equations for their angles. This follows
[Qvarfort–Pikovski](https://doi.org/10.1103/PRXQuantum.6.010201) and
[Altafini](https://arxiv.org/abs/quant-ph/0203005).

The coordinates can become singular even when the physical evolution is smooth.
Lizzy can restart in a fresh chart; those restarts add rotations. Integration is
numerical, so local solver tolerances are not a bound on final circuit error.
Declare pulse boundaries and use a step size that resolves the controls.
Static auto does not select Wei–Norman implicitly.

Magnus/Fer expansions are optional approximations for driven evolution. Their
resulting exponentials still need synthesis; they are not a shortcut to an
exact circuit.

## Fermionic and chemistry inputs

**Use the structure already present in the fermionic input.** OpenFermion
supplies quadratic (Gaussian) evolution; ffsim supplies molecular factorization
and simulation. Lizzy integrates these methods rather than replacing them with
a DLA construction. Interacting molecules are not generally Gaussian: keep their
orbital tensors and use the chemistry adapter. Finite-step and factorization
errors still need checking.

## What the gate counts promise

The default objective is CX cost. With `objective="t"`, Lizzy compares actual
Clifford+T circuits from supported exact routes. It first keeps a lowest-T
reference among its candidates, then accepts alternatives that reuse basis changes only if
**neither T nor CX increases**. This protects that reference, not a global optimum.
T mode has no Trotter or Wei–Norman fallback.

Check `emission_is_concrete` to distinguish counted gate lists from estimates.
Counts assume unrestricted connectivity. In T mode, `error` budgets rotation
approximation, not total simulation error. Gaussian and Wei–Norman numerical
checks likewise are not certified error bounds.

Finally, general BDI and Givens may match the target evolution only **up to
global phase**. That is enough for uncontrolled evolution, but controlled
evolution needs a separate phase check.
