# Lizzy

**Use the structure of a quantum system to build its evolution circuit.**

A Hamiltonian may contain many terms without exploring every possible quantum
operation. Repeated commutators of its supplied Pauli terms can stay within a
much smaller family of operators: the **dynamical Lie algebra (DLA)**.

Lizzy asks whether that structure supports a useful way to construct the
evolution, rather than treating every problem as an unrestricted unitary.

## Install

With Python 3.12 or newer and Git, run from the Lizzy repository directory:

```bash
python -m pip install -e .
```

This installs Lizzy in editable mode and automatically installs its required
dependencies, including the pinned `kak-tools` fork.

## The idea

```mermaid
flowchart LR
    H["Hamiltonian + evolution time"] --> S["Recognize useful structure"]
    S --> C["Construct candidate circuits"]
    C --> G["Compare gate costs"]
```

For static synthesis, structure determines which constructions are available;
gate cost helps choose between them. A supported factorization can avoid
repeating many small simulation steps. Approximate constructions remain useful
when an exact route is unavailable or more expensive.

Time-dependent evolution uses algebra-generated rotations. Fermionic inputs
use established specialized algorithms. These are different ways to exploit
structure, not competing claims that one method is always best.

**A smaller algebra is an opportunity, not a guarantee of fewer gates.**
Current results show savings on selected structured targets, not a universal
advantage or a chemistry-wide improvement.

[Get started](docs/getting_started.md) · [Results](docs/compiler_comparison.md) ·
[Methods](docs/methods.md)

The [package layout](docs/getting_started.md#development) separates algebra,
synthesis, emission and fermionic adapters. The main `Compiler`/`synthesize`
interface remains in `lizzy.synthesize`.

Build the Sphinx documentation locally:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. This builds a local site; it does not publish it.

Named in honor of Elizabeth Meckes.

## License

Lizzy's own code is available under the [MIT License](LICENSE). Dependencies
retain their own terms; upstream licensing for `kak-tools` remains to be clarified.
