# Getting started

[Overview](index.md) · [Methods](methods.md) · [Results](compiler_comparison.md)

## Install

Use Python 3.12 or newer and Git:

```bash
git clone https://github.com/AmanieOxana/Lizzy.git
cd Lizzy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.
The last command installs Lizzy in editable mode and automatically fetches the
tested `kak-tools` fork at its pinned Git commit; you do not need a separate
checkout. An internet connection is required for the initial dependency installation.

Optional extras: `ft` for Clifford+T, `gaussian` for OpenFermion quadratic
synthesis, `chemistry` for molecular tensors (also includes OpenFermion), and
`compare` for Qiskit/pytket. For example, `python -m pip install -e '.[ft]'`.

## Static evolution

```python
from lizzy.hamiltonian import hamiltonian
from lizzy.synthesize import synthesize

H = hamiltonian({"XX": 0.7, "ZI": 0.2, "IZ": -0.1})
result = synthesize(H, time=1.0, error=1e-3)
print(result.routes)
print(result.two_qubit_gates, result.emission_is_concrete)

from lizzy.emission.native import ladder_circuit

native = ladder_circuit(result.circuit, width=2)
qasm = native.to_qasm3()
```

Auto chooses among eligible exact and approximate routes by cost; you do not
have to choose BDI versus Givens. `result.circuit` contains logical Pauli
rotations. Counts may be analytical: `emission_is_concrete` says whether the
counted gate list is retained. The example also creates explicit native output.
For repeated requests with the same configuration:

```python
from lizzy.hamiltonian import hamiltonian
from lizzy.synthesize import Compiler

compiler = Compiler(error=1e-3)
H = hamiltonian({"XX": 0.7, "ZI": 0.2, "IZ": -0.1})
short = compiler.compile(H, time=0.1)
long = compiler.compile(H, time=1.0)
```

With `ft`, add `objective="t"` to compare actual T/T† counts at a common
rotation budget. This requires a supported exact route and has no Trotter
fallback. The rotation budget is not a total simulation-error certificate.
See [what the gate counts promise](methods.md#what-the-gate-counts-promise).

## Driven evolution

```python
import numpy as np
from lizzy.synthesis.driven import DrivenHamiltonian
from lizzy.synthesis.wei_norman import synthesize_wei_norman

drive = DrivenHamiltonian(
    ["X", "Y", "Z"],
    lambda t: [np.cos(t), np.sin(t), 0.3],
)
driven = synthesize_wei_norman(drive, (0.0, 0.4), max_step=0.02)
print(driven.charts, driven.two_qubit_gates)
```

Declare all potentially active controls and supply known pulse boundaries.
Local ODE tolerances do not certify final unitary error.
For orbital tensors use `lizzy.fermions.chemistry.synthesize_molecular_ffsim`;
for direct OpenFermion quadratic input use `lizzy.fermions.gaussian.from_quadratic`.
[Methods](methods.md) explains when each construction is useful.

## Development

The package groups specialist code by responsibility:

```text
lizzy/
├── synthesize.py    # Compiler and synthesize: main API
├── hamiltonian.py   # Inputs and logical circuits
├── dense.py         # Small verification references
├── hamlib.py        # Data loading
├── algebra/        # Classification, symmetries and representation changes
├── synthesis/      # Exact, product-formula and driven constructions
├── emission/       # Native and Clifford+T gate artifacts
└── fermions/       # OpenFermion and ffsim adapters
```

`lizzy.synthesize.Compiler` and `synthesize` keep their import paths. Direct
specialist imports use the subpackages, as in `lizzy.synthesis.driven` and
`lizzy.emission.native` above; the old flat paths are not compatibility aliases.
Persisted pickles naming moved specialist classes also require migration.

See [Run the checks](reproduce.md) for tests, compiler comparisons and figure
generation.

To build the Sphinx documentation locally:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. This does not publish the site.
