# Getting started

[Overview](index.md) · [Methods and limits](methods.md) · [Results](compiler_comparison.md)

## Install

Use Python 3.12 or newer. From the repository root:

```bash
python -m pip install \
  "kak_tools @ git+https://github.com/QPauLie/kak-tools.git@3980728596a060a7db2cf5f541a4aaf9011a9b1d"
python -m pip install -e '.[test]'
```

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

from lizzy.native import ladder_circuit

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
See [accuracy and phase contracts](methods.md).

## Driven evolution

```python
import numpy as np
from lizzy.driven import DrivenHamiltonian
from lizzy.wei_norman import synthesize_wei_norman

drive = DrivenHamiltonian(
    ["X", "Y", "Z"],
    lambda t: [np.cos(t), np.sin(t), 0.3],
)
driven = synthesize_wei_norman(drive, (0.0, 0.4), max_step=0.02)
print(driven.charts, driven.two_qubit_gates)
```

Declare all potentially active controls and supply known pulse boundaries.
Local ODE tolerances do not certify final unitary error.
For orbital tensors use `chemistry.synthesize_molecular_ffsim`; for a direct
OpenFermion quadratic input use `gaussian.from_quadratic`.
[The methods note](methods.md) explains their distinct contracts.

## Development

Run `python -m pytest -q`. Optional integrations skip when dependencies are
absent; tests do not download data. [Reproduce results](reproduce.md)
contains the three active validation commands. Regenerate the current figure
with `python docs/figures/generate.py` (requires `plot`).
