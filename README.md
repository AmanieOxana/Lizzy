# Lizzy

Lizzy is a Python package for compiling Hamiltonian evolution into quantum
circuits. It uses Lie-algebra and fermionic structure to choose a synthesis
method and compare gate costs. Static and time-dependent Hamiltonians are
supported.

From the repository directory, with Python 3.12 or newer and Git installed:

```bash
pip install -e .
```

This also installs the pinned `kak-tools` dependency.

Synthesize a Pauli Hamiltonian's evolution and print an OpenQASM circuit:

```python
from lizzy.hamiltonian import hamiltonian
from lizzy.synthesize import synthesize
from lizzy.emission.native import ladder_circuit

H = hamiltonian({"XX": 0.7, "ZI": 0.2, "IZ": -0.1})
result = synthesize(H, time=1.0, error=1e-3)
circuit = ladder_circuit(result.circuit, width=2)
print(circuit.to_qasm3())
```

- [Getting started](docs/getting_started.md): setup and examples.
- [Methods](docs/methods.md): constructions and their limits.
- [Results](docs/compiler_comparison.md): compiler comparison.

Named in honor of Elizabeth Meckes.

[MIT License](LICENSE) for Lizzy's own code. Upstream `kak-tools` licensing
remains to be clarified.
