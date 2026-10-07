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

[Documentation](https://amanieoxana.github.io/Lizzy/):

- [Getting started](https://amanieoxana.github.io/Lizzy/getting_started.html): setup and examples.
- [Methods](https://amanieoxana.github.io/Lizzy/methods.html): constructions and their limits.
- [Results](https://amanieoxana.github.io/Lizzy/compiler_comparison.html): compiler comparison.

Named in honor of Elizabeth Meckes.

[MIT License](LICENSE) for Lizzy's own code. Upstream `kak-tools` licensing
remains to be clarified.
