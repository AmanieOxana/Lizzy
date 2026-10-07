# Getting started

[The idea](../README.md) · [Synthesis guide](synthesis.md)

## Install

Use Python 3.12 or newer. From the repository root, install the tested
representation dependency and Lizzy:

```bash
python -m pip install \
  "kak_tools @ git+https://github.com/QPauLie/kak-tools.git@3980728596a060a7db2cf5f541a4aaf9011a9b1d"
python -m pip install -e '.[test]'
```

Optional extras: `ft` for Clifford+T output, `gaussian` for OpenFermion quadratic
synthesis, `chemistry` for the molecular stack (already includes OpenFermion),
and `compare` for Qiskit/pytket. For example:

```bash
python -m pip install -e '.[ft]'
```

## Compile a Hamiltonian

```python
from lizzy.hamiltonian import hamiltonian
from lizzy.synthesize import synthesize

H = hamiltonian({"XX": 0.7, "ZI": 0.2, "IZ": -0.1})
result = synthesize(H, time=1.0, error=1e-3)
print(result.routes)
print(result.two_qubit_gates, result.emission_is_concrete)
```

`result.circuit` contains Pauli rotations. Gate counts can be analytical;
`emission_is_concrete` tells you whether the counted gate list is retained.
See [emission and export](synthesis.md#logical-sequences-versus-emitted-gates).

The default objective is CX cost. With `ft`, `objective="t"` compares actual
T/T† counts at a common rotation-error budget. T mode requires a supported exact
route and has no Trotter fallback. Its rotation budget does not certify total
simulation error. Always inspect the result's accuracy and phase contract.

For time-dependent inputs, use the Wei–Norman example in the [synthesis guide](synthesis.md).
For orbital tensors, use the [chemistry interface](chemistry.md).
Routing limits and explicit method choices are in the [synthesis guide](synthesis.md).

## Development

Run tests with `python -m pytest -q`. Optional integration tests skip when their
dependencies are absent. Regenerate figures from saved measurements with
`python docs/figures/generate.py` (requires the `plot` extra).

[Architecture](architecture.md) · [Experiments](../experiments/README.md) ·
[Paper references](references.md).
