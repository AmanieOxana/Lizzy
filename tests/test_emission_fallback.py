"""The optional emission portfolio must retain a dependency-free builtin path."""

import builtins

from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.emit import best_emission
from lizzy.hamiltonian import Circuit


def test_best_emission_falls_back_when_pytket_is_unavailable(monkeypatch) -> None:
    circuit = Circuit()
    circuit.add(get_pauli_string("XYZ"), 0.3, "test")
    real_import = builtins.__import__

    def without_pytket(name, *args, **kwargs):
        if name == "pytket":
            raise ImportError("simulated optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pytket)
    quote = best_emission(circuit, 3)

    assert quote.backend == "builtin"
    assert quote.circuit is circuit
    assert quote.two_qubit_gates == circuit.two_qubit_gates

    before = quote.two_qubit_gates
    circuit.add(get_pauli_string("ZZI"), 0.1, "test")
    assert quote.two_qubit_gates == circuit.two_qubit_gates
    assert quote.two_qubit_gates > before


def test_missing_pytket_passes_cannot_suppress_builtin(monkeypatch) -> None:
    circuit = Circuit()
    circuit.add(get_pauli_string("XYZ"), 0.3, "test")
    real_import = builtins.__import__

    def without_passes(name, *args, **kwargs):
        if name == "pytket.passes":
            raise ImportError("simulated partial pytket installation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_passes)
    quote = best_emission(circuit, 3)

    assert quote.backend == "builtin"
