"""Emission portfolio contracts, independent of optional SDK installations."""

import builtins

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.dense import circuit_matrix
from lizzy.emit import EmissionQuote, best_emission, native_emission_candidates
from lizzy.hamiltonian import Circuit
from lizzy.native import ladder_circuit
from lizzy.synthesize import Result


def _logical_pair() -> Circuit:
    logical = Circuit()
    logical.add(get_pauli_string("XX"), 0.17, "test")
    logical.add(get_pauli_string("YY"), -0.23, "test")
    return logical


def test_builtin_quote_is_logical_even_when_pair_cost_is_exact() -> None:
    logical = _logical_pair()
    quote = EmissionQuote("builtin", logical.two_qubit_gates, logical)

    assert not quote.is_concrete
    assert quote.two_qubit_gates == 2
    assert ladder_circuit(logical, 2).two_qubit_gates == 4
    assert quote.circuit is logical
    assert not Result(logical, emission=quote).emission_is_concrete
    result = Result(logical)
    assert not result.emission_is_concrete
    assert result.emitted_circuit is logical
    assert result.two_qubit_gates == logical.two_qubit_gates


def test_opaque_artifact_does_not_claim_concrete_count() -> None:
    quote = EmissionQuote("external", 7, object())

    assert not quote.is_concrete
    assert quote.two_qubit_gates == 7
    assert not Result(Circuit(), emission=quote).emission_is_concrete


def test_native_portfolio_preserves_phase_order_and_caller_failure_policies(monkeypatch):
    import lizzy.synthesize as synthesis
    from lizzy import driven_bench, emit, synthesis_bench, wei_norman

    logical = _logical_pair()
    logical.add(get_pauli_string("II"), 0.37, "test")
    candidates = native_emission_candidates(logical, 2)
    assert [name for name, _ in candidates] == ["native-ladder", "native-frame"]
    target = circuit_matrix(logical, 2)
    for _, emitted in candidates:
        np.testing.assert_allclose(emitted.get_unitary(), target, atol=1e-12)
    assert native_emission_candidates(logical, 2, include_frame=False) == candidates[:1]

    for failure in (ValueError, StopIteration):
        def failed_frame(*args):
            raise failure("optional frame failed")

        for failure_point in ("native_frame_candidate", "native_frame_circuit"):
            with monkeypatch.context() as patch:
                patch.setattr(emit, failure_point, failed_frame)
                fallback = synthesis._gaussian_emission(logical, 2)
                assert fallback.backend == "native-ladder"
                np.testing.assert_allclose(fallback.circuit.get_unitary(), target, atol=1e-12)
                assert wei_norman._emit(logical, 2, "ladder").backend == "native-ladder"
                assert wei_norman._emit(logical, 2, "none") is None
                for compile_candidate in (
                    lambda: wei_norman._emit(logical, 2, "native"),
                    lambda: synthesis_bench._emission(logical, 2),
                    lambda: driven_bench._quote(logical, 2, target),
                ):
                    with pytest.raises(failure, match="optional frame failed"):
                        compile_candidate()

        with monkeypatch.context() as patch:
            patch.setattr(emit, "ladder_circuit", failed_frame)
            with pytest.raises(failure, match="optional frame failed"):
                synthesis._gaussian_emission(logical, 2)


def test_optional_sdk_failure_keeps_a_live_builtin_quote(monkeypatch) -> None:
    """A missing SDK or incompatible pass cannot suppress the baseline artifact."""
    circuit = Circuit()
    circuit.add(get_pauli_string("XYZ"), 0.3, "test")
    real_import = builtins.__import__

    def without_module(name, *args, **kwargs):
        if name == "pytket" or name.startswith("pytket."):
            raise ImportError("simulated optional dependency failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_module)
    quote = best_emission(circuit, 3)

    assert quote.backend == "builtin"
    assert quote.circuit is circuit
    assert not quote.is_concrete
    before = quote.two_qubit_gates
    circuit.add(get_pauli_string("ZZI"), 0.1, "test")
    assert quote.two_qubit_gates == circuit.two_qubit_gates
    assert quote.two_qubit_gates > before
