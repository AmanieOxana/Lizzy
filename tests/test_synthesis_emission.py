"""Contracts for full-sequence pricing and result-level emission metadata."""

import math

from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.dense import circuit_matrix, evolution, infidelity
from lizzy.emit import best_emission
from lizzy.exact import free_part
from lizzy.hamiltonian import Circuit, hamiltonian, model, n_qubits
from lizzy.symmetry import commuting_clusters
from lizzy.synthesize import _formula_plan, _hybrid_plan, _repeated_plan, synthesize


def test_formula_plan_prices_the_complete_folded_sequence() -> None:
    h = model("heisenberg", 4, seed=1)

    plan = _formula_plan(h, commuting_clusters(h), 1.0, 1e-3, 1.0, 3)

    assert plan.circuit is not None
    assert plan.cost == best_emission(plan.circuit, n_qubits(h)).two_qubit_gates
    assert infidelity(evolution(h, 1.0), circuit_matrix(plan.circuit, 4)) < 0.1


def test_hybrid_plan_retains_the_artifact_that_was_priced() -> None:
    h = model("heisenberg", 4, seed=1)
    free, rest = free_part(h)

    plan = _hybrid_plan(free, rest, 1.0, 1e-3, 1.0, None)

    assert plan.circuit is not None
    assert plan.emission.circuit is plan.circuit
    assert plan.cost == plan.circuit.two_qubit_gates


def test_result_is_quoted_once_more_as_a_complete_circuit(monkeypatch) -> None:
    import lizzy.synthesize as synthesis_module

    h = hamiltonian({"XI": 0.7, "IZ": -0.2})
    seen = []
    real_best_emission = synthesis_module.best_emission

    def recording_quote(circuit, width, **kwargs):
        seen.append(circuit)
        return real_best_emission(circuit, width, **kwargs)

    monkeypatch.setattr(synthesis_module, "best_emission", recording_quote)
    result = synthesize(h, time=0.8, steps=2)

    assert result.summands == 2
    assert seen[-1] is result.circuit
    assert result.emitted_circuit is result.emission.circuit
    assert result.two_qubit_gates == result.emission.two_qubit_gates
    assert infidelity(evolution(h, 0.8), circuit_matrix(result.circuit, 2)) < 1e-12


def test_high_weight_fixed_depth_considers_independent_set(monkeypatch) -> None:
    import lizzy.synthesize as synthesis_module

    h = hamiltonian(
        {
            "YIIXX": 1.0,
            "YIYXX": 2.0,
            "ZYYXX": 3.0,
            "YXXXX": 4.0,
            "XZXXX": 5.0,
            "ZZZXX": 6.0,
            "YZIXX": 7.0,
        }
    )
    strategies = []
    real_clusters = synthesis_module.commuting_clusters

    def recording_clusters(operator, strategy="largest_first"):
        strategies.append(strategy)
        return real_clusters(operator, strategy=strategy)

    monkeypatch.setattr(synthesis_module, "commuting_clusters", recording_clusters)
    _formula_plan(h, real_clusters(h), 1.0, 1e-3, 1.0, 2)

    assert "independent_set" in strategies


def test_result_keeps_logical_circuit_separate_from_backend_artifact() -> None:
    result = synthesize(model("tfim", 4, seed=0), time=1.0, steps=2)

    assert isinstance(result.circuit, Circuit)
    assert result.logical_two_qubit_gates == result.circuit.two_qubit_gates
    assert result.two_qubit_gates <= result.logical_two_qubit_gates


def test_deferred_guard_uses_raw_not_folded_rotation_count() -> None:
    calls = []

    def build(repetitions):
        calls.append(repetitions)
        circuit = Circuit()
        for _ in range(repetitions):
            circuit.add(get_pauli_string("X"), math.pi / 2, "test")
        return circuit

    plan = _repeated_plan(build, 25_001, 1, 1)

    assert calls == [4, 1, 2]
    assert plan.circuit is None
    assert plan.used_estimates


def test_large_route_estimation_is_reported_separately_from_error_guarantee() -> None:
    result = synthesize(model("tfim", 4, seed=0), time=1.0, error=1e-10)

    assert result.routing_estimated
    assert result.error_guaranteed
