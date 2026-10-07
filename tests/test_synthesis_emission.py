"""Contracts for full-sequence pricing and result-level emission metadata."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.dense import circuit_matrix, evolution, infidelity
from lizzy.emission.emit import EmissionQuote
from lizzy.hamiltonian import Circuit, hamiltonian, model, terms_of
from lizzy.synthesis._routing import _formula_plan, _repeated_plan
from lizzy.synthesize import Compiler, synthesize


def test_compiler_reuse_has_independent_results_and_snapshots_options():
    order = ["X", "Y", "Z"]
    options = {"max_dimension": 3, "basis_order": order}
    compiler = Compiler(method="wei-norman", numerical_options=options)
    operator = hamiltonian({"X": 0.7, "Z": -0.2})
    first = compiler.compile(operator, 0.3)
    options["max_dimension"] = 1
    order.clear()
    first.circuit.rotations.clear()
    second = compiler.compile(operator, -0.2)
    assert second.circuit is not first.circuit
    assert second.emitted_circuit is not first.emitted_circuit
    np.testing.assert_allclose(
        second.emitted_circuit.get_unitary(), evolution(operator, -0.2), atol=1e-8,
    )
    with pytest.raises(TypeError):
        compiler.numerical_options["max_dimension"] = 1


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
    import lizzy.synthesis._routing as synthesis_module

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


def test_t_objective_keeps_reference_when_optional_gauge_compilation_fails(monkeypatch):
    import lizzy.emission.clifford_t as backend
    import lizzy.synthesize as synthesis_module

    logical = Circuit()
    logical.add(get_pauli_string("Z"), 0.3, "exact-bdi")

    def prepare(*args, optimize="none", **kwargs):
        return SimpleNamespace(
            parameter_bound=1,
            circuit=lambda *args, **kwargs: logical,
            optimization=SimpleNamespace(optimized=optimize == "t"),
        )

    counts = iter([30, None])
    seen = []

    def compile_t(circuit, width, *, error):
        assert width == 1 and error == 1e-5
        count = next(counts)
        if count is None:
            raise ValueError("candidate failed its numerical budget")
        emitted = SimpleNamespace(t_count=count, n_2qb_gates=lambda: 0)
        seen.append((circuit, emitted))
        return emitted

    monkeypatch.setattr(synthesis_module.exact, "prepare_bdi", prepare)
    monkeypatch.setattr(backend, "compile_clifford_t", compile_t)
    monkeypatch.setattr(backend, "compile_frame_candidates", lambda *a, **k: ((), ()))
    result = synthesize(hamiltonian({"Z": 0.3}), 1.0, error=1e-5,
                        method="bdi", objective="t")
    assert result.t_count == 30
    assert result.circuit is seen[0][0]
    assert result.emitted_circuit is seen[0][1]
    assert result.emission_is_concrete and result.emission_backend == "clifford-t"
    assert result.t_selection.selected == "reference"
    assert result.t_selection.rejected_candidates == (
        ("nullspace-optimized", "ValueError: candidate failed its numerical budget"),)
    assert not result.error_guaranteed  # A rotation budget is not a global certificate.


def test_t_objective_combines_commuting_parts_and_preserves_phase_without_dependency():
    result = synthesize(hamiltonian({"XI": math.pi / 4, "IZ": -math.pi / 8}),
                        1.0, error=1e-6, method="bdi", objective="t")
    assert result.summands == 2 and result.t_count == 1
    assert dict(result.t_selection.candidate_t_counts)["reference"] == 1
    assert result.t_selection.no_regression_reference == "reference"
    np.testing.assert_allclose(
        result.emitted_circuit.get_unitary(), circuit_matrix(result.circuit, 2), atol=1e-12,
    )


@pytest.mark.parametrize("counts,selected", [
    ((20, 12, 10), "givens-reference"),
    ((20, 12, 12), "bdi-nullspace-optimized"),
    ((20, 20, 20), "bdi-reference"),
])
def test_auto_t_prices_complete_candidates_at_one_budget_and_retains_winner(
    monkeypatch, counts, selected,
):
    import lizzy.emission.clifford_t as backend
    import lizzy.synthesize as synthesis_module

    def prepare(operator, *, optimize="none", **kwargs):
        def circuit(time, route):
            logical = Circuit()
            coefficient, word = terms_of(operator)[0]
            logical.add(word, float(coefficient.real) * time, route)
            return logical
        return SimpleNamespace(
            parameter_bound=1, circuit=circuit,
            optimization=SimpleNamespace(optimized=optimize == "t"),
        )

    costs, seen = iter(counts), []

    def compile_t(circuit, width, *, error):
        # Both commuting components enter every compilation, never two separate
        # per-component allocations whose costs could mispredict the full result.
        assert width == 2 and error == 1e-5
        assert {str(word) for word, _ in circuit.rotations} == {"XI", "IZ"}
        emitted = SimpleNamespace(t_count=next(costs), n_2qb_gates=lambda: 0)
        seen.append((circuit, emitted))
        return emitted

    monkeypatch.setattr(synthesis_module.exact, "prepare_bdi", prepare)
    monkeypatch.setattr(backend, "compile_clifford_t", compile_t)
    monkeypatch.setattr(backend, "compile_frame_candidates", lambda *a, **k: ((), ()))
    result = synthesize(hamiltonian({"XI": 0.7, "IZ": -0.2}), 0.8,
                        error=1e-5, objective="t")
    names = ("bdi-reference", "bdi-nullspace-optimized", "givens-reference")
    winner = names.index(selected)
    assert result.t_selection.candidate_t_counts == tuple(zip(names, counts, strict=True))
    assert result.t_selection.selected == selected
    assert result.t_count == min(counts)
    assert result.circuit is seen[winner][0]
    assert result.emitted_circuit is seen[winner][1]
    assert result.routes == ["exact-givens" if winner == 2 else "exact-bdi"]
    assert not result.error_guaranteed


def test_auto_t_keeps_valid_fallback_and_distinguishes_unavailable_from_unsupported(monkeypatch):
    import lizzy.emission.clifford_t as backend
    import lizzy.synthesize as synthesis_module
    from lizzy.fermions import gaussian

    operator = hamiltonian({"Z": math.pi / 8})
    real_compile = backend.compile_clifford_t

    def reject_bdi(circuit, width, *, error):
        if "exact-bdi" in circuit.provenance:
            raise ValueError("BDI candidate exceeded numerical budget")
        return real_compile(circuit, width, error=error)

    monkeypatch.setattr(backend, "compile_clifford_t", reject_bdi)
    fallback = synthesize(operator, 1.0, objective="t")
    assert fallback.t_selection.selected == "givens-reference"
    assert fallback.t_count == 1
    assert fallback.t_selection.rejected_candidates[0][0] == "bdi-reference"
    np.testing.assert_allclose(fallback.emitted_circuit.get_unitary(), evolution(operator, 1.0),
                               atol=1e-12)

    def missing_backend(*args, **kwargs):
        raise ImportError("Install the optional ft extra")

    monkeypatch.setattr(backend, "compile_clifford_t", missing_backend)
    with pytest.raises(ImportError, match="optional ft extra"):
        synthesize(operator, 1.0, objective="t")

    def unsupported(*args, **kwargs):
        raise NotImplementedError("unsupported so(m) presentation")

    monkeypatch.setattr(synthesis_module.exact, "prepare_bdi", unsupported)
    monkeypatch.setattr(synthesis_module.exact, "is_decomposable", lambda _: False)
    monkeypatch.setattr(gaussian, "is_gaussian", lambda _: False)
    with pytest.raises(NotImplementedError, match="No supported exact T"):
        synthesize(operator, 1.0, objective="t")


def test_auto_t_rejects_product_formula_controls_and_numerical_dispatch():
    operator = hamiltonian({"Z": 1.0})
    for controls in ({"steps": 2}, {"order": 2}, {"randomized": True}, {"calibration": 0.5}):
        with pytest.raises(ValueError, match="product-formula controls"):
            synthesize(operator, 1.0, objective="t", **controls)
    with pytest.raises(ValueError, match="requires method='auto'"):
        synthesize(operator, 1.0, objective="t", method="wei-norman")


def test_shared_frames_obey_global_not_per_route_cost_caps(monkeypatch):
    import lizzy.emission.clifford_t as backend
    import lizzy.synthesize as synthesis_module
    from lizzy.fermions import gaussian

    def logical(word, route):
        circuit = Circuit()
        circuit.add(get_pauli_string(word), 0.3, route)
        return circuit

    bdi, givens = logical("Z", "exact-bdi"), logical("X", "exact-givens")

    def artifact(t, cx):
        return SimpleNamespace(t_count=t, n_2qb_gates=lambda: cx)

    old, other, unsafe, safe = artifact(10, 3), artifact(20, 8), artifact(9, 4), artifact(10, 2)
    monkeypatch.setattr(gaussian, "is_gaussian", lambda _: False)
    monkeypatch.setattr(synthesis_module.exact, "prepare_bdi", lambda *a, **k: SimpleNamespace(
        circuit=lambda *a, **k: bdi, parameter_bound=1,
        optimization=SimpleNamespace(optimized=False)))
    monkeypatch.setattr(synthesis_module.exact, "decompose", lambda *a, **k: givens)
    monkeypatch.setattr(backend, "compile_clifford_t", lambda c, *a, **k:
                        old if c.provenance[0] == "exact-bdi" else other)
    for alternatives, expected in (((('frame-tradeoff', unsafe),), old),
                                   ((('frame-tradeoff', unsafe), ('frame-safe', safe)), safe)):
        def frames(circuit, width, *, error):
            assert width == 1 and error == 1e-6
            return (alternatives, ()) if circuit.provenance[0] == "exact-givens" else (
                (), (("frame-failed", "ValueError: optional frame failed"),))

        monkeypatch.setattr(backend, "compile_frame_candidates", frames)
        result = synthesize(hamiltonian({"Z": 0.3}), 1, objective="t", error=1e-6)
        assert result.emitted_circuit is expected
        assert result.t_selection.no_regression_reference == "bdi-reference"
        assert result.t_count <= old.t_count and result.two_qubit_gates <= old.n_2qb_gates()
        assert dict(result.t_selection.candidate_t_counts)["givens-reference/frame-tradeoff"] == 9
        assert dict(result.t_selection.candidate_cx_counts)["givens-reference/frame-tradeoff"] == 4
        assert result.t_selection.rejected_candidates == (
            ("bdi-reference/frame-failed", "ValueError: optional frame failed"),)
        assert result.routes == ["exact-bdi" if expected is old else "exact-givens"]


def test_public_shared_frame_improves_encoded_su2_with_retained_artifact():
    pytest.importorskip("pygridsynth")
    operator = hamiltonian({"XX": 0.31, "XY": -0.47, "IZ": 0.23})
    result = synthesize(operator, 0.7, objective="t", error=5e-7)
    selection = result.t_selection
    t, cx = dict(selection.candidate_t_counts), dict(selection.candidate_cx_counts)
    assert selection.no_regression_reference == "bdi-reference"
    assert t[selection.selected] == result.t_count == t[selection.no_regression_reference]
    assert cx[selection.selected] == result.two_qubit_gates == 2 < cx[selection.no_regression_reference]
    assert "/frame-" in selection.selected
    assert np.linalg.norm(result.emitted_circuit.get_unitary() - evolution(operator, 0.7), 2) < 1e-6
    assert result.emitted_circuit.error_budget == 5e-7
    assert not result.error_guaranteed


def test_auto_cx_gives_both_exact_algorithms_equal_emission_effort(monkeypatch):
    import lizzy.synthesis._routing as synthesis_module

    operator = hamiltonian({"XXX": 0.7, "XXY": -0.2, "IIZ": 0.4})
    calls = []
    winner = "exact-givens"

    def quote(circuit, width):
        route = circuit.provenance[0]
        calls.append(route)
        return EmissionQuote("pytket-greedy", 0 if route == winner else 10, object())

    monkeypatch.setattr(synthesis_module, "greedy_emission_quote", quote)
    for winner in ("exact-givens", "exact-bdi"):
        calls.clear()
        result = synthesize(operator, 0.8, steps=2)
        assert result.routes == [winner]
        assert result.two_qubit_gates == 0
        assert calls.count("exact-bdi") == calls.count("exact-givens") == 1
        assert result.error_guaranteed  # Exact selections keep this even with fixed steps.
        assert infidelity(evolution(operator, 0.8), circuit_matrix(result.circuit, 3)) < 1e-12


@pytest.mark.parametrize("objective", ["cx", "t"])
def test_gaussian_delegation_skips_dla_work_and_reports_comparison_cap(monkeypatch, objective):
    import lizzy.emission.emit as emission_module
    import lizzy.synthesize as synthesis_module
    from lizzy.fermions import gaussian

    def no_dla_work(*args, **kwargs):
        pytest.fail("Explicit or large delegated Gaussian synthesis must not classify or split a DLA")

    def diagonal_circuit(operator, time, **kwargs):
        result = Circuit()
        for coefficient, word in terms_of(operator):
            result.add(word, float(coefficient.real) * time, "exact-gaussian")
        return result

    def unavailable_shared_frame(*args, **kwargs):
        raise ValueError("Optional frame optimization failed")

    monkeypatch.setattr(gaussian, "decompose", diagonal_circuit)
    monkeypatch.setattr(emission_module, "native_frame_circuit", unavailable_shared_frame)
    for name in ("classify", "summands", "z2_symmetries", "commuting_clusters"):
        monkeypatch.setattr(synthesis_module, name, no_dla_work)
    for method, width in (("gaussian", 2), ("auto", 9)):
        operator = hamiltonian({"Z" + "I" * (width - 1): math.pi / 8, "I" * width: 0.19})
        result = synthesize(operator, 1.0, method=method, objective=objective)
        assert result.routes == ["exact-gaussian"]
        assert result.algebra == "JW quadratic (Gaussian)"
        assert result.symmetries is result.clusters is None
        assert not result.error_guaranteed
        assert result.emission_is_concrete
        assert result.two_qubit_gates == 0
        assert bool(result.routing_notes) == (method == "auto")
        if objective == "t":
            label = "gaussian-reference" if method == "auto" else "reference"
            assert dict(result.t_selection.candidate_t_counts)[label] == 1
            assert result.t_selection.selected == label
            assert bool(result.t_selection.rejected_candidates) == (method == "auto")
        else:
            assert result.emission_backend == "native-ladder"
        if width == 2:
            np.testing.assert_allclose(result.emitted_circuit.get_unitary(), evolution(operator, 1.0),
                                       atol=1e-12)

    def failed_residual(*args, **kwargs):
        raise ValueError("Gaussian residual exceeded the requested tolerance")

    monkeypatch.setattr(gaussian, "decompose", failed_residual)
    with pytest.raises(ValueError, match="Automatic DLA fallback was not attempted"):
        synthesize(hamiltonian({"Z" + "I" * 8: 0.7}), 1.0, objective=objective)


def test_gaussian_optional_dependency_falls_back_only_in_auto(monkeypatch):
    from lizzy.fermions import gaussian

    operator = hamiltonian({"Z": math.pi / 8})

    def unavailable(*args, **kwargs):
        raise ImportError("Install lizzy[gaussian] for OpenFermion synthesis")

    monkeypatch.setattr(gaussian, "decompose", unavailable)
    for objective in ("cx", "t"):
        with pytest.raises(ImportError, match=r"lizzy\[gaussian\]"):
            synthesize(operator, 1.0, method="gaussian", objective=objective)
        result = synthesize(operator, 1.0, objective=objective)
        assert "exact-gaussian" not in result.routes
        assert "OpenFermion" in result.routing_notes[0]
        if objective == "t":
            assert any(label == "gaussian-reference" and "ImportError" in reason
                       for label, reason in result.t_selection.rejected_candidates)


def test_auto_gaussian_t_candidate_receives_whole_input_and_common_budget(monkeypatch):
    import lizzy.emission.clifford_t as backend
    from lizzy.fermions import gaussian

    operator = hamiltonian({"ZI": math.pi / 8, "IZ": -math.pi / 4, "II": 0.19})
    calls, compiled = [], []

    def decompose(actual, time, *, error_tolerance):
        assert error_tolerance == pytest.approx(1e-11)
        calls.append(actual)
        circuit = Circuit()
        for coefficient, word in terms_of(actual):
            circuit.add(word, float(coefficient.real) * time, "exact-gaussian")
        return circuit

    def compile_t(circuit, width, *, error):
        assert width == 2 and error == 1e-10
        if "exact-gaussian" in circuit.provenance:
            assert {str(word) for word, _ in circuit.rotations} == {"ZI", "IZ", "II"}
        # Deliberately make the whole Gaussian candidate win, without using
        # estimates or deciding separately for its three commuting terms.
        count = 1 if "exact-gaussian" in circuit.provenance else 7
        emitted = SimpleNamespace(t_count=count, n_2qb_gates=lambda: 0)
        compiled.append((circuit, emitted))
        return emitted

    monkeypatch.setattr(gaussian, "decompose", decompose)
    monkeypatch.setattr(backend, "compile_clifford_t", compile_t)
    monkeypatch.setattr(backend, "compile_frame_candidates", lambda *a, **k: ((), ()))
    result = synthesize(operator, 1.0, error=1e-10, objective="t")
    assert calls == [operator]
    assert result.t_selection.selected == "gaussian-reference"
    assert result.t_count == 1
    assert result.emitted_circuit is compiled[-1][1]
    assert result.circuit is compiled[-1][0]
    assert result.routes == ["exact-gaussian"]


def test_auto_gaussian_cx_compares_complete_artifacts_and_retains_scalar_phase(monkeypatch):
    pytest.importorskip("openfermion")
    import lizzy.synthesize as synthesis_module
    from lizzy.emission.native import NativeGate, ladder_circuit

    operator = hamiltonian({"ZI": 0.31, "IZ": -0.17, "II": 0.19})
    previous = []

    def redundant_quote(circuit, width):
        native = ladder_circuit(circuit, width)
        # A valid but suboptimal whole-circuit emitter, not a fabricated cost:
        # the two extra CNOTs really exist and multiply to the identity.
        native.append(NativeGate("cx", (0, 1)))
        native.append(NativeGate("cx", (0, 1)))
        previous.append(native)
        return EmissionQuote("test-native", native.two_qubit_gates, native)

    monkeypatch.setattr(synthesis_module, "best_emission", redundant_quote)
    result = synthesize(operator, 0.7, steps=2)
    assert result.routes == ["exact-gaussian"]
    assert result.two_qubit_gates == 0 < previous[-1].two_qubit_gates
    assert result.emission_is_concrete
    assert not result.error_guaranteed and not result.randomized
    np.testing.assert_allclose(result.emitted_circuit.get_unitary(), evolution(operator, 0.7), atol=1e-11)

    def broken_gaussian_quote(*args, **kwargs):
        raise ValueError("Gaussian emitter failed its numerical validation")

    monkeypatch.setattr(synthesis_module, "_gaussian_emission", broken_gaussian_quote)
    fallback = synthesize(operator, 0.7, steps=2)
    assert "exact-gaussian" not in fallback.routes
    assert fallback.emitted_circuit is previous[-1]
    assert fallback.two_qubit_gates == 2
    assert "Gaussian emission rejected" in fallback.routing_notes[-1]


def test_explicit_gaussian_full_unitary_retains_pairing_scalar_phase_and_rejects_interactions():
    pytest.importorskip("openfermion")
    operator = hamiltonian({"XX": 0.31, "YY": -0.23, "XY": 0.17,
                            "YX": -0.11, "ZI": 0.19, "IZ": -0.27, "II": 0.37})
    result = synthesize(operator, -0.7, method="gaussian")
    np.testing.assert_allclose(result.emitted_circuit.get_unitary(), evolution(operator, -0.7), atol=1e-11)
    assert not result.error_guaranteed
    assert result.emission_is_concrete
    with pytest.raises(ValueError, match="not Jordan--Wigner quadratic"):
        synthesize(hamiltonian({"XX": 0.3, "ZZ": 0.2}), 0.7, method="gaussian")
