"""Exact synthesis: mathematical reconstruction, reusable plans, and API contracts.

The separate mapping tests verify every Lie bracket. Here independent SO and
qubit references check factor order, spin-cover phase, and public dispatch.
"""

import importlib
import math
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from lizzy.dense import circuit_matrix, evolution
from lizzy.emission.emit import EmissionQuote
from lizzy.hamiltonian import hamiltonian, model, terms_of
from lizzy.synthesis import exact
from lizzy.synthesize import synthesize

synthesis = importlib.import_module("lizzy.synthesize")


def _assert_up_to_phase(actual, target, *, atol=5e-11):
    phase = np.trace(target.conj().T @ actual) / len(target)
    assert abs(phase) == pytest.approx(1.0, abs=atol)
    np.testing.assert_allclose(actual, phase * target, atol=atol, rtol=0)


@pytest.mark.parametrize("family,width,time", [
    ("tfim", 3, -3.7),
    ("tfxy", 4, 12.3),
])
@pytest.mark.parametrize("method", ["givens", "bdi"])
def test_exact_product_order_sign_and_parameter_bound(family, width, time, method):
    operator = model(family, width, seed=11)
    terms = terms_of(operator)
    generators, _, _, size = exact._irrep(tuple(str(p) for _, p in terms), width)
    irrep_hamiltonian = sum(c.real * generators[str(p)] for c, p in terms)
    circuit = exact.decompose(operator, time, method=method)
    bound = (
        exact.prepare_bdi(operator).parameter_bound if method == "bdi"
        else size * (size - 1) // 2
    )
    assert len(circuit) <= bound

    # rho(+iP) = B_P, but Circuit emits exp(-i theta P) in application order.
    recovered = np.eye(size)
    for pauli, angle in circuit.rotations:
        recovered = expm(-angle * generators[str(pauli)]) @ recovered
    np.testing.assert_allclose(recovered, expm(-time * irrep_hamiltonian), atol=5e-12)
    _assert_up_to_phase(circuit_matrix(circuit, width), evolution(operator, time))


@pytest.mark.parametrize("method", ["givens", "bdi"])
def test_half_turn_is_retained_and_spin_cover_phase_contract_is_explicit(method):
    operator = hamiltonian({"X": 1.0})
    # SO(2) endpoint -I: a zero off-diagonal entry must not skip a negative pivot.
    half_turn = exact.decompose(operator, math.pi / 2, method=method)
    assert len(half_turn) == 1
    _assert_up_to_phase(circuit_matrix(half_turn, 1), evolution(operator, math.pi / 2))

    # exp(-i pi X) = -I, but endpoint-only SO synthesis loses the covering sign.
    full_turn = circuit_matrix(exact.decompose(operator, math.pi, method=method), 1)
    expected = -np.eye(2) if method == "bdi" else np.eye(2)
    np.testing.assert_allclose(full_turn, expected, atol=1e-12)
    np.testing.assert_allclose(evolution(operator, math.pi), -np.eye(2), atol=1e-12)


@pytest.mark.parametrize("size,case", [(5, "generic"), (8, "generic"), (7, "half-turn")])
def test_recursive_bdi_reconstructs_odd_even_and_degenerate_blocks(size, case):
    # SO targets independent of the Pauli mapper exercise recursion and determinant fixes.
    if case == "half-turn":
        target = np.diag([-1.0, -1.0, *([1.0] * (size - 2))])
    else:
        raw = np.random.default_rng(71).normal(size=(size, size))
        target = expm(raw - raw.T)
    planes = list(exact._bdi_planes(target))
    assert len(planes) == size * (size - 1) // 2
    recovered = np.eye(size)
    for i, j, angle in planes:
        cosine, sine = np.cos(angle), np.sin(angle)
        plane = np.eye(size)
        plane[np.ix_([i, j], [i, j])] = [[cosine, sine], [-sine, cosine]]
        recovered = plane @ recovered
    np.testing.assert_allclose(recovered, target, atol=2e-12)


@pytest.mark.parametrize("terms,partition", [
    ({"XXX": 0.31, "XXY": -0.72}, (1, 2)),
    ({"XI": 0.2, "YI": -0.31, "ZX": 0.47, "ZY": 0.63}, (1, 4)),
], ids=["encoded", "unbalanced"])
def test_horizontal_bdi_is_generic_and_preserves_full_evolution(terms, partition):
    operator = hamiltonian(terms)
    plan = exact.prepare_bdi(operator)
    width = len(next(iter(terms)))
    assert plan.mapping_kind == "horizontal-graph" and plan.phase_preserving
    assert tuple(sorted(plan.partition)) == partition
    p, q = partition
    assert plan.parameter_bound == p * (p - 1) + q * (q - 1) + min(p, q)
    if q - p > 1:
        assert plan.parameter_bound > plan.irrep_size * (plan.irrep_size - 1) // 2
    circuit = plan.circuit(-2.7)
    assert len(circuit) <= plan.parameter_bound
    np.testing.assert_allclose(
        circuit_matrix(circuit, width), evolution(operator, -2.7), atol=2e-11,
    )


def test_prepared_wings_are_reused_and_cache_depends_on_coefficients(monkeypatch):
    operator = hamiltonian({"XX": 0.47, "ZI": -0.33, "IZ": 0.21})
    plan = exact.prepare_bdi(operator)
    assert plan._wing
    assert exact.prepare_bdi(operator) is plan

    def no_refactor(*args, **kwargs):
        pytest.fail("evaluating a prepared horizontal circuit must not refactor K")

    with monkeypatch.context() as patch:
        patch.setattr(exact, "horizontal_generator_decomposition", no_refactor)
        patch.setattr(exact, "_bdi_planes", no_refactor)
        forward, reverse = plan.circuit(2.3), plan.circuit(-2.3)
        np.testing.assert_allclose(
            circuit_matrix(forward, 2), evolution(operator, 2.3), atol=1e-11,
        )
        np.testing.assert_allclose(
            circuit_matrix(reverse, 2), circuit_matrix(forward, 2).conj().T, atol=1e-11,
        )
        assert len(plan.circuit(0)) == 0
        assert forward.rotations[-len(plan._wing):] == reverse.rotations[-len(plan._wing):]

    changed = hamiltonian({"XX": 0.48, "ZI": -0.33, "IZ": 0.21})
    changed_plan = exact.prepare_bdi(changed)
    assert changed_plan is not plan
    np.testing.assert_allclose(
        circuit_matrix(changed_plan.circuit(2.3), 2), evolution(changed, 2.3), atol=1e-11,
    )


def test_mutating_an_emitted_pauli_cannot_corrupt_a_cached_plan():
    operator = hamiltonian({"X": 0.731})
    plan = exact.prepare_bdi(operator)
    circuit = plan.circuit(0.7)
    circuit.rotations[0][0].bits[0] = False
    circuit.rotations.clear()
    repeated = plan.circuit(0.7)
    assert str(repeated.rotations[0][0]) == "X"
    np.testing.assert_allclose(circuit_matrix(repeated, 1), evolution(operator, 0.7), atol=1e-12)


def test_t_aware_bdi_reduces_wing_cost_without_changing_paper_recursion(monkeypatch):
    operator = hamiltonian({"XI": 0.2, "YI": -0.31, "ZX": 0.47, "ZY": 0.63, "ZZ": 0.89})
    reference = exact.prepare_bdi(operator)
    actual_recursion = exact.recursive_bdi
    calls = []

    def traced_recursion(*args, **kwargs):
        calls.append(kwargs)
        return actual_recursion(*args, **kwargs)

    monkeypatch.setattr(exact, "recursive_bdi", traced_recursion)
    plan = exact.prepare_bdi(operator, optimize="t", cache=False)
    diagnostics = plan.optimization
    assert reference.optimization is None
    assert diagnostics.eligible and diagnostics.optimized
    assert diagnostics.selected_t_estimate < diagnostics.original_t_estimate
    assert diagnostics.selected_nonclifford < diagnostics.original_nonclifford
    assert 1 < diagnostics.candidates <= 9
    assert len(calls) == diagnostics.candidates  # The SO(1) block emits nothing.
    assert all(call == {"first_is_horizontal": False, "validate": True} for call in calls)
    assert plan._cartan == reference._cartan
    assert plan.parameter_bound == reference.parameter_bound
    assert len(plan._wing) == len(reference._wing)  # No unproved quotient bound.
    for time in (-2.7, 0.61, 17.2):
        np.testing.assert_allclose(
            circuit_matrix(plan.circuit(time), 2), evolution(operator, time), atol=2e-11,
        )
    repeated = exact.prepare_bdi(operator, optimize="t", cache=False)
    assert repeated == plan
    cached = exact.prepare_bdi(operator, optimize="t")
    assert exact.prepare_bdi(operator, optimize="t") is cached
    assert exact.prepare_bdi(operator, optimize="t", rotation_error=1e-7) is not cached
    assert exact.prepare_bdi(operator) is reference


@pytest.mark.parametrize("partition", [(2, 5), (5, 2)])
def test_bdi_gauge_completion_handles_both_orientations_and_rank_deficiency(partition):
    p, q = partition
    # Rank one leaves extra accidental freedom; search must only use the
    # structural |p-q| null columns, retaining every original active column.
    rectangular = np.outer(np.linspace(-0.3, 0.8, p), np.linspace(0.2, 1.1, q))
    generator = np.zeros((p + q, p + q))
    generator[:p, p:] = rectangular
    generator[p:, :p] = -rectangular.T
    k, rates, planes = exact.horizontal_generator_decomposition(generator, p)
    central = np.zeros_like(generator)
    for (i, j), rate in zip(planes, rates, strict=True):
        central[i, j], central[j, i] = rate, -rate
    candidates = list(exact._bdi_nullspace_candidates(k, p))
    assert 0 < len(candidates) <= 8
    active = sorted({axis for plane in planes for axis in plane})
    for candidate in candidates:
        np.testing.assert_array_equal(candidate[:, active], k[:, active])
        np.testing.assert_allclose(candidate.T @ candidate, np.eye(p + q), atol=1e-12)
        assert np.linalg.det(candidate[:p, :p]) == pytest.approx(1.0)
        assert np.linalg.det(candidate[p:, p:]) == pytest.approx(1.0)
        np.testing.assert_allclose(candidate @ central @ candidate.T, generator, atol=1e-12)


def test_t_aware_bdi_rejects_invalid_gauges_and_retains_reference(monkeypatch):
    operator = hamiltonian({"XI": 0.2, "YI": -0.31, "ZX": 0.47, "ZY": 0.63, "ZZ": 0.89})
    reference = exact.prepare_bdi(operator)

    def invalid_candidates(k, p):
        # A null reflection preserves H but is not an SO gauge.
        reflection = k.copy()
        reflection[:, p] *= -1
        yield reflection
        # An orthogonal, determinant-one change of an active frame changes H.
        changed = k.copy()
        changed[:, [p, -1]] *= -1
        yield changed
        yield 2 * k

    monkeypatch.setattr(exact, "_bdi_nullspace_candidates", invalid_candidates)
    plan = exact.prepare_bdi(operator, optimize="t", cache=False)
    assert plan._wing == reference._wing
    assert plan.optimization.candidates == 1
    assert not plan.optimization.optimized
    assert plan.optimization.selected_t_estimate == plan.optimization.original_t_estimate


def test_t_aware_bdi_reports_ineligible_partitions_and_validates_controls():
    plan = exact.prepare_bdi(hamiltonian({"X": 0.71}), optimize="t")
    assert not plan.optimization.eligible and not plan.optimization.optimized
    assert plan.optimization.candidates == 1
    for options in (
        {"optimize": "unknown"}, {"rotation_error": 0},
        {"rotation_error": float("nan")}, {"rotation_error": 1},
    ):
        with pytest.raises(ValueError):
            exact.prepare_bdi(hamiltonian({"X": 0.71}), **options)


def test_nonhorizontal_bdi_uses_general_mapping_without_givens(monkeypatch):
    words = ["".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")]
    operator = hamiltonian({word: (index + 1) / 29 for index, word in enumerate(words)})
    plan = exact.prepare_bdi(operator)
    assert plan.mapping_kind == "general-graph" and not plan.phase_preserving
    assert plan.irrep_size == 6 and plan.parameter_bound == 15

    def no_legacy_mapping(*args, **kwargs):
        pytest.fail("paper BDI must not reuse the Givens mapping/order")

    monkeypatch.setattr(exact, "_irrep", no_legacy_mapping)
    result = synthesize(operator, 0.81, method="bdi")
    _assert_up_to_phase(circuit_matrix(result.circuit, 2), evolution(operator, 0.81))


@pytest.mark.parametrize("full_su4", [False, True], ids=["encoded-su2", "generic-su4"])
def test_nonhorizontal_givens_uses_full_mapping_and_public_dispatch(monkeypatch, full_su4):
    if full_su4:
        words = ["".join(word) for word in product("IXYZ", repeat=2) if word != ("I", "I")]
        operator = hamiltonian({word: (index + 1) / 29 for index, word in enumerate(words)})
        dimension = 15
    else:
        operator = hamiltonian({"XX": 0.31, "XY": -0.47, "IZ": 0.23})
        dimension = 3
    assert exact.is_decomposable(operator)

    def no_bdi_algorithm(*args, **kwargs):
        pytest.fail("sharing an algebra mapping must not replace Givens with BDI")

    monkeypatch.setattr(exact, "_bdi_planes", no_bdi_algorithm)
    monkeypatch.setattr(exact, "horizontal_generator_decomposition", no_bdi_algorithm)
    for time in (-2.7, 0.7):
        circuit = exact.decompose(operator, time, method="givens")
        assert len(circuit) <= dimension
        _assert_up_to_phase(circuit_matrix(circuit, 2), evolution(operator, time))
        result = synthesize(operator, time, method="givens")
        assert result.routes == ["exact-givens"]
        _assert_up_to_phase(circuit_matrix(result.circuit, 2), evolution(operator, time))


def test_givens_keeps_horizontal_mapping_and_propagates_unrelated_errors(monkeypatch):
    operator = model("tfim", 3, seed=11)
    reference = exact.decompose(operator, 0.7, method="givens")
    monkeypatch.setattr(exact, "_irrep_cache", {})

    def no_fallback(*args, **kwargs):
        pytest.fail("a successful horizontal mapping must retain its original plane ordering")

    monkeypatch.setattr(exact, "map_orthogonal", no_fallback)
    repeated = exact.decompose(operator, 0.7, method="givens")
    assert repeated.rotations == reference.rotations
    _assert_up_to_phase(circuit_matrix(repeated, 3), evolution(operator, 0.7))

    def invalid_mapping(*args, **kwargs):
        raise ValueError("invalid Lie mapping")

    monkeypatch.setattr(exact, "_irrep_cache", {})
    monkeypatch.setattr(exact, "map_dla_to_irrep", invalid_mapping)
    with pytest.raises(ValueError, match="invalid Lie mapping"):
        exact.decompose(operator, 0.7, method="givens")


@pytest.mark.parametrize("method", ["bdi", "givens"])
def test_explicit_exact_method_compiles_all_summands_without_auto_routing(monkeypatch, method):
    operator = model("xy", 4, seed=2)
    actual_decompose = exact.decompose
    dispatched = []

    def trace_decompose(part, time, route, *, method):
        dispatched.append((route, method))
        return actual_decompose(part, time, route=route, method=method)

    def no_auto_routing(*args, **kwargs):
        pytest.fail("an explicit exact method must not compete with product formulas")

    monkeypatch.setattr(exact, "decompose", trace_decompose)
    monkeypatch.setattr(synthesis, "select_part", no_auto_routing)
    result = synthesize(operator, -0.73, method=method)
    assert result.summands == len(dispatched) == 2
    assert dispatched == [(f"exact-{method}", method)] * 2
    assert result.routes == [f"exact-{method}"]
    assert result.error_guaranteed and not result.routing_estimated
    assert result.numerical is None and result.emission is not None
    assert result.emission_is_concrete == result.emission.is_concrete
    achieved = circuit_matrix(result.circuit, 4)
    _assert_up_to_phase(achieved, evolution(operator, -0.73))
    if result.emission_is_concrete:
        _assert_up_to_phase(result.emitted_circuit.get_unitary(), achieved)


@pytest.mark.parametrize("method", ["bdi", "givens"])
def test_explicit_exact_method_rejects_unsupported_algebra(method):
    with pytest.raises(NotImplementedError, match=r"so\(m\)"):
        synthesize(model("heisenberg", 4, seed=2), 0.2, method=method)


@pytest.mark.parametrize("method,options,message", [
    ("bdi", {"steps": 2}, "product-formula controls"),
    ("givens", {"numerical_options": {}}, "requires method='wei-norman'"),
])
def test_explicit_exact_method_rejects_other_solver_controls(method, options, message):
    with pytest.raises(ValueError, match=message):
        synthesize(hamiltonian({"X": 1.0}), 0.2, method=method, **options)


def test_bdi_backend_failure_is_propagated_without_fallback(monkeypatch):
    def unavailable(*args, **kwargs):
        assert kwargs["method"] == "bdi"
        raise NotImplementedError("unsupported BDI presentation")

    monkeypatch.setattr(exact, "decompose", unavailable)
    with pytest.raises(NotImplementedError, match="unsupported BDI presentation"):
        synthesize(hamiltonian({"X": 1.0}), 0.2, method="bdi")


def test_zero_time_preserves_explicit_route_and_logical_emission_contract(monkeypatch):
    def logical_quote(circuit, width):
        assert width == 1
        return EmissionQuote("builtin", circuit.two_qubit_gates, circuit)

    monkeypatch.setattr(synthesis, "best_emission", logical_quote)
    result = synthesize(hamiltonian({"X": 1.0}), 0.0, method="bdi")
    assert result.routes == ["exact-bdi"] and len(result.circuit) == 0
    assert result.error_guaranteed
    assert not result.emission_is_concrete
    assert result.emitted_circuit is result.circuit
