"""Full-unitary contracts for optional OpenFermion Gaussian synthesis."""

import builtins

import numpy as np
import pytest

from lizzy.dense import circuit_matrix, evolution
from lizzy.emission.native import native_frame_circuit
from lizzy.fermions import gaussian
from lizzy.hamiltonian import hamiltonian, model


def test_general_quadratic_paulis_preserve_full_unitary_without_dense_synthesis(monkeypatch):
    openfermion = pytest.importorskip("openfermion")
    cirq = pytest.importorskip("cirq")
    original = openfermion.bogoliubov_transform
    state_assumptions = []

    def full_transform(*args, **kwargs):
        state_assumptions.append(kwargs.get("initial_state", "not supplied"))
        return original(*args, **kwargs)

    def no_dense(*args, **kwargs):
        raise AssertionError("Synthesis must not form a full Hilbert-space matrix.")

    monkeypatch.setattr(openfermion, "bogoliubov_transform", full_transform)
    monkeypatch.setattr(openfermion, "get_sparse_operator", no_dense)
    monkeypatch.setattr(cirq, "unitary", no_dense)
    rng = np.random.default_rng(731)
    for width in (2, 3, 4):
        terms = {"I" * width: 0.23}
        for first in range(width):
            terms["I" * first + "Z" + "I" * (width - first - 1)] = rng.normal()
            for last in range(first + 1, width):
                for left in "XY":
                    for right in "XY":
                        word = "I" * first + left + "Z" * (last - first - 1) + right + "I" * (width - last - 1)
                        terms[word] = rng.normal()
        operator = hamiltonian(terms)
        assert gaussian.is_gaussian(operator)
        for time in (0.0, 0.37, -0.51):
            logical = gaussian.decompose(operator, time)
            assert set(logical.provenance) <= {"exact-gaussian"}
            assert np.linalg.norm(circuit_matrix(logical, width) - evolution(operator, time), 2) < 2e-12
    assert state_assumptions and all(state is None for state in state_assumptions)


def test_upstream_tensor_api_includes_chemical_potential_and_validates_inputs():
    openfermion = pytest.importorskip("openfermion")
    matrix = np.array([[1.3, 0.7 + 0.2j], [0.7 - 0.2j, -0.5]])
    pairing = np.array([[0, 0.4 - 0.6j], [-0.4 + 0.6j, 0]])
    operator = openfermion.QuadraticHamiltonian(matrix, pairing, constant=0.17, chemical_potential=0.11)
    terms = {"II": 0.46, "ZI": -0.595, "IZ": 0.305, "XX": 0.55, "YY": 0.15, "YX": -0.2, "XY": -0.4}
    logical = gaussian.from_quadratic(operator, -0.71)
    target = evolution(hamiltonian(terms), -0.71)
    assert np.linalg.norm(circuit_matrix(logical, 2) - target, 2) < 2e-12
    assert np.linalg.norm(native_frame_circuit(logical, 2).get_unitary() - target, 2) < 2e-12
    for invalid, error in (
        (openfermion.QuadraticHamiltonian(matrix + np.array([[0, 1e-12], [0, 0]]), pairing), "Hermitian"),
        (openfermion.QuadraticHamiltonian(matrix, pairing + np.eye(2) * 1e-12), "antisymmetric"),
        (openfermion.QuadraticHamiltonian(matrix, pairing, constant=1j), "constant"),
        (openfermion.QuadraticHamiltonian(matrix * np.nan, pairing), "finite"),
    ):
        with pytest.raises(ValueError, match=error):
            gaussian.from_quadratic(invalid, 0.2)
    with pytest.raises(TypeError, match="QuadraticHamiltonian"):
        gaussian.from_quadratic(matrix, 0.2)


def test_tiny_terms_are_never_silently_discarded_and_dependencies_remain_optional(monkeypatch):
    pytest.importorskip("openfermion")
    weak = {"XX": 0.5e-9, "YY": -0.5e-9, "II": 0.2e-9}
    # Do not construct the oracle through OF's PolynomialTensor iterator:
    # upstream iteration itself truncates tiny coefficients.
    circuit = gaussian.decompose(weak, 1e9)
    assert np.linalg.norm(circuit_matrix(circuit, 2) - evolution(hamiltonian(weak), 1e9), 2) < 2e-12
    mixed_scale = {**weak, "ZI": 0.7, "IZ": -0.4}
    with pytest.raises(ValueError, match="numerical guard"):
        gaussian.decompose(mixed_scale, 1e8)
    with pytest.raises(ValueError, match="numerical guard"):
        gaussian.decompose(
            {"XX": 5e-10, "YY": -5e-10, "ZI": 1.0, "IZ": -1.0},
            1.0, error_tolerance=1e-13,
        )

    original_import = builtins.__import__

    def without_optional(name, *args, **kwargs):
        if name.split(".")[0] in {"openfermion", "cirq"}:
            raise ImportError("optional dependency deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional)
    assert gaussian.is_gaussian({"XZZY": 0.3, "IIII": 0.7})
    assert not gaussian.is_gaussian({"XX": 1.0, "ZZ": 1e-30})
    assert not gaussian.is_gaussian({"XIY": 0.3})
    assert not gaussian.is_gaussian({"Z": 1j})
    with pytest.raises(ImportError, match=r"lizzy\[gaussian\]"):
        gaussian.decompose({"XX": 0.3, "IZ": 0.1}, 0.2)


def test_gaussian_circuit_uses_existing_clifford_t_backend_with_scalar_phase():
    pytest.importorskip("openfermion")
    pytest.importorskip("pygridsynth")
    from lizzy.emission.clifford_t import compile_native_clifford_t

    terms = {"XX": 0.31, "YY": -0.17, "ZI": 0.13, "II": 0.27}
    logical = gaussian.decompose(terms, -0.4)
    native = native_frame_circuit(logical, 2)
    compiled = compile_native_clifford_t(native, error=1e-6)
    assert compiled.t_count > 0
    assert np.linalg.norm(compiled.get_unitary() - evolution(hamiltonian(terms), -0.4), 2) < 1e-6


def test_fourier_stability_fallback_preserves_the_target_and_scales_without_dense(monkeypatch):
    openfermion = pytest.importorskip("openfermion")
    cirq = pytest.importorskip("cirq")
    replay = gaussian._basis_transform
    calls = 0

    def reject_initial_basis(circuit, width):
        nonlocal calls
        calls += 1
        left, right = replay(circuit, width)
        # Force the single documented fallback, then validate the real output.
        return (np.zeros_like(left), np.zeros_like(right)) if calls == 1 else (left, right)

    monkeypatch.setattr(gaussian, "_basis_transform", reject_initial_basis)
    terms = {"XXI": 0.4, "YYI": -0.2, "YZX": 0.7, "XZY": 0.1, "ZII": -0.3, "III": 0.27}
    circuit = gaussian.decompose(terms, -0.71)
    assert calls == 2
    assert np.linalg.norm(circuit_matrix(circuit, 3) - evolution(hamiltonian(terms), -0.71), 2) < 2e-12
    monkeypatch.setattr(gaussian, "_basis_transform", replay)

    def no_dense(*args, **kwargs):
        raise AssertionError("24-mode synthesis must not use a full Hilbert-space matrix.")

    monkeypatch.setattr(openfermion, "get_sparse_operator", no_dense)
    monkeypatch.setattr(cirq, "unitary", no_dense)
    # This localized input triggers near-zero elimination loss upstream; the
    # emitted composite basis must pass the SAME polynomial reconstruction guard.
    large = gaussian.decompose(model("tfxy", 24, seed=817), 0.73, error_tolerance=1e-10)
    assert 0 < len(large.rotations) < 30 * 24**2
