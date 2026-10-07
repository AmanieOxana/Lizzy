"""Small dense oracles and fixed cases for the retained validation programs.

These exponentially sized references are experiment-only tools, never compiler
inputs. The corpus, seeds, ODE settings and error metrics are kept fixed so
moving the runners out of the installed package does not change their evidence.
"""

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lizzy.dense import operator_errors, pauli_matrix
from lizzy.emission.emit import native_emission_candidates
from lizzy.synthesis.driven import DrivenHamiltonian


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    hamiltonian: DrivenHamiltonian
    time_span: tuple[float, float]
    target: np.ndarray
    reference: str
    static_coefficients: tuple[float, ...] | None = None
    expected_cap: bool = False


def _dense_reference(hamiltonian, time_span):
    """Integrate the full Schrodinger equation independently of Lie coordinates."""
    matrices = np.array([pauli_matrix(w) for w in hamiltonian.paulis])
    dimension = 2**hamiltonian.n_qubits

    def rhs(time, flattened):
        generator = np.einsum("k,kij->ij", hamiltonian.at(time), matrices)
        return (-1j * generator @ flattened.reshape(dimension, dimension)).ravel()

    solution = solve_ivp(
        rhs,
        time_span,
        np.eye(dimension, dtype=complex).ravel(),
        method="DOP853",
        rtol=3e-13,
        atol=1e-14,
        max_step=0.005,
    )
    if not solution.success:
        raise RuntimeError(f"Dense reference failed: {solution.message}")
    return solution.y[:, -1].reshape(dimension, dimension)


def _static_case(name, words, coefficients, duration, *, expected_cap=False):
    coefficients = tuple(float(c) for c in coefficients)
    driven = DrivenHamiltonian(words, lambda t: coefficients)
    generator = sum(c * pauli_matrix(p) for p, c in zip(words, coefficients))
    return BenchmarkCase(
        name, driven, (0.0, duration), expm(-1j * duration * generator),
        "dense expm", coefficients, expected_cap,
    )


def benchmark_cases():
    """Fixed families/seeds; at most three qubits, including unfavorable regimes."""
    yield _static_case("static-commuting", ["ZZI", "IZZ", "ZIZ"], [0.7, -0.4, 0.3], 1.2)
    span = (0.2, 1.4)
    commuting = DrivenHamiltonian(["ZZI", "IZZ", "ZIZ"], lambda t: [t, -0.4, 0.3 * t])
    integral = [0.5 * (span[1]**2 - span[0]**2), -0.4 * (span[1] - span[0])]
    generator = (integral[0] * pauli_matrix("ZZI") + integral[1] * pauli_matrix("IZZ")
                 + 0.3 * integral[0] * pauli_matrix("ZIZ"))
    yield BenchmarkCase("driven-commuting", commuting, span, expm(-1j * generator), "analytic integral")
    yield _static_case("static-spin", ["X", "Z"], [0.8, -0.3], 1.2)
    yield _static_case("static-encoded", ["XXX", "XXY", "IIZ"], [0.8, 0.3, -0.2], 1.2)
    yield _static_case("commuting-su2", ["XII", "ZII", "IXI", "IZI", "IIX", "IIZ"],
                       [0.7, 0.2, -0.4, 0.5, 0.3, -0.6], 1.2)

    omega, detuning = 1.7, -0.24
    for name, words in (("driven-spin", ["X", "Y", "Z"]),
                        ("driven-encoded", ["XXX", "XXY", "IIZ"])):
        driven = DrivenHamiltonian(words, lambda t: [np.cos(omega*t), np.sin(omega*t), detuning])
        x_axis, _, z_axis = map(pauli_matrix, words)
        target = (expm(-0.5j * omega * span[1] * z_axis)
                  @ expm(-1j * (span[1] - span[0]) * (x_axis + (detuning - omega/2)*z_axis))
                  @ expm(0.5j * omega * span[0] * z_axis))
        yield BenchmarkCase(name, driven, span, target, "analytic rotating frame")

    words = ["ZZI", "IZZ", "XII", "IXI", "IIX"]
    coefficients = [0.7, -0.45, 0.35, 0.5, -0.4]
    for name, duration in (("short-tfim3", 0.003), ("static-tfim3", 1.0), ("long-tfim3", 5.0)):
        yield _static_case(name, words, coefficients, duration)
    driven = DrivenHamiltonian(words, lambda t: [0.7 + 0.12*np.sin(0.9*t),
        -0.45 + 0.08*np.cos(1.1*t), 0.35 + 0.11*np.cos(0.7*t),
        0.5 + 0.13*np.sin(1.3*t), -0.4 + 0.09*np.cos(1.7*t)])
    yield BenchmarkCase("driven-tfim3", driven, (0.0, 1.2),
                        _dense_reference(driven, (0.0, 1.2)), "dense DOP853, max_step=.005")

    rng = np.random.default_rng(20260921)
    words = ["".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I")]
    yield _static_case("static-su4", words, rng.uniform(-0.3, 0.3, len(words)), 0.8)
    words = ["XI", "ZI", "IX", "IZ", "ZZ"]
    driven = DrivenHamiltonian(words, lambda t: [0.4 + 0.1*np.sin(t), 0.2*np.cos(1.3*t),
        -0.3 + 0.07*np.cos(t), 0.25 + 0.04*np.sin(0.8*t), 0.6 + 0.11*np.sin(1.1*t)])
    yield BenchmarkCase("driven-su4", driven, (0.1, 0.9),
                        _dense_reference(driven, (0.1, 0.9)), "dense DOP853, max_step=.005")
    yield _static_case("su8-closure-cap", ["XII", "ZII", "IXI", "IZI", "IIX", "IIZ", "ZZI", "IZZ"],
                       [0.3, -0.4, 0.2, 0.25, -0.35, 0.1, 0.45, -0.3], 0.4, expected_cap=True)


def _errors(achieved, target):
    """Strict norm and trace-phase-aligned norm (not an asserted phase optimum)."""
    errors = operator_errors(achieved, target)
    return errors["phase_aligned"], errors["strict"]


def _emission(circuit, width):
    """Concrete, fixed portfolio: counts are from artifacts, not block estimates."""
    candidates = native_emission_candidates(circuit, width)
    return min(candidates, key=lambda item: (item[1].two_qubit_gates, len(item[1].gates)))


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("lizzy", "numpy", "scipy", "paulie", "kak_tools"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    return versions
