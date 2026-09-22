"""Exactness and structural contracts for the dependency-free GF(2) emitter."""

import builtins

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.dense import circuit_matrix, infidelity
from lizzy.emit import best_emission
from lizzy.hamiltonian import Circuit, anticommutation_matrix, hamiltonian
from lizzy.native import (
    NativeCircuit,
    NativeGate,
    frame_profile,
    ladder_circuit,
    native_frame_circuit,
)
from lizzy.synthesize import synthesize


def _circuit(rotations: list[tuple[str, float]]) -> Circuit:
    circuit = Circuit()
    for word, angle in rotations:
        circuit.add(get_pauli_string(word), angle, "test")
    return circuit


def _assert_exact(logical: Circuit, emitted: NativeCircuit, width: int) -> None:
    target = circuit_matrix(logical, width)
    assert infidelity(target, emitted.get_unitary()) < 1e-12
    assert np.allclose(target, emitted.get_unitary(), rtol=0, atol=2e-12)
    assert emitted.n_2qb_gates() == sum(gate.kind == "cx" for gate in emitted.gates)
    assert all(isinstance(gate, NativeGate) for gate in emitted.gates)


def test_native_emission_preserves_identity_rotations_as_absolute_phase() -> None:
    large = 10_000 * np.pi + 0.37
    logical = _circuit(
        [
            ("II", large),
            ("XI", 0.23),
            ("ZI", -0.41),
            ("II", -7 * np.pi + 0.19),
            ("YI", 0.17),
        ]
    )

    emitted = native_frame_circuit(logical, 2)

    assert emitted.global_phase == pytest.approx(-large + 7 * np.pi - 0.19)
    _assert_exact(logical, emitted, 2)


def test_ladder_circuit_is_an_absolute_phase_preserving_generic_fallback() -> None:
    logical = _circuit(
        [
            ("XYZ", 0.23),
            ("III", 13 * np.pi + 0.11),
            ("ZXI", -0.31),
            ("IYY", 0.19),
        ]
    )

    emitted = ladder_circuit(logical, 3)

    assert emitted.two_qubit_gates == 8
    assert emitted.global_phase == pytest.approx(-(13 * np.pi + 0.11))
    _assert_exact(logical, emitted, 3)


def test_native_qasm3_is_deterministic_and_round_trip_precise() -> None:
    phase = float(np.nextafter(-0.2, -1.0))
    angle = float(np.nextafter(0.1, 1.0))
    emitted = NativeCircuit(2, global_phase=phase)
    for gate in [
        NativeGate("h", (0,)),
        NativeGate("s", (1,)),
        NativeGate("sdg", (0,)),
        NativeGate("cx", (0, 1)),
        NativeGate("rz", (1,), angle),
    ]:
        emitted.append(gate)

    expected = (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[2] q;\n"
        f"gphase({phase!r});\n"
        "h q[0];\n"
        "s q[1];\n"
        "inv @ s q[0];\n"
        "cx q[0], q[1];\n"
        f"rz({angle!r}) q[1];\n"
    )
    assert emitted.to_qasm3() == expected
    assert emitted.to_qasm3() == expected
    assert NativeCircuit(0, global_phase=phase).to_qasm3() == (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        f"gphase({phase!r});\n"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NativeCircuit(1, global_phase=np.inf),
        lambda: NativeCircuit(1.5),
        lambda: NativeGate("rz", (0,), np.nan),
        lambda: NativeGate("h", (0.5,)),
    ],
)
def test_native_values_reject_nonfinite_angles_and_fractional_indices(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_so3_frame_localizes_three_dependent_axes_once() -> None:
    """The shared prefix is one disguised qubit, not three separate ladders.

    The three words have the K3 anticommutation graph but only GF(2) rank two:
    ``IIZ`` is the product axis of ``XXX`` and ``XXY``.  Two fanout CXs enter that
    one-qubit frame and the same two leave it, independent of the rotation angles.
    """
    logical = _circuit([("XXX", 0.21), ("XXY", 0.37), ("IIZ", -0.13)])

    profile = frame_profile(logical)
    emitted = native_frame_circuit(logical, 3)

    assert profile.span_rank == 2
    assert profile.gram_rank == 2
    assert profile.canonical_qubits == 1
    assert profile.single_qubit_dla
    assert logical.two_qubit_gates == 8
    assert emitted.n_2qb_gates() == 4
    _assert_exact(logical, emitted, 3)


def test_commuting_ghz_stabilizers_share_one_global_frame() -> None:
    """Independent commuting stabilizers are diagonalized together.

    ``H(0); CX(0, 1); CX(0, 2)`` carries the three local Z generators onto these
    GHZ stabilizers.  Entering and leaving that frame costs four CXs instead of the
    eight CXs of three independent Pauli ladders.
    """
    logical = _circuit([("XXX", 0.19), ("ZZI", -0.31), ("ZIZ", 0.43)])

    emitted = native_frame_circuit(logical, 3)

    assert logical.two_qubit_gates == 8
    assert emitted.n_2qb_gates() == 4
    _assert_exact(logical, emitted, 3)


def test_dependent_commuting_word_keeps_its_operator_sign() -> None:
    """GF(2) dependence alone does not contain the Pauli product phase.

    In bits ``YYX = XXX + ZZI``, while as Hermitian operators
    ``XXX * ZZI = -YYX``.  Losing that minus sign produces a plausible frame with
    the wrong third rotation, so this is checked against the complete dense unitary.
    """
    logical = _circuit([("XXX", 0.17), ("ZZI", 0.29), ("YYX", 0.41)])

    emitted = native_frame_circuit(logical, 3)

    _assert_exact(logical, emitted, 3)


def test_same_anticommutation_graph_does_not_hide_a_rank_three_dla() -> None:
    """A K3 graph is insufficient evidence that three Paulis are one qubit.

    The first set is the dependent ``so(3)`` above.  The second has the same
    pairwise anticommutation graph but GF(2) rank three, so it needs at least two
    canonical qubits and must not take the single-qubit-DLA shortcut.
    """
    dependent = _circuit([("XXX", 0.1), ("XXY", 0.2), ("IIZ", 0.3)])
    independent = _circuit([("XXX", 0.1), ("XXY", 0.2), ("XXZ", 0.3)])
    dependent_paulis = [pauli for pauli, _ in dependent.rotations]
    independent_paulis = [pauli for pauli, _ in independent.rotations]

    assert np.array_equal(
        anticommutation_matrix(dependent_paulis),
        anticommutation_matrix(independent_paulis),
    )
    assert frame_profile(dependent).single_qubit_dla
    profile = frame_profile(independent)
    assert profile.span_rank == 3
    assert profile.canonical_qubits == 2
    assert not profile.single_qubit_dla

    _assert_exact(independent, native_frame_circuit(independent, 3), 3)


def test_native_frame_emission_is_deterministic() -> None:
    logical = _circuit(
        [("XXX", 0.17), ("ZZI", 0.29), ("YYX", 0.41), ("IIZ", -0.23)]
    )

    first = native_frame_circuit(logical, 3)
    second = native_frame_circuit(logical, 3)

    assert first.gates == second.gates
    assert first.n_2qb_gates() == second.n_2qb_gates()
    _assert_exact(logical, first, 3)


def test_native_rz_uses_the_standard_half_angle_convention() -> None:
    """A Lizzy rotation exp(-i theta Z) is a hardware Rz(2 theta)."""
    theta = 0.37
    logical = _circuit([("Z", theta)])

    emitted = native_frame_circuit(logical, 1)
    rotations = [gate for gate in emitted.gates if gate.kind == "rz"]

    assert len(rotations) == 1
    assert rotations[0].angle == pytest.approx(2 * theta)
    assert emitted.n_2qb_gates() == 0
    _assert_exact(logical, emitted, 1)


def test_best_emission_selects_native_frame_without_pytket(monkeypatch) -> None:
    logical = _circuit([("XXX", 0.21), ("XXY", 0.37), ("IIZ", -0.13)])
    real_import = builtins.__import__

    def without_pytket(name, *args, **kwargs):
        if name == "pytket" or name.startswith("pytket."):
            raise ImportError("simulated optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pytket)
    quote = best_emission(logical, 3)

    assert quote.backend == "native-frame"
    assert isinstance(quote.circuit, NativeCircuit)
    assert quote.two_qubit_gates == 4
    _assert_exact(logical, quote.circuit, 3)

    before = quote.two_qubit_gates
    quote.circuit.append(NativeGate("cx", (0, 1)))
    assert quote.two_qubit_gates == before + 1


def test_synthesis_retains_the_native_artifact_it_prices() -> None:
    """A repeated formula pays one shared DLA frame, not one ladder per gadget."""
    operator = hamiltonian({"XXX": 0.7, "XXY": -0.2, "IIZ": 0.4})

    result = synthesize(operator, time=1.0, steps=2)

    assert result.emission_backend == "native-frame"
    assert isinstance(result.emitted_circuit, NativeCircuit)
    assert len(result.circuit.rotations) == 9
    assert result.logical_two_qubit_gates == 24
    assert result.two_qubit_gates == 4
    assert (
        infidelity(
            circuit_matrix(result.circuit, 3), result.emitted_circuit.get_unitary()
        )
        < 1e-12
    )
