"""Focused checks for the optional pytket emission backends."""

import pytest
from paulie.common.pauli_string_factory import get_pauli_string

pytest.importorskip("pytket")

from pytket import OpType
from pytket.passes import AutoRebase, DecomposeBoxes

from lizzy.dense import circuit_matrix, infidelity
from lizzy.emit import (
    direct_tket_circuit,
    tket_circuit,
)
from lizzy.hamiltonian import Circuit


def _small_circuit() -> Circuit:
    circuit = Circuit()
    for word, angle in [
        ("XYZ", 0.21),
        ("YXI", -0.37),
        ("ZZX", 0.13),
        ("IYZ", 0.44),
    ]:
        circuit.add(get_pauli_string(word), angle, "test")
    return circuit


def test_direct_emission_optimizes_boxes_before_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin both pass order and the tuned, deterministic default configuration."""
    from pytket import passes

    circuit = _small_circuit()
    calls = []
    parameters = {}
    real_decompose_boxes = DecomposeBoxes
    real_auto_rebase = AutoRebase

    class RecordingPass:
        def __init__(self, name, inner=None):
            self.name = name
            self.inner = inner

        def apply(self, emitted):
            calls.append((self.name, emitted.n_gates_of_type(OpType.PauliExpBox)))
            if self.inner is not None:
                return self.inner.apply(emitted)
            return False

    def greedy_pauli_simp(**kwargs):
        parameters.update(kwargs)
        return RecordingPass("greedy")

    def decompose_boxes():
        return RecordingPass("decompose", real_decompose_boxes())

    def auto_rebase(gateset):
        assert gateset == {OpType.CX, OpType.TK1}
        return RecordingPass("rebase", real_auto_rebase(gateset))

    monkeypatch.setattr(passes, "GreedyPauliSimp", greedy_pauli_simp)
    monkeypatch.setattr(passes, "DecomposeBoxes", decompose_boxes)
    monkeypatch.setattr(passes, "AutoRebase", auto_rebase)

    direct_tket_circuit(circuit, 3)

    assert parameters == {
        "discount_rate": 0.9,
        "depth_weight": 0.0,
        "seed": 0,
    }
    assert calls == [
        ("greedy", len(circuit.rotations)),
        ("decompose", len(circuit.rotations)),
        ("rebase", 0),
    ]


@pytest.mark.parametrize("emitter", [direct_tket_circuit, tket_circuit])
def test_emission_is_exact_and_reports_its_actual_count(emitter) -> None:
    circuit = _small_circuit()

    emitted = emitter(circuit, 3)
    commands = emitted.get_commands()
    actual_two_qubit_gates = sum(len(command.args) == 2 for command in commands)

    assert {command.op.type for command in commands} <= {OpType.CX, OpType.TK1}
    assert infidelity(circuit_matrix(circuit, 3), emitted.get_unitary()) < 1e-12
    assert emitted.n_2qb_gates() == actual_two_qubit_gates

