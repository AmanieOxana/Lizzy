"""Meaningful phase, precision, inverse-reuse and optional-backend contracts."""

import builtins
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.emission.clifford_t import (
    CliffordTCircuit,
    CliffordTGate,
    compile_clifford_t,
    compile_frame_candidates,
    compile_native_clifford_t,
    estimate_rotation_t_count,
)
from lizzy.emission.native import NativeCircuit, NativeGate, ladder_circuit
from lizzy.hamiltonian import Circuit


def _circuit(*rotations):
    return Circuit([(get_pauli_string(word), angle) for word, angle in rotations],
                   ["test"] * len(rotations))


def test_exact_clifford_and_t_angles_keep_absolute_phase_and_count() -> None:
    # All eight residues, inverse-word signs, and both spin-cover/full turns.
    for multiple in (*range(8), -1, -4, -8, 8, 16):
        logical = _circuit(("XY", multiple * np.pi / 8), ("II", 0.37))
        emitted = compile_clifford_t(logical, 2)
        assert emitted.t_count == abs(multiple) % 2
        assert np.linalg.norm(emitted.get_unitary() - ladder_circuit(logical, 2).get_unitary(), 2) < 2e-14
        assert emitted.n_2qb_gates() == (0 if multiple % 8 == 0 else 2)
        assert "gphase(" in emitted.to_qasm3()
    assert estimate_rotation_t_count(np.pi / 4, 1e-6) == 0
    assert estimate_rotation_t_count(np.pi / 8, 1e-6) == 1
    assert estimate_rotation_t_count(0.123, 1e-8) > estimate_rotation_t_count(0.123, 1e-4)


def test_inverse_and_immutable_artifact() -> None:
    emitted = compile_clifford_t(_circuit(("XI", np.pi / 8), ("YZ", -np.pi / 8)), 2)
    inverse = emitted.inverse()
    assert inverse.t_count == emitted.t_count == 2
    assert np.allclose(inverse.get_unitary() @ emitted.get_unitary(), np.eye(4), atol=1e-14)
    assert isinstance(emitted.gates, tuple)
    with pytest.raises(FrozenInstanceError):
        emitted.width = 3
    with pytest.raises(ValueError):
        CliffordTCircuit(1, (CliffordTGate("t", (1,)),))

    # QASM keeps inverse gate spelling and the absolute phase byte-for-byte.
    phase = float(np.nextafter(-0.2, -1.0))
    serializable = CliffordTCircuit(2, tuple(
        CliffordTGate(kind, wires) for kind, wires in (
            ("h", (0,)), ("s", (1,)), ("sdg", (0,)), ("t", (1,)),
            ("tdg", (0,)), ("x", (0,)), ("z", (1,)), ("cx", (1, 0)),
        )
    ), phase)
    assert serializable.to_qasm3() == (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[2] q;\n"
        f"gphase({phase!r});\n"
        "h q[0];\ns q[1];\ninv @ s q[0];\nt q[1];\ninv @ t q[0];\n"
        "x q[0];\nz q[1];\ncx q[1], q[0];\n"
    )
    assert CliffordTCircuit(0, (), phase).to_qasm3() == (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\n' + f"gphase({phase!r});\n"
    )


def test_snapping_is_charged_and_not_silently_exact() -> None:
    logical = _circuit(("Z", np.pi / 4 + 1e-8), ("X", 1e-9))
    emitted = compile_clifford_t(logical, 1, error=1e-6)
    measured = np.linalg.norm(emitted.get_unitary() - ladder_circuit(logical, 1).get_unitary(), 2)
    assert emitted.t_count == 0
    assert 0 < measured <= emitted.rotation_error_bound <= emitted.error_budget
    assert emitted.error_bound_kind == "numerical triangle bound"


def test_optional_dependency_only_required_for_generic_angles(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_gridsynth(name, *args, **kwargs):
        if name == "pygridsynth" or name.startswith("pygridsynth."):
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gridsynth)
    assert compile_clifford_t(_circuit(("Z", np.pi / 8)), 1).t_count == 1
    with pytest.raises(ImportError, match=r"lizzy\[ft\]"):
        compile_clifford_t(_circuit(("Z", 0.123)), 1)


def test_invalid_budget_and_failed_local_bound_are_rejected(monkeypatch) -> None:
    for budget in (0, -1, 1, np.inf, np.nan):
        with pytest.raises(ValueError):
            compile_clifford_t(Circuit(), 0, error=budget)
    import lizzy.emission.clifford_t as backend
    monkeypatch.setattr(backend, "_gridsynth_rotation", lambda *_: ((), 0.0, 0.5))
    with pytest.raises(ValueError, match="exceeded"):
        compile_clifford_t(_circuit(("Z", 0.123)), 1, error=1e-6)


def test_real_backend_multipauli_phase_order_and_error() -> None:
    pytest.importorskip("pygridsynth")
    logical = _circuit(("XY", 0.173), ("ZZ", -0.241), ("II", -0.32), ("ZI", np.pi / 8))
    emitted = compile_clifford_t(logical, 2, error=1e-5)
    measured = np.linalg.norm(emitted.get_unitary() - ladder_circuit(logical, 2).get_unitary(), 2)
    assert 0 < measured <= emitted.rotation_error_bound <= 1e-5
    assert emitted.t_count == sum(gate.kind in {"t", "tdg"} for gate in emitted.gates)
    assert emitted.t_count > 2
    assert all(gate.kind != "rz" for gate in emitted.gates)
    assert emitted.n_2qb_gates() == 4


def test_real_backend_negative_angles_reuse_exact_inverse_word(monkeypatch) -> None:
    pytest.importorskip("pygridsynth")
    import lizzy.emission.clifford_t as backend
    original = backend._gridsynth_rotation
    calls = []

    def observed(theta, precision):
        calls.append(theta)
        return original(theta, precision)

    monkeypatch.setattr(backend, "_gridsynth_rotation", observed)
    emitted = compile_clifford_t(_circuit(("XY", 0.217), ("XY", -0.217)), 2)
    assert calls == [0.434]
    assert emitted.gates == ()
    assert emitted.global_phase == 0
    assert emitted.t_count == 0


def test_real_backend_rejects_wrong_gate_word(monkeypatch) -> None:
    pytest.importorskip("pygridsynth")
    from importlib import import_module
    gridsynth = import_module("pygridsynth.gridsynth")
    monkeypatch.setattr(gridsynth, "gridsynth_gates", lambda **_: "H")
    with pytest.raises(ValueError, match="failed its numerical error budget"):
        compile_clifford_t(_circuit(("Z", 0.217)), 1)


def test_real_backend_zero_and_clifford_padding_do_not_consume_precision_slots(monkeypatch) -> None:
    pytest.importorskip("pygridsynth")
    import lizzy.emission.clifford_t as backend
    original = backend._gridsynth_rotation
    precisions = []

    def observed(theta, precision):
        precisions.append(precision)
        return original(theta, precision)

    monkeypatch.setattr(backend, "_gridsynth_rotation", observed)
    rz = NativeGate("rz", (0,), 0.3)
    plain = compile_native_clifford_t(NativeCircuit(1, [rz]))
    padded = compile_native_clifford_t(
        NativeCircuit(1, [rz] + [NativeGate("rz", (0,), 0.0)] * 11)
    )
    assert precisions[0] == precisions[1] == 5e-7
    assert plain.gates == padded.gates
    assert plain.t_count == padded.t_count

    # S = exp(i*pi/4) Rz(pi/2). The Rz representation must cost only its
    # floating-point special-angle error, not another generic precision slot.
    fixed = NativeCircuit(1, [NativeGate("s", (0,)), rz])
    represented = NativeCircuit(1, [NativeGate("rz", (0,), np.pi / 2), rz], np.pi / 4)
    for native in (fixed, represented):
        emitted = compile_native_clifford_t(native)
        measured = np.linalg.norm(emitted.get_unitary() - native.get_unitary(), 2)
        assert measured <= emitted.rotation_error_bound <= 1e-6
    assert precisions[2] == precisions[0]
    assert 0 < precisions[2] - precisions[3] < 5e-15


def test_frame_portfolio_isolates_failures_deduplicates_and_normalizes_phase(monkeypatch) -> None:
    import lizzy.emission.clifford_t as backend

    logical = _circuit(("II", 1e16), ("II", 0.37), ("II", -1e16), ("XI", 0.2))
    bad = NativeCircuit(2, [NativeGate("h", (0,))], 19.0)
    good = NativeCircuit(2, [NativeGate("rz", (0,), 0.4)], 19.0)
    build_calls, compile_calls = [], []

    def alternatives(circuit, width, *, lookahead):
        build_calls.append(lookahead)
        if lookahead == 0:
            raise ValueError("deliberate frame build failure")
        return [("fixed-0", bad), ("rolling", good), ("duplicate", good)]

    def compile_one(native, *, error):
        compile_calls.append((native.gates[0].kind, native.global_phase, error))
        if native.gates[0].kind == "h":
            raise ValueError("deliberate candidate compilation failure")
        return CliffordTCircuit(native.width, (), native.global_phase, error, 0.0)

    monkeypatch.setattr(backend, "native_frame_candidates", alternatives)
    monkeypatch.setattr(backend, "compile_native_clifford_t", compile_one)
    outputs, rejected = compile_frame_candidates(logical, 2, error=2e-6)
    assert build_calls == [0, 8]
    assert compile_calls == [("h", -0.37, 2e-6), ("rz", -0.37, 2e-6)]
    assert [name for name, _ in outputs] == ["frame-rolling-8"]
    assert [name for name, _ in rejected] == ["frame-build-0", "frame-fixed-0"]
    assert all("deliberate" in reason for _, reason in rejected)
    assert good.global_phase == bad.global_phase == 19.0  # No caller-owned mutation.
    with pytest.raises(ValueError, match="error tolerance"):
        compile_frame_candidates(logical, 2, error=0)
    with pytest.raises(ValueError, match="width"):
        compile_frame_candidates(logical, 1)
    assert build_calls == [0, 8]  # Invalid public inputs are not optional failures.
    for capped, width in ((_circuit(("Z" + "I" * 8, 0.2)), 9),
                          (_circuit(*[("XI", 0.2)] * 257), 2)):
        outputs, rejected = compile_frame_candidates(capped, width)
        assert outputs == () and rejected[0][0] == "frame-cap"
        assert "Skipped:" in rejected[0][1]
    assert build_calls == [0, 8]  # A cap prevents building, not just compiling, frames.


def test_real_frame_portfolio_preserves_absolute_phase_budget_and_actual_counts() -> None:
    pytest.importorskip("pygridsynth")
    logical = _circuit(("XXX", 0.217), ("XXY", -0.131), ("IIZ", np.pi / 8), ("III", 0.27))
    target = ladder_circuit(logical, 3).get_unitary()
    outputs, rejected = compile_frame_candidates(logical, 3, error=1e-6)
    assert outputs and not rejected
    for name, emitted in outputs:
        assert name.startswith("frame-fixed-")
        assert emitted.error_budget == 1e-6
        measured = np.linalg.norm(emitted.get_unitary() - target, 2)
        assert measured <= emitted.rotation_error_bound + 2e-14 <= 1e-6
        assert emitted.t_count == sum(gate.kind in {"t", "tdg"} for gate in emitted.gates)
        assert emitted.two_qubit_gates == sum(gate.kind == "cx" for gate in emitted.gates)
    assert min(emitted.two_qubit_gates for _, emitted in outputs) < ladder_circuit(logical, 3).two_qubit_gates
