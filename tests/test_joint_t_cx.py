"""A small joint-cost experiment must keep real tradeoffs and accuracy failures."""

import hashlib

import pytest

from experiments import joint_t_cx as joint
from lizzy.native import NativeCircuit, NativeGate


def _row(name, t, cx, stage="delivered", status="PASS"):
    return dict(name=name, t_count=t, cx_count=cx, stage=stage, status=status, qasm_sha256=name)


def test_frontier_constraints_and_stable_ties():
    rows = [_row("old", 240, 6), _row("better", 240, 4), _row("tie", 240, 4),
            _row("tradeoff", 244, 2), _row("worse", 250, 7),
            _row("invalid-cheap", 0, 0, status="FAIL_FINAL_ERROR")]
    assert [r["name"] for r in joint.frontier(rows)] == ["better", "tradeoff"]
    assert joint.choose(rows)["name"] == "better"
    assert joint.choose(rows, priority="cx")["name"] == "tradeoff"
    assert joint.choose(rows, priority="cx", t_cap=240)["name"] == "better"
    assert joint.choose(rows, t_cap=240, cx_cap=2) is None


def test_nested_stages_retain_reference_and_do_not_hide_a_tradeoff():
    rows = [_row("old", 100, 20), _row("frame", 100, 10, "same-frames"),
            _row("route", 90, 30, "route-frames"), _row("gauge", 110, 5, "gauge-frames")]
    report = joint.summarize(rows)
    assert report["delivered"]["no_compromise"]["name"] == "old"
    assert report["gauge-frames"]["no_compromise"]["name"] == "frame"
    assert report["gauge-frames"]["t_first"]["name"] == "route"
    assert report["gauge-frames"]["cx_first"]["name"] == "gauge"
    assert len(report["gauge-frames"]["frontier"]) == 3


def test_measure_rejects_bad_frame_and_retains_real_frontier_artifacts(monkeypatch):
    pytest.importorskip("pygridsynth")
    frames = joint.native_frame_candidates
    compile_native = joint.compile_native_clifford_t
    budgets = []

    def with_bad_frame(logical, width, **kwargs):
        return [("wrong", NativeCircuit(width, [NativeGate("h", (0,))])),
                *frames(logical, width, **kwargs)]

    def observed_compile(native, *, error):
        budgets.append(error)
        return compile_native(native, error=error)

    monkeypatch.setattr(joint, "native_frame_candidates", with_bad_frame)
    monkeypatch.setattr(joint, "compile_native_clifford_t", observed_compile)
    result = joint.measure(next(c for c in joint.cases() if c.name == "encoded-su2"), 1e-6)
    assert result["status"] == "PARTIAL"
    rows = result["candidates"]
    assert any(r["status"] == "FAIL_FRAME" for r in rows)
    assert budgets and set(budgets) == {5e-7}
    for summary in result["stages"].values():
        for point in summary["frontier"]:
            assert "/wrong-" not in point["name"]
            qasm = result["artifacts"][point["qasm_sha256"]]
            assert hashlib.sha256(qasm.encode()).hexdigest() == point["qasm_sha256"]
            assert point["cx_count"] == sum(line.startswith("cx ") for line in qasm.splitlines())
            assert point["t_count"] == sum(line.startswith(("t ", "inv @ t "))
                                           for line in qasm.splitlines())
    reference = result["stages"]["delivered"]["no_compromise"]
    assert reference["qasm_sha256"] in result["artifacts"]
    for row in rows:
        if row["status"] == "PASS":
            assert row["frame_error"]["strict"] <= 1e-10
            assert row["final_error"]["phase_aligned"] <= 1e-6
            assert row["rotation_error_bound"] <= 5e-7
