"""Keep the comparison baseline and its pass/fail reporting honest."""

from types import SimpleNamespace

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string
from scipy.linalg import expm

from lizzy import driven_bench
from lizzy.dense import circuit_matrix, pauli_matrix
from lizzy.driven import DrivenHamiltonian
from lizzy.hamiltonian import Circuit
from lizzy.native import ladder_circuit


def test_midpoint_baseline_integrates_linear_commuting_drive_backwards():
    h = DrivenHamiltonian(["ZI", "IZ", "II"], lambda t: [t, -2 * t, 0.3])
    circuit = driven_bench._midpoint_formula(h, (1, 0), 4)
    integral = -0.5 * pauli_matrix("ZI") + pauli_matrix("IZ") - 0.3 * np.eye(4)
    assert np.linalg.norm(circuit_matrix(circuit, 2) - expm(-1j * integral), 2) < 1e-12


def test_midpoint_baseline_has_second_order_convergence():
    case = next(driven_bench._cases())
    errors = []
    for steps in (16, 32):
        circuit = driven_bench._midpoint_formula(case.hamiltonian, case.time_span, steps)
        errors.append(np.linalg.norm(circuit_matrix(circuit, 1) - case.target, 2))
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_quote_counts_concrete_gates_not_the_logical_pair_estimate():
    circuit = Circuit()
    for word, angle in [("XX", .2), ("YY", -.3), ("ZZ", .4)]:
        circuit.add(get_pauli_string(word), angle, "test")
    logical = circuit_matrix(circuit, 2)
    ladder = ladder_circuit(circuit, 2)
    frame = driven_bench.native_frame_circuit(circuit, 2)
    assert circuit.two_qubit_gates == 3
    cost, backend, actual = driven_bench._quote(circuit, 2, logical)
    assert cost == min(ladder.two_qubit_gates, frame.two_qubit_gates)
    assert backend in {"native-ladder", "native-frame"}
    assert np.linalg.norm(actual - logical, 2) < 1e-12


def test_quote_does_not_offer_a_lower_unemitted_block_cost(monkeypatch):
    circuit = Circuit()
    for word, angle in [("XX", .2), ("YY", -.3), ("ZZ", .4)]:
        circuit.add(get_pauli_string(word), angle, "test")
    monkeypatch.setattr(driven_bench, "native_frame_candidate", lambda *args: False)
    cost, backend, _ = driven_bench._quote(circuit, 2, circuit_matrix(circuit, 2))
    assert circuit.two_qubit_gates == 3
    assert cost == 6 and backend == "native-ladder"


def test_benchmark_reports_failure_when_baseline_exhausts_budget(monkeypatch, capsys):
    case = next(driven_bench._cases())
    monkeypatch.setattr(driven_bench, "_cases", lambda: iter([case]))
    monkeypatch.setattr("sys.argv", ["driven_bench", "--max-steps", "1"])
    assert driven_bench.main() == 1
    report = capsys.readouterr().out
    rows = [line for line in report.splitlines() if line.startswith(case.name)]
    assert len(rows) == 4
    assert "Wei-Norman" in rows[0] and rows[0].endswith("PASS")
    assert "magnus4" in rows[1] and rows[1].endswith("PASS")
    assert "fer4" in rows[2] and rows[2].endswith("PASS")
    assert "midpoint2" in rows[3] and rows[3].endswith("FAIL")


def test_benchmark_default_threshold_passes(monkeypatch, capsys):
    case = next(driven_bench._cases())
    monkeypatch.setattr(driven_bench, "_cases", lambda: iter([case]))
    monkeypatch.setattr("sys.argv", ["driven_bench"])
    assert driven_bench.main() == 0
    report = capsys.readouterr().out
    assert sum(line.endswith("PASS") for line in report.splitlines()) == 4
    assert "FAIL" not in report


def test_benchmark_reports_expansion_search_caps(monkeypatch, capsys):
    case = next(driven_bench._cases())
    monkeypatch.setattr(driven_bench, "_cases", lambda: iter([case]))
    monkeypatch.setattr(
        "sys.argv", ["driven_bench", "--max-steps", "1", "--max-expansion-steps", "1"]
    )
    assert driven_bench.main() == 1
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith(case.name)]
    assert len(rows) == 4
    assert rows[0].endswith("PASS")
    assert all(line.endswith("FAIL") for line in rows[1:])


def test_ideal_plan_screen_uses_circuit_application_order():
    plan = SimpleNamespace(basis=("X", "Y"), factors=((0.21, 0.0), (0.0, -0.34)))
    expected = expm(0.34j * pauli_matrix("Y")) @ expm(-0.21j * pauli_matrix("X"))
    assert np.linalg.norm(driven_bench._plan_matrix(plan) - expected, 2) < 1e-14


def test_expansion_search_requires_final_circuit_accuracy(monkeypatch):
    """An accurate ideal plan must not hide an inaccurate compiled artifact."""
    target = expm(-0.17j * pauli_matrix("X"))
    case = driven_bench._Case(
        "test", DrivenHamiltonian(["X"], lambda t: [0.17]), (0.0, 1.0), target, "analytic"
    )
    plan = SimpleNamespace(basis=("X",), factors=((0.17,),))
    monkeypatch.setattr(driven_bench, "expand_driven", lambda *args, **kwargs: plan)
    compiled_steps = []

    def compile_plan(*args, steps, **kwargs):
        compiled_steps.append(steps)
        circuit = Circuit()
        circuit.add(get_pauli_string("X"), 0.17 if steps == 2 else 0.4, "test")
        return SimpleNamespace(circuit=circuit, steps=steps)

    monkeypatch.setattr(driven_bench, "synthesize_expansion", compile_plan)
    result, plan_error, measurement = driven_bench._search_expansion(case, "magnus4", 1e-8, 2)
    assert compiled_steps == [1, 2]
    assert result.steps == 2
    assert plan_error < 1e-14
    assert measurement[2] < 1e-14
