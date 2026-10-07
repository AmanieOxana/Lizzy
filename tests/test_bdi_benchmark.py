"""The paired benchmark exposes failures and keeps both routes comparable."""

import numpy as np

from experiments import bdi_benchmark as benchmark
from lizzy.hamiltonian import Circuit, hamiltonian


def test_warmed_timings_alternate_order_and_exclude_preparation():
    calls = []

    def compiler(method):
        def compile_at(time):
            assert time == -0.7
            calls.append(method)
            return Circuit()
        return compile_at

    outcomes = {
        method: {"preparation_seconds": 100, "prepared_evaluation_samples_seconds": []}
        for method in benchmark.METHODS
    }
    compilers = {method: compiler(method) for method in benchmark.METHODS}
    outcomes, circuits = benchmark._compile_pair(outcomes, compilers, -0.7, 3)
    assert calls == ["givens", "bdi", "givens", "bdi", "bdi", "givens", "givens", "bdi"]
    assert set(circuits) == {"givens", "bdi"}
    for row in outcomes.values():
        assert len(row["prepared_evaluation_samples_seconds"]) == 3
        assert 0 <= row["prepared_evaluation_median_seconds"] < 100
        assert row["preparation_plus_first_evaluation_seconds"] == (
            100 + row["first_evaluation_seconds"]
        )


def test_cold_preparation_keeps_method_specific_mapping_and_reuses_bdi(monkeypatch):
    operator = hamiltonian({"X": 1})
    calls = []
    monkeypatch.setattr(benchmark.exact, "_irrep_cache", {("X",): object()})

    def irrep(words, width):
        assert words == ("X",) and width == 1
        assert words not in benchmark.exact._irrep_cache
        return {}, {}, [1, 0], 2

    class Plan:
        mapping_kind = "horizontal-graph"
        partition = (1, 1)
        irrep_size = 2
        parameter_bound = 1
        phase_preserving = True

        def circuit(self, time, route):
            calls.append((time, route))
            return Circuit()

    def prepare_bdi(part, *, cache):
        assert part is operator and cache is False
        calls.append("prepare")
        return Plan()

    monkeypatch.setattr(benchmark.exact, "_irrep", irrep)
    monkeypatch.setattr(benchmark.exact, "prepare_bdi", prepare_bdi, raising=False)
    outcomes, compilers = benchmark._prepare_routes([operator], 1)
    compilers["bdi"](0.1)
    compilers["bdi"](5.0)
    assert calls == ["prepare", (0.1, "exact-bdi"), (5.0, "exact-bdi")]
    assert outcomes["givens"]["mappings"][0]["order"] == [1, 0]
    assert outcomes["bdi"]["mappings"][0]["partition"] == [1, 1]
    assert outcomes["bdi"]["phase_preserving"]
    assert not outcomes["givens"]["phase_preserving"]


def test_emission_is_strict_even_when_target_phase_is_ignored(monkeypatch):
    circuit = Circuit()
    correct = benchmark.assess(circuit, 1, -np.eye(2))
    assert correct["status"] == "PASS"
    assert correct["aligned_error"] == 0
    assert correct["strict_error"] == 2
    assert correct["native_frame"]["status"] == "PASS"
    assert correct["native_portfolio"]["cx"] == 0
    strict = benchmark.assess(circuit, 1, -np.eye(2), require_strict_phase=True)
    assert strict["status"] == "FAIL_ACCURACY"
    assert strict["native_frame"]["status"] == "FAIL_ACCURACY"

    class WrongEmission:
        gates = []
        two_qubit_gates = 0

        def get_unitary(self):
            return -np.eye(2)

    monkeypatch.setattr(benchmark, "ladder_circuit", lambda *args: WrongEmission())
    wrong = benchmark.assess(circuit, 1, -np.eye(2))
    assert wrong["aligned_error"] == 0
    assert wrong["emission_discrepancy"] == 2
    assert wrong["status"] == "FAIL_ACCURACY"


def test_failed_backend_remains_visible_and_cli_returns_failure(monkeypatch, capsys):
    case = benchmark.Case("failure", hamiltonian({"X": 1}), 0.0)

    def compile_bdi(time):
        raise RuntimeError("intentional diagnostic")

    def prepare_routes(parts, width):
        outcomes = {
            method: {
                "preparation_seconds": 0, "prepared_evaluation_samples_seconds": [],
                "phase_preserving": method == "bdi",
            }
            for method in benchmark.METHODS
        }
        return outcomes, {"givens": lambda time: Circuit(), "bdi": compile_bdi}

    monkeypatch.setattr(benchmark, "cases", lambda: iter([case]))
    monkeypatch.setattr(benchmark, "_prepare_routes", prepare_routes)
    assert benchmark.main(["--json"]) == 1
    output = capsys.readouterr().out
    assert '"status": "PASS"' in output
    assert '"status": "FAIL_COMPILE"' in output
    assert "intentional diagnostic" in output
