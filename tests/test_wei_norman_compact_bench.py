"""The experimental comparison preserves controls and strict phase checking."""

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from experiments import wei_norman_compact_bench as bench
from lizzy import wei_norman
from lizzy.driven import IntegrationFailure
from lizzy.hamiltonian import Circuit


def test_strict_phase_is_required_for_every_numerical_variant():
    case = bench._static_case("phase", ["X"], [0.0], 1.0)
    wrong = Circuit()
    wrong.add(get_pauli_string("I"), np.pi, "diagnostic")
    assert bench.assess(case, wrong)["status"] == "FAIL_ACCURACY"
    assert bench.assess(case, wrong, strict=False)["status"] == "PASS"


def test_variants_retain_component_splitting_and_restore_solver_binding():
    trial = bench.Trial(bench._static_case("split", ["XI", "ZI", "IX", "IZ"],
                                         [0.2, 0.3, -0.1, 0.4], 0.2))
    original = wei_norman.synthesize_driven
    results = [bench.run_variant(trial, variant) for variant in
               ("production", "single-product", "condition-restart")]
    assert wei_norman.synthesize_driven is original
    assert all(result.component_dimensions == (3, 3) for result in results)
    assert all(bench.assess(trial.case, result.circuit)["status"] == "PASS"
               for result in results)


def test_failed_experiment_restores_production_solver(monkeypatch):
    def fail(*args, **kwargs):
        raise IntegrationFailure("intentional experiment failure")

    trial = bench.Trial(bench._static_case("failure", ["X", "Z"], [0.2, 0.3], 0.2))
    original = wei_norman.synthesize_driven
    monkeypatch.setattr(bench, "synthesize_compact", fail)
    with pytest.raises(IntegrationFailure):
        bench.run_variant(trial, "single-product")
    assert wei_norman.synthesize_driven is original


def test_historical_production_comparator_pins_the_old_radius(monkeypatch):
    options = {}

    def capture(*args, **kwargs):
        options.update(kwargs)
        return "captured"

    trial = bench.Trial(bench._static_case("historical", ["X", "Z"], [0.2, 0.3], 0.2))
    monkeypatch.setattr(bench, "synthesize_wei_norman", capture)
    assert bench.run_variant(trial, "production") == "captured"
    assert options["chart_radius"] == 0.5


def test_comparison_preserves_refusals_and_existing_pair_kernel():
    trial = bench.Trial(bench._static_case("pair", ["XI", "ZI", "IX", "IZ", "ZZ"],
                                         [0.2, 0.3, -0.1, 0.4, 0.5], 0.2))
    report = bench.run([trial])
    numerical = [r for r in report["rows"] if r["variant"] in
                 {"production", "single-product", "condition-restart"}]
    assert len(numerical) == 3
    assert all(row["status"] == "PASS" and row["strict_required"] for row in numerical)
    kernel = next(r for r in report["rows"] if r["variant"] == "existing-pair-kernel")
    assert kernel["status"] == "PASS"
    assert kernel["cx"] <= 6


def test_case_inventory_contains_adverse_and_raised_cap_trials():
    cases = list(bench.trials())
    by_name = {trial.case.name: trial for trial in cases}
    assert len(cases) == len(by_name) == 27
    assert {"euler-singularity", "fast-euler-crossing", "very-long-tfim3",
            "reverse-tfim3", "driven-tfim3-identity", "pulse-breakpoint"} <= by_name.keys()
    assert by_name["su8-enabled"].options == {"max_dimension": 64}
    assert by_name["su8-closure-cap"].options == {}
