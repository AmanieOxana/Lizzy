"""The validation harness checks production behavior, including visible failures."""

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from experiments import wei_norman_validation as validation
from lizzy.driven import AlgebraTooLarge, IntegrationFailure
from lizzy.hamiltonian import Circuit


def test_strict_global_phase_is_not_hidden_by_aligned_error():
    case = validation._static_case("phase", ["X"], [0.0], 1.0)
    wrong = Circuit()
    wrong.add(get_pauli_string("I"), np.pi, "diagnostic")
    row = validation.assess(case, wrong)
    assert row["aligned_error"] < 1e-12
    assert row["status"] == "FAIL_ACCURACY"


def test_modes_call_public_compiler_with_declared_step_policy(monkeypatch):
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return "compiled"

    trial = validation.Trial(validation._static_case("static", ["X"], [0.2], 0.2))
    monkeypatch.setattr(validation, "synthesize_wei_norman", capture)
    assert validation.compile_trial(trial, "resolved-step") == "compiled"
    assert validation.compile_trial(trial, "adaptive-default-step") == "compiled"
    assert calls[0]["max_step"] == 0.025
    assert "max_step" not in calls[1]
    assert calls[0]["chart_radius"] is calls[1]["chart_radius"] is None
    assert calls[0]["max_rhs_evaluations"] == calls[1]["max_rhs_evaluations"] == 20000
    with pytest.raises(ValueError, match="unknown mode"):
        validation.compile_trial(trial, "obsolete-prototype")


def test_small_production_run_preserves_commuting_components():
    trial = validation.Trial(validation._static_case(
        "split", ["XI", "ZI", "IX", "IZ"], [0.2, 0.3, -0.1, 0.4], 0.2,
    ))
    result = validation.compile_trial(trial, "resolved-step")
    assert result.component_dimensions == (3, 3)
    report = validation.run([trial])
    assert len(report["rows"]) == 2
    assert all(row["status"] == "PASS" and row["strict_required"] for row in report["rows"])


def test_expected_and_unexpected_caps_are_distinguished(monkeypatch):
    def refuse(*args):
        raise AlgebraTooLarge("dimension cap")

    cases = [validation.Trial(validation._static_case(
        "cap", ["X", "Y"], [0.2, 0.3], 0.2, expected_cap=expected,
    )) for expected in (True, False)]
    monkeypatch.setattr(validation, "compile_trial", refuse)
    rows = validation.run(cases)["rows"]
    assert [row["status"] for row in rows] == ["CAP", "CAP", "FAIL_CAP", "FAIL_CAP"]


def test_integration_failure_is_retained_and_cli_fails(monkeypatch, capsys):
    trial = validation.Trial(validation._static_case("failure", ["X"], [0.2], 0.2))

    def refuse(*args):
        raise IntegrationFailure("work exhausted")

    monkeypatch.setattr(validation, "trials", lambda: iter([trial]))
    monkeypatch.setattr(validation, "compile_trial", refuse)
    assert validation.main(["--json"]) == 1
    assert "FAIL_COMPILE" in capsys.readouterr().out


def test_inventory_retains_adverse_cases_and_explicit_raised_cap():
    cases = list(validation.trials())
    by_name = {trial.case.name: trial for trial in cases}
    assert len(cases) == len(by_name) == 27
    assert {"euler-singularity", "fast-euler-crossing", "very-long-tfim3",
            "reverse-tfim3", "driven-tfim3-identity", "pulse-breakpoint"} <= by_name.keys()
    assert by_name["su8-enabled"].options == {"max_dimension": 64}
    assert not by_name["su8-enabled"].case.expected_cap
    assert by_name["su8-closure-cap"].case.expected_cap
