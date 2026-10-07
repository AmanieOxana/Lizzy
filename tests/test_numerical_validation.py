"""Public numerical contracts on direct and zero-duration paths."""

import pytest

from lizzy.hamiltonian import hamiltonian
from lizzy.synthesis.driven import DrivenHamiltonian, synthesize_driven
from lizzy.synthesis.expansions import expand_driven
from lizzy.synthesis.wei_norman import synthesize_wei_norman


def _forbidden(*args, **kwargs):
    raise AssertionError("input validation must not invoke the compiler or controls")


@pytest.mark.parametrize("case, option, value", [
    ("static-direct", "rtol", 0),
    ("driven-zero", "condition_limit", 1),
])
def test_direct_and_zero_duration_paths_still_validate_solver_controls(case, option, value):
    source = (DrivenHamiltonian(["X", "Z"], _forbidden) if case == "driven-zero"
              else hamiltonian({"Z": 0.2}))
    with pytest.raises(ValueError, match=option):
        synthesize_wei_norman(source, 0.0 if case == "driven-zero" else 1.0, **{option: value})


@pytest.mark.parametrize("compiler", [synthesize_driven, expand_driven])
def test_low_level_time_spans_still_require_two_endpoints(compiler):
    with pytest.raises(ValueError, match="time_span must contain two finite times"):
        compiler(DrivenHamiltonian(["Z"], _forbidden), 1.0)
