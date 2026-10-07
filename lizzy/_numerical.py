"""Shared input validation for the opt-in numerical synthesis routes.

These checks are independent of Hamiltonian construction and ODE integration, so
direct/zero-duration paths enforce the same contracts without compiling a probe.
Algorithm-specific equations and work-budget accounting stay with their solvers.
"""

from operator import index

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    try:
        parsed = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _time_span(time_span, *, allow_duration: bool = False) -> tuple[float, float]:
    """Validate finite endpoints/duration; scalar shorthand is opt-in per API."""
    span = np.asarray(time_span, dtype=float)
    if allow_duration and span.ndim == 0:
        span = np.array([0.0, float(span)])
    if span.shape != (2,) or not np.all(np.isfinite(span)):
        message = ("time_span must be a finite duration or two finite times" if allow_duration
                   else "time_span must contain two finite times")
        raise ValueError(message)
    start, end = map(float, span)
    if not np.isfinite(end - start):
        raise ValueError("time_span duration must be finite")
    return start, end


def _validate_ode_tolerances(rtol: float, atol: float) -> None:
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _validate_chart_controls(
    *, rtol: float, atol: float, max_step: float,
    chart_radius: float | None, condition_limit: float,
) -> None:
    """Validate Wei–Norman controls even when no numerical chart is needed."""
    _validate_ode_tolerances(rtol, atol)
    if chart_radius is not None and (not np.isfinite(chart_radius) or not 0 < chart_radius <= 0.5):
        raise ValueError("chart_radius must be None or finite in (0, 0.5]")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    if np.isnan(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive")
