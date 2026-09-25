"""Isolated test of condition-only Wei--Norman chart management.

This experiment reuses Lizzy's Pauli closure and exact adjoint Jacobian. Unlike
the production prototype it does not impose a fixed L1 radius on coordinates.
It does not change the default router, optimize the factor order, or provide a
global error/conditioning certificate. Conditioning is sampled at RHS calls
and the accepted ODE mesh; singularities between those samples can be missed.
"""

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from lizzy.driven import (
    DrivenHamiltonian,
    DrivenResult,
    IntegrationFailure,
    _adjoint_pairs,
    _closure,
    _coordinate_matrix,
    _positive_integer,
)
from lizzy.hamiltonian import Circuit, fold_phases


class _ConditionFailure(Exception):
    """An attempted coordinate chart is nonfinite or poorly conditioned."""


def synthesize_compact(
    drive: DrivenHamiltonian,
    time_span: tuple[float, float],
    *,
    max_dimension: int = 32,
    basis_order: Sequence[str] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.025,
    condition_limit: float = 100.0,
    max_rhs_evaluations: int = 100_000,
    max_segments: int = 1024,
    restart: bool = False,
) -> DrivenResult:
    """Return a numerical circuit, or raise without exposing partial output.

    ``restart=False`` requires one chart for the full interval. ``restart=True``
    retries ill-conditioned intervals by the production prototype's bisection
    scheme and appends independently integrated left increments. Only the fixed
    L1 angle guard is removed. The RHS budget is global, including failed tries.

    Full closure includes declared controls with zero coefficients and central
    identity terms. ``basis_order`` must permute that complete closure. No dense
    Hilbert-space matrices or reference propagators are used in this solver.
    Local ODE tolerances do not certify global operator error. Smooth controls
    are expected; callers must split known discontinuities and resolve fast
    pulses with ``max_step``. A sampled condition guard is not an atlas.
    """
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    max_rhs_evaluations = _positive_integer(max_rhs_evaluations, "max_rhs_evaluations")
    max_segments = _positive_integer(max_segments, "max_segments")
    if not isinstance(restart, (bool, np.bool_)):
        raise TypeError("restart must be a boolean")
    span = np.asarray(time_span, dtype=float)
    if span.shape != (2,) or not np.all(np.isfinite(span)):
        raise ValueError("time_span must contain two finite times")
    start, end = map(float, span)
    if not np.isfinite(end - start):
        raise ValueError("time_span duration must be finite")
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if np.isnan(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")

    # Reject over-budget closures before evaluating any user callback.
    basis = _closure(drive.paulis, max_dimension)
    if basis_order is not None:
        order = tuple(str(p) for p in basis_order)
        if len(order) != len(basis) or set(order) != {str(p) for p in basis}:
            raise ValueError("basis_order must be a permutation of the full Pauli closure")
        by_word = {str(p): p for p in basis}
        basis = [by_word[word] for word in order]
    words = tuple(str(p) for p in basis)
    positions = [words.index(word) for word in drive.paulis]
    pairs = _adjoint_pairs(basis)
    evaluations, rejected, peak_condition = 0, 0, 1.0

    def check_chart(time, theta):
        nonlocal peak_condition
        if not np.all(np.isfinite(theta)):
            raise _ConditionFailure(f"nonfinite coordinates at t={time:.17g}")
        matrix = _coordinate_matrix(theta, pairs)
        if not np.all(np.isfinite(matrix)):
            raise _ConditionFailure(f"nonfinite coordinate Jacobian at t={time:.17g}")
        try:
            condition = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError as exc:
            raise _ConditionFailure(f"condition estimation failed at t={time:.17g}") from exc
        peak_condition = max(peak_condition, condition) if np.isfinite(condition) else np.inf
        if not np.isfinite(condition) or condition > condition_limit:
            raise _ConditionFailure(
                f"condition_limit={condition_limit:g} exceeded at t={time:.17g} "
                f"(sampled condition={condition:.6g})"
            )
        return matrix

    def rhs(time, theta):
        nonlocal evaluations
        if evaluations >= max_rhs_evaluations:
            raise IntegrationFailure("max_rhs_evaluations exhausted; no partial circuit returned")
        evaluations += 1
        matrix = check_chart(time, theta)
        coefficients = np.zeros(len(basis))
        coefficients[positions] = drive.at(time)
        try:
            derivative = np.linalg.solve(matrix, coefficients)
        except np.linalg.LinAlgError as exc:
            raise _ConditionFailure(f"coordinate solve failed at t={time:.17g}") from exc
        if not np.all(np.isfinite(derivative)):
            raise _ConditionFailure(f"nonfinite coordinate derivative at t={time:.17g}")
        return derivative

    circuit, intervals = Circuit(), []
    current, target = start, end
    direction = 1 if end >= start else -1
    while current != end:
        if len(intervals) >= max_segments:
            raise IntegrationFailure("max_segments exhausted; no partial circuit returned")
        try:
            solution = solve_ivp(
                rhs, (current, target), np.zeros(len(basis)), method="DOP853",
                rtol=rtol, atol=atol, max_step=max_step,
            )
            if not solution.success:
                raise IntegrationFailure(f"ODE solver failed: {solution.message}")
            # Also check accepted mesh states; RHS stages need not equal all of
            # them, and accepted output must not silently bypass the guard.
            for time, angles in zip(solution.t, solution.y.T):
                check_chart(time, angles)
            angles = solution.y[:, -1]
            check_chart(target, angles)
        except _ConditionFailure as exc:
            rejected += 1
            if not restart:
                raise IntegrationFailure(f"single-chart integration failed: {exc}") from exc
            midpoint = current + (target - current) / 2
            if midpoint == current or midpoint == target:
                raise IntegrationFailure(
                    "condition-limited interval cannot be resolved at this time scale"
                ) from exc
            target = midpoint
            continue
        # Circuit application order reverses the displayed WN factor product.
        for pauli, angle in reversed(list(zip(basis, angles))):
            if angle != 0:
                circuit.add(pauli, float(angle), "wei-norman-compact-experiment")
        intervals.append((current, target))
        duration = target - current
        current = target
        remaining = abs(end - current)
        target = end if abs(duration) >= remaining / 2 else current + 2 * duration
        if current != end and direction * (target - current) <= 0:
            raise IntegrationFailure("integration made no time progress")

    return DrivenResult(
        fold_phases(circuit, tolerance=0), words, tuple(intervals),
        evaluations, rejected, peak_condition,
    )
