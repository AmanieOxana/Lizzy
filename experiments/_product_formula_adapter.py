"""Bounded Hamiltonian-aware Qiskit product-formula comparison candidates.

Choose the first accuracy-passing default lowering for each declared Suzuki
order, then offer default and Rustiq lowerings of that same formula at exact
transpiler levels zero and three. This is a bounded first-passing heuristic,
not a search for the globally cheapest T count.
The benchmark still verifies every emitted candidate and uses its shared
Clifford+T backend; the dense target is used here only to size the time steps.
"""

from copy import deepcopy
from importlib.metadata import version

import numpy as np

from experiments._compiler_adapters import (
    CompilerCandidate,
    _target_width,
    qiskit_to_native,
)
from lizzy.dense import operator_errors

PRODUCT_FORMULA_ORDERS = (2, 4)
PRODUCT_FORMULA_REPS = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_EXPANDED_ROTATIONS = 4096
_BASIS = ["rz", "rx", "ry", "h", "s", "sdg", "x", "z", "cx"]


class ProductFormulaBudgetExceeded(TimeoutError):
    """The declared search could not reach accuracy; retain all search evidence."""

    def __init__(self, metadata: dict):
        super().__init__(
            "No Suzuki family reached the decomposition-error budget within "
            "the declared repetition and expanded-rotation caps."
        )
        self.metadata = metadata


def _lower(circuit, optimization_level: int = 0):
    """Exact SDK basis conversion, with no layout or approximate optimization."""
    from qiskit import transpile

    emitted = transpile(
        circuit, basis_gates=_BASIS, optimization_level=optimization_level,
        approximation_degree=1.0, seed_transpiler=0,
    )
    return qiskit_to_native(emitted)


def qiskit_product_formula_candidates(
    terms: dict[str, float], time: float, target: np.ndarray,
    decomposition_error: float,
) -> list[CompilerCandidate]:
    """Return first-passing order-2/4 Suzuki formulas and their Rustiq variants.

    All inputs use Lizzy's big-endian tensor convention. Formula term order is
    exactly the input dictionary's iteration order. Identity terms and their
    phases are retained. Missing Qiskit is an import error; exhausting the
    fixed accuracy search raises ``ProductFormulaBudgetExceeded`` with JSON-safe
    ``metadata`` containing every attempt, including upstream failures.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter, synth_pauli_network_rustiq

    matrix, width = _target_width(target)
    if not np.isfinite(time):
        raise ValueError("The evolution time must be finite.")
    if not np.isfinite(decomposition_error) or decomposition_error <= 0:
        raise ValueError("The decomposition-error budget must be finite and positive.")
    if not terms:
        raise ValueError("At least one Hamiltonian term is required.")
    for word, coefficient in terms.items():
        if len(word) != width or set(word) - set("IXYZ"):
            raise ValueError("Hamiltonian labels must match the target width and use IXYZ.")
        if not np.isreal(coefficient) or not np.isfinite(coefficient):
            raise ValueError("Hamiltonian coefficients must be finite and real.")

    # Qiskit label rightmost character acts on wire zero; native wire zero is
    # Lizzy's leftmost tensor factor. Preserve labels' insertion order.
    operator = SparsePauliOp.from_list([
        (word[::-1], float(np.real(coefficient))) for word, coefficient in terms.items()
    ])
    metadata = {
        "package": "qiskit", "version": version("qiskit"),
        "algorithm": "SuzukiTrotter; optional synth_pauli_network_rustiq",
        "input_tensor_order": "big-endian",
        "basis_conversion": "reverse Pauli labels; wire indices preserved",
        "term_order": list(terms),
        "orders": list(PRODUCT_FORMULA_ORDERS),
        "repetition_schedule": list(PRODUCT_FORMULA_REPS),
        "max_expanded_rotations": MAX_EXPANDED_ROTATIONS,
        "decomposition_error_budget": float(decomposition_error),
        "selection": "first default-lowering pass per order; no later-step T search",
        "step_selection_optimization_level": 0,
        "optimization_levels": [0, 3], "seed_transpiler": 0,
        "approximation_degree": 1.0, "preserve_order": True,
        "search": [],
    }
    candidates = []
    for order in PRODUCT_FORMULA_ORDERS:
        search = {"order": order, "status": "CAP", "attempts": []}
        metadata["search"].append(search)
        for reps in PRODUCT_FORMULA_REPS:
            attempt = {"reps": reps}
            search["attempts"].append(attempt)
            try:
                formula = SuzukiTrotter(order=order, reps=reps, preserve_order=True)
                evolution = PauliEvolutionGate(operator, time=time, synthesis=formula)
                network = formula.expand(evolution)
                attempt["expanded_rotations"] = len(network)
                if len(network) > MAX_EXPANDED_ROTATIONS:
                    attempt.update(status="CAP", reason="expanded-rotation cap")
                    search["stop_reason"] = "expanded-rotation cap"
                    break
                default_upstream = formula.synthesize(evolution)
                native = _lower(default_upstream)
                error = operator_errors(native.get_unitary(), matrix)
                attempt["error"] = error
                if error["phase_aligned"] > decomposition_error:
                    attempt["status"] = "REJECT_ERROR"
                    continue
            except Exception as exc:
                attempt.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                continue

            attempt["status"] = "PASS"
            search.update(status="PASS", selected_reps=reps, stop_reason="first passing step")
            common = {"order": order, "reps": reps, "source_rotation_count": len(network)}
            name = f"qiskit-pf-o{order}-r{reps}"
            rustiq_metadata = {
                "optimize_count": True,
                "upto_clifford": False, "upto_phase": False,
                "resynth_clifford_method": 1,
            }
            try:
                rustiq_upstream = synth_pauli_network_rustiq(
                    num_qubits=width, pauli_network=network,
                    optimize_count=True, preserve_order=True,
                    upto_clifford=False, upto_phase=False, resynth_clifford_method=1,
                )
            except Exception as exc:
                rustiq_upstream = None
                rustiq_metadata.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            for lowering, upstream, details in (
                ("default", default_upstream, {}),
                ("rustiq", rustiq_upstream, rustiq_metadata),
            ):
                for level in (0, 3):
                    candidate_metadata = {
                        **common, **details, "lowering": lowering,
                        "optimization_level": level,
                    }
                    emitted = None
                    if upstream is not None:
                        try:
                            # Retain exactly the artifact used to accept this
                            # time step; optional level-three failures stay visible.
                            emitted = native if lowering == "default" and level == 0 else _lower(
                                upstream, optimization_level=level,
                            )
                            candidate_metadata.update(
                                status="OK", precheck_error=operator_errors(emitted.get_unitary(), matrix),
                            )
                        except Exception as exc:
                            emitted = None
                            candidate_metadata.update(
                                status="ERROR", error=f"{type(exc).__name__}: {exc}",
                            )
                    candidates.append(CompilerCandidate(
                        f"{name}-{lowering}-opt{level}", emitted, candidate_metadata,
                    ))
            break
        else:
            search["stop_reason"] = "repetition cap"

    if not candidates:
        raise ProductFormulaBudgetExceeded(metadata)
    return [CompilerCandidate(
        candidate.name, candidate.native,
        {**deepcopy(metadata), **candidate.metadata},
    ) for candidate in candidates]
