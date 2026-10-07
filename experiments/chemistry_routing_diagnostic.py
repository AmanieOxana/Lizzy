"""Reproduce the bounded LiH-8 emission-mismatch diagnosis, without reclassification.

Run ``python -m experiments.chemistry_routing_diagnostic``. This inspects exactly
one case (t=.25, one DF step), using one retained factorization and the original
four sector probes. JSON goes to stdout only. It neither changes the measurement
harness nor relaxes its frozen acceptance criteria.

The pass boundary where a discrepancy first appears can be identified by replay;
the internal numerical/decomposition mechanism cannot be inferred from that alone.
"""

import json
import os
import platform
from importlib.metadata import version

import numpy as np
from scipy.sparse.linalg import expm_multiply

from experiments.chemistry_routing_measure import (
    COMPILER_SEED,
    STATE_SEED,
    _df_outputs,
    _factorization_fingerprint,
    _infidelities,
    _load_case,
)


def diagnose():
    """Compare optimization levels 0--3 and locate the first changing level-3 pass."""
    import ffsim
    from qiskit import QuantumCircuit
    from qiskit.converters import dag_to_circuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import generate_preset_pass_manager

    molecular, _, factorized, indices, states, sector, metadata = _load_case("LiH-8")
    time, steps = 0.25, 1
    logical = _df_outputs(factorized, molecular.norb, (2, 2), time, steps, states)
    reference = np.zeros_like(states)
    reference[indices] = expm_multiply(-1j * time * sector, states[indices])
    threshold = 1e-9

    def replay(circuit):
        return np.column_stack([
            Statevector(states[:, column]).evolve(circuit).data for column in range(4)
        ])

    pass_evidence = {"last_verified_below_guard": None, "first_exceeding_guard": None,
                     "unreplayable_passes_before_first_change": []}

    def callback(**context):
        if pass_evidence["first_exceeding_guard"] is not None:
            return
        item = {"index": int(context["count"]), "name": type(context["pass_"]).__name__}
        try:
            circuit = dag_to_circuit(context["dag"])
            errors, norm_error = _infidelities(replay(circuit), logical)
        except Exception as exc:  # noqa: BLE001 - record unreplayable upstream pass boundaries
            item["reason"] = f"{type(exc).__name__}: {exc}"
            pass_evidence["unreplayable_passes_before_first_change"].append(item)
            return
        item.update(total_gates=len(circuit), max_logical_mismatch_infidelity=max(errors),
                    normalization_error=norm_error)
        key = "first_exceeding_guard" if max(errors) > threshold else "last_verified_below_guard"
        pass_evidence[key] = item

    rows = []
    for level in (0, 1, 2, 3):
        raw = QuantumCircuit(molecular.n_qubits)
        raw.append(ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            factorized, time, n_steps=steps, order=1, tol=1e-10), range(molecular.n_qubits))
        manager = generate_preset_pass_manager(
            basis_gates=["cx", "rz", "sx", "x"], optimization_level=level,
            seed_transpiler=COMPILER_SEED, approximation_degree=1.0,
        )
        manager.pre_init = ffsim.qiskit.PRE_INIT
        emitted = manager.run(raw, callback=callback if level == 3 else None)
        output = replay(emitted)
        mismatches, logical_norm_error = _infidelities(output, logical)
        errors, reference_norm_error = _infidelities(output, reference)
        rows.append({
            "optimization_level": level,
            "cx": int(emitted.count_ops().get("cx", 0)),
            "total_gates": len(emitted),
            "state_reference_infidelities": errors,
            "max_reference_infidelity": max(errors),
            "state_logical_mismatch_infidelities": mismatches,
            "max_logical_mismatch_infidelity": max(mismatches),
            "normalization_error": max(logical_norm_error, reference_norm_error),
            "meets_primary_sampled_target": max(errors) <= 1e-3,
            "meets_frozen_emission_guard": max(mismatches) <= threshold,
        })

    logical_errors, _ = _infidelities(logical, reference)
    assert _factorization_fingerprint(factorized) == metadata["factorization_fingerprint"]
    return {
        "scope": "Post-measurement diagnostic only; primary data and classifications unchanged",
        "case": "LiH-8", "time": time, "steps": steps, "assumed_sector": [2, 2],
        "sector_dimension": len(indices), "state_seed": STATE_SEED,
        "compiler_seed": COMPILER_SEED, "approximation_degree": 1.0,
        "factorization_fingerprint": metadata["factorization_fingerprint"],
        "retained_factorization_reused_for_every_level": True,
        "tensor_sha256": metadata["tensor_sha256"],
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "openfermion", "ffsim", "qiskit")}},
        "thread_environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "logical_df_max_reference_infidelity": max(logical_errors),
        "sampled_accuracy_threshold": 1e-3,
        "frozen_emission_guard": threshold,
        "levels": rows, "level3_pass_evidence": pass_evidence,
        "interpretation": (
            "Pass replay identifies where the sampled logical discrepancy first appears. "
            "It does not prove which internal Weyl specialization or numerical tolerance causes it. "
            "Passing the evolution target does not retrospectively override the frozen emission guard."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, allow_nan=False))
