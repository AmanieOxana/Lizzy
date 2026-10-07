"""Frozen, bounded chemistry routing measurement; no production routing changes.

Run ``python -m experiments.chemistry_routing_measure``. JSON is written only to
stdout and progress to stderr. See chemistry_routing_protocol.md for the fixed
accuracy task and stopping rules. Existing cached HamLib files are required.
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import expm_multiply

from lizzy import gf2
from lizzy.chemistry import (
    _factorization_fingerprint,
    factorize_molecular_ffsim,
    fermion_operator_openfermion,
    to_ffsim,
    to_pauli_openfermion,
)
from lizzy.dense import monomial_form
from lizzy.hamiltonian import anticommutation_matrix, symplectic_vectors, terms_of
from lizzy.hamlib import DEFAULT_CACHE, load_molecular

CASES = {"H2-4": ("H2", 4, (1, 1), "training"),
         "LiH-8": ("LiH", 8, (2, 2), "training"),
         "BH-10": ("BH", 10, (3, 3), "family-holdout"),
         "LiH-12": ("LiH", 12, (2, 2), "size-holdout")}
BACKENDS = ("qiskit-default", "qiskit-rustiq", "pytket-direct", "pytket-greedy", "pytket-peephole")
STATE_SEED, COMPILER_SEED, THRESHOLD = 20260921, 1729, 1e-3


def _sector_indices(norb, nelec):
    return np.array(sorted(sum(1 << q for q in alpha) + sum(1 << (norb+q) for q in beta)
        for alpha in combinations(range(norb), nelec[0])
        for beta in combinations(range(norb), nelec[1])), dtype=int)


def _probe_states(norb, nelec):
    indices = _sector_indices(norb, nelec)
    vectors = np.zeros((2**(2*norb), 4), dtype=complex)
    hf = (1 << nelec[0]) - 1 + (((1 << nelec[1]) - 1) << norb)
    vectors[hf, 0] = 1
    rng = np.random.default_rng(STATE_SEED)
    random = rng.normal(size=(len(indices), 3)) + 1j*rng.normal(size=(len(indices), 3))
    random /= np.linalg.norm(random, axis=0)
    vectors[indices, 1:] = random
    return indices, vectors


def _bit_reverse(width):
    values = np.arange(2**width, dtype=np.int64)
    return sum(((values >> q) & 1) << (width-1-q) for q in range(width))


def _bk_permutation(width):
    """Map JW occupation integers to BK integers, both Qiskit little-endian."""
    import openfermion
    encoder = openfermion.bravyi_kitaev_code(width).encoder.toarray().astype(np.int64)
    values = np.arange(2**width, dtype=np.int64)
    bits = (values[:, None] >> np.arange(width)) & 1
    encoded = (bits @ encoder.T) % 2
    permutation = encoded @ (1 << np.arange(width))
    if len(np.unique(permutation)) != len(values):
        raise ValueError("OpenFermion BK encoder is not a permutation")
    return permutation


def _encode(vectors, permutation):
    output = np.empty_like(vectors)
    output[permutation] = vectors
    return output


def _infidelities(outputs, targets):
    norms = np.linalg.norm(outputs, axis=0)
    target_norms = np.linalg.norm(targets, axis=0)
    norm_error = float(max(np.max(np.abs(norms-1)), np.max(np.abs(target_norms-1))))
    overlaps = np.sum(targets.conj()*outputs, axis=0) / (norms*target_norms)
    errors = np.maximum(0.0, 1.0 - np.abs(overlaps)**2)
    return errors.tolist(), norm_error


@lru_cache(maxsize=1024)
def _pauli_action(word):
    return monomial_form(word[::-1])


def _pauli_outputs(words, coefficients, time, steps, states):
    output = states.copy()
    half = [(word, coefficient*time/(2*steps)) for word, coefficient in zip(words, coefficients)]
    for _ in range(steps):
        for word, angle in half + half[::-1]:
            rows, phases = _pauli_action(word)
            applied = np.empty_like(output)
            applied[rows] = phases[:, None]*output
            output = np.cos(angle)*output - 1j*np.sin(angle)*applied
    return output


def _df_outputs(factorized, norb, nelec, time, steps, states):
    import ffsim
    addresses = np.asarray(ffsim.addresses_to_strings(np.arange(ffsim.dim(norb, nelec)), norb, nelec), dtype=int)
    output = np.zeros_like(states)
    for column in range(states.shape[1]):
        output[addresses, column] = ffsim.simulate_trotter_double_factorized(
            states[addresses, column], factorized, time, norb=norb, nelec=nelec,
            n_steps=steps, order=1,
        )
    return output


def _compile_backend(backend, payload):
    """Only existing library routes; all returned artifacts are concrete CX gates."""
    if backend == "ffsim-df":
        import ffsim
        from qiskit import QuantumCircuit
        from qiskit.transpiler import generate_preset_pass_manager
        factorized = payload["factorized"]
        raw = QuantumCircuit(payload["width"])
        raw.append(ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(factorized, payload["time"],
            n_steps=payload["steps"], order=1, tol=1e-10), range(payload["width"]))
        manager = generate_preset_pass_manager(basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=3, seed_transpiler=COMPILER_SEED)
        manager.pre_init = ffsim.qiskit.PRE_INIT
        return manager.run(raw), {"factorization_fingerprint": _factorization_fingerprint(factorized),
            "factor_count": len(factorized.diag_coulomb_mats)}
    words, coefficients, width = payload["words"], payload["coefficients"], payload["width"]
    if backend.startswith("qiskit"):
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        raw = QuantumCircuit(width)
        operator = SparsePauliOp.from_list(list(zip([word[::-1] for word in words], coefficients)))
        raw.append(PauliEvolutionGate(operator, time=payload["time"],
            synthesis=SuzukiTrotter(order=2, reps=payload["steps"])), range(width))
        if backend == "qiskit-rustiq":
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import HighLevelSynthesis
            from qiskit.transpiler.passes.synthesis import HLSConfig
            raw = PassManager([HighLevelSynthesis(hls_config=HLSConfig(
                PauliEvolution=[("rustiq", {"preserve_order": True})]))]).run(raw)
        return transpile(raw, basis_gates=["cx", "u"], optimization_level=3,
                          seed_transpiler=COMPILER_SEED), {}
    from pytket import OpType
    from pytket.passes import (
        AutoRebase,
        DecomposeBoxes,
        FullPeepholeOptimise,
        GreedyPauliSimp,
    )
    from pytket.qasm import circuit_to_qasm_str
    from qiskit import qasm2

    from lizzy.emit import pauli_exp_boxes
    half = [(word, coefficient*payload["time"]/(2*payload["steps"]))
            for word, coefficient in zip(words, coefficients)]
    raw = pauli_exp_boxes((half+half[::-1])*payload["steps"], width)
    if backend != "pytket-direct":
        DecomposeBoxes().apply(raw)
    if backend == "pytket-peephole":
        FullPeepholeOptimise().apply(raw)
    else:
        GreedyPauliSimp(seed=COMPILER_SEED).apply(raw)
    DecomposeBoxes().apply(raw)
    # Materialize the output permutation before exporting an ordinary gate
    # list; otherwise QASM drops implicit swaps and validates a different map.
    raw.replace_implicit_wire_swaps()
    AutoRebase({OpType.CX, OpType.TK1}).apply(raw)
    # pytket qubit q[0] and Qiskit wire 0 denote the same physical mode.
    # Its QASM2 writer lowers TK1 to standard gates; global phase is irrelevant
    # for this explicitly phase-insensitive sampled-state accuracy metric.
    return qasm2.loads(circuit_to_qasm_str(raw)), {}


def _worker(connection, backend, payload):
    start = perf_counter()
    try:
        circuit, metadata = _compile_backend(backend, payload)
        connection.send(("OK", circuit, metadata, perf_counter()-start))
    except Exception as exc:  # noqa: BLE001 - retain independent optional-backend failures
        connection.send(("UNAVAILABLE" if isinstance(exc, ImportError) else "FAIL_BACKEND",
                         None, {"reason": f"{type(exc).__name__}: {exc}"}, perf_counter()-start))
    finally:
        connection.close()


def _compile_bounded(backend, payload, timeout):
    """Spawn avoids inherited native-library thread locks; timeout is a hard cap."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, backend, payload))
    started = perf_counter()
    process.start()
    child.close()
    try:
        if parent.poll(timeout):
            try:
                status, circuit, metadata, compile_seconds = parent.recv()
            except EOFError:
                status, circuit, metadata, compile_seconds = "FAIL_BACKEND", None, {"reason": "worker exited without a result"}, 0.0
        else:
            status, circuit, metadata, compile_seconds = "TIMEOUT", None, {"reason": f"backend exceeded {timeout:g}s including startup"}, timeout
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join()
    return status, circuit, metadata, compile_seconds, perf_counter()-started


def _features(molecular, paulis, factorized):
    import ffsim
    features = {"n_qubits": molecular.n_qubits, "df_factors": len(factorized.diag_coulomb_mats)}
    for encoding, h in paulis.items():
        terms = terms_of(h)
        words = [p for _, p in terms]
        weights = np.array([sum(c != "I" for c in str(p)) for p in words])
        degrees = anticommutation_matrix(words).sum(axis=1)
        vectors = symplectic_vectors(words)
        features.update({f"{encoding}_terms": len(words),
            f"{encoding}_mean_weight": float(weights.mean()),
            f"{encoding}_ladder_cx": int(2*np.maximum(weights-1, 0).sum()),
            f"{encoding}_gf2_span_rank": gf2.rank(vectors),
            f"{encoding}_gf2_gram_rank": gf2.rank(gf2.gram(vectors)),
            f"{encoding}_mean_anticommutation_degree": float(degrees.mean())})
    mats = factorized.diag_coulomb_mats
    rotations = factorized.orbital_rotations
    distances = [len(ffsim.linalg.givens_decomposition(rotations[i].conj().T @ rotations[j], tol=1e-10)[0])
        for i in range(len(rotations)) for j in range(len(rotations)) if i != j]
    denominator = molecular.norb*(molecular.norb-1)/2
    features.update({"df_coulomb_density": float(np.mean(np.abs(mats)>1e-12)),
        "normalized_pairwise_givens": float(np.mean(distances)/denominator) if distances and denominator else 0.0,
        "pairwise_givens_mean": float(np.mean(distances)) if distances else 0.0})
    return features


def _load_case(name):
    import ffsim
    import openfermion
    molecule_name, width, nelec, role = CASES[name]
    path = DEFAULT_CACHE / "chemistry/electronic/standard" / f"{molecule_name}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"cached file required; no automatic downloads: {path}")
    molecule = load_molecular(path, f"ham_molec-{width}")
    paulis = {encoding: to_pauli_openfermion(molecule, encoding, qubit_order="alpha-then-beta")
              for encoding in ("jw", "bk")}
    factorized = factorize_molecular_ffsim(molecule, tol=1e-8).to_z_representation()
    source = fermion_operator_openfermion(molecule, qubit_order="alpha-then-beta")
    jw_sparse = openfermion.get_sparse_operator(openfermion.jordan_wigner(source), n_qubits=width).tocsr()
    indices, states = _probe_states(molecule.norb, nelec)
    reversed_indices = _bit_reverse(width)[indices]
    sector = jw_sparse[reversed_indices][:, reversed_indices]
    # Independently check that the chosen spin/number sector is invariant.
    outside = np.ones(2**width, dtype=bool)
    outside[reversed_indices] = False
    leakage = jw_sparse[:, reversed_indices][outside].tocoo()
    if len(leakage.data) and np.max(np.abs(leakage.data)) > 1e-10:
        raise ValueError("reference Hamiltonian does not preserve the declared sector")
    permutation = _bk_permutation(width)
    bk_sparse = openfermion.get_sparse_operator(openfermion.bravyi_kitaev(source, n_qubits=width), n_qubits=width).tocsr()
    bk_indices = _bit_reverse(width)[permutation[indices]]
    difference = (bk_sparse[bk_indices][:, bk_indices] - sector).tocoo()
    bk_error = float(np.max(np.abs(difference.data), initial=0.0))
    bk_outside = np.ones(2**width, dtype=bool)
    bk_outside[bk_indices] = False
    bk_leak = bk_sparse[:, bk_indices][bk_outside].tocoo()
    bk_error = max(bk_error, float(np.max(np.abs(bk_leak.data), initial=0.0)))
    addresses = np.asarray(ffsim.addresses_to_strings(np.arange(ffsim.dim(molecule.norb, nelec)), molecule.norb, nelec), dtype=int)
    ff_action = np.zeros_like(states)
    ff_action[addresses] = ffsim.linear_operator(to_ffsim(molecule), molecule.norb, nelec) @ states[addresses]
    ff_error = float(np.max(np.abs(ff_action[indices] - sector @ states[indices])))
    if max(bk_error, ff_error) > 1e-9:
        raise ValueError(f"Hamiltonian/encoding preflight mismatch: BK={bk_error}, ffsim={ff_error}")
    return molecule, paulis, factorized, indices, states, sector, {
        "name": name, "role": role, "source_path": str(path), "dataset": f"ham_molec-{width}",
        "assumed_sector": nelec, "sector_dimension": len(indices),
        "tensor_sha256": sha256(molecule.one_body_tensor.tobytes()+molecule.two_body_tensor.tobytes()
                                + np.asarray(molecule.constant).tobytes()).hexdigest(),
        "preflight_bk_sector_max_abs_error": bk_error, "preflight_ffsim_action_max_abs_error": ff_error,
        "features": _features(molecule, paulis, factorized),
        "factorization_fingerprint": _factorization_fingerprint(factorized)}


def _measure_backend(backend, payload, states, targets, logical, timeout):
    from qiskit.quantum_info import Statevector
    status, artifact, info, compiled, wall = _compile_bounded(backend, payload, timeout)
    record = {"backend": backend, "status": status, "compile_seconds": compiled,
              "worker_wall_seconds": wall, **info}
    if artifact is None:
        return record
    start = perf_counter()
    outputs = np.column_stack([Statevector(states[:, col]).evolve(artifact).data for col in range(4)])
    errors, norm_error = _infidelities(outputs, targets)
    mismatches, _ = _infidelities(outputs, logical)
    record.update({"two_qubit_gates": int(artifact.count_ops().get("cx", 0)),
        "total_gates": int(sum(artifact.count_ops().values())), "state_infidelities": errors,
        "max_infidelity": max(errors), "normalization_error": norm_error,
        "emission_mismatch_infidelity": max(mismatches),
        "verification_seconds": perf_counter()-start})
    if not np.all(np.isfinite(errors)) or norm_error > 1e-8 or max(mismatches) > 1e-9:
        record.update(status="FAIL_VERIFICATION", reason="nonfinite, normalization, or logical/emission mismatch")
    elif max(errors) <= THRESHOLD:
        record["status"] = "PASS"
    else:
        record["status"] = "FAIL_ACCURACY"
    return record


def run_measurements(case_names=None, times=(0.25, 1.0), *, backend_timeout=20.0):
    """Collect frozen grid, retaining every failed backend and formula candidate."""
    names = list(CASES if case_names is None else case_names)
    if any(name not in CASES for name in names):
        raise ValueError("unknown benchmark case")
    if not times or any(time not in (0.25, 1.0) for time in times):
        raise ValueError("only the predeclared times 0.25 and 1.0 are allowed")
    if not np.isfinite(backend_timeout) or backend_timeout <= 0:
        raise ValueError("backend_timeout must be finite and positive")
    data = {"configuration": {"state_seed": STATE_SEED, "compiler_seed": COMPILER_SEED,
        "times": list(times), "steps": [1, 2, 3], "threshold": THRESHOLD,
        "metric": "maximum sampled-state infidelity over reference determinant and three random sector states",
        "sectors": "assumed benchmark sectors; not verified ground-state sectors",
        "qubit_order": "alpha-then-beta; Qiskit little-endian statevectors",
        "state_preparation_and_encoding_conversion_cost": "excluded: encoding chosen before simulation",
        "backend_timeout_seconds": backend_timeout, "tensor_tolerance": 1e-8,
        "givens_tolerance": 1e-10, "df_frame_order": "input", "coulomb_cutoff": 0.0,
        "thread_environment": {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "accuracy_certified": False, "reference": "OpenFermion sparse JW sector matrix + scipy expm_multiply"},
        "versions": {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "scipy", "openfermion", "ffsim", "qiskit", "pytket")}}, "cases": [], "rows": []}
    for name in names:
        start = perf_counter()
        molecule, paulis, factorized, indices, states, sector, metadata = _load_case(name)
        metadata["preparation_seconds"] = perf_counter()-start
        data["cases"].append(metadata)
        permutation = _bk_permutation(molecule.n_qubits)
        for time in times:
            target = np.zeros_like(states)
            target[indices] = expm_multiply(-1j*time*sector, states[indices])
            for steps in (1, 2, 3):
                for route in ("jw", "bk", "df"):
                    print(f"{name} t={time:g} steps={steps} {route}", file=sys.stderr, flush=True)
                    payload = {"width": molecule.n_qubits, "time": time, "steps": steps}
                    encoded_states = _encode(states, permutation) if route == "bk" else states
                    encoded_target = _encode(target, permutation) if route == "bk" else target
                    if route == "df":
                        payload["factorized"] = factorized
                        backends = ("ffsim-df",)
                        logical = _df_outputs(factorized, molecule.norb, CASES[name][2], time, steps, states)
                    else:
                        terms = terms_of(paulis[route])
                        payload.update(words=[str(p) for _, p in terms], coefficients=[float(c.real) for c, _ in terms])
                        backends = BACKENDS
                        logical = _pauli_outputs(payload["words"], payload["coefficients"], time, steps, encoded_states)
                    records = []
                    for backend in backends:
                        try:
                            record = _measure_backend(backend, payload, encoded_states, encoded_target, logical, backend_timeout)
                        except Exception as exc:  # noqa: BLE001 - retain validation failures per artifact
                            record = {"backend": backend, "status": "FAIL_VERIFICATION", "reason": f"{type(exc).__name__}: {exc}"}
                        if route == "df" and record.get("factorization_fingerprint", metadata["factorization_fingerprint"]) != metadata["factorization_fingerprint"]:
                            record.update(status="FAIL_FACTORIZATION_VARIATION", reason="factorization fingerprint changed across candidates")
                        records.append(record)
                    passing = [record for record in records if record["status"] == "PASS"]
                    best = min(passing, key=lambda record: (record["two_qubit_gates"], record["total_gates"])) if passing else None
                    data["rows"].append({"case": name, "time": time, "steps": steps, "route": route,
                        "route_description": f"{route.upper()}+canonical-order S2" if route != "df" else "ffsim DF original-order S2",
                        "status": "PASS" if best else "NO_PASSING_BACKEND", "backends": records,
                        "best_backend": best["backend"] if best else None,
                        "two_qubit_gates": best["two_qubit_gates"] if best else None,
                        "max_infidelity": best["max_infidelity"] if best else None})
        _pauli_action.cache_clear()
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES), dest="cases")
    parser.add_argument("--time", action="append", type=float, choices=(0.25, 1.0), dest="times")
    parser.add_argument("--backend-timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    data = run_measurements(args.cases, tuple(args.times) if args.times else (0.25, 1.0), backend_timeout=args.backend_timeout)
    print(json.dumps(data, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
