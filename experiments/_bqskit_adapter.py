"""Bounded official BQSKit baseline, without another SDK's resynthesis.

The fixed policy is one level-one, seed-zero search on at most three qubits.
Its Hilbert--Schmidt synthesis threshold is not a spectral-error certificate;
the comparison harness independently checks the actual target-unitary error.
"""

import json
import os
import signal
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from experiments._compiler_adapters import (
    CompilerCandidate,
    native_append_clifford,
    native_append_rotation,
)
from lizzy.native import NativeCircuit, NativeGate

MAX_WIDTH = 3
TIMEOUT_SECONDS = 120
SYNTHESIS_EPSILON = 1e-14
_RESULT_PREFIX = "LIZZY_BQSKIT_RESULT="


class BQSKitResourceCap(TimeoutError):
    """A declared benchmark resource limit, not lack of upstream support."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def _settings():
    versions = {}
    for package in ("bqskit", "bqskitrs"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        "package": "bqskit", "versions": versions,
        "algorithm": "bqskit.compile standard unitary workflow",
        "optimization_level": 1, "seed": 0,
        "machine_gate_set": ["CNOT", "U3"], "connectivity": "all-to-all",
        "ancillas": 0, "num_workers": 1, "num_blas_threads": 1,
        "max_synthesis_size": 3, "width_cap": MAX_WIDTH,
        "timeout_seconds": TIMEOUT_SECONDS,
        "synthesis_epsilon": SYNTHESIS_EPSILON,
        "synthesis_metric": "upstream HilbertSchmidtResiduals cost; not spectral norm",
        "acceptance_metric": "independent harness phase-aligned spectral norm",
        "input_tensor_order": "big-endian", "target_phase_adjustment": False,
        "search_policy": "one fixed seed and preset; no per-target retries",
    }


def bqskit_to_native(circuit) -> NativeCircuit:
    """Translate only CNOT/U3, strictly preserving BQSKit's full gate unitary."""
    from bqskit.ir.gates import CNOTGate, U3Gate

    if any(radix != 2 for radix in circuit.radixes):
        raise ValueError("The BQSKit adapter accepts qubits only.")
    native = NativeCircuit(circuit.num_qudits)
    for operation in circuit:
        qubits = tuple(operation.location)
        if isinstance(operation.gate, CNOTGate):
            native_append_clifford(native, "cx", qubits)
        elif isinstance(operation.gate, U3Gate):
            theta, phi, lam = map(float, operation.params)
            native_append_rotation(native, "rz", lam, qubits[0])
            native_append_rotation(native, "ry", theta, qubits[0])
            native_append_rotation(native, "rz", phi, qubits[0])
            # U3(theta,phi,lambda) = exp(i(phi+lambda)/2) Rz(phi) Ry(theta) Rz(lambda).
            native.add_global_phase((phi + lam) / 2)
        else:
            raise ValueError(f"Unsupported BQSKit output gate {operation.gate!r}.")
    return native


def _free_runtime_ports():
    """Choose separate ephemeral ports for this owned localhost runtime."""
    with socket.socket() as first, socket.socket() as second:
        first.bind(("localhost", 0))
        second.bind(("localhost", 0))
        return first.getsockname()[1], second.getsockname()[1]


def _compile_in_worker(target):
    """Run the standard public API on an owned, single-worker local runtime."""
    from bqskit import compile as bq_compile
    from bqskit.compiler import Compiler, MachineModel
    from bqskit.ir.gates import CNOTGate, U3Gate

    width = len(target).bit_length() - 1
    model = MachineModel(width, gate_set={CNOTGate(), U3Gate()})
    port, worker_port = _free_runtime_ports()
    launch = (
        "from bqskit.runtime.attached import start_attached_server; "
        f"start_attached_server(1, port={port}, worker_port={worker_port}, num_blas_threads=1)"
    )
    server = subprocess.Popen([sys.executable, "-c", launch])
    try:
        with Compiler(ip="localhost", port=port) as compiler:
            circuit = bq_compile(
                target, model=model, optimization_level=1, max_synthesis_size=3,
                synthesis_epsilon=SYNTHESIS_EPSILON, seed=0, compiler=compiler,
            )
        native = bqskit_to_native(circuit)
        source = np.asarray(circuit.get_unitary())
        error = float(np.linalg.norm(native.get_unitary() - source, ord=2))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"BQSKit gate translation failed its strict matrix check: {error}")
        return {
            "width": native.width, "global_phase": native.global_phase,
            "gates": [[gate.kind, gate.qubits, gate.angle] for gate in native.gates],
            "metadata": {"adapter_strict_error": error,
                         "source_cx": native.two_qubit_gates,
                         "source_u3": sum(isinstance(op.gate, U3Gate) for op in circuit),
                         "upstream_branch": "QSearch" if width == 2 else "LEAP" if width == 3 else "single-qudit"},
        }
    finally:
        # Compiler.close disconnects this attached server and stops its workers.
        # The outer process-group deadline also covers startup/shutdown failures.
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.terminate()
            try:
                server.wait(timeout=1)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=1)


def _stop_owned_group(process):
    """Stop only the new process session started by this adapter, including workers."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("The owned BQSKit worker group did not terminate.")


def _run_bounded(target, metadata):
    if os.name != "posix":
        raise NotImplementedError("The bounded BQSKit benchmark currently needs POSIX process groups.")
    payload = json.dumps({"real": target.real.tolist(), "imag": target.imag.tolist()})
    process = subprocess.Popen(
        [sys.executable, "-m", "experiments._bqskit_adapter", "--worker"],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        _stop_owned_group(process)
        raise BQSKitResourceCap("BQSKit exceeded the declared 120-second wall-time cap.", metadata) from exc
    except BaseException:
        _stop_owned_group(process)
        raise
    lines = [line[len(_RESULT_PREFIX):] for line in stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if process.returncode or len(lines) != 1:
        raise RuntimeError(f"BQSKit worker failed (exit {process.returncode}): {stderr[-3000:]}")
    return json.loads(lines[0])


def bqskit_candidates(target: np.ndarray) -> list[CompilerCandidate]:
    """One fixed official BQSKit candidate; widths above three are explicitly capped."""
    matrix = np.asarray(target, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("BQSKit requires a square unitary target.")
    dimension = len(matrix)
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("BQSKit requires a positive-qubit power-of-two target dimension.")
    if not np.isfinite(matrix).all() or np.linalg.norm(matrix.conj().T @ matrix - np.eye(dimension), ord=2) > 1e-10:
        raise ValueError("The BQSKit target must be finite and unitary.")
    metadata = _settings()
    width = dimension.bit_length() - 1
    if width > MAX_WIDTH:
        raise BQSKitResourceCap("BQSKit width exceeds this benchmark's three-qubit search cap.", metadata)
    if any(value == "unavailable" for value in metadata["versions"].values()):
        raise ImportError("This benchmark requires the optional official bqskit and bqskitrs dependencies.")
    result = _run_bounded(matrix, metadata)
    native = NativeCircuit(result["width"], [NativeGate(kind, tuple(qubits), angle)
                                           for kind, qubits, angle in result["gates"]],
                           result["global_phase"])
    return [CompilerCandidate("bqskit-level1-seed0", native, {**metadata, **result["metadata"]})]


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal worker entry point; use the comparison harness.")
    supplied = json.load(sys.stdin)
    target = np.asarray(supplied["real"]) + 1j * np.asarray(supplied["imag"])
    print(_RESULT_PREFIX + json.dumps(_compile_in_worker(target)), flush=True)
