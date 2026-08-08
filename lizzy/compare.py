"""
    The compiler benchmark behind the README table.

    Two contracts, because no single one is fair at every size. At eight qubits every
    compiler gets the fewest steps that reach the budget against a dense reference --
    oracle information nobody has at scale, so nobody is favoured by their sizing
    method. At a hundred qubits nothing can be verified densely, so every compiler
    gets the same fixed-depth task instead.

    Needs the ``compare`` extra (qiskit, pytket). Run with ``python -m lizzy.compare``.
"""

import warnings

from lizzy.dense import circuit_matrix, evolution, infidelity
from lizzy.hamiltonian import Circuit, model, n_qubits, terms_of
from lizzy.hamlib import fetch, load
from lizzy.synthesize import synthesize

HAMLIB_CASES = [
    ("tfim-1D-100", "condensedmatter/tfim/tfim.zip", "graph-1D-grid-nonpbc-qubitnodes_Lx-100_h-1"),
    ("tfim-1D-pbc", "condensedmatter/tfim/tfim.zip", "graph-1D-grid-pbc-qubitnodes_Lx-100_h-1"),
    ("tfim-2D-10x10", "condensedmatter/tfim/tfim.zip", "graph-2D-grid-nonpbc-qubitnodes_Lx-10_Ly-10_h-1"),
    ("tfim-hex-4x4", "condensedmatter/tfim/tfim.zip", "graph-2D-hex-nonpbc-qubitnodes_Lx-4_Ly-4_h-1"),
    ("tfim-3D-3x3x3", "condensedmatter/tfim/tfim.zip", "graph-3D-grid-nonpbc-qubitnodes_Lx-3_Ly-3_Lz-3_h-1"),
    ("heis-1D-100", "condensedmatter/heisenberg/heis.zip", "graph-1D-grid-nonpbc-qubitnodes_Lx-100_h-1"),
    ("heis-2D-10x10", "condensedmatter/heisenberg/heis.zip", "graph-2D-grid-nonpbc-qubitnodes_Lx-10_Ly-10_h-1"),
    ("heis-2D-pbc", "condensedmatter/heisenberg/heis.zip", "graph-2D-grid-pbc-qubitnodes_Lx-10_Ly-10_h-1"),
    ("fh-1D-jw", "condensedmatter/fermihubbard/FH_D-1.zip", "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-50_U-4_enc-jw"),
    ("fh-1D-bk", "condensedmatter/fermihubbard/FH_D-1.zip", "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-50_U-4_enc-bk"),
    ("maxcut-circ100", "binaryoptimization/maxcut/random/ham-graph-circulant.zip", "circ-n-100_offsets-1-2"),
    ("H2-BK", "chemistry/electronic/standard/H2.zip", "ham_BK-4"),
    ("BH-BK", "chemistry/electronic/standard/BH.zip", "ham_BK-10"),
    ("LiH-BK", "chemistry/electronic/standard/LiH.zip", "ham_BK-12"),
    ("LiH-JW", "chemistry/electronic/standard/LiH.zip", "ham_JW-12"),
]

ORACLE_CASES = [
    ("tfim", 1.0),
    ("tfim", 8.0),
    ("heisenberg", 1.0),
    ("heisenberg", 8.0),
    ("heisenberg_all_to_all", 1.0),
    ("heisenberg_all_to_all", 4.0),
]


def _s2_sequence(hamiltonian_, time_, steps):
    """The plain symmetric second-order rotation sequence, for the competitors."""
    terms = [(c.real, str(p)) for c, p in terms_of(hamiltonian_)]
    sequence = []
    for _ in range(steps):
        sequence += [(w, c * 0.5 * time_ / steps) for c, w in terms]
        sequence += [(w, c * 0.5 * time_ / steps) for c, w in reversed(terms)]
    return sequence


def qiskit_cx(hamiltonian_, width, time_, steps, rustiq=False, level=3):
    """Two-qubit count of one Qiskit synthesis path, at ``optimization_level=3``.

    Level 3 is Qiskit's strongest setting; level 1 leaves roughly a factor of two on
    the table and would flatter this compiler. Prefer :func:`qiskit_best`, which also
    tries the Rustiq plugin.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import HighLevelSynthesis
    from qiskit.transpiler.passes.synthesis import HLSConfig

    op = SparsePauliOp.from_list(
        [(str(p)[::-1], c.real) for c, p in terms_of(hamiltonian_)]
    )
    gate = PauliEvolutionGate(op, time=time_, synthesis=SuzukiTrotter(order=2, reps=steps))
    circuit = QuantumCircuit(width)
    circuit.append(gate, range(width))
    if rustiq:
        config = HLSConfig(PauliEvolution=[("rustiq", {"preserve_order": True})])
        circuit = PassManager([HighLevelSynthesis(hls_config=config)]).run(circuit)
    transpiled = transpile(circuit, basis_gates=["cx", "u"], optimization_level=level)
    return transpiled.count_ops().get("cx", 0)


def qiskit_best(hamiltonian_, width, time_, steps):
    """Best two-qubit count Qiskit reaches, over its synthesis plugins.

    The default Suzuki synthesis and the Rustiq Pauli-network plugin win on different
    instances -- Rustiq on high-weight chemistry, the default everywhere else -- so
    both run and the better stands, the same courtesy pytket gets from its passes.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        width (int): Number of qubits.
        time_ (float): Evolution time.
        steps (int): Suzuki-2 step count.
    Returns:
        int: The lower of the two counts.
    """
    return min(
        qiskit_cx(hamiltonian_, width, time_, steps),
        qiskit_cx(hamiltonian_, width, time_, steps, rustiq=True),
    )


def tket_cx(hamiltonian_, width, time_, steps):
    """Best two-qubit count pytket reaches on the same formula.

    Which pass wins is instance-dependent -- GreedyPauliSimp is the Pauli-aware one,
    but FullPeepholeOptimise beats it on grids -- so all of them run and the best
    result stands, the same courtesy Qiskit gets from optimization_level=3.
    """
    from pytket import OpType
    from pytket.passes import AutoRebase, FullPeepholeOptimise, GreedyPauliSimp

    from lizzy.emit import pauli_boxes

    counts = []
    for optimization in (GreedyPauliSimp(), FullPeepholeOptimise()):
        circuit = pauli_boxes(_s2_sequence(hamiltonian_, time_, steps), width)
        optimization.apply(circuit)
        AutoRebase({OpType.CX, OpType.TK1}).apply(circuit)
        counts.append(circuit.n_gates_of_type(OpType.CX))
    return min(counts)


def _check_tket_convention() -> None:
    """Pin pytket's half-turn phase convention against the dense reference."""
    from lizzy.emit import pauli_boxes

    h = model("heisenberg", 3, seed=1)
    sequence = _s2_sequence(h, 0.7, 2)
    ours = Circuit()
    for word, angle in sequence:
        ours.add(word, angle, "x")
    boxes = pauli_boxes(sequence, 3)
    assert infidelity(circuit_matrix(ours, 3), boxes.get_unitary()) < 1e-9


def _bisect(check, high=64):
    low = 1
    while low < high:
        middle = (low + high) // 2
        if check(middle):
            high = middle
        else:
            low = middle + 1
    return low


def oracle_table(error: float = 1e-3) -> None:
    """Every compiler gets the fewest steps that reach the budget, dense-verified."""
    from qiskit.quantum_info import Operator

    print(f"\nOracle for everyone, n=8, eps={error}, dense-verified")
    print(f"{'model':22}{'t':>4}{'lizzy':>9}{'qiskit':>9}{'tket':>9}")
    for name, time_ in ORACLE_CASES:
        h = model(name, 8, seed=1)
        target = evolution(h, time_)

        def ours_ok(steps, h=h, time_=time_, target=target):
            built = synthesize(h, time=time_, steps=steps)
            return infidelity(target, circuit_matrix(built.circuit, 8)) < error

        ours = synthesize(h, time=time_, steps=_bisect(ours_ok))

        def qiskit_ok(steps, h=h, time_=time_, target=target):
            from qiskit import QuantumCircuit, transpile
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.synthesis import SuzukiTrotter

            op = SparsePauliOp.from_list(
                [(str(p)[::-1], c.real) for c, p in terms_of(h)]
            )
            gate = PauliEvolutionGate(
                op, time=time_, synthesis=SuzukiTrotter(order=2, reps=steps)
            )
            circuit = QuantumCircuit(8)
            circuit.append(gate, range(8))
            unitary = Operator(
                transpile(circuit, basis_gates=["cx", "u"]).reverse_bits()
            ).data
            return infidelity(target, unitary) < error

        steps = _bisect(qiskit_ok)
        row = (
            f"{name:22}{time_:>4.0f}{ours.two_qubit_gates:>9,}"
            f"{qiskit_best(h, 8, time_, steps):>9,}"
            f"{tket_cx(h, 8, time_, steps):>9,}"
        )
        print(row, flush=True)


def hamlib_table(steps: int = 2, time_: float = 1.0) -> None:
    """Every compiler gets the same fixed-depth task on HamLib at 100 qubits."""
    from lizzy.emit import tket_two_qubit_gates

    print(f"\nHamLib, fixed steps={steps}, t={time_}")
    print(f"{'instance':16}{'n':>5}{'terms':>7}{'lizzy':>9}{'qiskit':>9}{'tket':>9}  route")
    for label, archive, key in HAMLIB_CASES:
        try:
            h = load(fetch(archive), key)
        except Exception as failure:  # noqa: BLE001 - a bad row must not kill the table
            print(f"{label:16} {type(failure).__name__}: {failure}", flush=True)
            continue
        width = n_qubits(h)
        ours = synthesize(h, time=time_, steps=steps)
        # The router prices the emissions too: the builtin ladder/block count against
        # the shared-frame backend when it is available, cheapest wins.
        best, emission = ours.two_qubit_gates, ""
        shared = tket_two_qubit_gates(ours.circuit, width)
        if shared is not None and shared < best:
            best, emission = shared, "+frame"
        row = (
            f"{label:16}{width:>5}{len(terms_of(h)):>7}{best:>9,}"
            f"{qiskit_best(h, width, time_, steps):>9,}"
            f"{tket_cx(h, width, time_, steps):>9,}"
        )
        print(row + f"  {','.join(ours.routes)}{emission}", flush=True)


def main() -> None:
    """Run both benchmark tables."""
    warnings.filterwarnings("ignore")
    _check_tket_convention()
    hamlib_table()
    oracle_table()


if __name__ == "__main__":
    main()
