"""
    Native and optional shared-frame emission for Pauli-rotation sequences.

    The builtin candidate retains logical Pauli rotations and quotes their ladder
    cost, with runs on one qubit pair priced by their canonical 0/1/2/3-CNOT class.
    It does not materialize those CNOTs. A dependency-free signed-GF(2) backend
    emits concrete gates for abelian and one-logical-qubit DLA frames. At higher
    Pauli weight, optional pytket passes can share a Clifford frame across a wider
    sequence; intact-box and decomposed-box heuristics win on different sequences.

    The native backend has no SDK dependency; without pytket installed it and the
    builtin emission remain. Equivalence is not taken on faith: every native algebraic
    case and both pytket paths are pinned against dense exact sequences in the tests.
"""

from dataclasses import dataclass

from lizzy.emission.native import (
    NativeCircuit,
    ladder_circuit,
    native_frame_candidate,
    native_frame_circuit,
)
from lizzy.hamiltonian import Circuit


@dataclass(frozen=True, init=False)
class EmissionQuote:
    """A logical cost quote or concrete emission of a Lizzy rotation sequence.

    ``circuit`` deliberately has a backend-dependent type: it is the logical
    :class:`~lizzy.hamiltonian.Circuit` for ``builtin``, a concrete
    :class:`lizzy.emission.native.NativeCircuit` for ``native-frame``, and a rebased pytket
    circuit for the optional backends. The logical sequence remains available on
    :class:`lizzy.synthesize.Result` for provenance and dense verification. Use
    :attr:`is_concrete` to distinguish an emitted gate count from the builtin
    analytical block/ladder quote.
    """

    backend: str
    circuit: object
    _quoted_two_qubit_gates: int

    def __init__(self, backend: str, two_qubit_gates: int, circuit: object):
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "_quoted_two_qubit_gates", two_qubit_gates)

    @property
    def two_qubit_gates(self) -> int:
        """Current gate count, or logical block cost for the builtin candidate."""
        if isinstance(self.circuit, Circuit):
            return self.circuit.two_qubit_gates
        if hasattr(self.circuit, "n_2qb_gates"):
            return self.circuit.n_2qb_gates()
        return self._quoted_two_qubit_gates

    @property
    def is_concrete(self) -> bool:
        """Whether the retained non-logical artifact exposes its gate count.

        Native and pytket circuits provide ``n_2qb_gates``. A logical
        :class:`Circuit`, or an opaque artifact with only a supplied quote, does
        not provide a materialized gate count.
        """
        return not isinstance(self.circuit, Circuit) and callable(
            getattr(self.circuit, "n_2qb_gates", None)
        )


def pauli_exp_boxes(rotations, width: int):
    r"""
    Build a pytket circuit containing one ``PauliExpBox`` per rotation.

    The boxes deliberately stay intact.  This is the representation expected by
    Pauli-gadget synthesis passes, which can use relations between whole rotations
    before committing to individual CNOT ladders.

    The one place the angle convention lives: pytket's ``PauliExpBox`` takes half
    turns, ours is :math:`e^{-i\theta P}`, so the box parameter is
    :math:`2\theta/\pi`. Identity-only rotations are omitted, so this adapter's
    equivalence contract is up to global phase.

    Args:
        rotations: ``(word_or_PauliString, angle)`` pairs.
        width (int): Number of qubits.
    Returns:
        pytket.Circuit: A circuit whose nontrivial rotations are intact boxes.

    Raises:
        ImportError: If pytket is not installed.
    """
    import numpy as np
    from pytket import Circuit as TketCircuit
    from pytket.circuit import PauliExpBox
    from pytket.pauli import Pauli

    letters = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    circuit = TketCircuit(width)
    for pauli, angle in rotations:
        word = str(pauli)
        support = [q for q, letter in enumerate(word) if letter != "I"]
        if support:
            box = PauliExpBox([letters[word[q]] for q in support], 2 * angle / np.pi)
            circuit.add_pauliexpbox(box, support)
    return circuit


def pauli_boxes(rotations, width: int):
    r"""
    Build the pytket box circuit of a rotation sequence, boxes decomposed.

    Args:
        rotations: ``(word_or_PauliString, angle)`` pairs.
        width (int): Number of qubits.
    Returns:
        pytket.Circuit: The circuit, ready for an optimization pass.

    Raises:
        ImportError: If pytket is not installed.
    """
    from pytket.passes import DecomposeBoxes

    circuit = pauli_exp_boxes(rotations, width)
    DecomposeBoxes().apply(circuit)
    return circuit


def tket_circuit(circuit: Circuit, width: int):
    """
    Re-synthesize a rotation sequence in a shared Clifford frame with pytket.

    Args:
        circuit (Circuit): The rotations.
        width (int): Number of qubits.
    Returns:
        pytket.Circuit: An equivalent circuit in the CX/TK1 basis.

    Raises:
        ImportError: If pytket is not installed.
    """
    from pytket import OpType
    from pytket.passes import AutoRebase, GreedyPauliSimp

    synthesized = pauli_boxes(circuit.rotations, width)
    GreedyPauliSimp().apply(synthesized)
    AutoRebase({OpType.CX, OpType.TK1}).apply(synthesized)
    return synthesized


def direct_tket_circuit(
    circuit: Circuit,
    width: int,
    *,
    discount_rate: float = 0.9,
    depth_weight: float = 0.0,
    seed: int = 0,
):
    """
    Emit a Lizzy circuit through pytket's direct Pauli-gadget synthesis.

    Unlike :func:`tket_circuit`, this path presents intact ``PauliExpBox`` objects
    to ``GreedyPauliSimp``.  Boxes are decomposed only after the shared Clifford
    frame has been chosen, then the result is rebased to CX/TK1.  All heuristic
    controls are explicit and the fixed default seed makes repeated emission
    reproducible.

    Args:
        circuit (Circuit): The rotations.
        width (int): Number of qubits.
        discount_rate (float): Lookahead discount used by ``GreedyPauliSimp``.
        depth_weight (float): Weight assigned to depth by ``GreedyPauliSimp``.
        seed (int): Deterministic tie-breaking seed for ``GreedyPauliSimp``.
    Returns:
        pytket.Circuit: An equivalent circuit in the CX/TK1 basis.

    Raises:
        ImportError: If pytket is not installed.
    """
    from pytket import OpType
    from pytket.passes import AutoRebase, DecomposeBoxes, GreedyPauliSimp

    synthesized = pauli_exp_boxes(circuit.rotations, width)
    GreedyPauliSimp(
        discount_rate=discount_rate,
        depth_weight=depth_weight,
        seed=seed,
    ).apply(synthesized)
    DecomposeBoxes().apply(synthesized)
    AutoRebase({OpType.CX, OpType.TK1}).apply(synthesized)
    return synthesized


def _optional_tket_quote(backend, emitter, circuit: Circuit, width: int) -> EmissionQuote | None:
    """Isolate optional SDK failures and retain the artifact that was priced."""
    try:
        emitted = emitter(circuit, width)
        return EmissionQuote(backend, emitted.n_2qb_gates(), emitted)
    except Exception:  # noqa: BLE001 - an optional candidate must fail closed
        return None


def direct_emission_quote(circuit: Circuit, width: int) -> EmissionQuote | None:
    """Return the direct pytket candidate without letting it suppress fallback."""
    return _optional_tket_quote("pytket-direct", direct_tket_circuit, circuit, width)


def greedy_emission_quote(circuit: Circuit, width: int) -> EmissionQuote | None:
    """Return the decomposed-box pytket candidate, isolated from the portfolio."""
    return _optional_tket_quote("pytket-greedy", tket_circuit, circuit, width)


def native_emission_candidates(
    circuit: Circuit, width: int, *, include_frame: bool = True,
    frame_failures: tuple[type[Exception], ...] = (),
) -> list[tuple[str, NativeCircuit]]:
    """Build concrete, phase-preserving ladder and eligible-frame artifacts.

    Order is stable and ladder-first. No cost selection, dense verification or
    SDK lookup happens here. Failures propagate by default; a caller may name
    exceptions from optional frame eligibility/construction that should retain
    the ladder. Ladder construction errors always propagate. Logical cost
    estimates are never included in this portfolio.
    """
    candidates = [("native-ladder", ladder_circuit(circuit, width))]
    try:
        if include_frame and native_frame_candidate(circuit, width):
            candidates.append(("native-frame", native_frame_circuit(circuit, width)))
    except frame_failures:
        pass
    return candidates


def native_emission_quote(circuit: Circuit, width: int) -> EmissionQuote:
    """Return the dependency-free signed-GF(2) Clifford-frame candidate."""
    emitted = native_frame_circuit(circuit, width)
    return EmissionQuote("native-frame", emitted.n_2qb_gates(), emitted)


def emission_candidates(
    circuit: Circuit,
    width: int,
    *,
    exhaustive: bool = True,
) -> list[EmissionQuote]:
    """Quote the builtin logical cost and eligible concrete emission candidates.

    The builtin circuit is always a candidate. A dependency-free native candidate is
    added when the GF(2) span is abelian or is one non-abelian logical qubit. When
    pytket is installed and :func:`shared_frame_candidate` passes, the direct
    intact-box pass is also run, and with ``exhaustive`` the established
    decomposed-box pass too. Keeping the candidates separate matters: their
    heuristics win on different Hamiltonians, while the minimum never exceeds the
    builtin quoted cost. Each optional pytket candidate is isolated
    so an absent or incompatible pass leaves the builtin artifact available.

    Args:
        circuit (Circuit): Folded logical Pauli rotations.
        width (int): Number of qubits in the backend circuit.
        exhaustive (bool): Also run the slower decomposed-box heuristic. The direct
            path is the fast quote used while shortlisting synthesis routes.
    Returns:
        list[EmissionQuote]: Deterministically ordered logical and concrete quotes.
    """
    candidates = [
        EmissionQuote(
            backend="builtin",
            two_qubit_gates=circuit.two_qubit_gates,
            circuit=circuit,
        )
    ]
    if not circuit.rotations:
        return candidates

    # A rank-two non-abelian Pauli span is an su(2) ~= so(3) representation: one
    # global Clifford maps every rotation to X/Y/Z on a single qubit, independently
    # of sequence length. Abelian stabilizer frames are the other native exact case.
    if native_frame_candidate(circuit, width):
        candidates.append(native_emission_quote(circuit, width))

    if not shared_frame_candidate(circuit):
        return candidates

    direct_quote = direct_emission_quote(circuit, width)
    if direct_quote is not None:
        candidates.append(direct_quote)

    # The legacy pass is slower because it starts from decomposed boxes. During route
    # search the fast direct quote is sufficient; shortlisted routes and the final
    # complete result request the exhaustive portfolio.
    if exhaustive:
        shared_quote = greedy_emission_quote(circuit, width)
        if shared_quote is not None:
            candidates.append(shared_quote)
    return candidates


def best_emission(
    circuit: Circuit,
    width: int,
    *,
    exhaustive: bool = True,
) -> EmissionQuote:
    """Return the cheapest quote and retained artifact; inspect ``is_concrete``."""
    return min(
        emission_candidates(circuit, width, exhaustive=exhaustive),
        key=lambda quote: quote.two_qubit_gates,
    )


def shared_frame_candidate(circuit: Circuit) -> bool:
    """Cheap structural filter for circuits where a shared frame can plausibly pay.

    Any weight-three-or-higher rotation has ladder pressure the direct backend can
    share. Weight-two sequences are tried only when they commute globally (MaxCut is
    the motivating case); ordinary two-local product formulas stay on the fast builtin
    path. This replaces the old mean-weight-four cliff without compiling every small
    candidate in calibration and unit tests.
    """
    paulis = [pauli for pauli, _ in circuit.rotations]
    weights = [sum(letter != "I" for letter in str(pauli)) for pauli in paulis]
    if max(weights, default=0) >= 3:
        return True
    if max(weights, default=0) < 2 or len(paulis) > 2048:
        return False

    from lizzy.hamiltonian import anticommutation_matrix

    return not anticommutation_matrix(paulis).any()
