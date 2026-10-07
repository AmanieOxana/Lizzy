"""Dependency-free Clifford-frame emission over GF(2).

The logical circuit stores Pauli rotations because that is the representation in
which synthesis and dense verification are simplest.  This module lowers such a
sequence to an inspectable ``H/S/Sdg/CX/Rz`` circuit while keeping a signed
symplectic tableau for the Clifford frame.

Two algebraic cases receive a fixed global frame:

* an abelian Pauli span is stabilizer-reduced to diagonal generators;
* a non-degenerate rank-two span is one logical qubit (``su(2) ~= so(3)``), so every
  word becomes one of ``X``, ``Y`` and ``Z`` on the same physical qubit.

Everything else can still use the exact rolling-frame construction.  The emission
portfolio retains the result only when its concrete two-qubit count wins, so this
native backend is a monotone addition rather than a new routing assumption.
"""

from dataclasses import dataclass, field
from operator import index as integer_index

import numpy as np

from lizzy.algebra import gf2
from lizzy.hamiltonian import Circuit, symplectic_vectors

_ARITY = {"h": 1, "s": 1, "sdg": 1, "cx": 2, "rz": 1}


def _finite_float(value: object, name: str) -> float:
    """Return one finite real scalar with a stable plain-Python representation."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite real number.")
    return parsed


def _qasm3(width: int, gates, phase: float) -> str:
    """Common byte-stable serializer for native and Clifford+T artifacts."""
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    if width:
        lines.append(f"qubit[{width}] q;")
    if phase:
        lines.append(f"gphase({phase!r});")
    for gate in gates:
        operands = ", ".join(f"q[{qubit}]" for qubit in gate.qubits)
        if gate.kind == "rz":
            assert gate.angle is not None
            kind = f"rz({gate.angle!r})"
        else:
            kind = {"sdg": "inv @ s", "tdg": "inv @ t"}.get(gate.kind, gate.kind)
        lines.append(f"{kind} {operands};")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class NativeGate:
    r"""One gate in a dependency-free native emission.

    ``rz`` uses the standard hardware convention
    :math:`R_Z(\phi)=\exp(-i\phi Z/2)`.  Lizzy rotations therefore emit an angle
    ``2 * theta`` after the Pauli has been reduced to ``Z``.
    """

    kind: str
    qubits: tuple[int, ...]
    angle: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ARITY:
            raise ValueError(f"Unsupported native gate {self.kind!r}.")
        try:
            qubits = tuple(integer_index(qubit) for qubit in self.qubits)
        except TypeError as exc:
            raise ValueError("Native gate qubits must be integer indices.") from exc
        if any(isinstance(qubit, (bool, np.bool_)) for qubit in self.qubits):
            raise ValueError("Native gate qubits must be integer indices.")
        object.__setattr__(self, "qubits", qubits)
        if len(self.qubits) != _ARITY[self.kind]:
            raise ValueError(
                f"Gate {self.kind!r} needs {_ARITY[self.kind]} qubits, "
                f"got {len(self.qubits)}."
            )
        if any(qubit < 0 for qubit in self.qubits):
            raise ValueError("Native gate qubits must be non-negative.")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("A native two-qubit gate needs distinct qubits.")
        if (self.kind == "rz") != (self.angle is not None):
            raise ValueError("Only rz gates carry an angle, and every rz needs one.")
        if self.kind == "rz":
            object.__setattr__(self, "angle", _finite_float(self.angle, "rz angle"))

    def inverse(self) -> "NativeGate":
        """Return the exact inverse gate."""
        if self.kind == "s":
            return NativeGate("sdg", self.qubits)
        if self.kind == "sdg":
            return NativeGate("s", self.qubits)
        if self.kind == "rz":
            assert self.angle is not None
            return NativeGate("rz", self.qubits, -self.angle)
        return self


@dataclass
class NativeCircuit:
    r"""Concrete ``H/S/Sdg/CX/Rz`` circuit emitted without an SDK dependency.

    ``global_phase`` is in radians: the represented unitary is
    :math:`e^{i\,\mathtt{global\_phase}}` times the ordered gate product.
    """

    width: int
    gates: list[NativeGate] = field(default_factory=list)
    global_phase: float = 0.0

    def __post_init__(self) -> None:
        try:
            width = integer_index(self.width)
        except TypeError as exc:
            raise ValueError("A circuit width must be a non-negative integer.") from exc
        if isinstance(self.width, (bool, np.bool_)):
            raise TypeError("A circuit width must be a non-negative integer.")
        self.width = width
        if self.width < 0:
            raise ValueError("A circuit width must be a non-negative integer.")
        self.global_phase = _finite_float(self.global_phase, "global_phase")
        for gate in self.gates:
            self._validate_gate(gate)

    def _validate_gate(self, gate: NativeGate) -> None:
        if any(qubit >= self.width for qubit in gate.qubits):
            raise ValueError(
                f"Gate {gate.kind!r} addresses {gate.qubits} in width {self.width}."
            )

    def append(self, gate: NativeGate) -> None:
        """Append one validated gate."""
        self._validate_gate(gate)
        self.gates.append(gate)

    def add_global_phase(self, angle: float) -> None:
        """Add a finite phase angle in radians."""
        current = _finite_float(self.global_phase, "global_phase")
        contribution = _finite_float(angle, "global phase contribution")
        self.global_phase = _finite_float(
            current + contribution, "global_phase"
        )

    @property
    def two_qubit_gates(self) -> int:
        """Number of concrete two-qubit gates in the retained artifact."""
        return sum(gate.kind == "cx" for gate in self.gates)

    def n_2qb_gates(self) -> int:
        """Backend-neutral count hook used by :class:`lizzy.emission.emit.EmissionQuote`."""
        return self.two_qubit_gates

    def get_unitary(self) -> np.ndarray:
        """Build the dense unitary, intended for small-width verification."""
        if self.width > 10:
            raise ValueError("Dense native-circuit verification is capped at 10 qubits.")
        dimension = 2**self.width
        phase = _finite_float(self.global_phase, "global_phase")
        total = np.exp(1j * phase) * np.eye(dimension, dtype=complex)
        for gate in self.gates:
            total = self._gate_matrix(gate) @ total
        return total

    def to_qasm3(self) -> str:
        """Serialize deterministically as dependency-free OpenQASM 3 text."""
        phase = _finite_float(self.global_phase, "global_phase")
        return _qasm3(self.width, self.gates, phase)

    def _gate_matrix(self, gate: NativeGate) -> np.ndarray:
        if gate.kind == "cx":
            control, target = gate.qubits
            states = np.arange(2**self.width)
            control_bit = 1 << (self.width - 1 - control)
            target_bit = 1 << (self.width - 1 - target)
            rows = states ^ np.where(states & control_bit, target_bit, 0)
            matrix = np.zeros((states.size, states.size), dtype=complex)
            matrix[rows, states] = 1
            return matrix

        if gate.kind == "h":
            local = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif gate.kind == "s":
            local = np.diag([1, 1j]).astype(complex)
        elif gate.kind == "sdg":
            local = np.diag([1, -1j]).astype(complex)
        else:
            assert gate.kind == "rz" and gate.angle is not None
            local = np.diag(
                [np.exp(-0.5j * gate.angle), np.exp(0.5j * gate.angle)]
            )

        matrix = np.array([[1]], dtype=complex)
        qubit = gate.qubits[0]
        for index in range(self.width):
            matrix = np.kron(matrix, local if index == qubit else np.eye(2))
        return matrix


@dataclass(frozen=True)
class GF2FrameProfile:
    """Representation-dependent invariants used to decide native-frame eligibility."""

    width: int
    span_rank: int
    gram_rank: int

    @property
    def canonical_qubits(self) -> int:
        r"""Fewest symplectic axes needed by the span's canonical form.

        A rank-``2s`` symplectic part consumes ``s`` qubits and every radical axis
        consumes one more, hence ``rank - gram_rank / 2``.
        """
        return self.span_rank - self.gram_rank // 2

    @property
    def is_abelian(self) -> bool:
        """Whether every distinct Pauli in the span commutes."""
        return self.gram_rank == 0

    @property
    def single_qubit_dla(self) -> bool:
        """Whether the Pauli span is exactly one non-abelian logical qubit."""
        return self.span_rank == 2 and self.gram_rank == 2


def _unique_vectors(circuit: Circuit, width: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    paulis = []
    seen: set[str] = set()
    all_vectors = []
    for pauli, _ in circuit.rotations:
        word = str(pauli)
        if len(word) != width:
            raise ValueError(f"Pauli width {len(word)} does not match circuit width {width}.")
        vector = symplectic_vectors([pauli])[0].astype(np.uint8)
        all_vectors.append(vector)
        if word not in seen and vector.any():
            seen.add(word)
            paulis.append(pauli)
    unique = (
        [row.astype(np.uint8) for row in symplectic_vectors(paulis)] if paulis else []
    )
    return all_vectors, unique


def _frame_profile_from_vectors(unique: list[np.ndarray], width: int) -> GF2FrameProfile:
    """Measure an already parsed Pauli span without parsing the circuit again."""
    vectors = (
        np.array(unique, dtype=np.uint8).reshape(len(unique), 2 * width)
        if unique
        else np.zeros((0, 2 * width), dtype=np.uint8)
    )
    gram_matrix = gf2.gram(vectors)
    return GF2FrameProfile(
        width=width,
        span_rank=gf2.rank(vectors),
        gram_rank=gf2.rank(gram_matrix),
    )


def frame_profile(circuit: Circuit, width: int | None = None) -> GF2FrameProfile:
    """Measure the GF(2) span and restricted symplectic form of a rotation sequence."""
    if width is None:
        width = len(str(circuit.rotations[0][0])) if circuit.rotations else 0
    if width < 0:
        raise ValueError("A circuit width cannot be negative.")
    _, unique = _unique_vectors(circuit, width)
    return _frame_profile_from_vectors(unique, width)


def native_frame_candidate(circuit: Circuit, width: int) -> bool:
    """Return whether a measured-profitable canonical frame is available.

    Abelian stabilizer spans and a single non-abelian logical qubit have constructive
    global canonicalizers. Larger spans can be emitted by :func:`native_frame_circuit`
    explicitly, but its rolling heuristic is not run repeatedly during route search:
    measurements show that broader chemistry frames remain pytket territory for now.
    """
    if len(circuit.rotations) < 2:
        return False
    profile = frame_profile(circuit, width)
    return profile.is_abelian or profile.single_qubit_dla


class _SignedFrame:
    """Images and signs of ``X_0..X_n,Z_0..Z_n`` under a Clifford prefix."""

    def __init__(self, width: int):
        self.width = width
        self.tableau = np.eye(2 * width, dtype=np.uint8)
        self.signs = np.zeros(2 * width, dtype=np.uint8)

    def copy(self) -> "_SignedFrame":
        clone = _SignedFrame(self.width)
        clone.tableau = self.tableau.copy()
        clone.signs = self.signs.copy()
        return clone

    def apply(self, gate: NativeGate) -> None:
        """Conjugate every generator image by one physical Clifford gate."""
        if gate.kind == "h":
            q = gate.qubits[0]
            x, z = self.tableau[:, q].copy(), self.tableau[:, self.width + q].copy()
            self.signs ^= x & z
            self.tableau[:, q], self.tableau[:, self.width + q] = z, x
        elif gate.kind in {"s", "sdg"}:
            q = gate.qubits[0]
            x, z = self.tableau[:, q].copy(), self.tableau[:, self.width + q].copy()
            if gate.kind == "s":
                self.signs ^= x & z
            else:
                self.signs ^= x & (z ^ 1)
            self.tableau[:, self.width + q] ^= x
        elif gate.kind == "cx":
            control, target = gate.qubits
            xc = self.tableau[:, control].copy()
            xt = self.tableau[:, target].copy()
            zc = self.tableau[:, self.width + control].copy()
            zt = self.tableau[:, self.width + target].copy()
            self.signs ^= xc & zt & (xt ^ zc ^ 1)
            self.tableau[:, target] ^= xc
            self.tableau[:, self.width + control] ^= zt
        else:
            raise ValueError("A non-Clifford rz gate cannot update a Clifford frame.")

    def transform(self, vector: np.ndarray) -> tuple[int, np.ndarray]:
        """Conjugate one Hermitian Pauli, returning its sign and symplectic bits."""
        vector = np.asarray(vector, dtype=np.uint8) % 2
        if vector.shape != (2 * self.width,):
            raise ValueError("Pauli vector has the wrong width for this frame.")

        x, z = vector[: self.width], vector[self.width :]
        phase = int(np.dot(x, z)) % 4
        accumulated = np.zeros(2 * self.width, dtype=np.uint8)
        generator_indices = list(np.flatnonzero(x)) + [
            self.width + int(index) for index in np.flatnonzero(z)
        ]
        for index in generator_indices:
            image = self.tableau[index]
            phase = (phase + 2 * int(self.signs[index])) % 4
            phase = (phase + _product_phase(accumulated, image, self.width)) % 4
            accumulated ^= image

        if phase not in (0, 2):
            raise ValueError("A Clifford mapped a Hermitian Pauli to a non-Hermitian one.")
        return phase // 2, accumulated


def _product_phase(left: np.ndarray, right: np.ndarray, width: int) -> int:
    """Power of ``i`` in ``P(left) P(right) / P(left xor right)``."""
    lx, lz = left[:width], left[width:]
    rx, rz = right[:width], right[width:]
    combined = left ^ right
    cx, cz = combined[:width], combined[width:]
    exponent = (
        int(np.dot(lx, lz))
        + int(np.dot(rx, rz))
        + 2 * int(np.dot(lz, rx))
        - int(np.dot(cx, cz))
    )
    return exponent % 4


def _apply_bits(rows: np.ndarray, gate: NativeGate, width: int) -> None:
    """Apply only the unsigned symplectic action to working generator rows."""
    if gate.kind == "h":
        q = gate.qubits[0]
        rows[:, [q, width + q]] = rows[:, [width + q, q]]
    elif gate.kind in {"s", "sdg"}:
        q = gate.qubits[0]
        rows[:, width + q] ^= rows[:, q]
    elif gate.kind == "cx":
        control, target = gate.qubits
        rows[:, target] ^= rows[:, control]
        rows[:, width + control] ^= rows[:, width + target]
    else:
        raise ValueError("Only Clifford gates act on symplectic work rows.")


def _apply_frame_gate(
    frame: _SignedFrame,
    gates: list[NativeGate],
    gate: NativeGate,
    rows: np.ndarray | None = None,
) -> None:
    frame.apply(gate)
    if rows is not None:
        _apply_bits(rows, gate, frame.width)
    gates.append(gate)


def _support(vector: np.ndarray, width: int) -> list[int]:
    return list(np.flatnonzero(vector[:width] | vector[width:]))


def _reduce_to_z(
    frame: _SignedFrame,
    vector: np.ndarray,
    pivot: int,
    gates: list[NativeGate],
) -> None:
    """Extend ``frame`` until ``vector`` maps to signed ``Z[pivot]``."""
    _, transformed = frame.transform(vector)
    if pivot not in _support(transformed, frame.width):
        raise ValueError("The requested reduction pivot is outside the Pauli support.")

    for qubit in _support(transformed, frame.width):
        x, z = transformed[qubit], transformed[frame.width + qubit]
        if x:
            if z:
                _apply_frame_gate(frame, gates, NativeGate("sdg", (qubit,)))
            _apply_frame_gate(frame, gates, NativeGate("h", (qubit,)))
    _, transformed = frame.transform(vector)
    if transformed[: frame.width].any():
        raise ValueError("Local Clifford reduction failed to diagonalize a Pauli.")

    for qubit in _support(transformed, frame.width):
        if qubit != pivot:
            _apply_frame_gate(frame, gates, NativeGate("cx", (qubit, pivot)))
    _, transformed = frame.transform(vector)
    expected = np.zeros(2 * frame.width, dtype=np.uint8)
    expected[frame.width + pivot] = 1
    if not np.array_equal(transformed, expected):
        raise ValueError("Parity reduction did not produce the requested Z axis.")


def _independent_rows(vectors: list[np.ndarray]) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    for vector in vectors:
        trial = np.array(chosen + [vector], dtype=np.uint8)
        if gf2.rank(trial) > len(chosen):
            chosen.append(vector.copy())
    return chosen


def _commuting_frames(
    vectors: list[np.ndarray], width: int
) -> list[tuple[_SignedFrame, list[NativeGate]]]:
    """Construct stabilizer frames under a few deterministic generator orderings."""
    independent = _independent_rows(vectors)
    if not independent:
        return [(_SignedFrame(width), [])]
    orderings = [independent]
    weighted = sorted(independent, key=lambda row: (-len(_support(row, width)), tuple(row)))
    if any(not np.array_equal(a, b) for a, b in zip(independent, weighted)):
        orderings.append(weighted)

    answers = []
    for ordering in orderings:
        basis = np.array(ordering, dtype=np.uint8)
        frame, gates = _SignedFrame(width), []
        pivots: list[int] = []
        for row in range(len(basis)):
            current_support = _support(basis[row], width)
            if not current_support:
                raise ValueError("Independent commuting generator vanished in reduction.")
            for qubit in current_support:
                x, z = basis[row, qubit], basis[row, width + qubit]
                if x:
                    if z:
                        _apply_frame_gate(
                            frame, gates, NativeGate("sdg", (qubit,)), basis
                        )
                    _apply_frame_gate(frame, gates, NativeGate("h", (qubit,)), basis)

            diagonal = _support(basis[row], width)
            available = [qubit for qubit in diagonal if qubit not in pivots]
            if not available:
                raise ValueError("Commuting canonicalization ran out of pivot qubits.")
            pivot = min(available)
            for qubit in diagonal:
                if qubit != pivot:
                    _apply_frame_gate(
                        frame, gates, NativeGate("cx", (qubit, pivot)), basis
                    )

            expected = np.zeros(2 * width, dtype=np.uint8)
            expected[width + pivot] = 1
            if not np.array_equal(basis[row], expected):
                raise ValueError("Commuting generator did not reduce to one Z axis.")
            for later in range(row + 1, len(basis)):
                if basis[later, width + pivot]:
                    basis[later] ^= basis[row]
            pivots.append(pivot)
        answers.append((frame, gates))
    return answers


def _anticommutes(left: np.ndarray, right: np.ndarray, width: int) -> bool:
    return bool(
        (np.dot(left[:width], right[width:]) + np.dot(left[width:], right[:width]))
        % 2
    )


def _single_qubit_frames(
    vectors: list[np.ndarray], width: int
) -> list[tuple[_SignedFrame, list[NativeGate]]]:
    """Enumerate frames mapping a rank-two non-abelian span onto one qubit."""
    answers = []
    for first_index, first in enumerate(vectors):
        for second_index, second in enumerate(vectors):
            if first_index == second_index or not _anticommutes(first, second, width):
                continue
            for pivot in _support(first, width):
                frame, gates = _SignedFrame(width), []
                _reduce_to_z(frame, first, pivot, gates)
                _reduce_to_z(frame, second, pivot, gates)
                mapped = [frame.transform(vector)[1] for vector in vectors]
                if all(_support(vector, width) == [pivot] for vector in mapped):
                    answers.append((frame, gates))
    if not answers:
        raise ValueError("A rank-two non-abelian span had no one-qubit Clifford frame.")
    return answers


def _append_pauli_gadget(
    emitted: NativeCircuit,
    vector: np.ndarray,
    angle: float,
) -> None:
    """Append a standard, immediately uncomputed Pauli-rotation gadget."""
    width = emitted.width
    support = _support(vector, width)
    if not support:
        # exp(-i angle I) is observable only as a global phase, but retaining it is
        # necessary when this artifact is composed with controlled operations.
        emitted.add_global_phase(-angle)
        return

    basis = []
    for qubit in support:
        x, z = vector[qubit], vector[width + qubit]
        if x and z:
            basis.extend([NativeGate("sdg", (qubit,)), NativeGate("h", (qubit,))])
        elif x:
            basis.append(NativeGate("h", (qubit,)))
    pivot = support[-1]
    parity = [NativeGate("cx", (qubit, pivot)) for qubit in support if qubit != pivot]

    for gate in basis + parity:
        emitted.append(gate)
    emitted.append(NativeGate("rz", (pivot,), 2 * angle))
    for gate in reversed(basis + parity):
        emitted.append(gate.inverse())


def _fixed_frame_circuit(
    circuit: Circuit,
    vectors: list[np.ndarray],
    width: int,
    frame: _SignedFrame,
    frame_gates: list[NativeGate],
) -> NativeCircuit:
    emitted = NativeCircuit(width)
    for gate in frame_gates:
        emitted.append(gate)
    for vector, (_, angle) in zip(vectors, circuit.rotations):
        sign, transformed = frame.transform(vector)
        _append_pauli_gadget(emitted, transformed, -angle if sign else angle)
    for gate in reversed(frame_gates):
        emitted.append(gate.inverse())
    return emitted


def _rolling_frame_circuit(
    circuit: Circuit,
    vectors: list[np.ndarray],
    width: int,
    lookahead: int,
    discount_rate: float,
) -> NativeCircuit:
    """Emit arbitrary order exactly while retaining the Clifford between gadgets."""
    frame = _SignedFrame(width)
    frame_gates: list[NativeGate] = []
    emitted = NativeCircuit(width)

    for index, (vector, (_, angle)) in enumerate(zip(vectors, circuit.rotations)):
        _, transformed = frame.transform(vector)
        for qubit in _support(transformed, width):
            x, z = transformed[qubit], transformed[width + qubit]
            if x:
                if z:
                    gate = NativeGate("sdg", (qubit,))
                    _apply_frame_gate(frame, frame_gates, gate)
                    emitted.append(gate)
                gate = NativeGate("h", (qubit,))
                _apply_frame_gate(frame, frame_gates, gate)
                emitted.append(gate)

        sign, transformed = frame.transform(vector)
        support = _support(transformed, width)
        if not support:
            _append_pauli_gadget(
                emitted, transformed, -angle if sign else angle
            )
            continue
        candidates = []
        for pivot in support:
            trial = frame.copy()
            gates = [NativeGate("cx", (qubit, pivot)) for qubit in support if qubit != pivot]
            for gate in gates:
                trial.apply(gate)
            score = 0.0
            for offset, future in enumerate(
                vectors[index + 1 : index + 1 + lookahead]
            ):
                future_weight = len(_support(trial.transform(future)[1], width))
                score += discount_rate**offset * max(future_weight - 1, 0)
            candidates.append((score, pivot, gates))

        _, pivot, parity = min(candidates, key=lambda item: (item[0], item[1]))
        for gate in parity:
            _apply_frame_gate(frame, frame_gates, gate)
            emitted.append(gate)
        sign, reduced = frame.transform(vector)
        expected = np.zeros(2 * width, dtype=np.uint8)
        expected[width + pivot] = 1
        if not np.array_equal(reduced, expected):
            raise ValueError("Rolling Pauli frame failed to expose its Z rotation.")
        emitted.append(NativeGate("rz", (pivot,), 2 * (-angle if sign else angle)))

    for gate in reversed(frame_gates):
        emitted.append(gate.inverse())
    return emitted


def ladder_circuit(circuit: Circuit, width: int) -> NativeCircuit:
    """Lower every Pauli rotation to an independently uncomputed native gadget.

    This is the concrete dependency-free fallback: it preserves input order and
    absolute global phase, and makes no shared-frame optimization assumption.
    """
    try:
        parsed_width = integer_index(width)
    except TypeError as exc:
        raise ValueError("A circuit width must be a non-negative integer.") from exc
    if isinstance(width, (bool, np.bool_)) or parsed_width < 0:
        raise ValueError("A circuit width must be a non-negative integer.")

    vectors, _ = _unique_vectors(circuit, parsed_width)
    emitted = NativeCircuit(parsed_width)
    for vector, (_, angle) in zip(vectors, circuit.rotations):
        _append_pauli_gadget(emitted, vector, angle)
    return emitted


def native_frame_candidates(
    circuit: Circuit,
    width: int,
    *,
    lookahead: int = 8,
    discount_rate: float = 0.9,
) -> list[tuple[str, NativeCircuit]]:
    """Construct named exact frame alternatives without choosing by gate cost.

    The deterministic portfolio contains at most two abelian fixed frames, at
    most ``6 * width`` single-logical-qubit fixed frames, or one rolling frame
    for a general span. Fixed alternatives are named ``fixed-0``, ``fixed-1``,
    etc.; the rolling alternative is named ``rolling``. An empty input returns
    one ``empty`` circuit. Independent Pauli ladders are not included here.

    Every alternative preserves input rotation order, absolute Rz angles
    (``2 * abs(theta)``), and scalar phase. Signed Clifford conjugation may
    change rotation signs and target wires. No rotation merging, approximation,
    or gate-cost preselection is performed. ``lookahead`` and ``discount_rate``
    configure only the rolling heuristic; callers may compare a bounded set of
    their own settings after concrete downstream compilation.
    """
    if lookahead < 0:
        raise ValueError("Native-frame lookahead cannot be negative.")
    if not 0 <= discount_rate <= 1:
        raise ValueError("Native-frame discount_rate must lie in [0, 1].")
    if width < 0:
        raise ValueError("A circuit width cannot be negative.")

    vectors, unique = _unique_vectors(circuit, width)
    if not vectors:
        return [("empty", NativeCircuit(width))]
    profile = _frame_profile_from_vectors(unique, width)
    if profile.is_abelian:
        frames = _commuting_frames(unique, width)
    elif profile.single_qubit_dla:
        frames = _single_qubit_frames(unique, width)
    else:
        return [(
            "rolling",
            _rolling_frame_circuit(circuit, vectors, width, lookahead, discount_rate),
        )]
    return [
        (f"fixed-{index}", _fixed_frame_circuit(circuit, vectors, width, frame, gates))
        for index, (frame, gates) in enumerate(frames)
    ]


def native_frame_circuit(
    circuit: Circuit,
    width: int,
    *,
    lookahead: int = 8,
    discount_rate: float = 0.9,
) -> NativeCircuit:
    """Return the cheapest native Clifford-frame emission constructed here.

    The input order is never changed. Every candidate is a concrete exact circuit;
    the minimum is therefore safe even when the structural heuristic predicts badly.
    Cost selection remains CX first, then total gates, with stable ties.
    """
    candidates = native_frame_candidates(
        circuit, width, lookahead=lookahead, discount_rate=discount_rate,
    )
    return min(
        (emitted for _, emitted in candidates),
        key=lambda item: (item.two_qubit_gates, len(item.gates)),
    )
