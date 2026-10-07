"""Optional OpenFermion synthesis of full Jordan--Wigner Gaussian unitaries.

Eligibility is a property of the Pauli words, not a named spin model. Quadratic
tensors and basis validation use polynomial-size matrices; no full Hilbert-space
matrix is formed. OpenFermion owns diagonalization and Bogoliubov gate synthesis.
Numerical reconstruction checks are guards, not formal error certificates.
"""

from collections.abc import Mapping

import numpy as np
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.hamiltonian import Circuit, terms_of

_ROUTE = "exact-gaussian"
_EVOLUTION_RESIDUAL_LIMIT = 1e-8


def _word_pair(word: str) -> tuple[int, int, float] | None:
    """Return ``a,b,s`` for P=s*i*gamma_a*gamma_b, or None for identity.

    Majoranas are interleaved X/Y, with their Jordan--Wigner Z prefixes.
    Reject every nonquadratic word, without magnitude-based truncation.
    """
    support = [index for index, letter in enumerate(word) if letter != "I"]
    if not support:
        return None
    if len(support) == 1 and word[support[0]] == "Z":
        index = support[0]
        return 2 * index, 2 * index + 1, -1.0
    first, last = support[0], support[-1]
    if (
        first == last or word[first] not in "XY" or word[last] not in "XY"
        or any(letter != "Z" for letter in word[first + 1:last])
    ):
        raise ValueError(f"Pauli word {word!r} is not Jordan--Wigner quadratic.")
    # XX -> -i gamma_Y gamma_X; XY -> -i gamma_Y gamma_Y;
    # YX -> +i gamma_X gamma_X; YY -> +i gamma_X gamma_Y.
    return (
        2 * first + (word[first] == "X"),
        2 * last + (word[last] == "Y"),
        -1.0 if word[first] == "X" else 1.0,
    )


def _validated_terms(operator) -> tuple[int, dict[str, float]]:
    pairs = (
        [(coefficient, word) for word, coefficient in operator.items()]
        if isinstance(operator, Mapping) else terms_of(operator)
    )
    if not pairs:
        raise ValueError("A Gaussian Hamiltonian must specify its qubit width.")
    width = len(str(pairs[0][1]))
    if width == 0:
        raise ValueError("A Gaussian Hamiltonian needs at least one mode.")
    combined = {}
    for coefficient, pauli in pairs:
        word = str(pauli)
        if len(word) != width or set(word) - set("IXYZ"):
            raise ValueError("Pauli words must have equal width and use only IXYZ.")
        value = complex(coefficient)
        if not np.isfinite(value) or value.imag != 0:
            raise ValueError("Gaussian Hamiltonian coefficients must be finite and real.")
        combined[word] = combined.get(word, 0.0) + value.real
        if not np.isfinite(combined[word]):
            raise ValueError("Hamiltonian coefficient accumulation overflowed.")
    combined = {word: value for word, value in combined.items() if value != 0.0}
    for word in combined:
        _word_pair(word)
    return width, combined


def is_gaussian(operator) -> bool:
    """Whether Pauli input is exactly JW-quadratic, without optional imports.

    Includes scalar terms, local Z fields, and X/Y endpoints with the complete
    intervening Z string. A tiny nonzero nonquadratic term is still rejected.
    No Clifford-frame search or interacting-fermion approximation is performed.
    """
    try:
        _validated_terms(operator)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _quadratic_tensors(width: int, terms: dict[str, float]):
    """Exact sparse-Pauli conversion, avoiding reverse-JW expression expansion."""
    matrix = np.zeros((width, width), dtype=complex)
    pairing = np.zeros_like(matrix)
    constant = 0.0
    for word, coefficient in terms.items():
        support = [index for index, letter in enumerate(word) if letter != "I"]
        if not support:
            constant += coefficient
        elif len(support) == 1:
            matrix[support[0], support[0]] -= 2 * coefficient
            constant += coefficient
        else:
            first, last = support[0], support[-1]
            endpoints = word[first] + word[last]
            hopping_factor = {"XX": 1, "YY": 1, "YX": 1j, "XY": -1j}[endpoints]
            pairing_factor = {"XX": 1, "YY": -1, "YX": 1j, "XY": 1j}[endpoints]
            hopping = hopping_factor * coefficient
            pair = pairing_factor * coefficient
            matrix[first, last] += hopping
            matrix[last, first] += hopping.conjugate()
            pairing[first, last] += pair
            pairing[last, first] -= pair
    return matrix, pairing, constant


def _append(circuit: Circuit, width: int, letters: dict[int, str], angle: float):
    if not np.isfinite(angle):
        raise ValueError("Gaussian synthesis produced a nonfinite rotation angle.")
    if angle != 0.0:
        word = "".join(letters.get(index, "I") for index in range(width))
        circuit.add(get_pauli_string(word), float(angle), _ROUTE)


def _basis_circuit(operations, qubits, cirq) -> Circuit:
    """Lower OpenFermion's actual gates, preserving their absolute phase."""
    indices = {qubit: index for index, qubit in enumerate(qubits)}
    width = len(qubits)
    result = Circuit()
    for operation in cirq.flatten_op_tree(operations):
        gate = operation.gate
        wires = tuple(indices[qubit] for qubit in operation.qubits)
        if isinstance(gate, (cirq.ZPowGate, cirq.XPowGate)):
            angle = float(gate.exponent) * np.pi / 2
            axis = "Z" if isinstance(gate, cirq.ZPowGate) else "X"
            if axis == "X" and float(gate.exponent) % 1 != 0.0:
                raise ValueError("OpenFermion emitted a non-Gaussian fractional X gate.")
            _append(result, width, {wires[0]: axis}, angle)
            _append(result, width, {}, -2 * angle * (float(gate.global_shift) + 0.5))
        elif isinstance(gate, cirq.PhasedISwapPowGate) and gate.phase_exponent == 0.25:
            # Ryxxy(theta)=exp[-i theta(YX-XY)/2]. These are circuit-wire
            # Paulis, not Hamiltonian JW strings; OF emits adjacent Givens gates.
            first, second = wires
            if abs(first - second) != 1:
                raise ValueError("OpenFermion emitted a nonadjacent Gaussian Givens gate.")
            angle = float(gate.exponent) * np.pi / 4
            _append(result, width, {first: "Y", second: "X"}, angle)
            _append(result, width, {first: "X", second: "Y"}, -angle)
            _append(result, width, {}, -4 * angle * float(gate.global_shift))
        else:
            raise ValueError(f"Unsupported OpenFermion Gaussian circuit gate: {gate!r}.")
    return result


def _basis_transform(circuit: Circuit, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Independently replay the emitted basis in its 2n-Majorana representation."""
    orthogonal = np.eye(2 * width)
    for pauli, angle in circuit.rotations:
        word = str(pauli)
        support = [index for index, letter in enumerate(word) if letter != "I"]
        if len(support) == 1 and word[support[0]] == "X":
            # Particle-hole X also reverses the JW prefixes on all later modes.
            if int(round(angle / (np.pi / 2))) % 2:
                orthogonal[:, 2 * support[0] + 1:] *= -1
            continue
        pair = _word_pair(word)
        if pair is None:
            continue
        first, second, sign = pair
        cosine, sine = np.cos(2 * angle), sign * np.sin(2 * angle)
        col_first, col_second = orthogonal[:, first].copy(), orthogonal[:, second].copy()
        orthogonal[:, first] = cosine * col_first + sine * col_second
        orthogonal[:, second] = -sine * col_first + cosine * col_second
    # U a† U† expressed as creation/annihilation coefficients (not state action).
    coefficients = (orthogonal[0::2] - 1j * orthogonal[1::2]) / 2
    left = coefficients[:, 0::2] + 1j * coefficients[:, 1::2]
    right = coefficients[:, 0::2] - 1j * coefficients[:, 1::2]
    return left, right


def _validate_tensors(matrix, pairing, constant):
    matrix = np.asarray(matrix, dtype=complex)
    pairing = np.asarray(pairing, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The one-body tensor must be a nonempty square matrix.")
    if pairing.shape != matrix.shape or not np.isfinite(matrix).all() or not np.isfinite(pairing).all():
        raise ValueError("Quadratic tensors must be finite square matrices of equal size.")
    constant = complex(constant)
    if not np.isfinite(constant) or constant.imag != 0.0:
        raise ValueError("The quadratic scalar constant must be finite and real.")
    if not np.array_equal(matrix, matrix.conj().T):
        raise ValueError("The one-body tensor must be Hermitian; inputs are not projected.")
    if not np.array_equal(pairing, -pairing.T):
        raise ValueError("The pairing tensor must be antisymmetric; inputs are not projected.")
    return matrix, pairing, float(constant.real)


def _coefficient_residual(left, right, energies, shift, matrix, pairing) -> float:
    """Coefficient l1 controls the represented Hamiltonian's operator residual."""
    reconstructed_matrix = left.T @ (energies[:, None] * left.conj()) - right.conj().T @ (
        energies[:, None] * right
    )
    creation = left.T @ (energies[:, None] * right.conj())
    reconstructed_pairing = creation - creation.T
    reconstructed_constant = shift + np.sum(energies[:, None] * np.abs(right)**2)
    return float(
        np.sum(np.abs(reconstructed_matrix - matrix))
        + np.sum(np.abs(reconstructed_pairing - pairing))
        + abs(reconstructed_constant)
    )


def _conditioned_basis(transform, qubits, openfermion, cirq) -> Circuit:
    """Exactly change the orbital basis before upstream's elimination.

    Localized/sparse Bogoliubov matrices can trigger upstream near-zero
    shortcuts. A deterministic Fourier basis mixes the orbitals without
    changing the target: [L F†, R Fᵀ] followed by F has action [L, R]. This
    adds a basis circuit; it is a numerical-stability fallback, not a cost
    optimization or a relaxation of the reconstruction guard.
    """
    width = len(qubits)
    indices = np.arange(width)
    fourier = np.exp(2j * np.pi * np.outer(indices, indices) / width) / np.sqrt(width)
    transformed = transform[:, :width] @ fourier.conj().T
    if transform.shape[1] == 2 * width:
        transformed = np.concatenate([transformed, transform[:, width:] @ fourier.T], axis=1)
    basis = _basis_circuit(
        openfermion.bogoliubov_transform(qubits, transformed, initial_state=None), qubits, cirq,
    )
    basis.extend(_basis_circuit(
        openfermion.bogoliubov_transform(qubits, fourier, initial_state=None), qubits, cirq,
    ))
    return basis


def _decompose_tensors(matrix, pairing, constant, time: float, error_tolerance: float) -> Circuit:
    matrix, pairing, constant = _validate_tensors(matrix, pairing, constant)
    time = float(time)
    if not np.isfinite(time):
        raise ValueError("Evolution time must be finite.")
    error_tolerance = float(error_tolerance)
    if not np.isfinite(error_tolerance) or error_tolerance <= 0.0:
        raise ValueError("The Gaussian numerical guard tolerance must be finite and positive.")
    width = len(matrix)
    result = Circuit()
    if time == 0.0:
        return result
    scale = float(max(np.max(np.abs(matrix)), np.max(np.abs(pairing))))
    if scale == 0.0:
        _append(result, width, {}, time * constant)
        return result
    try:
        import cirq
        import openfermion
    except ImportError as exc:
        raise ImportError("Gaussian synthesis requires pip install 'lizzy[gaussian]'.") from exc

    # Normalize small overall energy scales so upstream absolute tolerances do
    # not mistake an entirely weak Hamiltonian for zero. Relative tiny entries
    # can still be omitted upstream; the emitted-basis reconstruction below
    # detects omissions and rejects time-amplified error instead of hiding it.
    source = openfermion.QuadraticHamiltonian(matrix / scale, pairing / scale, constant=0.0)
    energies, transform, shift = source.diagonalizing_bogoliubov_transform()
    energies, shift = np.asarray(energies) * scale, float(shift) * scale
    # First check diagonalization itself: changing the synthesis basis cannot
    # recover input coefficients already discarded by upstream diagonalization.
    left = transform[:, :width]
    right = transform[:, width:] if transform.shape[1] == 2 * width else np.zeros_like(left)
    angle_roundoff = 8 * np.finfo(float).eps * (abs(constant) + np.sum(np.abs(energies)))

    def evolution_residual(left, right):
        return abs(time) * (
            _coefficient_residual(left, right, energies, shift, matrix, pairing) + angle_roundoff
        )

    diagonal_residual = evolution_residual(left, right)
    if not np.isfinite(diagonal_residual) or diagonal_residual > error_tolerance:
        raise ValueError(
            "OpenFermion Gaussian diagonalization exceeds the numerical guard "
            f"({diagonal_residual:.3g} > {error_tolerance:g}); "
            "tiny couplings may have been discarded by upstream tolerances."
        )
    qubits = cirq.LineQubit.range(width)
    basis = _basis_circuit(
        openfermion.bogoliubov_transform(qubits, transform, initial_state=None), qubits, cirq,
    )

    # Verify the Hamiltonian represented by the ACTUAL emitted basis, not just
    # the requested Bogoliubov matrix. This also catches upstream spin-block
    # shortcuts that silently discard small cross-block entries.
    # Coefficient l1 controls an operator-norm residual, without dense matrices.
    # The extra term guards large-time angle roundoff; this is still numerical,
    # not an interval-arithmetic certificate for the complete emitted circuit.
    original_residual = evolution_residual(*_basis_transform(basis, width))
    emitted_residual = original_residual
    if not np.isfinite(emitted_residual) or emitted_residual > error_tolerance:
        basis = _conditioned_basis(transform, qubits, openfermion, cirq)
        emitted_residual = evolution_residual(*_basis_transform(basis, width))
    if not np.isfinite(emitted_residual) or emitted_residual > error_tolerance:
        raise ValueError(
            "OpenFermion Gaussian reconstruction exceeds the numerical guard "
            f"({emitted_residual:.3g} > {error_tolerance:g}; "
            f"unconditioned {original_residual:.3g}); "
            "upstream elimination remains inaccurate after Fourier conditioning."
        )
    for pauli, angle in reversed(basis.rotations):
        result.add(pauli, -angle, _ROUTE)
    for index, energy in enumerate(energies):
        _append(result, width, {index: "Z"}, -time * float(energy) / 2)
    _append(result, width, {}, time * (constant + shift + float(np.sum(energies)) / 2))
    result.extend(basis)
    return result


def decompose(operator, time: float, *, error_tolerance: float = _EVOLUTION_RESIDUAL_LIMIT) -> Circuit:
    """Synthesize full evolution of a JW-quadratic Pauli Hamiltonian.

    Returns logical Pauli rotations with ``exact-gaussian`` provenance, including
    identity rotations for the scalar phase. Native/Clifford+T lowering uses the
    existing Lizzy emitters. There is no particle-number-sector restriction,
    state preparation, or full-unitary matrix decomposition.
    ``error_tolerance`` controls the independent numerical reconstruction guard,
    not a formal error certificate or the later Clifford+T approximation budget.
    """
    width, terms = _validated_terms(operator)
    return _decompose_tensors(*_quadratic_tensors(width, terms), time, error_tolerance)


def from_quadratic(
    operator, time: float, *, error_tolerance: float = _EVOLUTION_RESIDUAL_LIMIT,
) -> Circuit:
    """Synthesize an upstream OpenFermion ``QuadraticHamiltonian`` directly.

    Uses ``combined_hermitian_part``, so chemical potential is included. Finite
    values, exact tensor Hermiticity/antisymmetry and real scalar are required;
    malformed inputs are not silently projected or approximately truncated.
    ``error_tolerance`` has the same numerical-guard meaning as in ``decompose``.
    """
    try:
        import openfermion
    except ImportError as exc:
        raise ImportError("Gaussian synthesis requires pip install 'lizzy[gaussian]'.") from exc
    if not isinstance(operator, openfermion.QuadraticHamiltonian):
        raise TypeError("from_quadratic requires an OpenFermion QuadraticHamiltonian.")
    return _decompose_tensors(
        operator.combined_hermitian_part, operator.antisymmetric_part, operator.constant, time,
        error_tolerance,
    )
