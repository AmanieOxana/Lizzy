"""Thin molecular adapters around OpenFermion and ffsim.

OpenFermion owns the fermionic operator algebra and JW/BK encodings.  ffsim owns
double factorization, number/Z representation conversion, orbital-rotation
decomposition, adjacent-frame merging, and the semantic DF-Trotter gate.  Lizzy's
scope here is deliberately smaller: a validated dependency-light tensor IR,
HamLib/layout bridges, provenance, and explicitly opt-in experimental policies.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real

import numpy as np

from lizzy.fermions._chemistry_extensions import (
    optimized_frame_order,
    prune_double_factorization,
    reorder_double_factorization,
    to_interleaved_jw,
)
from lizzy.hamiltonian import hamiltonian


@dataclass
class MolecularHamiltonian:
    r"""A validated real, spin-restricted Hamiltonian in spatial orbitals.

    The tensor convention matches :class:`ffsim.MolecularHamiltonian`,

    .. math::

        H = \sum_{pq\sigma} h_{pq} a^\dagger_{p\sigma}a_{q\sigma}
          + \frac12\sum_{pqrs\sigma\tau} h_{pqrs}
            a^\dagger_{p\sigma}a^\dagger_{r\tau}
            a_{s\tau}a_{q\sigma} + c.

    Keeping this small façade local lets HamLib data be inspected without importing
    the optional simulation stack.  Numerical validation happens here; actual
    chemistry operations are delegated by the conversion functions below.
    """

    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    constant: float = 0.0

    def __post_init__(self) -> None:
        one = _real_array(self.one_body_tensor, "one_body_tensor")
        two = _real_array(self.two_body_tensor, "two_body_tensor")
        if one.ndim != 2 or one.shape[0] != one.shape[1]:
            raise ValueError("one_body_tensor must be a square matrix")
        if one.shape[0] == 0:
            raise ValueError("a molecular Hamiltonian needs at least one orbital")
        expected = (one.shape[0],) * 4
        if two.shape != expected:
            raise ValueError(
                f"two_body_tensor must have shape {expected}, got {two.shape}"
            )
        if not np.allclose(one, one.T, atol=1e-10, rtol=1e-10):
            raise ValueError("one_body_tensor must be real symmetric")
        if not (
            np.allclose(two, two.transpose(1, 0, 2, 3), atol=1e-10, rtol=1e-10)
            and np.allclose(two, two.transpose(0, 1, 3, 2), atol=1e-10, rtol=1e-10)
            and np.allclose(two, two.transpose(2, 3, 0, 1), atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("two_body_tensor must have real chemist symmetries")
        constant = np.asarray(self.constant)
        if constant.ndim or not np.isfinite(constant):
            raise ValueError("constant must be a finite scalar")
        if np.iscomplexobj(constant) and abs(constant.imag) > 1e-10:
            raise ValueError("constant must be real")
        self.one_body_tensor = one
        self.two_body_tensor = two
        self.constant = float(constant.real)

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @property
    def norb(self) -> int:
        """ffsim-compatible alias for :attr:`n_orbitals`."""
        return self.n_orbitals

    @property
    def n_qubits(self) -> int:
        """Number of spin orbitals/qubits."""
        return 2 * self.n_orbitals


@dataclass
class DoubleFactorizedResult:
    """Concrete Qiskit artifact emitted from an ffsim DF-Trotter gate.

    Tensor reconstruction error is not a bound on evolution error.  In particular,
    finite-step Trotter error, Givens truncation, optional factor reordering, and
    compiler numerics remain outside it.  The explicit certification fields below
    prevent a gate-count improvement from being mistaken for an accuracy guarantee.
    """

    circuit: object
    _quoted_two_qubit_gates: int = field(repr=False)
    factors: int
    factorization_rank: int
    steps: int
    order: int
    tensor_error: float
    tensor_tolerance: float
    coulomb_cutoff: float
    givens_tolerance: float
    frame_ordered: bool
    ordering: tuple[int, ...]
    preoptimization_two_qubit_gates: int
    optimization_level: int
    candidate_counts: dict[str, int] = field(default_factory=dict)
    backend: str = "ffsim-double-factorized"
    emitter: str = "qiskit"
    qubit_order: str = "alpha-then-beta"
    routing_mode: str = "input"
    routing_attempted: bool = False
    factor_order_preserved: bool = True
    approximation_sources: tuple[str, ...] = ()
    max_vecs: int | None = None
    factorization_optimized: bool = False
    cholesky: bool = True
    factorization_fingerprint: str = ""
    ffsim_version: str = "unknown"
    qiskit_version: str = "unknown"
    accuracy_certified: bool = False
    unitary_error_bound: float | None = None
    trotter_error_bound: float | None = None
    error_guaranteed: bool = False

    @property
    def emitted_circuit(self) -> object:
        """The Qiskit circuit whose CX gates were quoted."""
        return self.circuit

    @property
    def two_qubit_gates(self) -> int:
        """Current CX count of the retained, mutable Qiskit circuit."""
        if hasattr(self.circuit, "count_ops"):
            return int(self.circuit.count_ops().get("cx", 0))
        return self._quoted_two_qubit_gates

    @property
    def quoted_two_qubit_gates(self) -> int:
        """CX-count snapshot taken when the result was constructed."""
        return self._quoted_two_qubit_gates

    @property
    def two_body_tensor_max_abs_error(self) -> float:
        """Explicit name for the diagnostic retained as ``tensor_error``."""
        return self.tensor_error

    @property
    def retained_factor_count(self) -> int:
        """Number of factors after Lizzy's optional Coulomb pruning."""
        return self.factors

    @property
    def unpruned_factor_count(self) -> int:
        """Number of factors returned by ffsim before optional pruning."""
        return self.factorization_rank


def to_ffsim(molecular):
    """Return an upstream :class:`ffsim.MolecularHamiltonian`.

    Existing ffsim objects pass through unchanged.  Lizzy's local façade is copied
    into ffsim's representation without changing tensor order or spin convention.
    """
    ffsim = _require_ffsim()
    if isinstance(molecular, ffsim.MolecularHamiltonian):
        return molecular
    if not isinstance(molecular, MolecularHamiltonian):
        raise TypeError(
            "molecular must be lizzy.fermions.chemistry.MolecularHamiltonian or "
            "ffsim.MolecularHamiltonian"
        )
    return ffsim.MolecularHamiltonian(
        molecular.one_body_tensor,
        molecular.two_body_tensor,
        molecular.constant,
    )


def molecular_from_openfermion_ffsim(
    operator,
    *,
    n_orbitals: int | None = None,
    qubit_order: str = "interleaved",
) -> MolecularHamiltonian:
    """Recover spatial tensors using ffsim's raw FermionOperator parser.

    OpenFermion supplies the parsed symbolic operator.  This function performs only
    the mode-layout translation into ffsim's spin-labelled actions; ffsim's
    ``MolecularHamiltonian.from_fermion_operator`` owns extraction and
    symmetrization of the molecular tensors.  No normal ordering is applied.
    """
    _validate_qubit_order(qubit_order)
    ffsim = _require_ffsim()
    try:
        terms = operator.terms
    except AttributeError as exc:
        raise TypeError("operator must be an OpenFermion FermionOperator") from exc

    occupied_modes = [mode for term in terms for mode, _ in term]
    inferred_modes = 1 + max(occupied_modes, default=-1)
    if n_orbitals is None:
        if inferred_modes <= 0 or inferred_modes % 2:
            raise ValueError(
                "cannot infer an even spin-orbital count; pass n_orbitals explicitly"
            )
        n_orbitals = inferred_modes // 2
    if (
        not isinstance(n_orbitals, Integral)
        or isinstance(n_orbitals, bool)
        or n_orbitals < 1
    ):
        raise ValueError("n_orbitals must be a positive integer")
    n_orbitals = int(n_orbitals)
    if inferred_modes > 2 * n_orbitals:
        raise ValueError("operator contains a mode outside n_orbitals")

    converted = {}
    for term, coefficient in terms.items():
        actions = []
        for mode, action in term:
            orbital, spin = _mode_to_spin_orbital(mode, n_orbitals, qubit_order)
            if action not in (0, 1):
                raise ValueError(f"invalid fermionic ladder action {action!r}")
            if spin == 0:
                factory = ffsim.cre_a if action else ffsim.des_a
            else:
                factory = ffsim.cre_b if action else ffsim.des_b
            actions.append(factory(orbital))
        converted[tuple(actions)] = coefficient

    if occupied_modes:
        parsed = ffsim.MolecularHamiltonian.from_fermion_operator(
            ffsim.FermionOperator(converted)
        )
        one = np.zeros((n_orbitals, n_orbitals), dtype=complex)
        two = np.zeros((n_orbitals,) * 4, dtype=complex)
        one[: parsed.norb, : parsed.norb] = parsed.one_body_tensor
        two[
            : parsed.norb,
            : parsed.norb,
            : parsed.norb,
            : parsed.norb,
        ] = parsed.two_body_tensor
        constant = parsed.constant
    else:
        one = np.zeros((n_orbitals, n_orbitals))
        two = np.zeros((n_orbitals,) * 4)
        constant = complex(terms.get((), 0.0))
    return MolecularHamiltonian(one, two, constant)


def interaction_operator_openfermion(molecular: MolecularHamiltonian):
    """Build OpenFermion's canonical spin-orbital ``InteractionOperator``.

    The returned tensor uses HamLib's interleaved alpha/beta mode order.  OpenFermion
    owns expansion into a symbolic ``FermionOperator``; the only local work is the
    documented ffsim-spatial to OpenFermion-spin-orbital tensor-axis bridge.
    """
    openfermion = _require_openfermion()
    n = molecular.n_orbitals
    modes = 2 * n
    alpha = np.arange(0, modes, 2)
    beta = np.arange(1, modes, 2)
    one = np.zeros((modes, modes), dtype=float)
    one[np.ix_(alpha, alpha)] = molecular.one_body_tensor
    one[np.ix_(beta, beta)] = molecular.one_body_tensor

    # OpenFermion stores coefficients in creation/creation/annihilation/
    # annihilation order (p, r, s, q).  ffsim stores h[p, q, r, s] and places a
    # factor 1/2 in the Hamiltonian definition.
    kernel = 0.5 * molecular.two_body_tensor.transpose(0, 2, 3, 1)
    two = np.zeros((modes,) * 4, dtype=float)
    for first_spin in (alpha, beta):
        for second_spin in (alpha, beta):
            two[np.ix_(first_spin, second_spin, second_spin, first_spin)] = kernel
    return openfermion.InteractionOperator(molecular.constant, one, two)


def fermion_operator_openfermion(
    molecular: MolecularHamiltonian,
    qubit_order: str = "interleaved",
):
    """Convert molecular tensors with OpenFermion's public operator APIs."""
    _validate_qubit_order(qubit_order)
    openfermion = _require_openfermion()
    operator = openfermion.get_fermion_operator(
        interaction_operator_openfermion(molecular)
    )
    if qubit_order == "alpha-then-beta":
        operator = openfermion.reorder(
            operator,
            openfermion.up_then_down,
            num_modes=molecular.n_qubits,
        )
    return operator


def fermion_operator(
    molecular: MolecularHamiltonian,
    qubit_order: str = "interleaved",
):
    """Compatibility alias for :func:`fermion_operator_openfermion`."""
    return fermion_operator_openfermion(molecular, qubit_order=qubit_order)


def to_pauli_openfermion(
    molecular: MolecularHamiltonian,
    encoding: str = "jw",
    qubit_order: str = "interleaved",
):
    """Encode the molecular operator with OpenFermion's JW or BK mapping."""
    openfermion = _require_openfermion()
    source = fermion_operator_openfermion(molecular, qubit_order=qubit_order)
    normalized = encoding.lower().replace("-", "_")
    if normalized in {"jw", "jordan_wigner"}:
        encoded = openfermion.jordan_wigner(source)
    elif normalized in {"bk", "bravyi_kitaev"}:
        encoded = openfermion.bravyi_kitaev(source, n_qubits=molecular.n_qubits)
    else:
        raise ValueError(f"Unknown fermion-to-qubit encoding {encoding!r}")

    terms: list[tuple[str, float]] = []
    for term, coefficient in sorted(encoded.terms.items()):
        if not term:
            continue
        if abs(coefficient.imag) > 1e-10:
            raise ValueError(
                "fermion-to-qubit transform produced a complex coefficient"
            )
        letters = ["I"] * molecular.n_qubits
        for qubit, letter in term:
            letters[qubit] = letter
        terms.append(("".join(letters), float(coefficient.real)))
    return hamiltonian(terms)


def to_pauli(
    molecular: MolecularHamiltonian,
    encoding: str = "jw",
    qubit_order: str = "interleaved",
):
    """Compatibility alias for :func:`to_pauli_openfermion`."""
    return to_pauli_openfermion(molecular, encoding=encoding, qubit_order=qubit_order)


def factorize_molecular_ffsim(
    molecular,
    *,
    tol: float = 1e-8,
    max_vecs: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: Mapping | None = None,
    diag_coulomb_indices: Sequence[tuple[int, int]] | None = None,
    cholesky: bool = True,
    z_representation: bool = False,
):
    """Delegate molecular double factorization directly to ffsim.

    The arguments mirror
    ``ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian``.  Keeping this
    boundary public makes it possible to inspect or reuse the upstream factorization
    without compiling a circuit or invoking a Lizzy routing policy.
    """
    tol = _nonnegative_real_scalar(tol, "tol")
    max_vecs = _optional_positive_integer(max_vecs, "max_vecs")
    if not isinstance(optimize, bool):
        raise ValueError("optimize must be a bool")
    if not isinstance(cholesky, bool):
        raise ValueError("cholesky must be a bool")
    if not isinstance(z_representation, bool):
        raise ValueError("z_representation must be a bool")
    ffsim = _require_ffsim()
    return ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        to_ffsim(molecular),
        z_representation=z_representation,
        tol=tol,
        max_vecs=max_vecs,
        optimize=optimize,
        method=method,
        callback=callback,
        options=None if options is None else dict(options),
        diag_coulomb_indices=(
            None if diag_coulomb_indices is None else list(diag_coulomb_indices)
        ),
        cholesky=cholesky,
    )


def synthesize_molecular_ffsim(
    molecular,
    time: float,
    *,
    steps: int = 1,
    formula_order: int = 2,
    tensor_tolerance: float = 1e-8,
    max_vecs: int | None = None,
    factorization_optimize: bool = False,
    cholesky: bool = True,
    coulomb_cutoff: float = 0.0,
    givens_tolerance: float = 1e-10,
    frame_ordering: str | Sequence[int] = "input",
    qubit_order: str = "alpha-then-beta",
    optimization_level: int = 1,
) -> DoubleFactorizedResult:
    r"""Compile molecular evolution through ffsim's DF-Trotter implementation.

    The conservative default preserves the factor order returned by ffsim.  The
    string ``"experimental-givens"`` opts into Lizzy's local Givens-distance
    nearest-neighbour/2-opt candidate; the original ffsim order is still compiled
    and wins ties.  Reordering changes the finite-step Trotter approximant, so that
    mode is never accuracy-certified.  The former name ``"portfolio"`` remains as a
    deprecated alias for the same experiment.

    ``formula_order`` uses the conventional physical names: 1 for Lie--Trotter and
    positive even integers for Suzuki formulas.  They map to ffsim's ``order=0`` and
    ``order=formula_order/2`` respectively.  The experimental ordering objective is
    defined only for S2; input and explicit orders can use every supported formula.

    ffsim owns factorization, number-to-Z conversion, Givens decomposition, adjacent
    orbital-frame merging, and the semantic ``SimulateTrotterDoubleFactorizedJW``
    gate.  Qiskit's preset pass manager is initialized with ``ffsim.qiskit.PRE_INIT``
    rather than relying on a fixed recursive ``decompose`` depth.
    """
    if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    ffsim_order = _ffsim_formula_order(formula_order)
    time = _finite_real_scalar(time, "time")
    tensor_tolerance = _nonnegative_real_scalar(tensor_tolerance, "tensor_tolerance")
    max_vecs = _optional_positive_integer(max_vecs, "max_vecs")
    if not isinstance(factorization_optimize, bool):
        raise ValueError("factorization_optimize must be a bool")
    if not isinstance(cholesky, bool):
        raise ValueError("cholesky must be a bool")
    coulomb_cutoff = _nonnegative_real_scalar(coulomb_cutoff, "coulomb_cutoff")
    givens_tolerance = _nonnegative_real_scalar(givens_tolerance, "givens_tolerance")
    _validate_qubit_order(qubit_order)
    if (
        not isinstance(optimization_level, Integral)
        or isinstance(optimization_level, bool)
        or optimization_level not in range(4)
    ):
        raise ValueError("optimization_level must be between 0 and 3")

    requested_mode = frame_ordering
    if isinstance(frame_ordering, str):
        if frame_ordering == "portfolio":
            warnings.warn(
                "frame_ordering='portfolio' is deprecated; use the explicit "
                "accuracy-sensitive name 'experimental-givens'",
                DeprecationWarning,
                stacklevel=2,
            )
            requested_mode = "experimental-givens"
        if requested_mode not in {"input", "experimental-givens"}:
            raise ValueError(
                "frame_ordering must be 'input', 'experimental-givens', "
                "or a permutation"
            )
        if requested_mode == "experimental-givens" and formula_order != 2:
            raise ValueError(
                "experimental-givens is currently defined only for formula_order=2"
            )

    ffsim, QuantumCircuit, generate_preset_pass_manager, EvolutionGate = (
        _require_ffsim_qiskit()
    )
    source = to_ffsim(molecular)
    number_form = factorize_molecular_ffsim(
        source,
        tol=tensor_tolerance,
        max_vecs=max_vecs,
        optimize=factorization_optimize,
        cholesky=cholesky,
        z_representation=False,
    )
    if coulomb_cutoff == 0.0:
        # The default is the unmodified ffsim baseline.  Do not even round-trip its
        # factors through Lizzy's optional post-processing policy.
        factorized = number_form.to_z_representation()
        reconstructed = number_form.to_molecular_hamiltonian()
    else:
        factorized, reconstructed = prune_double_factorization(
            source, number_form, coulomb_cutoff
        )
    matrices = factorized.diag_coulomb_mats
    tensor_error = float(
        np.max(np.abs(reconstructed.two_body_tensor - source.two_body_tensor))
    )

    original = tuple(range(len(matrices)))
    routing_attempted = False
    if isinstance(requested_mode, str):
        orders = [("original", original)]
        routing_mode = requested_mode
        if requested_mode == "experimental-givens" and len(matrices) > 1:
            routing_attempted = True
            optimized = optimized_frame_order(
                ffsim,
                factorized,
                time / steps,
                givens_tolerance,
                steps,
            )
            if optimized != original:
                orders.append(("experimental-givens", optimized))
    else:
        explicit = tuple(requested_mode)
        if any(
            not isinstance(index, Integral) or isinstance(index, bool)
            for index in explicit
        ) or sorted(explicit) != list(original):
            raise ValueError("explicit frame_ordering must permute every factor once")
        orders = [("explicit", explicit)]
        routing_mode = "explicit-permutation"

    compiled = []
    for label, ordering in orders:
        arranged = reorder_double_factorization(factorized, ordering)
        raw = QuantumCircuit(2 * source.norb)
        raw.append(
            EvolutionGate(
                arranged,
                time,
                n_steps=steps,
                order=ffsim_order,
                tol=givens_tolerance,
            ),
            range(2 * source.norb),
        )
        if qubit_order == "interleaved":
            raw = to_interleaved_jw(raw, source.norb, QuantumCircuit)

        level_zero = _transpile_ffsim(
            raw,
            ffsim,
            generate_preset_pass_manager,
            optimization_level=0,
        )
        pre = int(level_zero.count_ops().get("cx", 0))
        emitted = (
            level_zero
            if optimization_level == 0
            else _transpile_ffsim(
                raw,
                ffsim,
                generate_preset_pass_manager,
                optimization_level=optimization_level,
            )
        )
        count = int(emitted.count_ops().get("cx", 0))
        compiled.append((count, label, ordering, pre, emitted))

    count, _label, ordering, pre, emitted = min(compiled, key=lambda item: item[0])
    factor_order_preserved = ordering == original
    approximation_sources = ["finite-step-trotter", "double-factorization"]
    if coulomb_cutoff:
        approximation_sources.append("coulomb-cutoff")
    if givens_tolerance:
        approximation_sources.append("givens-truncation")
    if not factor_order_preserved:
        approximation_sources.append("fragment-reordering")

    return DoubleFactorizedResult(
        circuit=emitted,
        _quoted_two_qubit_gates=count,
        factors=len(matrices),
        factorization_rank=len(number_form.diag_coulomb_mats),
        steps=int(steps),
        order=int(formula_order),
        tensor_error=tensor_error,
        tensor_tolerance=tensor_tolerance,
        coulomb_cutoff=coulomb_cutoff,
        givens_tolerance=givens_tolerance,
        frame_ordered=not factor_order_preserved,
        ordering=ordering,
        preoptimization_two_qubit_gates=pre,
        optimization_level=int(optimization_level),
        candidate_counts={candidate[1]: candidate[0] for candidate in compiled},
        qubit_order=qubit_order,
        routing_mode=routing_mode,
        routing_attempted=routing_attempted,
        factor_order_preserved=factor_order_preserved,
        approximation_sources=tuple(approximation_sources),
        max_vecs=max_vecs,
        factorization_optimized=factorization_optimize,
        cholesky=cholesky,
        factorization_fingerprint=_factorization_fingerprint(factorized),
        ffsim_version=_package_version("ffsim"),
        qiskit_version=_package_version("qiskit"),
    )


def synthesize_molecular(molecular, time: float, **kwargs) -> DoubleFactorizedResult:
    """Compatibility alias for :func:`synthesize_molecular_ffsim`."""
    return synthesize_molecular_ffsim(molecular, time, **kwargs)


def _transpile_ffsim(
    circuit,
    ffsim,
    generate_preset_pass_manager,
    *,
    optimization_level: int,
):
    pass_manager = generate_preset_pass_manager(
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=optimization_level,
        seed_transpiler=0,
    )
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    return pass_manager.run(circuit)


def _ffsim_formula_order(formula_order: int) -> int:
    if (
        not isinstance(formula_order, Integral)
        or isinstance(formula_order, bool)
        or formula_order < 1
        or (formula_order != 1 and formula_order % 2)
    ):
        raise ValueError(
            "formula_order must be 1 or a positive even integer (2, 4, ...)"
        )
    return 0 if formula_order == 1 else int(formula_order) // 2


def _mode_to_spin_orbital(
    mode: int,
    n_orbitals: int,
    qubit_order: str,
) -> tuple[int, int]:
    if not isinstance(mode, Integral) or isinstance(mode, bool) or mode < 0:
        raise ValueError(f"invalid fermionic mode index {mode!r}")
    if mode >= 2 * n_orbitals:
        raise ValueError("operator contains a mode outside n_orbitals")
    if qubit_order == "interleaved":
        return mode // 2, mode % 2
    return mode % n_orbitals, mode // n_orbitals


def _validate_qubit_order(qubit_order: str) -> None:
    if qubit_order not in {"interleaved", "alpha-then-beta"}:
        raise ValueError("qubit_order must be 'interleaved' or 'alpha-then-beta'")


def _real_array(value, name: str) -> np.ndarray:
    """Copy an array as float after rejecting unsupported complex data."""
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.iscomplexobj(array) and np.max(np.abs(array.imag), initial=0.0) > 1e-10:
        raise ValueError(f"{name} must be real")
    return np.array(array.real, dtype=float, copy=True)


def _finite_real_scalar(value, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _nonnegative_real_scalar(value, name: str) -> float:
    scalar = _finite_real_scalar(value, name)
    if scalar < 0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:  # pragma: no cover - import already guards install
        return "unknown"


def _factorization_fingerprint(factorized) -> str:
    """Identify the exact upstream factor frames used for this circuit."""
    digest = sha256()
    for value in (
        factorized.one_body_tensor,
        factorized.diag_coulomb_mats,
        factorized.orbital_rotations,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    digest.update(np.asarray(factorized.constant).tobytes())
    digest.update(str(factorized.z_representation).encode("ascii"))
    return digest.hexdigest()


def _require_openfermion():
    try:
        import openfermion
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Molecular conversion needs openfermion: pip install 'lizzy[chemistry]'."
        ) from exc
    return openfermion


def _require_ffsim():
    try:
        import ffsim
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Molecular tensors and factorization need ffsim: "
            "pip install 'lizzy[chemistry]'."
        ) from exc
    return ffsim


def _require_ffsim_qiskit():
    try:
        import ffsim
        from ffsim.qiskit import SimulateTrotterDoubleFactorizedJW
        from qiskit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Double-factorized synthesis needs ffsim and qiskit: "
            "pip install 'lizzy[chemistry]'."
        ) from exc
    return (
        ffsim,
        QuantumCircuit,
        generate_preset_pass_manager,
        SimulateTrotterDoubleFactorizedJW,
    )
