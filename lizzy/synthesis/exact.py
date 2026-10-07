"""Fixed-depth synthesis for supported Pauli so(m) presentations.

Givens eliminates a group endpoint in a low-weight plane ordering. Paper BDI
uses an independent graph-derived mapping: horizontal inputs have reusable
K exp(-it A) K† factors; general inputs use recursive endpoint decomposition.
Only horizontal BDI preserves the spin-cover phase. Neither route is a generic
decomposition theorem for every small DLA.

``free_part`` extracts a supported subset for an exact layer inside a product
formula. Its internal splitting error disappears, but cross-layer error remains.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from kak_tools import (
    labelled_matrix_basis,
    map_dla_to_irrep,
    pauli_word_to_string,
    recursive_bdi,
)
from kak_tools._horizontal_bdi import horizontal_generator_decomposition
from kak_tools.map_to_irrep import HorizontalEmbeddingError
from paulie.classifier.classification import TypeGraph
from paulie.common.pauli_string_factory import get_pauli_string
from paulie.common.pauli_string_linear import PauliStringLinear
from scipy.linalg import expm

from lizzy.algebra._orthogonal_mapping import map_orthogonal
from lizzy.algebra.classify import classify, is_fast_forwardable, summands
from lizzy.hamiltonian import (
    Circuit,
    anticommutation_matrix,
    hamiltonian,
    n_qubits,
    terms_of,
)

# Irrep data by generator set; the split-step route decomposes the same free part once
# per Trotter step, and the mapping (with its Lie closure) is the expensive part.
_irrep_cache: dict = {}


def is_decomposable(hamiltonian_: PauliStringLinear) -> bool:
    """
    Check eligibility for the default Givens route in one piece.

    Two conditions: the algebra must be small enough to be worth decomposing, and it
    must have an ``so(m)`` presentation, which is what the Pauli-word pipeline of
    ``kak_tools`` implements.

    Parts with more than ``max(64, 8n)`` terms are declined without classifying them, on the
    same budget reasoning as :func:`free_part`: dense classification can be
    expensive. This is a search cutoff for the targeted sparse model families,
    not a necessary condition for a Hamiltonian to have a small DLA.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        bool: True if its classification is eligible for an exact attempt. The
        upstream representation mapping can still fail for a generator set.
    """
    width = n_qubits(hamiltonian_)
    if len(terms_of(hamiltonian_)) > max(64, 8 * width):
        return False
    classification = classify(hamiltonian_)
    if not (
        is_fast_forwardable(classification, width)
        and classification.get_orthogonal_size() is not None
    ):
        return False
    # Only canonical graphs of type A (a genuine so line) or NONE (a lone u(1))
    # are accepted. The low-rank coincidences -- su(2) as so(3), sp(2) as so(5) --
    # also present an orthogonal size, but their irrep matching is not the line-graph
    # embedding, and the subgraph search behind it can run for hours on the junk
    # sets the free-part greedy otherwise assembles out of them. Explicit BDI uses
    # its own verified generic mapper and does not apply this legacy filter.
    return all(
        morph.get_type() in (TypeGraph.A, TypeGraph.NONE)
        for morph in classification.get_morphs()
    )


def decomposes_by_summand(hamiltonian_: PauliStringLinear) -> bool:
    """
    Check whether the exact branch can handle a Hamiltonian after splitting it.

    Summands commute, so a Hamiltonian is exactly compilable as soon as each of its
    summands is -- which is a strictly weaker condition than :func:`is_decomposable`.
    The XY model is the standard example: as a whole it is ``2*so(n)``, which has no
    ``so(m)`` presentation, while each summand is ``so(n)`` and decomposes.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        bool: True if every summand is eligible for an exact attempt.
    """
    return all(is_decomposable(part) for part in summands(hamiltonian_))


def _irrep(words: tuple[str, ...], width: int) -> tuple:
    """Map a generator set into its so(m) irrep, cached.

    Returns the labelled basis for the generators (to build the irrep Hamiltonian), a
    per-plane table of qubit words and spinor scales, an index ordering that makes
    adjacent planes carry the cheapest words available, and the irrep size ``m``.
    Prefer the existing horizontal mapping so successful routes keep their plane
    ordering. Givens also accepts nonhorizontal Hamiltonians: use the verified
    full-algebra mapping when no horizontal embedding exists.
    """
    if words in _irrep_cache:
        return _irrep_cache[words]

    try:
        mapping, signs, info = map_dla_to_irrep(list(words))
        m = info.get_orthogonal_size()
    except ValueError as exc:
        if not isinstance(exc.__cause__, HorizontalEmbeddingError):
            raise
        # The endpoint elimination only needs a complete so(m) representation,
        # not H in a single BDI horizontal space. Keep mapping budgets and Lie
        # verification in the shared mapper; this does not run the BDI algorithm.
        mapping, signs, m, _ = map_orthogonal(words, width)
        info = classify(hamiltonian({word: 1.0 for word in words}))
    basis = labelled_matrix_basis(mapping, signs, info)

    generators = {
        str(pauli_word_to_string(word, width)): basis[word] for word in basis
    }
    planes = {
        pair: (pauli_word_to_string(word, width), 2.0 * signs[pair])
        for pair, word in mapping.items()
    }

    # An adjacent-plane rotation costs what its Pauli word weighs, and the mapping
    # scatters the cheap words over arbitrary index pairs. Relabelling the basis is
    # free, so order the indices so that adjacent ones are joined by low-weight
    # planes wherever possible. The weight-two planes are the Majorana line of the
    # model -- for a chain they form exactly a path over all m indices -- so the
    # ordering walks the connected paths of that subgraph and concatenates them,
    # paying an expensive plane only at the seams between components.
    cheap: dict[int, list[int]] = {index: [] for index in range(m)}
    for (i, j), (word, _) in planes.items():
        if sum(1 for letter in str(word) if letter != "I") <= 2:
            cheap[i].append(j)
            cheap[j].append(i)

    order = []
    visited: set[int] = set()
    for start in sorted(range(m), key=lambda k: (len(cheap[k]), k)):
        if start in visited:
            continue
        node = start
        while node is not None:
            order.append(node)
            visited.add(node)
            node = next((k for k in cheap[node] if k not in visited), None)

    _irrep_cache[words] = (generators, planes, order, m)
    return _irrep_cache[words]


def _bdi_planes(matrix: np.ndarray) -> Iterator[tuple[int, int, float]]:
    """Yield application-ordered SO(2) planes from the upstream BDI factor tree.

    This is the balanced CS recursion of arXiv:2503.19014, Appendix F.4,
    not the horizontal Hamiltonian variant. Upstream factors multiply left to
    right, while Circuit stores application order. Odd blocks pair axis i with
    i + ceil(block_size/2), leaving the middle axis unpaired (Appendix F.5).
    The recursion and determinant corrections belong to kak-tools; this adapter
    only reads its terminal matrices/angles, including exact half-turns.
    """
    if len(matrix) <= 1:
        return
    factors = recursive_bdi(
        matrix, len(matrix), first_is_horizontal=False, validate=True,
    )
    for factor, start, end, kind in reversed(factors):
        if kind.startswith("a"):
            q = (end - start + 1) // 2
            for index, angle in enumerate(factor):
                yield start + index, start + q + index, float(angle)
        else:
            if end - start != 2:
                raise ValueError("recursive BDI returned a nonterminal orthogonal block")
            yield start, start + 1, float(np.arctan2(factor[0, 1], factor[0, 0]))


@dataclass(frozen=True)
class BDIOptimization:
    """Bounded nullspace-gauge search, with heuristic costs for both K wings.

    Estimates are not compiled T counts or upper bounds. The unchanged Cartan
    rotations are excluded. ``eligible`` requires a structural nullspace of
    dimension at least two; no extra freedom from zero/repeated rates is assumed.
    """

    eligible: bool
    rotation_error: float
    candidates: int
    original_t_estimate: int
    selected_t_estimate: int
    original_nonclifford: int
    selected_nonclifford: int
    optimized: bool


@dataclass(frozen=True)
class BDIPlan:
    """Reusable algebra-derived BDI synthesis, with no model-specific templates.

    Horizontal inputs use the paper's VI.2 construction: only unwrapped Cartan
    rates depend on time; recursive BDI compiles a fixed K, whose physical inverse
    is emitted exactly. General inputs use the VI.1 endpoint decomposition and
    retain its global-phase ambiguity. ``phase_preserving`` distinguishes them.
    The private payload is immutable so cached plans cannot share mutable circuits.
    """

    irrep_size: int
    partition: tuple[int, int]
    mapping_kind: str
    parameter_bound: int
    _planes: tuple[tuple[int, int, str, float], ...] = field(repr=False)
    _generator: tuple[tuple[float, ...], ...] = field(repr=False)
    _wing: tuple[tuple[str, float], ...] = field(default=(), repr=False)
    _cartan: tuple[tuple[str, float], ...] = field(default=(), repr=False)
    optimization: BDIOptimization | None = None

    @property
    def phase_preserving(self) -> bool:
        return self.mapping_kind == "horizontal-graph"

    def circuit(self, time: float, route: str = "exact-bdi") -> Circuit:
        """Evaluate the prepared circuit; horizontal evaluation needs no new SVD."""
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        circuit = Circuit()
        if time == 0:
            return circuit
        if self.phase_preserving:
            # Application order: K†, exp(-it A), K. Mirror the same physical
            # wing instead of independently lifting K† from its SO endpoint.
            for word, angle in reversed(self._wing):
                circuit.add(get_pauli_string(word), -angle, route)
            for word, rate in self._cartan:
                angle = time * rate
                if not np.isfinite(angle):
                    raise ValueError("time times a Cartan rate must be finite")
                circuit.add(get_pauli_string(word), angle, route)
            for word, angle in self._wing:
                circuit.add(get_pauli_string(word), angle, route)
        else:
            scaled = -time * np.asarray(self._generator)
            if not np.isfinite(scaled).all():
                raise ValueError("time times the irrep generator must be finite")
            matrix = expm(scaled)
            planes = {(i, j): (word, scale) for i, j, word, scale in self._planes}
            for i, j, angle in _bdi_planes(matrix):
                word, scale = planes[(i, j)]
                circuit.add(get_pauli_string(word), -angle / scale, route)
        return circuit


def prepare_bdi(
    hamiltonian_: PauliStringLinear, *, cache: bool = True,
    optimize: str = "none", rotation_error: float = 1e-6,
) -> BDIPlan:
    """Prepare paper BDI from the supplied Pauli algebra, for any evolution time.

    App. F.6 horizontal mapping is attempted without model-name recognition or
    Givens-oriented reordering. If no such mapping exists, a verified orthogonal
    algebra embedding enables the general VI.1 algorithm. Unsupported algebras
    or resource budgets fail explicitly. ``cache=False`` measures fresh plan
    preparation; it does not clear unrelated PauLie caches.

    ``optimize='t'`` searches a bounded set of legal nullspace completions of
    horizontal K, scored by an estimated paired-wing T cost at the supplied
    per-rotation precision. Every candidate uses the same paper BDI recursion.
    The original factors remain a fallback; no globally minimal T count or
    reduced rotation-count bound is claimed. Defaults retain the reference path.
    """
    if optimize not in ("none", "t"):
        raise ValueError("BDI optimize must be 'none' or 't'")
    rotation_error = float(rotation_error)
    if not np.isfinite(rotation_error) or not 0 < rotation_error < 1:
        raise ValueError("rotation_error must be finite and between zero and one")
    terms = terms_of(hamiltonian_)
    if any(not np.isfinite(c) or c.imag != 0 for c, _ in terms):
        raise ValueError("BDI requires finite real Hamiltonian coefficients")
    ordered = sorted((str(word), float(c.real)) for c, word in terms)
    words, coefficients = zip(*ordered) if ordered else ((), ())
    builder = _prepare_bdi if cache else _prepare_bdi.__wrapped__
    return builder(
        tuple(words), tuple(coefficients), n_qubits(hamiltonian_), optimize, rotation_error,
    )


def _compile_bdi_wing(k: np.ndarray, p: int, planes: dict) -> tuple[tuple[str, float], ...]:
    wing = []
    for start, stop in ((0, p), (p, len(k))):
        for i, j, angle in _bdi_planes(k[start:stop, start:stop]):
            word, scale = planes[(start + i, start + j)]
            wing.append((word, -angle / scale))
    return tuple(wing)


def _bdi_nullspace_candidates(k: np.ndarray, p: int) -> Iterator[np.ndarray]:
    """Change only the structural null columns, not the active singular frame.

    QR supplies candidate orthonormal completions, never emitted Givens gates.
    The fixed BDI recursion subsequently compiles the resulting full matrices.
    The eight-order cap bounds work independently of the nullspace dimension.
    """
    q = len(k) - p
    if abs(p - q) < 2:
        return
    start, stop = (p, len(k)) if q > p else (0, p)
    block = k[start:stop, start:stop]
    size, rank = len(block), min(p, q)
    null = np.arange(size - rank) if q > p else np.arange(rank, size)
    active = np.arange(size - rank, size) if q > p else np.arange(rank)
    frame = block[:, active]
    for shift in range(min(size, 8)):
        axes = np.roll(np.eye(size), shift, axis=1)
        completed, _ = np.linalg.qr(np.column_stack((frame, axes)), mode="complete")
        candidate_block = block.copy()
        candidate_block[:, null] = completed[:, rank:]
        # A reflection on a null column leaves H unchanged and keeps K in SO.
        if np.linalg.det(candidate_block) < 0:
            candidate_block[:, null[0]] *= -1
        candidate = k.copy()
        candidate[start:stop, start:stop] = candidate_block
        yield candidate


def _validated_bdi_gauge_wings(k, central, generator, p, planes):
    """Yield legal bounded completions, compiled with the unchanged BDI recursion."""
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(generator))))
    for index, candidate in enumerate(_bdi_nullspace_candidates(k, p), start=1):
        # Protect the lift: candidates must be legal SO(p) x SO(q) gauges of H.
        blocks = (candidate[:p, :p], candidate[p:, p:])
        if any(
            not np.allclose(block.T @ block, np.eye(len(block)), rtol=0, atol=1e-11)
            or not np.isclose(np.linalg.det(block), 1.0, rtol=0, atol=1e-10)
            for block in blocks
        ) or not np.allclose(
            candidate @ central @ candidate.T, generator, rtol=0, atol=tolerance,
        ):
            continue
        try:
            wing = _compile_bdi_wing(candidate, p, planes)
        except (ValueError, np.linalg.LinAlgError):
            # A numerically singular candidate must not invalidate the reference.
            continue
        yield index, wing


def _optimize_bdi_wing(
    k: np.ndarray, central: np.ndarray, generator: np.ndarray, p: int,
    planes: dict, baseline: tuple[tuple[str, float], ...], rotation_error: float,
) -> tuple[tuple[tuple[str, float], ...], BDIOptimization]:
    from lizzy.emission.clifford_t import estimate_rotation_t_count

    def score(wing):
        costs = [estimate_rotation_t_count(angle, rotation_error) for _, angle in wing]
        return 2 * sum(costs), 2 * sum(cost > 0 for cost in costs)

    original = score(baseline)
    selected, selected_score = baseline, original
    candidates = 1
    q = len(k) - p
    eligible = abs(p - q) >= 2
    for _, wing in _validated_bdi_gauge_wings(k, central, generator, p, planes):
        candidates += 1
        candidate_score = score(wing)
        if candidate_score < selected_score:
            selected, selected_score = wing, candidate_score
    return selected, BDIOptimization(
        eligible=eligible,
        rotation_error=rotation_error,
        candidates=candidates,
        original_t_estimate=original[0],
        selected_t_estimate=selected_score[0],
        original_nonclifford=original[1],
        selected_nonclifford=selected_score[1],
        optimized=selected is not baseline,
    )


@lru_cache(maxsize=32)
def _prepare_bdi(
    words: tuple[str, ...], coefficients: tuple[float, ...], width: int,
    optimize: str, rotation_error: float,
) -> BDIPlan:
    mapping, signs, size, partition = map_orthogonal(words, width)
    planes = {
        pair: (str(pauli_word_to_string(word, width)), 2.0 * signs[pair])
        for pair, word in mapping.items()
    }
    inverse = {word: pair for pair, (word, _) in planes.items()}
    generator = np.zeros((size, size))
    for word, coefficient in zip(words, coefficients, strict=True):
        i, j = inverse[word]
        generator[i, j] = coefficient * planes[(i, j)][1]
        generator[j, i] = -generator[i, j]
    payload = tuple((i, j, word, scale) for (i, j), (word, scale) in planes.items())
    immutable_generator = tuple(tuple(row) for row in generator)
    if partition is None:
        return BDIPlan(
            irrep_size=size,
            partition=(size // 2, size - size // 2),
            mapping_kind="general-graph",
            parameter_bound=size * (size - 1) // 2,
            _planes=payload,
            _generator=immutable_generator,
            optimization=(
                BDIOptimization(False, rotation_error, 0, 0, 0, 0, 0, False)
                if optimize == "t" else None
            ),
        )

    p, q = partition
    # The tested upstream generator SVD is recorded in README. Its high-level
    # horizontal compiler uses Givens for K, so compile both K blocks here with BDI.
    k, rates, cartan_planes = horizontal_generator_decomposition(generator, p)
    central = np.zeros_like(generator)
    cartan = []
    for (i, j), rate in zip(cartan_planes, rates, strict=True):
        central[i, j], central[j, i] = rate, -rate
        word, scale = planes[(i, j)]
        cartan.append((word, float(rate / scale)))
    if not np.allclose(
        k @ central @ k.T, generator, rtol=0,
        atol=1e-10 * max(1.0, np.max(np.abs(generator))),
    ):
        raise ValueError("horizontal BDI factors do not reconstruct the Hamiltonian")
    wing = _compile_bdi_wing(k, p, planes)
    optimization = None
    if optimize == "t":
        wing, optimization = _optimize_bdi_wing(
            k, central, generator, p, planes, wing, rotation_error,
        )
    bound = p * (p - 1) + q * (q - 1) + min(p, q)
    return BDIPlan(
        irrep_size=size,
        partition=partition,
        mapping_kind="horizontal-graph",
        parameter_bound=bound,
        _planes=payload,
        _generator=immutable_generator,
        _wing=wing,
        _cartan=tuple(cartan),
        optimization=optimization,
    )


def decompose(
    hamiltonian_: PauliStringLinear, time: float, route: str = "exact", *,
    method: str = "givens",
) -> Circuit:
    r"""Synthesize :math:`e^{-itH}` with time-independent depth.

    ``method='givens'`` eliminates the SO endpoint using at most m(m-1)/2
    adjacent-plane rotations. The low-weight ordering from :func:`_irrep` is
    a gate-cost heuristic, not an optimality guarantee.

    ``method='bdi'`` uses :func:`prepare_bdi`, following arXiv:2503.19014,
    Sections VI.1--VI.2 and Appendices F.4--F.6. Its graph-derived mapping is
    independent of Hamiltonian names and Givens' ordering. Horizontal inputs
    preserve phase and reuse a fixed K; see :class:`BDIPlan` for their bounds.

    Givens and general endpoint BDI are exact only up to global phase and are
    unsuitable for controlled phase-sensitive evolution. The SO-to-Pauli
    conversion divides each plane angle by its signed spinor scale (±2).
    ``route`` labels the emitted rotations. Unsupported representations raise;
    neither method falls back to the other.
    """
    if method not in {"givens", "bdi"}:
        raise ValueError("exact method must be 'givens' or 'bdi'")
    if method == "bdi":
        return prepare_bdi(hamiltonian_).circuit(time, route)
    terms = terms_of(hamiltonian_)
    words = tuple(str(p) for _, p in terms)
    width = n_qubits(hamiltonian_)
    generators, planes, order, m = _irrep(words, width)

    irrep_h = np.zeros((m, m))
    for coefficient, pauli in terms:
        irrep_h = irrep_h + coefficient.real * generators[str(pauli)]

    if time == 0:
        return Circuit()

    # Reduce V = exp(+t H_irrep) to the identity with adjacent-row Givens rotations.
    # Their elimination product is V^-1, the representative of exp(-itH).
    matrix = expm(time * irrep_h)[np.ix_(order, order)]
    applied: list[tuple[int, float]] = []
    for column in range(m - 1):
        for row in range(m - 2, column - 1, -1):
            a, b = matrix[row, column], matrix[row + 1, column]
            # A negative pivot still needs a pi rotation even if b is zero.
            if abs(b) < 1e-14 and a >= 0:
                continue
            angle = float(np.arctan2(b, a))
            cosine, sine = np.cos(angle), np.sin(angle)
            upper = cosine * matrix[row] + sine * matrix[row + 1]
            matrix[row + 1] = -sine * matrix[row] + cosine * matrix[row + 1]
            matrix[row] = upper
            applied.append((row, angle))

    # rho(+iP) = scale * (E_ij - E_ji), so a circuit angle -angle/scale represents
    # the elimination exp(angle * (E_ij - E_ji)). Emitting R_1 first gives the matrix
    # product R_N ... R_1 = V^-1. Reversing a plane's relabeled indices negates its
    # angle once more. The SO(m) endpoint alone does not fix the spin-cover phase.
    circuit = Circuit()
    for row, angle in applied:
        i, j = order[row], order[row + 1]
        angle = -angle
        if i > j:
            i, j = j, i
            angle = -angle
        word, scale = planes[(i, j)]
        circuit.add(word, angle / scale, route)
    return circuit


def free_part(
    hamiltonian_: PauliStringLinear,
) -> tuple[PauliStringLinear | None, PauliStringLinear]:
    """Extract a noncommuting, exactly compilable subset and its remainder.

    Greedy growth offers whole interaction families in descending coefficient
    weight, then individual terms. If this finds only commuting terms, retry
    while deferring commuting families: heavy diagonal families can otherwise
    prevent useful noncommuting terms from joining the subset. This is a
    heuristic, not a maximal-subalgebra search.

    Candidates above ``max(64, 8n)`` terms are declined before classification
    to bound search cost. A supported subset becomes one exact product-formula
    layer, removing internal splitting error but not cross-layer commutators.
    Mutually commuting subsets provide no such benefit and are discarded.

    Return ``(None, hamiltonian_)`` if no qualifying subset exists. Otherwise
    the returned pair contains every original term, without overlap.
    """
    terms = [(c, str(p)) for c, p in terms_of(hamiltonian_)]
    budget = max(64, 8 * n_qubits(hamiltonian_))

    families: dict[str, list[tuple[complex, str]]] = {}
    for coefficient, pauli in terms:
        families.setdefault(_family(pauli), []).append((coefficient, pauli))
    ordered = sorted(
        families.values(), key=lambda f: -sum(abs(c) for c, _ in f)
    )

    free = _grow(ordered, terms, budget, defer_abelian=False)
    if not _is_non_abelian(free):
        free = _grow(ordered, terms, budget, defer_abelian=True)

    if not _is_non_abelian(free):
        # A set of mutually commuting terms contributes nothing to the Trotter error in
        # the first place, so extracting it buys no accuracy and only adds a branch.
        return None, hamiltonian_
    taken = {pauli for _, pauli in free}
    rest = [(c, p) for c, p in terms if p not in taken]
    return _build(free), _build(rest)


def _grow(
    ordered: list[list[tuple[complex, str]]],
    terms: list[tuple[complex, str]],
    budget: int,
    defer_abelian: bool,
) -> list[tuple[complex, str]]:
    """Grow a decomposable set greedily: whole families first, then single terms.

    With ``defer_abelian`` a family that would leave the set mutually commuting is held
    back and offered again at the end of the family pass, once something non-commuting
    has been found for it to join. See :func:`free_part` for why that ordering is the
    second one tried rather than the first.
    """
    free: list[tuple[complex, str]] = []
    deferred: list[list[tuple[complex, str]]] = []

    for family in ordered:
        candidate = free + family
        if not 2 <= len(candidate) <= budget:
            continue
        if defer_abelian and not _is_non_abelian(candidate):
            deferred.append(family)
            continue
        if decomposes_by_summand(_build(candidate)):
            free = candidate

    for family in deferred:
        candidate = free + family
        if 2 <= len(candidate) <= budget and decomposes_by_summand(_build(candidate)):
            free = candidate

    # Top up with any individual term the family pass could not take wholesale.
    taken = {pauli for _, pauli in free}
    for coefficient, pauli in sorted(terms, key=lambda t: -abs(t[0])):
        if pauli in taken:
            continue
        candidate = free + [(coefficient, pauli)]
        if 2 <= len(candidate) <= budget and decomposes_by_summand(_build(candidate)):
            free = candidate
            taken.add(pauli)
    return free


def _is_non_abelian(terms: list[tuple[complex, str]]) -> bool:
    """Check that at least one pair of terms fails to commute."""
    paulis = [p for _, p in terms_of(_build(terms))]
    return bool(anticommutation_matrix(paulis).any())


def _family(pauli: str) -> str:
    """Name the interaction family of a Pauli string, ignoring where it sits.

    ``"IXXI"`` and ``"XXII"`` are both ``"XX"``, so a chain's bond terms form one
    family and its field terms another.
    """
    return "".join(letter for letter in pauli if letter != "I")


def _build(terms: list[tuple[complex, str]]) -> PauliStringLinear:
    """Rebuild a Hamiltonian from coefficient/string pairs."""
    return hamiltonian([(pauli, coefficient) for coefficient, pauli in terms])
