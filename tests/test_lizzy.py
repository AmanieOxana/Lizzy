"""
    Tests for the synthesis pipeline.

    The ones that matter check circuits against a dense reference: exactness claims are
    worth exactly as much as their verification, and the upstream KAK decomposition
    verifies itself only in its m-dimensional irrep, never at the qubit level.
"""

import math

import numpy as np
import pytest

from lizzy.bench import calibrate
from lizzy.classify import classify, is_fast_forwardable, summands
from lizzy.dense import circuit_matrix, evolution, infidelity, pauli_matrix
from lizzy.exact import decompose, free_part, is_decomposable
from lizzy.frame import (
    clifford_to,
    conjugate,
    find_assignment,
    is_symplectic,
    pauli_vectors,
    witt_extend,
)
from lizzy.hamiltonian import (
    Circuit,
    hamiltonian,
    model,
    n_qubits,
    rotation_cost,
    terms_of,
    weight,
)
from lizzy.hamlib import fetch, load
from lizzy.symmetry import commuting_clusters, pair_clusters, taper, z2_symmetries
from lizzy.synthesize import synthesize
from lizzy.trotter import (
    _collected_commutator,
    cluster_error_constant,
    cluster_formula,
    coefficient_norm,
    commutator_sum,
    nested_commutator_sum,
    product_formula,
    product_formula_cost,
    qdrift,
    split_by_magnitude,
    steps_for,
    steps_for_clusters,
)

FAST_FORWARDABLE = ["tfim", "tfxy", "xy"]
SMALL = [3, 4, 5]


# ---------------------------------------------------------------------------
# The exact branch, checked at the qubit level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAST_FORWARDABLE)
@pytest.mark.parametrize("n", SMALL)
@pytest.mark.parametrize("time", [0.4, 3.0])
def test_exact_branch_implements_the_evolution(name: str, n: int, time: float) -> None:
    """Every summand's decomposition must be the evolution it claims to be.

    kak_tools validates its own output against exp(tH) in the irrep it decomposes in,
    which says nothing about whether the Pauli rotations compose into the right circuit
    on 2**n qubits. That is what is checked here.
    """
    h = model(name, n, seed=1)
    circuit = Circuit()
    for part in summands(h):
        assert is_decomposable(part)
        circuit.extend(decompose(part, time))

    assert infidelity(evolution(h, time), circuit_matrix(circuit, n)) < 1e-10


@pytest.mark.parametrize("n", SMALL)
def test_exact_branch_is_flat_in_time(n: int) -> None:
    """The gate count of a fixed-depth decomposition does not grow with the time."""
    h = model("tfxy", n, seed=0)
    counts = {decompose(h, t).two_qubit_gates for t in (0.1, 1.0, 10.0, 100.0)}
    assert len(counts) == 1


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_odd_irrep_sizes_are_reachable(n: int) -> None:
    """The Ising chain is so(2n-1) in PauLie's convention, and must still decompose."""
    h = model("tfim", n, seed=0)
    assert is_decomposable(h)


# ---------------------------------------------------------------------------
# Exact reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["xy", "heisenberg", "tfim"])
@pytest.mark.parametrize("n", [4, 5])
def test_summands_commute_and_are_complete(name: str, n: int) -> None:
    """Splitting must lose no term and must leave the parts mutually commuting."""
    h = model(name, n, seed=2)
    parts = summands(h)

    original = {str(p) for _, p in terms_of(h)}
    recovered = {str(p) for part in parts for _, p in terms_of(part)}
    assert recovered == original

    for i, first in enumerate(parts):
        for second in parts[i + 1:]:
            assert all(
                a.commutes_with(b)
                for _, a in terms_of(first)
                for _, b in terms_of(second)
            )


@pytest.mark.parametrize("n", [3, 4])
def test_summand_factorization_is_exact(n: int) -> None:
    """Because the parts commute, evolving them in sequence is not an approximation."""
    h = model("xy", n, seed=3)
    time = 0.7

    product = np.eye(2**n, dtype=complex)
    for part in summands(h):
        product = evolution(part, time) @ product

    assert infidelity(evolution(h, time), product) < 1e-12


@pytest.mark.parametrize("name", ["tfim", "heisenberg", "heisenberg_all_to_all"])
@pytest.mark.parametrize("n", [4, 6])
def test_clusters_are_internally_commuting(name: str, n: int) -> None:
    """Terms grouped together must commute, or the group cannot share a basis change."""
    for cluster in commuting_clusters(model(name, n, seed=0)):
        paulis = [p for _, p in terms_of(cluster)]
        assert all(a.commutes_with(b) for a in paulis for b in paulis)


@pytest.mark.parametrize("name", ["tfim", "tfxy", "heisenberg", "heisenberg_all_to_all"])
@pytest.mark.parametrize("n", [6, 10, 16])
def test_cluster_count_does_not_grow_with_size(name: str, n: int) -> None:
    """A Trotter step needs a basis change per cluster, and that count stays put."""
    assert len(commuting_clusters(model(name, n, seed=0))) <= 4


@pytest.mark.parametrize("name", ["tfim", "tfxy", "heisenberg"])
@pytest.mark.parametrize("n", [4, 5])
def test_tapering_keeps_the_spectrum_of_its_sector(name: str, n: int) -> None:
    """Every eigenvalue of the tapered Hamiltonian must be one of the original's.

    Tapering claims to remove qubits without changing physics, which is only true if
    the reduced spectrum sits inside the full one -- the claim worth checking, since a
    wrong Clifford or a dropped phase would still produce a plausible-looking operator.
    """
    h = model(name, n, seed=1)
    charges = z2_symmetries(h)
    tapered, removed = taper(h)
    assert tapered is not None
    # One qubit per *commuting* charge: anticommuting charges share no eigenbasis, so
    # an odd chain's X^n and Z^n cannot both be fixed and only one is used.
    assert 0 < len(removed) <= len(charges)

    def spectrum(hamiltonian_, width):
        matrix = sum(
            c.real * pauli_matrix(str(p)) for c, p in terms_of(hamiltonian_)
        )
        return np.sort(np.linalg.eigvalsh(matrix))

    full = spectrum(h, n)
    reduced = spectrum(tapered, n - len(removed))
    assert all(np.min(np.abs(full - value)) < 1e-8 for value in reduced)


def test_tapering_shortens_the_strings_it_keeps() -> None:
    """The saving would be illusory if the tapering Clifford inflated Pauli weight;
    on chemistry it does the opposite, which is why the reduction is worth taking."""
    h = load(fetch("chemistry/electronic/standard/BH.zip"), "ham_BK-10")
    tapered, removed = taper(h)
    assert len(removed) == 4

    before = np.mean([weight(p) for _, p in terms_of(h)])
    after = np.mean([weight(p) for _, p in terms_of(tapered)])
    assert after < before


@pytest.mark.parametrize("name", ["tfim", "tfxy", "heisenberg"])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_symmetries_commute_with_every_term(name: str, n: int) -> None:
    """A conserved charge that fails to commute with a term is not conserved."""
    h = model(name, n, seed=0)
    for symmetry in z2_symmetries(h):
        assert all(symmetry.commutes_with(p) for _, p in terms_of(h))


# ---------------------------------------------------------------------------
# Clifford frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sites", [2, 3, 4])
def test_clifford_carries_bravyi_kitaev_onto_jordan_wigner(sites: int) -> None:
    """The same model in two encodings is Clifford equivalent, and the map is built.

    This is the whole claim of lizzy.frame in one test: Bravyi-Kitaev and
    Jordan-Wigner are two representations of one algebra, the second is markedly more
    local, and the Clifford between them is constructible rather than searched for.
    """
    archive = fetch("condensedmatter/fermihubbard/FH_D-1.zip")
    jw = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-jw")
    bk = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-bk")
    width = n_qubits(jw)

    matrix = clifford_to(bk, jw)
    assert matrix is not None
    assert is_symplectic(matrix, width)

    images = (pauli_vectors(bk) @ matrix) % 2
    assert {tuple(row) for row in images} == {tuple(row) for row in pauli_vectors(jw)}


def test_matching_must_respect_algebraic_dependencies() -> None:
    """Anticommutation alone is not the equivalence criterion.

    Dependent generators have to map to the same dependencies, or the linear map
    breaks on exactly those terms. The assignment is what enforces it, so every term
    it returns must be reachable, not merely most of them.
    """
    archive = fetch("condensedmatter/fermihubbard/FH_D-1.zip")
    jw = load(archive, "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-2_U-4_enc-jw")
    bk = load(archive, "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-2_U-4_enc-bk")
    source, target = pauli_vectors(bk), pauli_vectors(jw)
    width = source.shape[1] // 2

    assignment = find_assignment(source, target, width)
    assert assignment is not None
    assert sorted(assignment) == list(range(source.shape[0]))  # a bijection

    matrix = clifford_to(bk, jw)
    images = (source @ matrix) % 2
    for index, image in enumerate(images):
        assert np.array_equal(image, target[assignment[index]])


@pytest.mark.parametrize("sites", [2, 3])
def test_conjugation_preserves_the_spectrum(sites: int) -> None:
    """A Clifford cannot move eigenvalues, so a phase slip shows up as a moved one.

    This is the check that matters for the sign bookkeeping: Y is XZ up to a phase, and
    getting that wrong yields a plausible-looking operator with the wrong spectrum.
    """
    archive = fetch("condensedmatter/fermihubbard/FH_D-1.zip")
    jw = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-jw")
    bk = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-bk")

    transformed = conjugate(bk, clifford_to(bk, jw))

    def spectrum(hamiltonian_):
        matrix = sum(c.real * pauli_matrix(str(p)) for c, p in terms_of(hamiltonian_))
        return np.sort(np.linalg.eigvalsh(matrix))

    assert np.abs(spectrum(bk) - spectrum(transformed)).max() < 1e-9


@pytest.mark.parametrize("sites", [2, 3])
def test_the_frame_recovers_the_better_representation_cost(sites: int) -> None:
    """The point of the frame: compiling the moved Hamiltonian costs what the good
    representation costs, not what the one handed over costs."""
    archive = fetch("condensedmatter/fermihubbard/FH_D-1.zip")
    jw = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-jw")
    bk = load(archive, f"fh-graph-1D-grid-nonpbc-qubitnodes_Lx-{sites}_U-4_enc-bk")

    moved = conjugate(bk, clifford_to(bk, jw))
    before = synthesize(bk, time=1.0, steps=2).two_qubit_gates
    after = synthesize(moved, time=1.0, steps=2).two_qubit_gates
    reference = synthesize(jw, time=1.0, steps=2).two_qubit_gates

    assert after <= reference
    assert after < before


def test_witt_extension_refuses_a_non_isometry() -> None:
    """A matching that does not preserve the form has no symplectic extension, and
    saying so beats returning a matrix that quietly is not a Clifford."""
    width = 2
    source = [np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])]   # anticommuting pair
    target = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]   # commuting pair
    with pytest.raises(ValueError):
        witt_extend(source, target, width)


# ---------------------------------------------------------------------------
# The free part
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 6, 7])
def test_free_part_of_a_heisenberg_chain_is_decomposable(n: int) -> None:
    """The extracted part has to be exactly compilable, or extracting it is pointless."""
    free, rest = free_part(model("heisenberg", n, seed=0))
    assert free is not None
    assert all(is_decomposable(part) for part in summands(free))
    assert len(terms_of(free)) + len(terms_of(rest)) == len(
        terms_of(model("heisenberg", n, seed=0))
    )


@pytest.mark.parametrize("n", [4, 5])
def test_hybrid_mechanism_meets_the_budget(n: int) -> None:
    """The free-part-as-summand formula delivers the error it was sized for.

    Tested as a mechanism, not a route: on 2-local chains the pair-kernel formula
    is cheaper and the router rightly prefers it, but the hybrid stays reachable
    for inputs the pair clustering declines, so its budget claim keeps its test.
    """
    from lizzy.synthesize import _hybrid_plan

    h = model("heisenberg", n, seed=1)
    free, rest = free_part(h)
    cost, build = _hybrid_plan(free, rest, 1.0, 1e-3, 1.0, None)
    circuit = build()
    assert circuit.two_qubit_gates <= cost  # folding can only improve on the price
    assert infidelity(evolution(h, 1.0), circuit_matrix(circuit, n)) < 1e-3


def test_hybrid_free_network_appears_once_per_step() -> None:
    """The free summand sits last in the symmetric step, so its two halves merge:
    one network per step, not two."""
    from lizzy.synthesize import _hybrid_plan

    h = model("heisenberg", 4, seed=1)
    free, rest = free_part(h)
    clusters = commuting_clusters(rest) + [free]
    steps = steps_for_clusters(clusters, 1.0, 1e-3)

    circuit = _hybrid_plan(free, rest, 1.0, 1e-3, 1.0, None)[1]()
    per_step = Circuit()
    for piece in summands(free):
        per_step.extend(decompose(piece, 1.0 / steps))
    exact_rotations = sum(
        1 for label in circuit.provenance if label == "exact-in-step"
    )
    assert exact_rotations == steps * len(per_step.rotations)


@pytest.mark.parametrize("n", [8, 30])
def test_free_part_declines_dense_families_within_budget(n: int) -> None:
    """All-to-all XX+YY is not hopping -- the Jordan-Wigner strings beyond nearest
    neighbours make it exponential -- so the right answer is no, and the budget makes
    that answer cheap instead of classifying n^2-term candidates to hear it."""
    free, rest = free_part(model("heisenberg_all_to_all", n, seed=0))
    assert free is None
    assert len(terms_of(rest)) == len(terms_of(model("heisenberg_all_to_all", n, seed=0)))


@pytest.mark.parametrize("n", [4, 5])
def test_pair_kernel_route_meets_the_budget(n: int) -> None:
    """The pair-kernel layers are a different S2 than the letter clusters, and any
    formula the router may pick has to deliver the budget it was sized for."""
    h = model("heisenberg_all_to_all", n, seed=1)
    result = synthesize(h, time=1.0, error=1e-3, order=4, seed=0)
    assert infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, n)) < 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noncommuting_kernels_compile_exactly(seed: int) -> None:
    """A kernel with fields folded in is not internally commuting, and its KAK
    emission must still equal the 4x4 exponential -- checked here through the dense
    machinery, on top of the self-check every kernel runs at synthesis time."""
    import numpy as np
    from scipy.linalg import expm as dense_expm

    from lizzy.kernels import compile_kernel

    rng = np.random.default_rng(seed)
    words = ["IXIX", "IYIZ", "IZIY", "IXII", "IIIZ", "IYIY"]
    chosen = {w: float(rng.normal()) for w in rng.choice(words, 4, replace=False)}
    h = hamiltonian(chosen)
    tau = float(rng.uniform(0.3, 1.5))

    circuit = compile_kernel(terms_of(h), tau, 4, "t")
    target = dense_expm(
        -1j * tau * sum(c.real * pauli_matrix(str(p)) for c, p in terms_of(h))
    )
    assert infidelity(target, circuit_matrix(circuit, 4)) < 1e-9
    assert circuit.two_qubit_gates <= 3


@pytest.mark.parametrize("name", ["tfim", "heisenberg"])
def test_field_carrying_models_take_the_kernel_route(name: str) -> None:
    """Weight-one fields fold into their pair kernels instead of blocking the pair
    clustering, and the resulting formula still meets its budget densely."""
    h = model(name, 5, seed=1)
    layers = pair_clusters(h)
    assert layers is not None

    result = synthesize(h, time=1.0, error=1e-3, order=4, seed=0)
    assert infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, 5)) < 1e-3


def test_pair_kernels_cost_less_than_their_ladders() -> None:
    """Three same-pair rotations compile as one canonical block: three CNOTs, not
    six. The block-aware count is what makes the pair clustering worth choosing."""
    h = model("heisenberg_all_to_all", 6, seed=0)
    layers = pair_clusters(h)
    assert layers is not None
    step = cluster_formula(layers, 1.0, 1)
    ladders = sum(2 for p, _ in step.rotations)  # weight-2 terms, 2 CNOTs each
    assert step.two_qubit_gates < ladders


def test_shared_frame_emission_implements_the_same_unitary() -> None:
    """The shared-frame backend must reproduce the rotation sequence exactly.

    GreedyPauliSimp conjugates rotations through a Clifford frame; if that frame were
    not unwound correctly the gate counts it reports would be for a different circuit.
    """
    pytest.importorskip("pytket")
    from lizzy.emit import tket_circuit

    h = model("heisenberg", 4, seed=1)
    result = synthesize(h, time=1.0, steps=2)
    synthesized = tket_circuit(result.circuit, 4)
    assert infidelity(circuit_matrix(result.circuit, 4), synthesized.get_unitary()) < 1e-9


def test_fixed_steps_mode_emits_without_sizing() -> None:
    """With ``steps`` given the circuit is the cluster formula at exactly that depth.

    This is the Qiskit-reps contract: the user chooses the depth, no error statement
    is made, and none of the bounds are paid for.
    """
    h = model("heisenberg_all_to_all", 5, seed=0)

    def cost(cluster):
        return sum(rotation_cost(p) for _, p in terms_of(cluster))

    result = synthesize(h, time=1.0, steps=3)
    assert result.routes == ["trotter2"]

    # Whichever clustering the router priced cheaper, the emitted circuit must be
    # that formula at exactly three steps -- checked as a unitary, since phase
    # folding makes the rotation count a poor proxy for the depth contract.
    emitted = circuit_matrix(result.circuit, 5)
    expected = []
    for candidate in (commuting_clusters(h), pair_clusters(h)):
        ordered = sorted(candidate, key=cost)
        expected.append(circuit_matrix(cluster_formula(ordered, 1.0, 3), 5))
    assert any(infidelity(emitted, reference) < 1e-10 for reference in expected)
    assert len(result.circuit.rotations) <= 3 * 2 * len(terms_of(h))


def test_routing_is_arithmetic_not_precedence() -> None:
    """The exact route wins exactly when it is cheaper, not because it is exact.

    Its cost is flat in the step count, a formula's is linear, so few steps favour
    the formula and many steps the fixed depth -- both directions in one model.
    """
    cheap = synthesize(model("tfim", 4, seed=0), time=1.0, steps=2)
    assert cheap.routes == ["trotter2"]
    deep = synthesize(model("tfim", 4, seed=0), time=1.0, steps=200)
    assert deep.routes == ["exact"]


def test_dense_parts_are_not_classified() -> None:
    """Naming the algebra is reporting, and at dense sizes it costs minutes; the
    routing budget gates it, and the report says so instead of guessing."""
    result = synthesize(model("heisenberg_all_to_all", 30, seed=0), time=1.0, steps=1)
    assert result.algebra.startswith("(unclassified")


def test_free_part_declines_a_commuting_set() -> None:
    """Commuting terms contribute no Trotter error, so extracting them wins nothing."""
    h = hamiltonian({"ZZII": 1.0, "IZZI": 1.0, "IIZZ": 1.0})
    free, rest = free_part(h)
    assert free is None
    assert len(terms_of(rest)) == 3


# ---------------------------------------------------------------------------
# Product formulas and sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 4])
def test_product_formula_converges(order: int) -> None:
    """More steps must mean less error, at the rate the order promises."""
    h = model("heisenberg", 3, seed=0)
    target = evolution(h, 1.0)

    errors = [
        infidelity(target, circuit_matrix(product_formula(h, 1.0, steps, order), 3))
        for steps in (2, 8, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_steps_scale_with_the_budget() -> None:
    """A tighter budget must buy more steps, and a looser one fewer."""
    h = model("heisenberg", 4, seed=0)
    assert steps_for(h, 1.0, 1e-2, 2) < steps_for(h, 1.0, 1e-4, 2)
    assert steps_for(h, 1.0, 1e-3, 2) < steps_for(h, 4.0, 1e-3, 2)


def test_commutator_sum_counts_only_anticommuting_pairs() -> None:
    """Commuting terms contribute nothing to the first-order error."""
    assert commutator_sum(hamiltonian({"ZII": 1.0, "IZI": 1.0})) == 0.0
    assert commutator_sum(hamiltonian({"XII": 1.0, "ZII": 1.0})) == pytest.approx(2.0)


def test_default_route_meets_its_budget() -> None:
    """The default route keeps its budget on an instance the sampled one does not."""
    h = model("heisenberg", 5, seed=2)
    result = synthesize(h, time=1.0, error=1e-3, order=4, seed=0)

    assert not result.randomized
    assert "qdrift" not in result.routes
    error = infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, 5))
    assert error < 1e-3


def _spread_hamiltonian():
    """A couple of large terms and a swarm of tiny weight-three ones.

    qDRIFT's regime, kept out of reach of the kernel route: weight three makes the
    pair clustering decline, so the deterministic side pays ladders per term.
    """
    rng = np.random.default_rng(0)
    terms = {"ZZIII": 1.0, "IIZZI": 1.0}
    for _ in range(40):
        q = int(rng.integers(0, 3))
        letters = "".join(rng.choice(list("XYZ"), size=3))
        s = "I" * q + letters + "I" * (2 - q)
        terms[s] = terms.get(s, 0.0) + float(rng.normal()) * 1e-3
    return hamiltonian(terms)


def test_sampling_stays_reachable_when_asked_for() -> None:
    """The branch is gated, not deleted: opting in routes into it where it wins.

    Deterministic formulas pay a gate per term per step, so many tiny terms make
    sampling cheaper at a loose budget; the same call without the flag must stay
    deterministic.
    """
    h = _spread_hamiltonian()
    result = synthesize(h, time=1.0, error=1e-1, order=4, seed=0, randomized=True)
    assert result.randomized
    assert "qdrift" in result.routes
    assert not synthesize(h, time=1.0, error=1e-1, order=4, seed=0).randomized


def test_nested_commutator_sum_matches_a_hand_computable_case() -> None:
    """Two anticommuting terms give chains whose weight is closed-form."""
    h = hamiltonian({"XI": 0.5, "ZI": 2.0})

    # At order 1 the chains of length two are (X, Z) and (Z, X); each is nonzero with
    # norm 2|c_a||c_b| = 2 * 0.5 * 2.0 = 2, and (X, X) and (Z, Z) vanish.
    assert nested_commutator_sum(h, 1) == pytest.approx(4.0)

    # Commuting terms have no chains at all, at any order.
    assert nested_commutator_sum(hamiltonian({"ZII": 1.0, "IZI": 1.0}), 4) == 0.0


def test_nested_commutator_sum_reports_exhaustion_rather_than_guessing() -> None:
    """Running out of budget must return None, not a truncated sum."""
    h = model("heisenberg_all_to_all", 5, seed=0)
    assert nested_commutator_sum(h, 4, budget=50) is None
    assert steps_for(h, 1.0, 1e-3, 4) > 0  # falls back rather than failing


def test_nested_bound_beats_the_coefficient_norm_by_more_as_size_grows() -> None:
    """Only neighbouring terms of a local Hamiltonian anticommute, so the chain sum
    grows linearly in the qubit count where the coefficient norm grows as a power of
    it, and the advantage widens with size rather than being a constant."""

    def coefficient_norm_steps(h, order: int) -> int:
        return max(1, math.ceil(((coefficient_norm(h) * 1.0) ** (order + 1) / 1e-3) ** (1 / order)))

    gains = []
    for n in (8, 16, 32):
        h = model("tfim", n, seed=1)
        gains.append(coefficient_norm_steps(h, 4) / steps_for(h, 1.0, 1e-3, 4))

    assert gains[0] < gains[1] < gains[2]
    assert gains[-1] > 4.0


def _dense_sum(terms) -> np.ndarray:
    matrix = None
    for coefficient, pauli in terms:
        term = complex(coefficient) * pauli_matrix(str(pauli))
        matrix = term if matrix is None else matrix + term
    return matrix


@pytest.mark.parametrize("name", ["heisenberg", "heisenberg_all_to_all"])
def test_collected_commutator_reproduces_the_dense_one(name: str) -> None:
    """The collected Pauli sum must equal [S, [S, A]] as matrices, phases included.

    The whole point of collecting is that opposite-phase chains cancel before the norm
    is taken, so a phase convention error would silently destroy the bound.
    """
    clusters = commuting_clusters(model(name, 3, seed=1))
    head = [(c.real, p) for c, p in terms_of(clusters[0])]
    tail = [(c.real, p) for cl in clusters[1:] for c, p in terms_of(cl)]

    inner = _collected_commutator(tail, head, [0], 10**9)
    outer = _collected_commutator(tail, inner, [0], 10**9)

    s, a = _dense_sum(tail), _dense_sum(head)
    dense = s @ (s @ a - a @ s) - (s @ a - a @ s) @ s
    assert np.allclose(_dense_sum(outer), dense)


@pytest.mark.parametrize("name", ["heisenberg", "heisenberg_all_to_all"])
@pytest.mark.parametrize("n", [4, 5])
def test_cluster_formula_meets_the_budget_it_was_sized_for(name: str, n: int) -> None:
    """The certified second-order step count must actually deliver the error."""
    h = model(name, n, seed=1)
    clusters = commuting_clusters(h)
    steps = steps_for_clusters(clusters, 1.0, 1e-3)
    assert steps is not None

    circuit = cluster_formula(clusters, 1.0, steps)
    assert infidelity(evolution(h, 1.0), circuit_matrix(circuit, n)) < 1e-3


def test_cluster_constant_is_cached() -> None:
    """The constant depends only on the clusters, so asking twice walks once.

    Calibration bisects over synthesize and benchmark sweeps vary the time, both
    recomputing an identical constant; the cache is what makes them affordable.
    """
    from lizzy import trotter

    clusters = commuting_clusters(model("heisenberg", 6, seed=0))
    trotter._constant_cache.clear()
    first = cluster_error_constant(clusters)
    assert len(trotter._constant_cache) == 1

    def exploding_walk(*_args, **_kwargs):
        raise AssertionError("cache miss: the walk ran again for identical clusters")

    original = trotter._collected_commutator
    trotter._collected_commutator = exploding_walk
    try:
        assert cluster_error_constant(clusters) == first
    finally:
        trotter._collected_commutator = original


def test_cluster_route_is_cheaper_and_taken() -> None:
    """On a Heisenberg model a clustered second-order step beats the chain-bounded
    formula, and the router follows the arithmetic."""
    h = model("heisenberg_all_to_all", 5, seed=1)
    chain = product_formula_cost(h, 1.0, 1e-3, 4)

    result = synthesize(h, time=1.0, error=1e-3, order=4)
    assert "trotter2" in result.routes
    assert result.two_qubit_gates < chain
    assert infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, 5)) < 1e-3


def test_calibration_shrinks_the_circuit_and_keeps_the_budget() -> None:
    """A factor measured on one size still meets the budget at a larger one."""
    factor = calibrate(model("heisenberg_all_to_all", 4, seed=1), 1.0, 1e-3, 4)
    assert factor > 1.0

    h = model("heisenberg_all_to_all", 5, seed=1)
    base = synthesize(h, time=1.0, error=1e-3, order=4)
    tuned = synthesize(h, time=1.0, error=1e-3, order=4, calibration=factor)

    assert tuned.two_qubit_gates < base.two_qubit_gates
    assert infidelity(evolution(h, 1.0), circuit_matrix(tuned.circuit, 5)) < 1e-3


def test_calibration_holds_on_the_route_taken() -> None:
    """A factor measured through synthesize keeps the budget on whatever route the
    instance triggers, because calibration bisects over synthesize itself."""
    h = model("heisenberg", 5, seed=1)
    factor = calibrate(h, 1.0, 1e-3, 4)
    assert factor > 1.0

    tuned = synthesize(h, time=1.0, error=1e-3, order=4, calibration=factor)
    assert infidelity(evolution(h, 1.0), circuit_matrix(tuned.circuit, 5)) < 1e-3


def test_split_by_magnitude_keeps_every_term() -> None:
    """The split is a partition, not a filter."""
    h = model("heisenberg", 4, seed=0)
    large, small = split_by_magnitude(h)
    assert len(terms_of(large)) + len(terms_of(small)) == len(terms_of(h))
    assert coefficient_norm(large) + coefficient_norm(small) == pytest.approx(
        coefficient_norm(h)
    )


def test_qdrift_gate_count_ignores_the_term_count() -> None:
    """Sampling pays for the coefficients, not for how many terms carry them.

    Splitting one term into ten of a tenth the size leaves the total weight, and so the
    gate count, unchanged -- which is what a product formula cannot do.
    """
    concentrated = hamiltonian({"XXII": 1.0})
    spread = hamiltonian({s: 0.1 for s in ["XXII", "IXXI", "IIXX", "ZIII", "IZII",
                                           "IIZI", "IIIZ", "YYII", "IYYI", "IIYY"]})
    assert coefficient_norm(concentrated) == pytest.approx(coefficient_norm(spread))
    assert len(qdrift(concentrated, 1.0, 0.1, seed=0)) == len(
        qdrift(spread, 1.0, 0.1, seed=0)
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAST_FORWARDABLE)
@pytest.mark.parametrize("n", [3, 4])
def test_synthesize_is_exact_where_the_algebra_allows(name: str, n: int) -> None:
    """A fast-forwardable model must come out exact, at any time, to machine precision."""
    h = model(name, n, seed=1)
    for time in (0.5, 5.0):
        result = synthesize(h, time=time, error=1e-3)
        assert result.routes == ["exact"]
        assert infidelity(
            evolution(h, time), circuit_matrix(result.circuit, n)
        ) < 1e-10


@pytest.mark.parametrize("n", [3, 4])
def test_synthesize_meets_its_budget_on_the_inexact_branch(n: int) -> None:
    """Where the evolution has to be approximated, the budget must still be met."""
    h = model("heisenberg", n, seed=1)
    result = synthesize(h, time=1.0, error=1e-3, order=4, seed=0)
    assert infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, n)) < 1e-3


def test_synthesize_reports_the_route_it_took() -> None:
    """The account of the route is what makes the cost attributable."""
    result = synthesize(model("tfim", 5, seed=0), time=1.0)
    assert result.algebra == "so(10)"
    assert result.summands == 1
    assert result.symmetries >= 1
    assert result.circuit.cost_by_route() == {"exact": result.two_qubit_gates}


def test_classification_routes_poly_and_exponential_apart() -> None:
    """The routing verdict has to follow the dimension, not the model's name."""
    tfxy = model("tfxy", 6, seed=0)
    heisenberg = model("heisenberg", 6, seed=0)
    assert is_fast_forwardable(classify(tfxy), n_qubits(tfxy))
    assert not is_fast_forwardable(classify(heisenberg), n_qubits(heisenberg))


def test_pauli_matrix_matches_the_string() -> None:
    """The dense reference is only useful if it agrees with the Pauli convention."""
    assert np.allclose(pauli_matrix("IZ"), np.diag([1, -1, 1, -1]))
    assert np.allclose(pauli_matrix("XI"), np.kron(pauli_matrix("X"), np.eye(2)))
