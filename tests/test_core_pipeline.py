"""Synthesis routing, hybrid construction, and end-to-end accuracy contracts."""

import numpy as np
import pytest
from paulie.common.pauli_string_factory import get_pauli_string

from lizzy.classify import summands
from lizzy.dense import (
    circuit_matrix,
    evolution,
    infidelity,
)
from lizzy.exact import decompose, free_part, is_decomposable
from lizzy.hamiltonian import (
    Circuit,
    anticommutation_matrix,
    hamiltonian,
    model,
    rotation_cost,
    terms_of,
)
from lizzy.symmetry import commuting_clusters, pair_clusters
from lizzy.synthesize import synthesize
from lizzy.trotter import (
    cluster_formula,
    product_formula_cost,
    steps_for_clusters,
)


def test_the_degenerate_hybrid_is_not_a_candidate() -> None:
    """A free part that swallows the whole summand leaves no remainder to Trotterize,
    so the hybrid would be the exact route relabelled. It must not be offered --
    whichever route then wins the pricing, and the cheapest here is not always the
    exact one."""
    h = model("tfim", 4, seed=0)
    free, rest = free_part(h)
    assert free is not None and not terms_of(rest)
    for error in (1e-3, 1e-10):
        assert "exact-in-step" not in synthesize(h, time=1.0, error=error).routes


def test_hybrid_mechanism_meets_the_budget() -> None:
    """The free-part-as-summand formula delivers the error it was sized for.

    Tested as a mechanism, not a route: on 2-local chains the pair-kernel formula
    is cheaper and the router rightly prefers it, but the hybrid stays reachable
    for inputs the pair clustering declines, so its budget claim keeps its test.
    """
    from lizzy._routing import _hybrid_plan

    n = 4
    h = model("heisenberg", n, seed=1)
    free, rest = free_part(h)
    plan = _hybrid_plan(free, rest, 1.0, 1e-3, 1.0, None)
    circuit = plan.circuit
    assert plan.cost <= circuit.two_qubit_gates
    assert infidelity(evolution(h, 1.0), circuit_matrix(circuit, n)) < 1e-3


def test_hybrid_free_network_appears_once_per_step() -> None:
    """The free summand sits last in the symmetric step, so its two halves merge:
    one network per step, not two."""
    from lizzy._routing import _hybrid_plan

    h = model("heisenberg", 4, seed=1)
    free, rest = free_part(h)
    clusters = commuting_clusters(rest) + [free]
    steps = steps_for_clusters(clusters, 1.0, 1e-3)

    circuit = _hybrid_plan(free, rest, 1.0, 1e-3, 1.0, None).circuit
    per_step = Circuit()
    for piece in summands(free):
        per_step.extend(decompose(piece, 1.0 / steps))
    exact_rotations = sum(1 for label in circuit.provenance if label == "exact-in-step")
    assert exact_rotations == steps * len(per_step.rotations)


def test_free_part_declines_dense_families_within_budget() -> None:
    """All-to-all XX+YY is not hopping -- the Jordan-Wigner strings beyond nearest
    neighbours make it exponential -- so the right answer is no, and the budget makes
    that answer cheap instead of classifying n^2-term candidates to hear it."""
    h = model("heisenberg_all_to_all", 30, seed=0)
    free, rest = free_part(h)
    assert free is None
    assert len(terms_of(rest)) == len(terms_of(h))


def test_folding_keeps_each_rotation_with_its_own_route() -> None:
    """Merging shortens the circuit, so a route label indexed by the output position
    slides onto the wrong rotation and the last one falls off the end entirely.

    The count survives that -- the labels are a permutation of themselves -- which is
    why the attribution has to be checked rather than the total.
    """
    from lizzy.hamiltonian import fold_phases

    circuit = Circuit()
    circuit.add(get_pauli_string("XXI"), 0.1, "first")
    circuit.add(get_pauli_string("XXI"), 0.1, "first")  # merges into the one before
    circuit.add(get_pauli_string("IZZ"), 0.3, "second")
    circuit.add(get_pauli_string("ZII"), 0.4, "third")

    folded = fold_phases(circuit)
    assert [str(p) for p, _ in folded.rotations] == ["XXI", "IZZ", "ZII"]
    assert folded.provenance == ["first", "second", "third"]


def test_folding_merges_only_across_commuting_rotations() -> None:
    """Two rotations about the same Pauli come together only when everything between
    them commutes with it; a barrier keeps them apart, however far back the match is."""
    from lizzy.hamiltonian import fold_phases

    barred = Circuit()
    barred.add(get_pauli_string("XX"), 0.3, "r")
    barred.add(get_pauli_string("ZI"), 0.2, "r")  # anticommutes with XX
    barred.add(get_pauli_string("XX"), 0.4, "r")
    assert len(fold_phases(barred)) == 3

    clear = Circuit()
    clear.add(get_pauli_string("XX"), 0.3, "r")
    clear.add(get_pauli_string("YY"), 0.2, "r")  # commutes with XX
    clear.add(get_pauli_string("XX"), 0.4, "r")
    folded = fold_phases(clear)
    assert len(folded) == 2
    assert folded.rotations[0][1] == pytest.approx(0.7)


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
    assert not result.error_guaranteed

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
    assert deep.routes and set(deep.routes) <= {"exact-bdi", "exact-givens"}
    assert deep.error_guaranteed


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


def test_a_commuting_answer_is_retried_rather_than_returned_as_none() -> None:
    """Ordering families by weight can grow a set that is entirely commuting, which is
    then discarded -- extracting it would remove no commutator from the bound -- so the
    search reports nothing after doing all of the work.

    Here the heavy Z family builds exactly that set, and holding it back until
    something non-commuting has been found turns the answer from none into five of the
    seven terms. This is the shape LiH has at six hundred terms.
    """
    terms = {
        "IIZI": 9.0,
        "ZIZI": 9.0,
        "ZIZZ": 9.0,
        "IIYI": 1.0,
        "XYZI": 1.0,
        "YIII": 1.0,
        "ZIIZ": 1.0,
    }

    free, rest = free_part(hamiltonian(terms))
    assert free is not None
    assert anticommutation_matrix([p for _, p in terms_of(free)]).any()
    assert all(is_decomposable(part) for part in summands(free))
    assert len(terms_of(free)) + len(terms_of(rest)) == len(terms)


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
    assert not result.error_guaranteed
    assert "qdrift" in result.routes
    assert not synthesize(h, time=1.0, error=1e-1, order=4, seed=0).randomized


def test_cluster_route_is_cheaper_and_taken() -> None:
    """On a Heisenberg model a clustered second-order step beats the chain-bounded
    formula, and the router follows the arithmetic."""
    h = model("heisenberg_all_to_all", 5, seed=1)
    chain = product_formula_cost(h, 1.0, 1e-3, 4)

    result = synthesize(h, time=1.0, error=1e-3, order=4)
    assert "trotter2" in result.routes
    assert not result.randomized
    assert result.error_guaranteed
    assert result.two_qubit_gates < chain
    assert infidelity(evolution(h, 1.0), circuit_matrix(result.circuit, 5)) < 1e-3


def test_calibration_shrinks_the_circuit_without_claiming_a_bound() -> None:
    """An explicitly relaxed formula budget must not retain its guarantee."""
    h = model("heisenberg_all_to_all", 5, seed=1)
    base = synthesize(h, time=1.0, error=1e-3, order=4)
    tuned = synthesize(h, time=1.0, error=1e-3, order=4, calibration=4.0)

    assert tuned.two_qubit_gates < base.two_qubit_gates
    assert not tuned.error_guaranteed


def test_auto_synthesis_preserves_exact_multipart_evolution() -> None:
    """A tight budget selects exact evolution across commuting XY summands."""
    n, time = 3, 5.0
    h = model("xy", n, seed=1)
    result = synthesize(h, time=time, error=1e-10)
    assert result.summands == 2
    assert result.routes and set(result.routes) <= {"exact-bdi", "exact-givens"}
    assert infidelity(evolution(h, time), circuit_matrix(result.circuit, n)) < 1e-10


def test_synthesize_reports_the_route_it_took() -> None:
    """The account of the route is what makes the cost attributable."""
    result = synthesize(model("tfim", 5, seed=0), time=1.0)
    assert result.algebra == "so(10)"
    assert result.summands == 1
    assert result.symmetries >= 1
    assert result.routes and set(result.circuit.cost_by_route()) <= {"exact-bdi", "exact-givens"}
    assert sum(result.circuit.cost_by_route().values()) == result.logical_two_qubit_gates
    assert result.two_qubit_gates <= result.logical_two_qubit_gates
    assert result.emitted_circuit is result.emission.circuit
