"""Product-formula convergence, certified bounds, and sampling invariants."""

import numpy as np
import pytest

from lizzy.algebra.symmetry import commuting_clusters
from lizzy.dense import (
    circuit_matrix,
    evolution,
    infidelity,
    pauli_matrix,
)
from lizzy.hamiltonian import (
    hamiltonian,
    model,
    terms_of,
)
from lizzy.synthesis.trotter import (
    _collected_commutator,
    cluster_formula,
    coefficient_norm,
    commutator_sum,
    nested_commutator_sum,
    product_formula,
    product_formula_cost,
    qdrift,
    qdrift_cost,
    split_by_magnitude,
    steps_for,
    steps_for_clusters,
)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_product_formula_converges(order: int) -> None:
    """More steps must reduce dense-reference error for every supported order."""
    h = model("heisenberg", 3, seed=0)
    target = evolution(h, 1.0)

    errors = [
        infidelity(target, circuit_matrix(product_formula(h, 1.0, steps, order), 3))
        for steps in (2, 8, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_commutator_sum_counts_only_anticommuting_pairs() -> None:
    """Commuting terms contribute nothing to the first-order error."""
    assert commutator_sum(hamiltonian({"ZII": 1.0, "IZI": 1.0})) == 0.0
    assert commutator_sum(hamiltonian({"XII": 1.0, "ZII": 1.0})) == pytest.approx(2.0)


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


def _dense_sum(terms) -> np.ndarray:
    matrix = None
    for coefficient, pauli in terms:
        term = complex(coefficient) * pauli_matrix(str(pauli))
        matrix = term if matrix is None else matrix + term
    return matrix


def test_collected_commutator_reproduces_the_dense_one() -> None:
    """The collected Pauli sum must equal [S, [S, A]] as matrices, phases included.

    The whole point of collecting is that opposite-phase chains cancel before the norm
    is taken, so a phase convention error would silently destroy the bound.
    """
    clusters = commuting_clusters(model("heisenberg_all_to_all", 3, seed=1))
    head = [(c.real, p) for c, p in terms_of(clusters[0])]
    tail = [(c.real, p) for cl in clusters[1:] for c, p in terms_of(cl)]

    inner = _collected_commutator(tail, head, [0], 10**9)
    outer = _collected_commutator(tail, inner, [0], 10**9)

    s, a = _dense_sum(tail), _dense_sum(head)
    dense = s @ (s @ a - a @ s) - (s @ a - a @ s) @ s
    assert np.allclose(_dense_sum(outer), dense)


@pytest.mark.parametrize("name,n", [("heisenberg", 3), ("heisenberg_all_to_all", 4)])
def test_cluster_formula_meets_the_budget_it_was_sized_for(name: str, n: int) -> None:
    """The certified second-order step count must actually deliver the error."""
    h = model(name, n, seed=1)
    clusters = commuting_clusters(h)
    steps = steps_for_clusters(clusters, 1.0, 1e-3)
    assert steps is not None

    circuit = cluster_formula(clusters, 1.0, steps)
    assert infidelity(evolution(h, 1.0), circuit_matrix(circuit, n)) < 1e-3


def test_negative_time_preserves_step_counts_and_reverses_evolution() -> None:
    """Sizing depends on |time| while emitted rotations retain its sign."""
    h = hamiltonian({"X": 0.7, "Z": -0.4})
    time, error = 0.8, 1e-3
    clusters = commuting_clusters(h)
    cluster_steps = steps_for_clusters(clusters, -time, error)
    assert cluster_steps == steps_for_clusters(clusters, time, error)
    chain_steps = steps_for(h, -time, error, order=4)
    assert chain_steps == steps_for(h, time, error, order=4)

    forward = cluster_formula(clusters, time, cluster_steps)
    backward = cluster_formula(clusters, -time, cluster_steps)
    assert np.allclose(
        circuit_matrix(backward, 1), circuit_matrix(forward, 1).conj().T
    )
    target = evolution(h, -time)
    assert np.linalg.norm(target - circuit_matrix(backward, 1), ord=2) <= error
    fourth_order = product_formula(h, -time, chain_steps, order=4)
    assert np.linalg.norm(target - circuit_matrix(fourth_order, 1), ord=2) <= error


def test_explicit_formula_controls_reject_invalid_counts_and_orders() -> None:
    """Invalid controls must fail instead of silently emitting an empty circuit."""
    h = hamiltonian({"X": 1.0})
    with pytest.raises(ValueError, match="steps"):
        product_formula(h, 1.0, steps=-1)
    with pytest.raises(ValueError, match="steps"):
        cluster_formula([h], 1.0, steps=0)
    with pytest.raises(ValueError, match="steps"):
        product_formula(h, 1.0, steps=True)
    with pytest.raises(ValueError, match="order"):
        product_formula(h, 1.0, steps=1, order=3)
    with pytest.raises(ValueError, match="order"):
        steps_for(h, 1.0, 1e-3, order=2.0)

    circuit = product_formula(h, -0.3, steps=np.int64(2), order=np.int64(2))
    assert np.allclose(circuit_matrix(circuit, 1), evolution(h, -0.3))


def test_sizing_controls_are_validated_before_exact_and_empty_shortcuts() -> None:
    """Trivial inputs must not hide an invalid error budget or calibration."""
    h = hamiltonian({"X": 1.0})
    empty = hamiltonian({})
    with pytest.raises(ValueError, match="error"):
        steps_for(h, 1.0, error=0.0)
    with pytest.raises(ValueError, match="calibration"):
        steps_for_clusters([h], 1.0, 1e-3, calibration=float("nan"))
    with pytest.raises(ValueError, match="calibration"):
        product_formula_cost(empty, 1.0, 1e-3, calibration=-1.0)
    with pytest.raises(ValueError, match="error"):
        qdrift(empty, 1.0, error=-1.0)
    with pytest.raises(ValueError, match="error"):
        qdrift_cost(empty, 1.0, error=float("inf"))
    with pytest.raises(ValueError, match="time"):
        cluster_formula([h], float("nan"), steps=1)


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
    spread = hamiltonian(
        dict.fromkeys(
            [
                "XXII",
                "IXXI",
                "IIXX",
                "ZIII",
                "IZII",
                "IIZI",
                "IIIZ",
                "YYII",
                "IYYI",
                "IIYY",
            ],
            0.1,
        )
    )
    assert coefficient_norm(concentrated) == pytest.approx(coefficient_norm(spread))
    assert len(qdrift(concentrated, 1.0, 0.1, seed=0)) == len(
        qdrift(spread, 1.0, 0.1, seed=0)
    )
