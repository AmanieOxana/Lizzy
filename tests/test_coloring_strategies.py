"""Focused tests for selectable commuting-cluster colouring strategies."""

import pytest

from lizzy.hamiltonian import hamiltonian, terms_of
from lizzy.symmetry import commuting_clusters

PAULIS = ["YII", "YIY", "ZYY", "YXX", "XZX", "ZZZ", "YZI"]


@pytest.fixture
def colouring_fixture():
    return hamiltonian([(pauli, index + 1) for index, pauli in enumerate(PAULIS)])


def _words(clusters):
    return [[str(pauli) for _, pauli in terms_of(cluster)] for cluster in clusters]


@pytest.mark.parametrize(
    "strategy",
    ["largest_first", "saturation_largest_first", "DSATUR", "independent_set"],
)
def test_colouring_strategies_are_complete_commuting_and_deterministic(
    colouring_fixture, strategy
) -> None:
    first = commuting_clusters(colouring_fixture, strategy=strategy)
    second = commuting_clusters(colouring_fixture, strategy=strategy)

    assert _words(first) == _words(second)
    assert sorted(word for group in _words(first) for word in group) == sorted(PAULIS)
    assert {
        str(pauli): coefficient
        for cluster in first
        for coefficient, pauli in terms_of(cluster)
    } == {pauli: complex(index + 1) for index, pauli in enumerate(PAULIS)}
    assert [len(group) for group in first] == sorted(
        (len(group) for group in first), reverse=True
    )
    for cluster in first:
        paulis = [pauli for _, pauli in terms_of(cluster)]
        assert all(left.commutes_with(right) for left in paulis for right in paulis)


def test_independent_set_improves_the_fixture_over_the_default(
    colouring_fixture,
) -> None:
    default = commuting_clusters(colouring_fixture)
    explicit_default = commuting_clusters(colouring_fixture, strategy="largest_first")
    independent = commuting_clusters(colouring_fixture, strategy="independent_set")

    assert _words(default) == _words(explicit_default)
    assert len(default) == 4
    assert len(independent) == 3


def test_dsatur_aliases_saturation_largest_first(colouring_fixture) -> None:
    saturation = commuting_clusters(
        colouring_fixture, strategy="saturation_largest_first"
    )
    dsatur = commuting_clusters(colouring_fixture, strategy="DSATUR")

    assert _words(dsatur) == _words(saturation)
