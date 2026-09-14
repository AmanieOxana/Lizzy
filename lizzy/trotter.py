"""
    Product formulas and randomized compilation, with the step count derived from an
    error budget rather than passed in.

    Trotter error is bounded by commutator norms over the terms being Trotterized, so
    anything handled exactly elsewhere -- a free subalgebra, a summand, a commuting
    cluster -- drops out of the bound along with its commutators. That is why the exact
    branch reduces the cost of the inexact one.

    qDRIFT's cost depends on the coefficients alone rather than on how many terms there
    are, which is what matters for Hamiltonians with many small terms. Its guarantee is
    on the averaged channel, not on any one sampled circuit.
"""

import math

import numpy as np
from paulie.common.pauli_string_linear import PauliStringLinear

from lizzy.hamiltonian import (
    Circuit,
    anticommutation_matrix,
    hamiltonian,
    rotation_cost,
    terms_of,
)


class _Exhausted(Exception):
    """The chain walk ran out of budget."""


# Cluster error constants by cluster contents; see cluster_error_constant.
_constant_cache: dict = {}


# Suzuki's recursion multiplies the stage count by 5 at each order, so a formula of
# order 2k applies 2 * 5^(k-1) exponentials per term per step.
def _stages(order: int) -> int:
    """Number of exponentials a formula of the given order applies per term per step."""
    if order == 1:
        return 1
    return 2 * 5 ** (order // 2 - 1)


def coefficient_norm(hamiltonian_: PauliStringLinear) -> float:
    r"""
    Get :math:`\Lambda = \sum_i |c_i|`, the quantity randomized compilation pays for.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        float: The sum of absolute coefficients.
    """
    return float(sum(abs(c) for c, _ in terms_of(hamiltonian_)))


def commutator_sum(hamiltonian_: PauliStringLinear) -> float:
    r"""
    Get :math:`\sum_{a<b} \lVert [H_a, H_b] \rVert`, the first-order Trotter error term.

    Two Pauli terms either commute, contributing nothing, or anticommute, in which case
    :math:`[c_aP_a, c_bP_b] = 2c_ac_bP_aP_b` has norm :math:`2|c_ac_b|`. So the sum is
    exact and costs :math:`O(L^{2})`, with no operator norms to estimate.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
    Returns:
        float: The commutator sum.
    """
    terms = terms_of(hamiltonian_)
    if len(terms) < 2:
        return 0.0
    adjacency = anticommutation_matrix([p for _, p in terms])
    magnitudes = np.array([abs(c) for c, _ in terms])
    return float(magnitudes @ adjacency @ magnitudes)


def nested_commutator_sum(
    hamiltonian_: PauliStringLinear, order: int, budget: int = 400_000
) -> float | None:
    r"""
    Get :math:`\alpha_{\mathrm{comm}}`, the order-:math:`p` Trotter error constant.

    The bound of `Childs et al. <https://doi.org/10.1103/PhysRevX.11.011020>`__ is

    .. math::

        \alpha_{\mathrm{comm}} = \sum_{\gamma_1 \ldots \gamma_{p+1}}
        \bigl\lVert [H_{\gamma_{p+1}}, [\ldots [H_{\gamma_2}, H_{\gamma_1}]]] \bigr\rVert

    which looks like :math:`O(L^{p+1})` work. For Pauli terms it is not. A nested
    commutator of Pauli strings is either zero or a Pauli string again: writing
    :math:`Q_k` for the accumulated product, the next commutator vanishes unless
    :math:`Q_k` and :math:`P_{\gamma_{k+1}}` anticommute, and when it does not vanish
    its norm is exactly :math:`2^{p}\prod_j |c_{\gamma_j}|`. The sum therefore runs over
    *anticommuting chains* only, each contributing a closed-form weight, with no operator
    norms to estimate.

    A term can only anticommute with :math:`Q_k` if it touches its support, so the
    branching is bounded by the local connectivity rather than by :math:`L`, and for a
    geometrically local Hamiltonian the chain count grows linearly in the qubit count.
    Hamiltonians without that locality can still blow up, so the walk is capped; hitting
    the cap returns ``None`` rather than a truncated number, and :func:`steps_for` falls
    back to its coarser bound.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        order (int): Formula order :math:`p`; chains have length :math:`p + 1`.
        budget (int): Maximum chain extensions to explore before giving up.
    Returns:
        float | None: The sum, or ``None`` if the budget was exhausted.
    """
    terms = terms_of(hamiltonian_)
    if not terms:
        return 0.0

    by_qubit: dict[int, list[int]] = {}
    for index, (_, pauli) in enumerate(terms):
        for qubit in pauli.get_support():
            by_qubit.setdefault(qubit, []).append(index)

    total = 0.0
    spent = 0

    def extend(product, weight: float, remaining: int) -> None:
        """Accumulate the weight of every chain extending ``product``."""
        nonlocal total, spent
        if remaining == 0:
            total += weight
            return

        for index in {i for q in product.get_support() for i in by_qubit.get(q, ())}:
            coefficient, pauli = terms[index]
            if product.commutes_with(pauli):
                continue
            spent += 1
            if spent > budget:
                raise _Exhausted
            extend(product @ pauli, 2 * weight * abs(coefficient), remaining - 1)

    try:
        for coefficient, pauli in terms:
            extend(pauli, abs(coefficient), order)
    except _Exhausted:
        return None
    return total


def _collected_commutator(
    left: list[tuple[complex, object]],
    right: list[tuple[complex, object]],
    spent: list[int],
    budget: int,
) -> list[tuple[complex, object]]:
    r"""
    Get :math:`[L, R]` of two Pauli sums as a single collected Pauli sum.

    Anticommuting terms contribute :math:`[c_lP_l, c_rP_r] = 2c_lc_r\,\sigma\,\hat{P}`
    with :math:`\hat{P}` the phaseless product and :math:`\sigma` its exact phase, so
    contributions landing on the same Pauli string cancel *before* any norm is taken.
    That cancellation is precisely what a chain-by-chain triangle inequality discards.
    """
    by_qubit: dict[int, list[int]] = {}
    for index, (_, pauli) in enumerate(left):
        for qubit in pauli.get_support():
            by_qubit.setdefault(qubit, []).append(index)

    out: dict[object, complex] = {}
    for c_r, p_r in right:
        for index in {i for q in p_r.get_support() for i in by_qubit.get(q, ())}:
            c_l, p_l = left[index]
            if p_l.commutes_with(p_r):
                continue
            spent[0] += 1
            if spent[0] > budget:
                raise _Exhausted
            product = p_l @ p_r
            out[product] = out.get(product, 0.0) + 2 * c_l * c_r * p_l.sign(p_r)
    return [(c, p) for p, c in out.items() if abs(c) > 1e-30]


def cluster_error_constant(clusters: list, budget: int = 300_000) -> float | None:
    r"""
    Get the tight second-order Trotter constant over a cluster ordering.

    For the symmetric second-order formula with summands :math:`H_1, \ldots, H_L`,
    `Childs et al. <https://doi.org/10.1103/PhysRevX.11.011020>`__ (Prop. 9) bound one
    step of size :math:`\tau` by

    .. math::

        \tau^{3}\sum_{\gamma}\Bigl(\tfrac{1}{12}\lVert[S_\gamma,[S_\gamma,H_\gamma]]\rVert
        + \tfrac{1}{24}\lVert[H_\gamma,[H_\gamma,S_\gamma]]\rVert\Bigr),
        \qquad S_\gamma = \sum_{\gamma' > \gamma} H_{\gamma'} .

    Taking the summands to be the commuting clusters does two things at once: the sum
    has as many terms as there are clusters -- at most three for a Heisenberg model of any size
    -- and every norm is of a *collected* commutator, so cancellations between chains
    survive. Each norm is bounded by the collected coefficient 1-norm, which keeps the
    whole computation polynomial. The honest prefactors and the surviving cancellation
    are what the chain bound of :func:`nested_commutator_sum` gives away.

    The constant depends only on the clusters, not on the time or the budget being
    asked about, so it is cached: the calibration bisection and benchmark sweeps over
    evolution times ask for the same constant many times over, and the walk is the
    expensive part.

    Args:
        clusters (list[PauliStringLinear]): The commuting clusters, in the order the
            formula will apply them.
        budget (int): Maximum anticommuting products to expand; ``None`` beyond it.
            The default keeps the give-up path at seconds -- the collected walk costs
            far more per product than the plain chain walk, and a bound that takes
            minutes to say "no answer" is worse than none.
    Returns:
        float | None: The constant, or ``None`` if the budget was exhausted.
    """
    key = (
        tuple(
            tuple((str(p), complex(c)) for c, p in terms_of(cluster))
            for cluster in clusters
        ),
        budget,
    )
    if key in _constant_cache:
        return _constant_cache[key]

    spent = [0]
    total = 0.0
    try:
        for g in range(len(clusters) - 1):
            head = [(c.real, p) for c, p in terms_of(clusters[g])]
            tail = [
                (c.real, p)
                for cluster in clusters[g + 1 :]
                for c, p in terms_of(cluster)
            ]
            inner = _collected_commutator(tail, head, spent, budget)  # [S, A]
            outer_s = _collected_commutator(tail, inner, spent, budget)  # [S, [S, A]]
            outer_a = _collected_commutator(head, inner, spent, budget)  # -[A, [A, S]]
            total += sum(abs(c) for c, _ in outer_s) / 12
            total += sum(abs(c) for c, _ in outer_a) / 24
    except _Exhausted:
        _constant_cache[key] = None
        return None
    _constant_cache[key] = total
    return total


def steps_for_clusters(
    clusters: list, time: float, error: float, calibration: float = 1.0
) -> int | None:
    """
    Get the second-order step count the cluster bound certifies.

    Args:
        clusters (list[PauliStringLinear]): The commuting clusters, in formula order.
        time (float): Evolution time.
        error (float): Target error.
        calibration (float): Divides the count, as in :func:`steps_for`.
    Returns:
        int | None: Number of steps, or ``None`` if the constant was not computable
        within budget.
    """
    constant = cluster_error_constant(clusters)
    if constant is None:
        return None
    if constant == 0.0:
        return 1
    return max(1, math.ceil(math.sqrt(constant * time**3 / error) / calibration))


def cluster_formula(
    clusters: list,
    time: float,
    steps: int,
    route: str = "trotter2",
    middle: Circuit | None = None,
    compile_cluster=None,
) -> Circuit:
    r"""
    Synthesize :math:`e^{-itH}` as the symmetric second-order formula over summands.

    One step applies the leading summands forward at half the step time, the last
    summand once in the middle at the full step time, and the leading ones backward.
    That is exactly :math:`S_2`: the last summand's two adjacent half-passes merge
    because it exponentiates error-free, and merging saves its rotations once per
    step.

    Each summand's circuit is built once and replayed across steps -- every step
    uses the same step time. ``compile_cluster`` supplies the per-summand emission
    (kernel layers use :func:`lizzy.kernels.compile_layer`); the default is one
    plain rotation per term, correct exactly when the summand commutes internally.
    ``middle`` overrides the last summand's emission outright, which is how the free
    part's Givens network enters.

    Args:
        clusters (list[PauliStringLinear]): The summands, in formula order.
        time (float): Evolution time.
        steps (int): Number of steps.
        route (str): Label recorded against the rotations produced.
        middle (Circuit, optional): Replacement emission for the last summand.
        compile_cluster (callable, optional): ``(summand, tau, route) -> Circuit``.
    Returns:
        Circuit: The rotations.
    """
    if compile_cluster is None:

        def compile_cluster(cluster, tau, label):
            built = Circuit()
            for coefficient, pauli in terms_of(cluster):
                built.add(pauli, coefficient.real * tau, label)
            return built

    halves = [
        compile_cluster(cluster, 0.5 * time / steps, route)
        for cluster in clusters[:-1]
    ]
    if middle is None:
        middle = compile_cluster(clusters[-1], time / steps, route)

    circuit = Circuit()
    for _ in range(steps):
        for built in halves:
            circuit.extend(built)
        circuit.extend(middle)
        for built in reversed(halves):
            circuit.extend(built)
    return circuit


def _steps_from(
    residual: float, time: float, error: float, order: int, calibration: float = 1.0
) -> int:
    r"""Turn an error constant into a step count.

    The error of ``N`` steps is :math:`\alpha t^{p+1}/N^{p}`, halved at first order
    where :math:`\alpha` is the pairwise commutator sum. A residual of zero means the
    formula is already exact. ``calibration`` divides the result.
    """
    if residual <= 0.0:
        return 1
    if order == 1:
        steps = residual * time**2 / (2 * error)
    else:
        steps = (residual * time ** (order + 1) / error) ** (1 / order)
    return max(1, math.ceil(steps / calibration))


def steps_for(
    hamiltonian_: PauliStringLinear,
    time: float,
    error: float,
    order: int = 2,
    calibration: float = 1.0,
) -> int:
    r"""
    Get the number of product-formula steps that fits an error budget.

    At first order the bound is the rigorous commutator one,
    :math:`\varepsilon \le t^{2}/(2N) \sum_{a<b}\lVert[H_a,H_b]\rVert`, so the step
    count follows directly.

    Above first order the error is :math:`\alpha_{\mathrm{comm}}t^{p+1}/N^{p}` with
    :math:`\alpha_{\mathrm{comm}}` from :func:`nested_commutator_sum`. On a
    geometrically local Hamiltonian only neighbouring terms anticommute, so
    :math:`\alpha_{\mathrm{comm}}` grows linearly in the qubit count where
    :math:`\Lambda^{p+1}` grows as a power of it -- the gain is a factor of n, not a
    constant. When the chain walk exceeds its budget this falls back to
    :math:`\bigl(\sum_{a<b}\lVert[H_a,H_b]\rVert\bigr)\Lambda^{p-1}`, which still uses
    the commutator structure once instead of not at all.

    The formula-dependent prefactor is taken as one throughout, so **the result is an
    estimate, not a certificate**. :mod:`lizzy.bench` checks the error actually
    achieved against a dense reference wherever the qubit count allows.

    Because the bound sums over every chain with a triangle inequality and the true error
    enjoys cancellation between them, the estimate overshoots -- on these models by 12x
    to 54x against the fewest steps that in fact meet the budget. ``calibration`` divides
    it by a measured factor. That factor is an observation on one instance, not a proof
    about another, so a calibrated count is a measurement rather than a bound; see
    :func:`lizzy.bench.calibrate`.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian being Trotterized.
        time (float): Evolution time.
        error (float): Target error.
        order (int): Formula order: 1, or an even Suzuki order such as 2 or 4.
        calibration (float): Divide the estimated count by this. One leaves the bound
            alone; larger values trade the guarantee for a measured factor.
    Returns:
        int: Number of steps, at least one.

    Raises:
        ValueError: If the order is neither 1 nor a positive even number, or if
            ``calibration`` is not positive.
    """
    if order < 1 or (order > 1 and order % 2):
        raise ValueError(f"Order must be 1 or a positive even number, got {order}.")
    if calibration <= 0:
        raise ValueError(f"Calibration must be positive, got {calibration}.")

    pairwise = commutator_sum(hamiltonian_)
    if order == 1:
        return _steps_from(pairwise, time, error, order, calibration)

    alpha = nested_commutator_sum(hamiltonian_, order)
    if alpha is None:
        alpha = pairwise * coefficient_norm(hamiltonian_) ** (order - 1)
    return _steps_from(alpha, time, error, order, calibration)


def _suzuki_sequence(count: int, order: int) -> list[tuple[int, float]]:
    """Build the (term index, time fraction) sequence of one Suzuki step."""
    if order == 1:
        return [(i, 1.0) for i in range(count)]
    if order == 2:
        return [(i, 0.5) for i in range(count)] + [
            (i, 0.5) for i in reversed(range(count))
        ]

    factor = 1 / (4 - 4 ** (1 / (order - 1)))
    inner = _suzuki_sequence(count, order - 2)
    sequence = []
    for weight in (factor, factor, 1 - 4 * factor, factor, factor):
        sequence += [(i, fraction * weight) for i, fraction in inner]
    return sequence


def product_formula(
    hamiltonian_: PauliStringLinear,
    time: float,
    steps: int,
    order: int = 2,
    route: str = "trotter",
) -> Circuit:
    r"""
    Synthesize :math:`e^{-itH}` as a product formula.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        steps (int): Number of steps.
        order (int): Formula order: 1, or an even Suzuki order.
        route (str): Label recorded against the rotations produced.
    Returns:
        Circuit: The rotations.
    """
    terms = terms_of(hamiltonian_)
    sequence = _suzuki_sequence(len(terms), order)

    circuit = Circuit()
    for _ in range(steps):
        for index, fraction in sequence:
            coefficient, pauli = terms[index]
            circuit.add(pauli, coefficient.real * fraction * time / steps, route)
    return circuit


def qdrift(
    hamiltonian_: PauliStringLinear,
    time: float,
    error: float,
    seed: int | None = None,
    route: str = "qdrift",
) -> Circuit:
    r"""
    Synthesize :math:`e^{-itH}` by randomly sampling terms.

    Terms are drawn with probability :math:`|c_i|/\Lambda` and applied with a fixed
    angle, so the channel converges to the target evolution in
    :math:`\lceil 2\Lambda^{2}t^{2}/\varepsilon \rceil` gates -- a count that does not
    depend on how many terms the Hamiltonian has. That independence is the point: it is
    what a deterministic product formula, which pays a gate per term per step, cannot
    offer.

    The guarantee differs in kind from a product formula's. The error is a diamond-norm
    bound on the *channel*, in expectation over the sampling, rather than a bound on a
    single circuit. Estimating an observable this way means averaging over draws.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        error (float): Target diamond-norm error.
        seed (int, optional): Seed for the sampling.
        route (str): Label recorded against the rotations produced.
    Returns:
        Circuit: One sampled circuit.
    """
    terms = terms_of(hamiltonian_)
    norm = coefficient_norm(hamiltonian_)
    if norm == 0.0:
        return Circuit()

    gates = max(1, math.ceil(2 * norm**2 * time**2 / error))
    weights = np.array([abs(c) for c, _ in terms]) / norm
    rng = np.random.default_rng(seed)

    circuit = Circuit()
    for index in rng.choice(len(terms), size=gates, p=weights):
        coefficient, pauli = terms[index]
        angle = norm * time / gates * (1 if coefficient.real >= 0 else -1)
        circuit.add(pauli, angle, route)
    return circuit


def split_by_magnitude(
    hamiltonian_: PauliStringLinear, fraction: float = 0.5
) -> tuple[PauliStringLinear, PauliStringLinear]:
    r"""
    Split a Hamiltonian into its large and small terms.

    A composite scheme runs the large terms through a product formula and samples the
    small ones, which beats either method alone when the coefficients are spread out.
    The cut keeps the terms carrying ``fraction`` of the total weight
    :math:`\Lambda` on the deterministic side.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        fraction (float): Share of :math:`\Lambda` to place on the deterministic side.
    Returns:
        tuple[PauliStringLinear, PauliStringLinear]: The large part and the small part.
        Either may be empty.
    """
    terms = sorted(terms_of(hamiltonian_), key=lambda t: -abs(t[0]))
    budget = fraction * coefficient_norm(hamiltonian_)

    large: list[tuple[str, complex]] = []
    small: list[tuple[str, complex]] = []
    running = 0.0
    for coefficient, pauli in terms:
        if running < budget:
            large.append((str(pauli), coefficient))
            running += abs(coefficient)
        else:
            small.append((str(pauli), coefficient))

    return hamiltonian(large), hamiltonian(small)


def product_formula_cost(
    hamiltonian_: PauliStringLinear,
    time: float,
    error: float,
    order: int = 2,
    calibration: float = 1.0,
) -> int:
    """
    Estimate the two-qubit gate count of a product formula, without building it.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        error (float): Target error.
        order (int): Formula order.
        calibration (float): Divides the step count, as in :func:`steps_for`.
    Returns:
        int: Estimated two-qubit gate count.
    """
    terms = terms_of(hamiltonian_)
    if not terms:
        return 0
    per_pass = sum(rotation_cost(p) for _, p in terms)
    steps = steps_for(hamiltonian_, time, error, order, calibration)
    return steps * _stages(order) * per_pass


def qdrift_cost(hamiltonian_: PauliStringLinear, time: float, error: float) -> int:
    r"""
    Estimate the two-qubit gate count of a qDRIFT channel, without sampling it.

    The gate count is :math:`\lceil 2\Lambda^{2}t^{2}/\varepsilon \rceil`, so this grows
    like :math:`\varepsilon^{-1}` where a product formula of order :math:`p` grows like
    :math:`\varepsilon^{-1/p}`. Sampling therefore wins at loose budgets and loses at
    tight ones, and which side of that crossover an instance falls on is a question of
    arithmetic rather than of judgement.

    Args:
        hamiltonian_ (PauliStringLinear): The Hamiltonian.
        time (float): Evolution time.
        error (float): Target error.
    Returns:
        int: Estimated two-qubit gate count.
    """
    terms = terms_of(hamiltonian_)
    norm = coefficient_norm(hamiltonian_)
    if not terms or norm == 0.0:
        return 0
    gates = max(1, math.ceil(2 * norm**2 * time**2 / error))
    mean_cost = sum(rotation_cost(p) for _, p in terms) / len(terms)
    return math.ceil(gates * mean_cost)
