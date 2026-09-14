"""
    Benchmark harness.

    Reports what each lever actually bought, per instance, rather than a single gate
    count: which route was taken, how many qubits the symmetries would remove, how many
    commuting clusters a Trotter step needs, and where the exact branch overtakes the
    product formula. Randomized routes are averaged over seeds and flagged because
    the sampled composite does not retain the requested error guarantee.

    Run with ``python -m lizzy.bench``.
"""

import argparse
import statistics

from lizzy.dense import circuit_matrix, evolution, infidelity
from lizzy.hamiltonian import model, n_qubits, terms_of
from lizzy.synthesize import synthesize
from lizzy.trotter import product_formula, steps_for

# Above this the dense reference needs more memory than it is worth.
VERIFIABLE_QUBITS = 10

# Verifying a circuit costs one pass over a 2**n matrix per rotation, so long circuits
# are reported unverified rather than left to run for hours. The same cap keeps the
# all-product-formula baseline from being built when it would run to millions of gates.
VERIFIABLE_ROTATIONS = 20_000

# The calibration search never divides the step count by more than this, and spends this
# many bisections narrowing it down. Each bisection costs one synthesis and one dense
# comparison, so the count trades resolution against the time calibration takes.
MAX_CALIBRATION = 64.0
CALIBRATION_BISECTIONS = 8


def run(
    name: str,
    n: int,
    time: float,
    error: float,
    order: int = 4,
    seeds: int = 5,
    verify: bool = True,
    randomized: bool = False,
) -> dict:
    """
    Synthesize one model and measure what it cost.

    Args:
        name (str): Model family, as accepted by :func:`lizzy.hamiltonian.model`.
        n (int): Number of qubits.
        time (float): Evolution time.
        error (float): Error budget.
        order (int): Product-formula order for the inexact branch.
        seeds (int): Repeats, which matter only for routes that sample.
        verify (bool): Check the circuit against a dense reference where feasible.
        randomized (bool): Allow the sampled branch, which does not keep its budget.
    Returns:
        dict: One row of results.
    """
    hamiltonian_ = model(name, n, seed=0)
    counts, achieved = [], []

    for seed in range(seeds):
        result = synthesize(
            hamiltonian_,
            time=time,
            error=error,
            order=order,
            seed=seed,
            randomized=randomized,
        )
        counts.append(result.two_qubit_gates)
        if verify and n <= VERIFIABLE_QUBITS and len(result.circuit) <= VERIFIABLE_ROTATIONS:
            achieved.append(
                infidelity(
                    evolution(hamiltonian_, time), circuit_matrix(result.circuit, n)
                )
            )
        if not result.randomized:
            break  # A deterministic route gives the same answer every time.

    row = {
        "model": name,
        "n": n,
        "t": time,
        "eps": error,
        "algebra": result.algebra,
        "terms": len(terms_of(hamiltonian_)),
        "summands": result.summands,
        "symmetries": result.symmetries,
        "clusters": result.clusters,
        "routes": ",".join(result.routes),
        "emission_backend": result.emission_backend,
        "randomized": result.randomized,
        "error_guaranteed": result.error_guaranteed,
        "routing_estimated": result.routing_estimated,
        "two_qubit": statistics.mean(counts),
        "two_qubit_sd": statistics.stdev(counts) if len(counts) > 1 else 0.0,
        "logical_two_qubit": result.logical_two_qubit_gates,
        # Provenance belongs to the logical sequence. A shared-frame backend can move
        # gates across route boundaries, so this need not sum to ``two_qubit``.
        "by_route": result.circuit.cost_by_route(),
        "achieved": statistics.mean(achieved) if achieved else None,
        # A sampled circuit's infidelity is not what qDRIFT bounds: the guarantee is on
        # the average channel, so a single draw says nothing about whether the budget
        # was met. Checking it here would compare two different objects.
        "met_budget": (
            None if result.randomized or not achieved else max(achieved) < error
        ),
    }

    if n <= VERIFIABLE_QUBITS:
        row["trotter_only"] = _product_formula_only(hamiltonian_, time, error, order)
    return row


def calibrate(
    hamiltonian_, time: float, error: float, order: int = 4, margin: float = 2.0
) -> float:
    r"""
    Measure how far the step-count bound overshoots on one instance.

    :func:`lizzy.trotter.steps_for` sums a bound over every anticommuting chain with
    a triangle inequality; the true error cancels between them, so the count it returns
    is larger than the smallest one that meets the budget -- here by 12x to 54x. This
    finds how far it can be divided before the budget is actually missed, and returns
    that factor for :func:`lizzy.synthesize.synthesize` to take as ``calibration``.

    The search runs against ``synthesize`` rather than against a bare product formula,
    because the routes do not share an error mechanism: a Hamiltonian whose free part is
    compiled exactly inside each step tolerates a very different divisor from one that
    goes straight through a product formula, and a factor measured on the wrong route
    misses the budget at every size.

    **The result is a measurement, not a bound.** It holds for the instance it was taken
    on; reusing it elsewhere assumes the overshoot is a property of the family, which is
    plausible for a translation-invariant model and unproven in general. ``margin``
    divides the measured factor to keep part of the gap in reserve.

    Args:
        hamiltonian_ (PauliStringLinear): The instance to measure on. Must be small
            enough for a dense reference.
        time (float): Evolution time.
        error (float): Error budget.
        order (int): Formula order.
        margin (float): Safety divisor on the measured factor.
    Returns:
        float: The factor to divide step counts by, at least one.
    """
    n = n_qubits(hamiltonian_)
    if n > VERIFIABLE_QUBITS:
        raise ValueError(
            f"Calibration needs a dense reference, so at most {VERIFIABLE_QUBITS} "
            f"qubits, got {n}."
        )

    target = evolution(hamiltonian_, time)

    def misses_budget(factor: float) -> bool:
        result = synthesize(
            hamiltonian_, time=time, error=error, order=order, calibration=factor
        )
        return infidelity(target, circuit_matrix(result.circuit, n)) >= error

    if misses_budget(1.0):
        return 1.0  # The bound is not overshooting on this instance.

    low, high = 1.0, float(MAX_CALIBRATION)
    for _ in range(CALIBRATION_BISECTIONS):
        middle = (low + high) / 2
        low, high = (low, middle) if misses_budget(middle) else (middle, high)

    return max(1.0, low / margin)


def _product_formula_only(hamiltonian_, time, error, order) -> int | None:
    """Gate count of putting everything through a product formula, for comparison."""
    steps = steps_for(hamiltonian_, time, error, order)
    if steps * len(terms_of(hamiltonian_)) > VERIFIABLE_ROTATIONS:
        return None  # Building it would cost more than the comparison is worth.
    return product_formula(hamiltonian_, time, steps, order).two_qubit_gates


def _print(rows: list[dict]) -> None:
    """Print the rows as a table."""
    header = (
        f"{'model':22}{'n':>3}{'t':>6}{'eps':>7}  {'algebra':12}"
        f"{'sum':>4}{'sym':>4}{'clu':>4}{'2q':>11}{'sd':>7}"
        f"{'trotter-only':>14}{'gain':>9}  {'err':>10}  routes"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        baseline = row.get("trotter_only")
        gain = f"{baseline / row['two_qubit']:.1f}x" if baseline and row["two_qubit"] else "-"
        achieved = f"{row['achieved']:.1e}" if row["achieved"] is not None else "-"
        if row["randomized"] and row["achieved"] is not None:
            achieved += "~"  # one draw, not the channel the bound is about
        flag = "" if row["met_budget"] in (True, None) else " OVER"
        print(
            f"{row['model']:22}{row['n']:>3}{row['t']:>6.1f}{row['eps']:>7.0e}  "
            f"{row['algebra']:12}{row['summands']:>4}{row['symmetries']:>4}"
            f"{row['clusters']:>4}{row['two_qubit']:>11.0f}{row['two_qubit_sd']:>7.0f}"
            f"{(str(baseline) if baseline else '-'):>14}{gain:>9}  {achieved:>10}{flag}  "
            f"{row['routes']} [{row['emission_backend']}]"
        )


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        nargs="+",
        default=["tfim", "tfxy", "xy", "heisenberg", "heisenberg_all_to_all"],
    )
    parser.add_argument("--n", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--time", nargs="+", type=float, default=[1.0, 8.0])
    parser.add_argument("--error", type=float, default=1e-3)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="measure the step-count overshoot at the smallest size and reuse it "
        "(a measurement, not a bound)",
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="allow the sampled branch, which does not keep its error budget",
    )
    args = parser.parse_args()

    rows = [
        run(
            name,
            n,
            time,
            args.error,
            args.order,
            args.seeds,
            verify=not args.no_verify,
            randomized=args.randomized,
        )
        for name in args.family
        for n in args.n
        for time in args.time
    ]
    _print(rows)

    randomized = [r for r in rows if r["randomized"]]
    if randomized:
        print(
            f"\n{len(randomized)} of {len(rows)} rows used sampling, marked ~. qDRIFT "
            "bounds the average channel, so the infidelity of one drawn circuit is a "
            "diagnostic rather than a budget check, and is not flagged against it."
        )


if __name__ == "__main__":
    main()
