"""Regenerate README figures from saved measurements, without rerunning benchmarks.

From the repository root: python docs/figures/generate.py
Add --preview-dir /tmp/lizzy-figure-previews to save PNGs for visual inspection.
Requires the existing ``plot`` extra; no optional quantum SDK is imported.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
from collections import Counter, defaultdict
from math import ceil, isfinite, log2
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BDI_SOURCE = ROOT / "experiments/bdi_paper_results.json"
DRIVEN_SOURCE = ROOT / "docs/driven_benchmark.txt"
T_SOURCE = ROOT / "experiments/bdi_t_results.json"
COMPILER_SOURCE = ROOT / "experiments/compiler_comparison_results.json"
INK = "#253449"
BDI_COLOR = "#087E8B"
GIVENS_COLOR = "#5665AD"
TIE_COLOR = "#CBD5E1"
BASELINE_COLOR = "#72839B"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "lizzy-saved-results",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _clean_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=12)


def _save(
    fig: plt.Figure,
    name: str,
    *,
    title: str,
    description: str,
    date: str,
    source: Path,
    preview_dir: Path | None,
    publish_png: bool = False,
) -> None:
    provenance = f"Source: {source.relative_to(ROOT)}. Regenerate: python docs/figures/generate.py."
    description = f"{description} {provenance}"
    stream = io.StringIO()
    fig.savefig(stream, format="svg", metadata={
        "Title": title,
        "Description": description,
        "Date": date,
        "Creator": "Lizzy docs/figures/generate.py (Matplotlib)",
    })
    svg = stream.getvalue()
    # Native SVG titles/descriptions complement Matplotlib's RDF metadata.
    svg = svg.replace("<svg ", f'<svg role="img" aria-labelledby="{name}-title {name}-desc" ', 1)
    svg = svg.replace("<title>", f'<title id="{name}-title">', 1)
    root_end = svg.index(">", svg.index("<svg ")) + 1
    accessible = f'\n <desc id="{name}-desc">{html.escape(description)}</desc>'
    (OUTPUT / f"{name}.svg").write_text(svg[:root_end] + accessible + svg[root_end:])
    if publish_png:
        fig.savefig(OUTPUT / f"{name}.png", dpi=160)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Approximately GitHub's README width: check the actual reading size.
        fig.savefig(preview_dir / f"{name}.png", dpi=64)
    plt.close(fig)
    print(f"Generated {(OUTPUT / f'{name}.svg').relative_to(ROOT)} from {source.relative_to(ROOT)}")


def bdi_figure(preview_dir: Path | None) -> None:
    record = json.loads(BDI_SOURCE.read_text())
    pairs = defaultdict(dict)
    for row in record["rows"]:
        if row["method"] in pairs[row["case"]]:
            raise ValueError(f"Duplicate BDI comparison row: {row['case']} / {row['method']}")
        if row["status"] != "PASS" or row["native_frame"]["status"] != "PASS":
            raise ValueError("This summary requires every logical output and artifact to pass")
        pairs[row["case"]][row["method"]] = row
    if set(pairs) != {case["name"] for case in record["cases"]}:
        raise ValueError("BDI comparison rows do not cover the declared cases")
    outcomes = {"ladder": Counter(), "portfolio": Counter()}
    first_ratios, reuse_ratios = [], []
    for pair in pairs.values():
        if set(pair) != {"bdi", "givens"}:
            raise ValueError("Every target needs exactly one BDI and one Givens result")
        bdi, givens = pair["bdi"], pair["givens"]
        for policy in outcomes:
            left = bdi if policy == "ladder" else bdi["native_portfolio"]
            right = givens if policy == "ladder" else givens["native_portfolio"]
            winner = "bdi" if left["cx"] < right["cx"] else "givens" if left["cx"] > right["cx"] else "tie"
            outcomes[policy][winner] += 1
        first_ratios.append(
            bdi["preparation_plus_first_evaluation_seconds"]
            / givens["preparation_plus_first_evaluation_seconds"]
        )
        reuse_ratios.append(
            bdi["prepared_evaluation_median_seconds"]
            / givens["prepared_evaluation_median_seconds"]
        )
    total = len(pairs)
    first, reuse = median(first_ratios), median(reuse_ratios)
    widths = [case["width"] for case in record["cases"]]
    date = record["recorded_on"]
    title = "BDI versus Givens: the trade-off"
    fig = plt.figure(figsize=(11.6, 6.7))
    fig.text(.035, .958, title, fontsize=21, weight="bold")
    fig.text(.035, .913,
             f"{total} paired targets · {min(widths)}–{max(widths)} qubits · snapshot {date}",
             fontsize=14, color=BASELINE_COLOR)
    fig.text(.035, .849, "Which method uses fewer emitted CX gates?", fontsize=15, weight="bold")
    gates = fig.add_axes((.265, .565, .685, .24))
    colors = {"bdi": BDI_COLOR, "givens": GIVENS_COLOR, "tie": TIE_COLOR}
    for y, policy in ((1, "ladder"), (0, "portfolio")):
        start = 0
        for winner, color in colors.items():
            count = outcomes[policy][winner]
            if count:
                gates.barh(y, count, left=start, height=.56, color=color)
                gates.text(start + count / 2, y, str(count), ha="center", va="center",
                           weight="bold", color=INK if winner == "tie" else "white")
            start += count
    gates.set(xlim=(0, total), ylim=(-.6, 1.6), xticks=[], yticks=[1, 0],
              yticklabels=["Ladders only", "Same best of\nladder + frame"])
    _clean_axes(gates)
    fig.legend(handles=[Patch(color=BDI_COLOR, label="BDI lower CX"),
                        Patch(color=GIVENS_COLOR, label="Givens lower CX"),
                        Patch(color=TIE_COLOR, label="Equal CX")],
               loc="center", bbox_to_anchor=(.605, .524), ncol=3, frameon=False,
               handlelength=1.2, columnspacing=2)
    fig.text(.035, .443, "Time cost depends on whether the plan is reused", fontsize=15, weight="bold")
    timing = fig.add_axes((.265, .177, .685, .205))
    timing.barh([1, 0], [first, reuse], height=.5, color=BDI_COLOR)
    timing.axvline(1, color=GIVENS_COLOR, lw=1.5, linestyle=(0, (3, 3)))
    timing.text(1, 1.66, "Givens = 1×", color=GIVENS_COLOR, ha="center", fontsize=13)
    for y, ratio in ((1, first), (0, reuse)):
        timing.text(ratio + .03, y, f"{ratio:.2f}×", va="center", weight="bold", color=BDI_COLOR)
    timing.set(xlim=(0, max(first, reuse, 1) * 1.24), ylim=(-.6, 1.5), xticks=[],
               yticks=[1, 0], yticklabels=["Preparation +\nfirst evaluation", "Prepared repeat\nevaluation"])
    _clean_axes(timing)
    fig.text(.035, .09,
             "Median paired BDI/Givens time ratios; excludes folding, emission and verification.",
             fontsize=12)
    fig.text(.035, .052,
             "All artifacts verified. Generic emitters, not the paper’s specialized gate construction.",
             fontsize=12)
    description = (
        f"{total} paired targets, snapshot {date}. Ladder-only results: "
        f"BDI wins {outcomes['ladder']['bdi']}, Givens wins {outcomes['ladder']['givens']}, "
        f"ties {outcomes['ladder']['tie']}. The same best-of-ladder/frame portfolio gives "
        f"BDI wins {outcomes['portfolio']['bdi']}, Givens wins {outcomes['portfolio']['givens']}, "
        f"ties {outcomes['portfolio']['tie']}. Median paired BDI/Givens timing ratios: "
        f"preparation plus first evaluation {first:.3f}; prepared repeat evaluation {reuse:.3f}. "
        "Timing excludes folding, emission and verification; a small horizontal corpus, not a universal ranking."
    )
    _save(fig, "bdi-results", title=title, description=description, date=date,
          source=BDI_SOURCE, preview_dir=preview_dir)


def driven_figure(preview_dir: Path | None) -> None:
    source = DRIVEN_SOURCE.read_text()
    date = re.search(r"^Recorded (\d{4}-\d{2}-\d{2})$", source, re.MULTILINE).group(1)
    threshold = float(re.search(r"Common operator-norm threshold: ([^;]+);", source).group(1))
    columns = next(line.split() for line in source.splitlines() if line.startswith("case "))
    rows = [dict(zip(columns, line.split(), strict=True)) for line in source.splitlines()
            if line.startswith("driven-tfim-3 ")]
    route_order = ["Wei-Norman", "magnus4", "fer4", "midpoint2"]
    if len(rows) != len(route_order) or {row["route"] for row in rows} != set(route_order):
        raise ValueError("Expected one result for each of the four driven methods")
    if any(row["status"] != "PASS" for row in rows):
        raise ValueError("This summary requires all four driven outputs to pass")
    by_route = {row["route"]: row for row in rows}
    values = [int(by_route[route]["emitted_CX"]) for route in route_order]
    labels = ["Wei–Norman", "Magnus4", "Fer4", "Midpoint-S2"]
    title = "Driven TFIM3: fewer CX with Wei–Norman"
    fig = plt.figure(figsize=(11.6, 4.9))
    fig.text(.035, .937, title, fontsize=21, weight="bold")
    fig.text(.035, .875,
             f"One three-qubit driven target · snapshot {date} · operator-error target {threshold:.0e}",
             fontsize=14, color=BASELINE_COLOR)
    ax = fig.add_axes((.19, .25, .745, .535))
    ax.barh([3, 2, 1, 0], values, height=.56,
            color=[BDI_COLOR, BASELINE_COLOR, BASELINE_COLOR, BASELINE_COLOR], zorder=3)
    for y, value in zip([3, 2, 1, 0], values, strict=True):
        ax.text(value + max(values) * .018, y, f"{value:,}", va="center", weight="bold")
    ax.set(yticks=[3, 2, 1, 0], yticklabels=labels, xlim=(0, max(values) * 1.13),
           xlabel="Actual emitted CX gates · linear scale from zero")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.grid(axis="x", color="#E7ECF2", zorder=0)
    _clean_axes(ax)
    ax.tick_params(axis="x", pad=6)
    fig.text(.035, .092,
             "One target, not a universal ranking. Step-grid settings are not guaranteed minima.",
             fontsize=12)
    fig.text(.035, .053,
             "Native-ladder / eligible native-frame portfolio. All four outputs passed.",
             fontsize=12)
    description = (
        f"Driven TFIM3 snapshot {date}, operator-error target {threshold:g}. "
        + "; ".join(f"{label}: {value} actual emitted CX gates" for label, value in zip(labels, values, strict=True))
        + ". All outputs passed. A linear scale begins at zero. One three-qubit target, not a universal ranking; "
        "step-grid settings need not be minimal. Same native-ladder and eligible native-frame portfolio."
    )
    _save(fig, "driven-results", title=title, description=description, date=date,
          source=DRIVEN_SOURCE, preview_dir=preview_dir)


def t_figure(preview_dir: Path | None) -> None:
    record = json.loads(T_SOURCE.read_text())
    rows = record["results"]
    if not rows or any(row["status"] != "PASS" for row in rows):
        raise ValueError("T-cost summary requires successful measurements")
    reference = [row["candidate_t_counts"]["reference"] for row in rows]
    selected = [row["t_count"] for row in rows]
    if any(new > old for old, new in zip(reference, selected, strict=True)):
        raise ValueError("T-cost selection must retain the reference fallback")
    labels = [row["case"].replace("-", " ") for row in rows]
    date = record["measured_at"][:10]
    improved = sum(new < old for old, new in zip(reference, selected, strict=True))
    title = f"T-aware BDI: fewer T gates in {improved} of {len(rows)} checks"
    fig = plt.figure(figsize=(11.6, 7.4))
    fig.text(.035, .956, title, fontsize=19, weight="bold")
    fig.text(.035, .911,
             f"Targeted checks · {min(row['width'] for row in rows)}–{max(row['width'] for row in rows)} qubits"
             f" · rotation budget {record['rotation_error_budget']:.0e} · {date}",
             fontsize=13, color=BASELINE_COLOR)
    ax = fig.add_axes((.25, .23, .67, .56))
    positions = list(reversed(range(len(rows))))
    ax.barh([y + .17 for y in positions], reference, height=.29,
            color=BASELINE_COLOR, label="Original BDI", zorder=3)
    ax.barh([y - .17 for y in positions], selected, height=.29,
            color=BDI_COLOR, label="T-aware BDI selection", zorder=3)
    maximum = max(reference)
    for y, old, new in zip(positions, reference, selected, strict=True):
        for offset, value in ((.17, old), (-.17, new)):
            ax.text(value + maximum * .012, y + offset, f"{value:,}", va="center", fontsize=11)
    ax.set(yticks=positions, yticklabels=labels, xlim=(0, maximum * 1.18),
           xlabel="Actual emitted T + T† gates · linear scale from zero")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.grid(axis="x", color="#E7ECF2", zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(-.32, 1.03), ncol=2, frameon=False, fontsize=12)
    _clean_axes(ax)
    ax.tick_params(axis="x", pad=6)
    fig.text(.035, .092, "Same BDI recursion and Clifford+T backend. Balanced/general controls are unchanged.",
             fontsize=12)
    fig.text(.035, .052, "Numerically checked, not formally certified. Small targeted set; not a FlagSynth comparison.",
             fontsize=12)
    description = "; ".join(
        f"{row['case']}: original {old}, selected {new} T gates"
        for row, old, new in zip(rows, reference, selected, strict=True)
    )
    _save(fig, "bdi-t-results", title=title, description=description, date=date,
          source=T_SOURCE, preview_dir=preview_dir)


def _compiler_overview(record, indexed, names, epsilon, preview_dir) -> None:
    """Compare T and CX on the same selected outputs, keeping coverage visible."""
    competitors = {
        "qiskit-pf": "Qiskit formulas",
        "flagsynth-sdm": "FlagSynth SDM*",
        "qiskit-qsd": "Qiskit QSD",
        "pytket": "pytket",
        "bqskit": "BQSKit",
    }
    metrics = {"t_count": "T", "cx_count": "CX"}
    counts = {}
    for metric in metrics:
        counts[metric] = []
        for method in competitors:
            outcomes = Counter()
            for name in names:
                lizzy, other = (indexed[(name, item, epsilon)] for item in ("lizzy-auto", method))
                if lizzy["status"] != "PASS" or other["status"] != "PASS":
                    outcomes["no-pair"] += 1
                else:
                    difference = lizzy[metric] - other[metric]
                    outcomes["lower" if difference < 0 else "higher" if difference > 0 else "tie"] += 1
            counts[metric].append(outcomes)

    categories = [
        ("lower", "Lizzy fewer", "#087E8B", "", "white"),
        ("tie", "Equal", "#F1C76A", "//", INK),
        ("higher", "Other fewer", "#B66553", "..", "black"),
        ("no-pair", "No valid pair", "#E5E9EE", "xx", INK),
    ]
    title = "Compiler comparison, at a glance"
    fig = plt.figure(figsize=(11.6, 5.8))
    fig.text(.035, .927, title, fontsize=23, weight="bold")
    fig.text(.035, .862, "12 targets · 2–4 qubits · error target 10⁻⁶, up to global phase", fontsize=14)
    y = np.arange(len(competitors))
    for panel, (metric, unit) in enumerate(metrics.items()):
        ax = fig.add_axes((.275 + panel * .365, .33, .325, .41))
        ax.set_title(f"{unit} gates", fontsize=17, weight="bold", pad=14)
        left = np.zeros(len(competitors))
        for key, _, color, hatch, text_color in categories:
            values = np.array([outcome[key] for outcome in counts[metric]])
            ax.barh(y, values, left=left, height=.65, color=color,
                    edgecolor=INK if hatch else "white", linewidth=.5, hatch=hatch)
            for index, value in enumerate(values):
                if value:
                    # Opaque label backing keeps hatching from crossing the text.
                    ax.text(left[index] + value / 2, index, str(value), ha="center", va="center",
                            fontsize=14, weight="bold", color=text_color,
                            bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
            left += values
        labels = list(competitors.values()) if panel == 0 else [""] * len(competitors)
        ax.set(yticks=y, yticklabels=labels, xticks=[], xlim=(0, len(names)))
        ax.invert_yaxis()
        _clean_axes(ax)
        ax.tick_params(axis="y", labelsize=15)
    handles = [
        Patch(facecolor=color, edgecolor=INK if hatch else "white", hatch=hatch, label=label)
        for _, label, color, hatch, _ in categories
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.025, .278),
               ncol=len(handles), frameon=False, fontsize=13, handlelength=1.7)
    fig.text(.035, .183, "Numbers count targets, not gates. Each bar includes all 12 targets.", fontsize=13)
    fig.text(.035, .124, "Same compiled circuits in both panels; not a separate CX-optimized run.", fontsize=12)
    fig.text(.035, .073, "No valid pair = one or both methods unsupported, capped or unsuccessful.", fontsize=12)
    fig.text(.035, .023, "*SDM includes local repairs. Small-system results, not a scaling ranking.", fontsize=12)
    description = (
        "At final phase-aligned operator error 1e-6, all twelve targets remain in each comparison. "
        "T includes T-dagger gates. Both panels use the same compiled circuits. "
        "Baselines are T-selected. Lizzy accepts shared frames only without increasing "
        "T or CX against its original T-only ladder winner. This is not a separate CX-optimized run. "
    )
    description += "; ".join(
        f"Against {label}: Lizzy fewer {unit} on {outcome['lower']}, equal on {outcome['tie']}, "
        f"more on {outcome['higher']}, no valid pair on {outcome['no-pair']}"
        for metric, unit in metrics.items()
        for label, outcome in zip(competitors.values(), counts[metric], strict=True)
    )
    _save(fig, "compiler-comparison", title=title, description=description,
          date=record["measured_at"][:10], source=COMPILER_SOURCE,
          preview_dir=preview_dir, publish_png=True)


def compiler_figure(preview_dir: Path | None) -> None:
    """Show all targets and coverage, without turning missing results into wins."""
    record = json.loads(COMPILER_SOURCE.read_text())
    if record.get("complete") is not True:
        raise ValueError("Compiler comparison figure requires a completed measurement record")
    # The public comparison has one automatic Lizzy route. Internal algorithms
    # remain in the raw record/report; they are not separate compiler brands.
    methods = ["lizzy-auto", "qiskit-pf", "flagsynth-sdm", "qiskit-qsd", "pytket", "bqskit"]
    all_methods = methods + ["lizzy-bdi", "lizzy-givens", "lizzy-wei-norman"]
    names = [case["name"] for case in record["cases"]]
    epsilons = record["epsilons"]
    epsilon = 1e-6
    patched_sdm = record.get("baseline_variants", {}).get("flagsynth-sdm") == "locally patched upstream SDM"
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("Expected the twelve distinct declared compiler targets")
    if len(record["methods"]) != len(all_methods) or set(record["methods"]) != set(all_methods):
        raise ValueError("Expected all external methods, public selector and internal diagnostics")
    if len(set(epsilons)) != len(epsilons) or epsilon not in epsilons:
        raise ValueError("Expected distinct accuracy settings including the primary 1e-6 target")
    indexed = {}
    for row in record["rows"]:
        key = (row["case"], row["method"], row["epsilon"])
        if key in indexed:
            raise ValueError(f"Duplicate compiler comparison row: {key}")
        indexed[key] = row
    expected = {(name, method, eps) for name in names for method in all_methods for eps in epsilons}
    if set(indexed) != expected:
        raise ValueError("Compiler comparison has missing or undeclared rows; none may be hidden")

    labels = {
        "commuting-z2": "Commuting Z · 2q", "encoded-su2": "Encoded SU(2) · 2q",
        "anticommuting-star2": "Anticommuting star · 2q", "generic-su4": "Generic SU(4) · 2q",
        "generic-su8": "Generic SU(8) · 3q", "tfim3": "TFIM · 3q", "tfxy3": "TFXY · 3q",
        "heisenberg3": "Heisenberg · 3q", "tfim4": "TFIM · 4q", "tfxy4": "TFXY · 4q",
        "heisenberg_all_to_all4": "All-to-all Heisenberg · 4q", "H2-JW4": "H₂ chemistry · 4q",
    }
    if set(names) != set(labels):
        raise ValueError("Update the figure's readable labels for the declared corpus")
    coverage = Counter()
    ratios = np.full((len(names), len(methods)), np.nan)
    annotations, outcomes = {}, []
    statuses = {"UNSUPPORTED": "N/A", "CAP": "CAP", "UNAVAILABLE": "MISSING", "FAIL": "FAIL"}
    for y, name in enumerate(names):
        rows = [indexed[(name, method, epsilon)] for method in methods]
        passing = [row for row in rows if row["status"] == "PASS"]
        for row in passing:
            counts = [row.get(metric) for metric in ("t_count", "cx_count")]
            error = row.get("final_error", {}).get("phase_aligned", float("nan"))
            if (any(type(count) is not int or count < 0 for count in counts) or not isfinite(error)
                    or not 0 <= error <= epsilon):
                raise ValueError(f"Invalid passing compiler record: {name} / {row['method']}")
        best = min((row["t_count"] for row in passing), default=None)
        for x, row in enumerate(rows):
            if row["status"] == "PASS":
                count = row["t_count"]
                ratios[y, x] = count / best if best else 1.0 if count == 0 else np.inf
                annotations[y, x] = f"{count:,}"
                coverage[row["method"]] += 1
                outcomes.append(f"{name}, {row['method']}: {count} T gates")
            else:
                status = row["status"]
                if status not in statuses and not status.startswith("FAIL"):
                    raise ValueError(f"Unknown compiler status: {status}")
                annotations[y, x] = statuses.get(status, "FAIL")
                outcomes.append(f"{name}, {row['method']}: {status}")

    finite = ratios[np.isfinite(ratios)]
    maximum = 2 ** max(1, ceil(log2(float(finite.max())))) if finite.size else 2
    cmap = LinearSegmentedColormap.from_list(
        "compiler_cost", ["#087E8B", "#D7ECE7", "#F7E6BC", "#D9A775"],
    )
    cmap.set_bad("#EDF0F4")
    norm = LogNorm(vmin=1, vmax=maximum)
    title = "One Lizzy selector, matched output accuracy"
    date = record["measured_at"][:10]
    fig = plt.figure(figsize=(11.6, 9.4))
    fig.text(.035, .954, title, fontsize=21, weight="bold")
    fig.text(.035, .915, f"Actual T + T† gates · 12 targets · 2–4 qubits · snapshot {date}",
             fontsize=13, color=BASELINE_COLOR)
    fig.text(.035, .879, "Same final operator-error target 10⁻⁶, up to global phase · no ancillas",
             fontsize=13)
    ax = fig.add_axes((.29, .267, .68, .552))
    # Clip infinite ratios only for color; the exact counts remain in every cell.
    colored = np.where(np.isinf(ratios), maximum, ratios)
    ax.imshow(colored, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set(xticks=range(len(methods)),
           xticklabels=["Lizzy\nauto", "Qiskit\nformulas",
                        "FlagSynth\nSDM*" if patched_sdm else "FlagSynth\nSDM",
                        "Qiskit\nQSD", "pytket", "BQSKit"],
           yticks=range(len(names)), yticklabels=[labels[name] for name in names])
    ax.xaxis.tick_top()
    _clean_axes(ax)
    ax.tick_params(axis="both", labelsize=12, pad=9)
    ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.axvline(1.5, color=INK, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for (y, x), text in annotations.items():
        if np.isnan(ratios[y, x]):
            color, weight = "#737F91", "normal"
        else:
            red, green, blue, _ = cmap(norm(colored[y, x]))
            color = "white" if .2126 * red + .7152 * green + .0722 * blue < .55 else INK
            weight = "bold" if ratios[y, x] == 1 else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=12, color=color, weight=weight)
    fig.text(.271, .228, "Targets passed", ha="right", fontsize=12, weight="bold")
    for x, method in enumerate(methods):
        fig.text(.29 + .68 * (x + .5) / len(methods), .228,
                 f"{coverage[method]}/12", ha="center", fontsize=12, weight="bold")
    fig.text(.035, .189, "First two columns receive H,t; the others receive dense U. No runtime ranking.", fontsize=11)
    fig.text(.035, .162, "Color: T gates / lowest passing T count in that target row", fontsize=12)
    bar = fig.add_axes((.35, .128, .57, .022))
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                           orientation="horizontal", extend="max" if np.isinf(ratios).any() else "neither")
    powers = list(range(0, int(log2(maximum)) + 1, 2 if maximum > 32 else 1))
    ticks = [2**power for power in powers]
    colorbar.set_ticks(ticks, labels=["1× (lowest)" if tick == 1 else f"{tick}×" for tick in ticks])
    colorbar.ax.tick_params(labelsize=11, length=0, pad=5)
    colorbar.outline.set_visible(False)
    fig.text(.035, .069, "N/A: unsupported · CAP: solver limit · FAIL: unsuccessful · MISSING: dependency unavailable",
             fontsize=11)
    footer = ("* SDM includes disclosed local repairs. Coverage differs; no overall ranking."
              if patched_sdm else "Coverage differs. These are per-target results, not an overall compiler ranking.")
    fig.text(.035, .031, footer, fontsize=12)
    description = (
        "All twelve targets and six public method columns at final phase-aligned operator-error tolerance 1e-6. "
        "Lizzy is its measured automatic exact selector; Qiskit formulas include default/Rustiq lowering, not an extra compiler brand. "
        "Separate internal BDI, Givens and Wei-Norman diagnostics remain in the complete measurement record. "
        "Numbers are actual T plus T-dagger counts. Shared logarithmic colors show each count divided "
        "by the lowest passing count for that target, not a global ranking. Gray cells retain every "
        "unsuccessful or unsupported result. " + "; ".join(outcomes)
    )
    if patched_sdm:
        description += ". SDM includes disclosed local numerical and zero-rotation repairs, not unmodified upstream."
    if np.isinf(ratios).any():
        description += ". Positive counts compared to a zero-T best have infinite ratios, clipped in color only."
    _save(fig, "compiler-comparison-detail", title=title, description=description, date=date,
          source=COMPILER_SOURCE, preview_dir=preview_dir)
    _compiler_overview(record, indexed, names, epsilon, preview_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional PNG preview directory outside the repository")
    parser.add_argument("--only", choices=("bdi", "driven", "t", "compiler"),
                        help="Regenerate just one figure, leaving the other snapshots untouched")
    args = parser.parse_args()
    _style()
    figures = {"bdi": bdi_figure, "driven": driven_figure, "t": t_figure, "compiler": compiler_figure}
    for name, generate in figures.items():
        if args.only is None or args.only == name:
            generate(args.preview_dir)


if __name__ == "__main__":
    main()
