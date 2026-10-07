"""Regenerate the compiler T/CX overview from saved measurements, without benchmarks.

From the repository root: python docs/figures/generate.py
Add --preview-dir /tmp/lizzy-figure-previews to save PNGs for visual inspection.
Requires the existing ``plot`` extra; no optional quantum SDK is imported.
"""

from __future__ import annotations

import argparse
import html
import io
import json
from collections import Counter
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
COMPILER_SOURCE = ROOT / "experiments/compiler_comparison_results.json"
INK = "#253449"


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
    fig.savefig(OUTPUT / f"{name}.png", dpi=160)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Approximately GitHub's README width: check the actual reading size.
        fig.savefig(preview_dir / f"{name}.png", dpi=64)
    plt.close(fig)
    print(f"Generated {(OUTPUT / f'{name}.svg').relative_to(ROOT)} from {source.relative_to(ROOT)}")


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
          preview_dir=preview_dir)


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

    expected_names = {
        "commuting-z2", "encoded-su2", "anticommuting-star2", "generic-su4",
        "generic-su8", "tfim3", "tfxy3", "heisenberg3", "tfim4", "tfxy4",
        "heisenberg_all_to_all4", "H2-JW4",
    }
    if set(names) != expected_names:
        raise ValueError("Expected the current twelve-target compiler corpus")
    statuses = {"UNSUPPORTED", "CAP", "UNAVAILABLE", "FAIL"}
    for name in names:
        for method in methods:
            row = indexed[(name, method, epsilon)]
            if row["status"] == "PASS":
                counts = [row.get(metric) for metric in ("t_count", "cx_count")]
                error = row.get("final_error", {}).get("phase_aligned", float("nan"))
                if (any(type(count) is not int or count < 0 for count in counts) or not isfinite(error)
                        or not 0 <= error <= epsilon):
                    raise ValueError(f"Invalid passing compiler record: {name} / {method}")
            elif row["status"] not in statuses and not row["status"].startswith("FAIL"):
                raise ValueError(f"Unknown compiler status: {row['status']}")

    _compiler_overview(record, indexed, names, epsilon, preview_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional PNG preview directory outside the repository")
    args = parser.parse_args()
    _style()
    compiler_figure(args.preview_dir)


if __name__ == "__main__":
    main()
