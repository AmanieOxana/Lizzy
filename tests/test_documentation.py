"""Keep the quick start executable and public documentation navigable."""

import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_quick_start_python_examples_run():
    for document in ("README.md", "docs/getting_started.md"):
        source = (ROOT / document).read_text()
        examples = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
        assert examples, f"{document} must retain an executable quick start"
        for index, code in enumerate(examples, 1):
            exec(compile(code, f"{document} example {index}", "exec"), {})


def test_local_documentation_links_resolve():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
                 *sorted((ROOT / "experiments").glob("*.md"))]
    missing = []
    for document in documents:
        source = document.read_text()
        targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", source)
        # MyST document roles and Sphinx navigation must not conceal broken links.
        for kind, role in re.findall(r"\{(doc|download)\}`([^`]+)`", source):
            target = role.rsplit("<", 1)[-1].removesuffix(">")
            targets.append(target + ".md" if kind == "doc" else target)
        for tree in re.findall(r"```\{toctree\}\n(.*?)```", source, re.DOTALL):
            for entry in tree.splitlines():
                entry = entry.strip()
                if entry and not entry.startswith(":"):
                    target = entry.rsplit("<", 1)[-1].removesuffix(">")
                    targets.append(target + ".md")
        for target in targets:
            target = urlsplit(target)
            if target.scheme or target.netloc or not target.path:
                continue
            if not (document.parent / unquote(target.path)).exists():
                missing.append(f"{document.relative_to(ROOT)}: {target.path}")
    assert not missing, "Missing documentation targets:\n" + "\n".join(missing)


def test_compiler_overview_retains_all_targets_and_accessible_formats():
    record = json.loads((ROOT / "experiments/compiler_comparison_results.json").read_text())
    rows = {(row["case"], row["method"]): row for row in record["rows"] if row["epsilon"] == 1e-6}
    svg = ET.parse(ROOT / "docs/figures/compiler-comparison.svg").getroot()
    elements = {element.get("id"): element for element in svg.iter() if element.get("id")}
    assert all(label in elements for label in svg.attrib["aria-labelledby"].split())
    description = elements["compiler-comparison-desc"].text
    assert "same compiled circuits" in description
    assert "without increasing T or CX" in description
    for method, label in {
        "qiskit-pf": "Qiskit formulas", "flagsynth-sdm": "FlagSynth SDM*",
        "qiskit-qsd": "Qiskit QSD", "pytket": "pytket", "bqskit": "BQSKit",
    }.items():
        for metric, unit in (("t_count", "T"), ("cx_count", "CX")):
            counts = Counter()
            for case in record["cases"]:
                ours, theirs = rows[case["name"], "lizzy-auto"], rows[case["name"], method]
                if ours["status"] != "PASS" or theirs["status"] != "PASS":
                    counts["missing"] += 1
                else:
                    counts[(ours[metric] > theirs[metric]) - (ours[metric] < theirs[metric])] += 1
            assert sum(counts.values()) == len(record["cases"])
            assert (
                f"Against {label}: Lizzy fewer {unit} on {counts[-1]}, equal on {counts[0]}, "
                f"more on {counts[1]}, no valid pair on {counts['missing']}"
            ) in description
    with (ROOT / "docs/figures/compiler-comparison.png").open("rb") as raster:
        assert raster.read(8) == b"\x89PNG\r\n\x1a\n"


def test_sphinx_builds_pages_math_and_downloads_without_warnings(tmp_path):
    application = pytest.importorskip("sphinx.application")
    pytest.importorskip("myst_parser")
    warnings = io.StringIO()
    app = application.Sphinx(
        srcdir=ROOT / "docs", confdir=ROOT / "docs",
        outdir=tmp_path / "html", doctreedir=tmp_path / "doctrees",
        buildername="html", status=io.StringIO(), warning=warnings,
        warningiserror=True, freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0, warnings.getvalue()
    for name in ("index", "getting_started", "methods", "compiler_comparison", "reproduce"):
        assert (tmp_path / "html" / f"{name}.html").is_file()
    methods = (tmp_path / "html" / "methods.html").read_text()
    assert 'class="math notranslate' in methods
    # Keep existing deep links usable as the reader-facing explanation changes.
    assert 'id="cartan-bdi-and-givens"' in methods
    assert 'id="weinorman"' in methods
    for name in ("compiler_comparison_results.json", "compiler_comparison_protocol.md"):
        downloads = list((tmp_path / "html" / "_downloads").rglob(name))
        assert len(downloads) == 1
        assert downloads[0].read_bytes() == (ROOT / "experiments" / name).read_bytes()
