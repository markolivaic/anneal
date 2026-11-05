"""Numbers in the README against the committed files they come from."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = (ROOT / "README.md").read_text(encoding="utf8")
README = re.sub(r"\s+", " ", RAW)  # so a number split across a line break still matches


def artefact(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf8"))


QUOTED = {
    "survey_cod_summary.json": lambda s: [
        f"{s['survival_pct']:.1f}%",
        f"{s['frame']['counts']['default']:,}",
        f"{s['survivors']:,}",
        f"hold more than {s['sample']['max_atoms']} atoms",
        f"within the {s['sample']['max_atoms']}-atom budget",
    ],
    "benchmark.json": lambda b: [
        f"{b['model']['total_parameters']:,}",
        f"{b['cost_model']['fitted_on']} structures",
    ],
    "catalog_summary.json": lambda c: [f"{c['kept']:,}"],
    "validation_summary.json": lambda v: [
        f"{v['statistics']['overall']['volume_error_pct']['median_signed']:+.2f}%",
    ],
}


@pytest.mark.parametrize("name", QUOTED)
def test_the_readme_quotes_the_artefact(name):
    absent = [text for text in QUOTED[name](artefact(name)) if text not in README]
    assert not absent, f"{name} says these and the README does not: {absent}"


def test_the_survivor_count_and_the_rate_agree():
    survey = artefact("survey_cod_summary.json")
    rate = 100 * survey["survivors"] / survey["sample"]["size"]
    assert abs(rate - survey["survival_pct"]) < 0.01


def test_the_timing_table_is_the_benchmark():
    table = RAW[RAW.index("| atoms | graph edges |") :].split("\n\n")[0]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in table.splitlines()[2:]]
    expected = [
        [
            str(r["n_atoms"]),
            f"{r['atom_graph_edges']:,}",
            f"{r['single_point_ms']:,.0f} ms",
            str(r["relax_steps"]),
            f"{r['ms_per_step']:,.0f}",
            f"{r['relax_s']:.1f} s",
        ]
        for r in sorted(artefact("benchmark.json")["measurements"], key=lambda r: r["n_atoms"])
    ]
    assert rows == expected, "regenerate with: python scripts/benchmark.py --table"


def test_every_r_squared_in_the_readme_is_the_cost_fit():
    fit = artefact("benchmark.json")["cost_model"]
    quoted = re.findall(r"R² \**([0-9]\.[0-9]+)", README)
    assert quoted, "the README quotes no cost model fit"
    assert set(quoted) <= {f"{fit['r_squared']:.3f}", f"{fit['r_squared_atoms']:.3f}"}


def test_the_element_bound_counts_match_both_runs():
    catalog = artefact("catalog_summary.json")
    rejected = catalog["rejected_by_stage"].get("elements_within_chgnet", 0)
    funnel = artefact("survey_cod_summary.json")["funnel"]
    in_survey = next(f["lost"] for f in funnel if f["stage"] == "elements_within_chgnet")
    if in_survey == 0 and rejected > 0:
        assert f"{rejected} of {catalog['fetched']:,}" in README
