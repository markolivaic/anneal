#!/usr/bin/env python3
"""Time CHGNet on this machine, so the README's numbers are the reader's numbers.

Every timing anneal quotes comes from here. Run it and the table regenerates
against your own CPU -- the point is that nobody has to trust a number measured
on the author's laptop.

What it measures, separately, because these are different things:

  model load     first call pays for importing torch as well as reading the
                 4.9 MB checkpoint; the second call pays for neither
  single point   one energy+forces evaluation, which is one relaxation step
  relaxation     atoms displaced by a seeded random 0.05 A and relaxed back in
                 the fixed cell, the relaxation the interface runs by default

Structures come from data/catalog.csv when it exists, chosen to span the atom
count range, so the timings describe what anneal actually offers rather than a
synthetic cell.

Outputs data/benchmark.json. `--table` prints the README's timing table from
that file without running anything.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --repeats 5 --max-structures 8
    python scripts/benchmark.py --table
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from importlib.metadata import version as dist_version
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from anneal import (
    DEFAULT_MAX_ATOMS,
    evaluate,
    fetch_cif,
    graph_edge_count,
    make_session,
    relax,
    single_point,
)
from anneal.relax import DEFAULT_FMAX

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CIF_CACHE = DATA_DIR / "cache" / "cif"

# Fallback if no catalogue has been built yet: a spread of COD entries with
# known-good geometry, small to large.
FALLBACK_IDS = (1000041, 9016675, 1534870, 2202845)

RATTLE_A = 0.05


def rattled(structure: Structure, seed: int) -> Structure:
    """Every atom moved by a normal random vector of `RATTLE_A` per axis, cell unchanged."""
    rng = np.random.default_rng(seed)
    out = structure.copy()
    for index, shift in enumerate(rng.normal(0.0, RATTLE_A, size=(len(out), 3))):
        out.translate_sites([index], shift, frac_coords=False, to_unit_cell=True)
    return out


def pick_structures(catalog: Path, wanted: int, max_atoms: int) -> list[int]:
    """COD ids spanning the atom-count range, so timing vs size is visible."""
    if not catalog.exists():
        return list(FALLBACK_IDS)[:wanted]
    # At least two sites: a rattled one-site cell only translates, so it has
    # nothing to relax. The catalogue does contain single-atom elemental cells.
    rows = [
        r
        for r in csv.DictReader(catalog.open(encoding="utf8"))
        if r["n_sites"] and 2 <= int(r["n_sites"]) <= max_atoms
    ]
    if not rows:
        return list(FALLBACK_IDS)[:wanted]
    rows.sort(key=lambda r: int(r["n_sites"]))
    # Even spread across the size range rather than the first N.
    picks = [rows[round(i * (len(rows) - 1) / max(1, wanted - 1))] for i in range(wanted)]
    seen: list[int] = []
    for r in picks:
        if int(r["cod_id"]) not in seen:
            seen.append(int(r["cod_id"]))
    return seen


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Least-squares line through (x, y): intercept, slope, predictions, R^2."""
    design = np.vstack([np.ones_like(x), x]).T
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = intercept + slope * x
    r_squared = 1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(intercept), float(slope), predicted, float(r_squared)


def fit_cost_model(rows: list[dict]) -> dict:
    """Fit cost per step against graph edges, which is what CHGNet convolves over.

    Atom count is the obvious predictor. Both fits are recorded, with the number
    of points they rest on, so any comparison quoted elsewhere comes from here.

    `rel_spread` is the largest relative residual of the estimate the UI uses,
    the line floored at the cheapest measured step, so the range it shows is
    wide enough to have contained every measurement here.
    """
    edges = np.array([r["atom_graph_edges"] for r in rows], dtype=float)
    atoms = np.array([r["n_atoms"] for r in rows], dtype=float)
    cost = np.array([r["ms_per_step"] for r in rows], dtype=float)
    steps = [r["relax_steps"] for r in rows]

    intercept, slope, predicted, r_squared = _linear_fit(edges, cost)
    residual = np.abs(np.maximum(predicted, cost.min()) - cost) / cost
    r_squared_atoms = _linear_fit(atoms, cost)[3]

    return {
        "form": "ms_per_step = intercept_ms + ms_per_edge * atom_graph_edges",
        "intercept_ms": round(intercept, 3),
        "ms_per_edge": round(slope, 6),
        "r_squared": round(r_squared, 4),
        "r_squared_atoms": round(r_squared_atoms, 4),
        "rel_spread": round(float(residual.max()), 3),
        # Small cells are dominated by fixed overhead the linear term cannot see,
        # so the estimate never drops below the cheapest step actually measured.
        "floor_ms": round(float(cost.min()), 1),
        "step_range": [min(steps), max(steps)],
        "median_steps": int(statistics.median(steps)),
        "fitted_on": len(rows),
    }


def timing_table(payload: dict) -> str:
    """The README's timing table, in Markdown."""
    lines = [
        "| atoms | graph edges | single point | steps | ms/step | full relax |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(payload["measurements"], key=lambda r: r["n_atoms"]):
        lines.append(
            f"| {r['n_atoms']} | {r['atom_graph_edges']:,} | {r['single_point_ms']:,.0f} ms"
            f" | {r['relax_steps']} | {r['ms_per_step']:,.0f} | {r['relax_s']:.1f} s |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-structures", type=int, default=6)
    ap.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    ap.add_argument("--catalog", type=Path, default=DATA_DIR / "catalog.csv")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "benchmark.json")
    ap.add_argument("--table", action="store_true", help="print the table from --out and exit")
    args = ap.parse_args()

    if args.table:
        print(timing_table(json.loads(args.out.read_text(encoding="utf8"))))
        return 0

    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # Model load, cold then warm. The cold number includes importing torch,
    # which is most of it, so quoting one figure for "load time" is misleading.
    t0 = time.perf_counter()
    from anneal.relax import load_model, model_info

    load_model()
    cold_load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    load_model()
    warm_load_s = time.perf_counter() - t1
    info = model_info()
    print(
        f"model: {info['total_parameters']:,} parameters, "
        f"pretrained {info['pretrained_version']}, package {info['package_version']}"
    )
    print(
        f"  cold load {cold_load_s:.2f}s (includes importing torch), "
        f"warm {warm_load_s * 1000:.2f}ms"
    )

    ids = pick_structures(args.catalog, args.max_structures, args.max_atoms)
    print(f"benchmarking {len(ids)} structures, {args.repeats} repeats each")

    rows = []
    for cod_id in ids:
        status, text = fetch_cif(session, cod_id, CIF_CACHE)
        if status != 200:
            print(f"  {cod_id}: fetch failed ({status}), skipped")
            continue
        result = evaluate(text, max_atoms=args.max_atoms)
        if not result.survived:
            print(f"  {cod_id}: {result.stage_failed}, skipped")
            continue
        structure = result.structure

        sp_times = []
        for _ in range(args.repeats):
            sp = single_point(structure)
            sp_times.append(sp.elapsed_s)

        displaced = rattled(structure, seed=cod_id)
        relax_times, step_counts = [], []
        for _ in range(args.repeats):
            r = relax(displaced, relax_cell=False)
            relax_times.append(r.elapsed_s)
            step_counts.append(len(r.steps))

        median_relax = statistics.median(relax_times)
        median_steps = statistics.median(step_counts)
        row = {
            "cod_id": cod_id,
            "formula": structure.composition.reduced_formula,
            "n_atoms": len(structure),
            "atom_graph_edges": graph_edge_count(structure),
            "single_point_ms": round(1000 * statistics.median(sp_times), 1),
            "relax_s": round(median_relax, 2),
            "relax_steps": int(median_steps),
            "ms_per_step": round(1000 * median_relax / max(1, median_steps), 1),
        }
        rows.append(row)
        print(
            f"  {cod_id:>8}  {row['formula']:<14} {row['n_atoms']:>3} atoms"
            f"  single {row['single_point_ms']:>7.1f} ms"
            f"  relax {row['relax_s']:>6.2f} s over {row['relax_steps']:>3} steps"
            f"  ({row['ms_per_step']:.0f} ms/step)"
        )

    if not rows:
        print("no structures benchmarked")
        return 1

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/benchmark.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": __import__("os").cpu_count(),
            "torch": dist_version("torch"),
            "chgnet": dist_version("chgnet"),
            "pymatgen": dist_version("pymatgen"),
            "pymatgen_core": dist_version("pymatgen-core"),
            "device": "cpu",
        },
        "model": info,
        "model_load": {
            "cold_s": round(cold_load_s, 3),
            "warm_s": round(warm_load_s, 6),
            "note": "cold includes importing torch; warm is the cached instance",
        },
        "repeats": args.repeats,
        "workload": {
            "relax_cell": False,
            "rattle_a": RATTLE_A,
            "rattle_seed": "the COD id",
            "fmax_ev_per_a": DEFAULT_FMAX,
        },
        "measurements": rows,
        "cost_model": fit_cost_model(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf8")

    print()
    print(timing_table(payload))
    try:
        print(f"\nwrote {args.out.resolve().relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
