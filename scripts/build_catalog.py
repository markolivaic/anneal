#!/usr/bin/env python3
"""Build the browsable catalogue of structures anneal offers.

This is NOT a random sample of COD and its size must never be quoted as a
survival rate. scripts/survey_cod.py owns that number.

Why it differs: a uniform sample of COD spends most of its fetches on cells far
too large to offer -- only 27.8% of a uniform sample survives at 150 atoms. Cell
volume separates the two populations well (entries under 1500 A^3 are 95% likely
to hold 150 atoms or fewer, and that cut still recalls 92.3% of all small cells),
so the catalogue is drawn from a volume-bounded COD query instead. That trades an
unbiased sample for roughly twice the yield per request, which is the right trade
for a catalogue and the wrong one for a statistic.

Both numbers above come from data/survey_cod.csv; recompute them with
scripts/survey_cod.py.

Outputs data/catalog.csv and data/catalog_summary.json.

Usage
-----
    python scripts/build_catalog.py --max-fetch 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version as dist_version
from pathlib import Path

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from anneal import (
    DEFAULT_CIF_BYTE_CAP,
    DEFAULT_MAX_ATOMS,
    evaluate,
    fetch_cif,
    make_session,
    search_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CIF_CACHE = CACHE_DIR / "cif"

# Chosen from the measured volume/atom-count separation, not by taste.
CATALOG_VMAX_A3 = 1500.0

CATALOG_FIELDS = (
    "cod_id",
    "formula_reduced",
    "formula_full",
    "n_sites",
    "n_sites_primitive",
    "volume_a3",
    "spacegroup_symbol",
    "spacegroup_number",
    "crystal_system",
    "n_elements",
    "elements",
    "contains_carbon",
)


def rel(path: Path) -> str:
    """Repo-relative path for printing, tolerating paths given on the command line."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def describe_symmetry(structure) -> tuple[str, int | None, str]:
    """Hermann-Mauguin symbol, space group number, crystal system.

    Recomputed from the coordinates rather than trusting the CIF's own symmetry
    block, which is sometimes absent and sometimes disagrees with the sites.
    """
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return sga.get_space_group_symbol(), sga.get_space_group_number(), sga.get_crystal_system()
    except Exception:
        return "", None, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-fetch", type=int, default=20000, help="how many volume-bounded COD entries to pull"
    )
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    ap.add_argument("--vmax", type=float, default=CATALOG_VMAX_A3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--byte-cap", type=int, default=DEFAULT_CIF_BYTE_CAP)
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "catalog.csv")
    ap.add_argument("--summary", type=Path, default=DATA_DIR / "catalog_summary.json")
    args = ap.parse_args()

    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    session = make_session(pool=args.workers * 2)

    frame_cache = CACHE_DIR / f"frame_vmax{int(args.vmax)}.json"
    if frame_cache.exists():
        ids = json.loads(frame_cache.read_text())
    else:
        print(f"querying COD for entries under {args.vmax:g} A^3...")
        ids = search_ids(session, vmax=args.vmax)
        frame_cache.parent.mkdir(parents=True, exist_ok=True)
        frame_cache.write_text(json.dumps(ids))
    print(f"  volume-bounded frame: {len(ids):,} ids")

    rng = random.Random(args.seed)
    n = min(args.max_fetch, len(ids))
    sample = sorted(rng.sample(sorted(set(ids)), n))

    print(f"fetching {n:,} CIFs with {args.workers} workers...")
    t0 = time.time()
    fetched: dict[int, tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_cif, session, cid, CIF_CACHE, byte_cap=args.byte_cap): cid
            for cid in sample
        }
        for done, (future, cid) in enumerate(futures.items(), 1):
            fetched[cid] = future.result()
            if done % 1000 == 0:
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done:,}/{n:,}  {time.time() - t0:.0f}s  {rate:.1f}/s")
    print(f"  fetched in {time.time() - t0:.0f}s")

    print("filtering and reading symmetry...")
    t1 = time.time()
    rows: list[dict] = []
    stage_counts: dict[str, int] = {}
    for done, cid in enumerate(sample, 1):
        status, text = fetched[cid]
        if status != 200 or not text.strip():
            stage_counts["fetched"] = stage_counts.get("fetched", 0) + 1
            continue
        result = evaluate(text, max_atoms=args.max_atoms)
        if not result.survived:
            stage_counts[result.stage_failed] = stage_counts.get(result.stage_failed, 0) + 1
            continue

        structure = result.structure
        symbol, number, system = describe_symmetry(structure)
        elements = sorted({str(el) for el in structure.composition.elements})
        rows.append(
            {
                "cod_id": cid,
                "formula_reduced": structure.composition.reduced_formula,
                "formula_full": result.formula_parsed,
                "n_sites": result.n_sites,
                "n_sites_primitive": result.n_sites_primitive,
                "volume_a3": result.volume_a3,
                "spacegroup_symbol": symbol,
                "spacegroup_number": number,
                "crystal_system": system,
                "n_elements": len(elements),
                "elements": " ".join(elements),
                "contains_carbon": result.contains_carbon,
            }
        )
        if done % 2000 == 0:
            print(f"  {done:,}/{n:,}  kept {len(rows):,}  {time.time() - t1:.0f}s")
    print(f"  filtered in {time.time() - t1:.0f}s")

    if not rows:
        print("no structures survived -- nothing written", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["cod_id"])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CATALOG_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    sizes = sorted(r["n_sites"] for r in rows)
    systems: dict[str, int] = {}
    for r in rows:
        systems[r["crystal_system"] or "unknown"] = (
            systems.get(r["crystal_system"] or "unknown", 0) + 1
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/build_catalog.py",
        "provenance": (
            "volume-bounded COD query, not a uniform sample of COD -- this count "
            "is a catalogue size, not a survival rate. See survey_cod_summary.json."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pymatgen": dist_version("pymatgen"),
        },
        "query": {"vmax_a3": args.vmax, "frame_size": len(ids), "seed": args.seed},
        "fetched": n,
        "kept": len(rows),
        "yield_pct": round(100 * len(rows) / n, 2),
        "max_atoms": args.max_atoms,
        "rejected_by_stage": dict(sorted(stage_counts.items(), key=lambda kv: -kv[1])),
        "sites": {
            "min": sizes[0],
            "median": sizes[len(sizes) // 2],
            "p90": sizes[max(0, round(0.9 * len(sizes)) - 1)],
            "max": sizes[-1],
        },
        "crystal_systems": dict(sorted(systems.items(), key=lambda kv: -kv[1])),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf8")

    print()
    print(f"  fetched {n:,}, kept {len(rows):,}  ({summary['yield_pct']:.1f}% yield)")
    print(
        f"  sites: min {sizes[0]} / median {summary['sites']['median']}"
        f" / p90 {summary['sites']['p90']} / max {sizes[-1]}"
    )
    print(f"  rejected: {summary['rejected_by_stage']}")
    print(f"wrote {rel(args.csv)}")
    print(f"wrote {rel(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
