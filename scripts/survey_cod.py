#!/usr/bin/env python3
"""Measure what fraction of the Crystallography Open Database anneal can offer.

The filter itself lives in backend/anneal/filters.py and is the same code the
application runs. This script only samples, drives it, and reports.

Method
------
1. Build a sampling frame: every COD id the default search returns (which
   excludes duplicates, error-flagged and theoretical entries). One request.
2. Draw a uniform random sample from that frame with a fixed seed.
3. Fetch each CIF, cached on disk.
4. Run the filter and record which stage each entry failed.

Outputs data/survey_cod.csv (one row per sampled structure) and
data/survey_cod_summary.json (the funnel, the survival rate, and a sweep over
alternative atom caps).

Usage
-----
    python scripts/survey_cod.py --sample-size 4000 --seed 20260806
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version as dist_version
from pathlib import Path

from anneal import (
    CHGNET_MAX_Z,
    DEFAULT_MAX_ATOMS,
    STAGES,
    FilterResult,
    build_frame,
    evaluate,
    fetch_cif,
    make_session,
)
from anneal.filters import ELEMENT_TOKEN

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CIF_CACHE = CACHE_DIR / "cif"

# Alternative atom-count caps, reported so the warn/refuse thresholds come from
# measured survival rather than a guess.
SIZE_CAPS = (16, 32, 48, 64, 96, 128, 150, 200)


def rel(path: Path) -> str:
    """Repo-relative path for printing, tolerating paths given on the command line."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval. Behaves at the extremes where normal-approx does not."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def report(rows: list[dict], frame, args: argparse.Namespace) -> dict:
    total = len(rows)
    survivors = [r for r in rows if r["survived"]]
    frame_size = frame.counts.get("default", 0)

    funnel = []
    remaining = total
    for stage in STAGES:
        lost = sum(1 for r in rows if r["stage_failed"] == stage)
        remaining -= lost
        funnel.append(
            {
                "stage": stage,
                "lost": lost,
                "remaining": remaining,
                "pct_of_sample": round(100 * remaining / total, 2) if total else 0.0,
            }
        )

    # Pass rates for each criterion on its own, over every entry that got far
    # enough to be judged on it. The funnel hides this: a structure rejected for
    # disorder is never asked about its size.
    judged = [r for r in rows if r["n_sites"] is not None]
    independent = {
        "evaluated_on": len(judged),
        "ordered": sum(1 for r in judged if r["is_ordered"]),
        "size_within_limit": sum(1 for r in judged if r["n_sites"] <= args.max_atoms),
        "elements_within_chgnet": sum(
            1 for r in judged if r["max_z"] is not None and r["max_z"] <= CHGNET_MAX_Z
        ),
    }

    # Rows that cleared every criterion except possibly size. Because size is the
    # last stage, this set is exact, so the sweep costs no extra parsing.
    size_limited = [
        r
        for r in rows
        if r["stage_failed"] in ("", "size_within_limit") and r["n_sites"] is not None
    ]
    size_sweep = []
    for cap in SIZE_CAPS:
        kept = sorted(r["n_sites"] for r in size_limited if r["n_sites"] <= cap)
        lo, hi = wilson_interval(len(kept), total)
        size_sweep.append(
            {
                "max_atoms": cap,
                "survivors": len(kept),
                "survival_pct": round(100 * len(kept) / total, 2) if total else 0.0,
                "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
                "projected_entries": round(len(kept) / total * frame_size) if total else 0,
                "median_sites": kept[len(kept) // 2] if kept else None,
                "p90_sites": kept[max(0, math.ceil(0.9 * len(kept)) - 1)] if kept else None,
            }
        )

    rate = len(survivors) / total if total else 0.0
    low, high = wilson_interval(len(survivors), total)
    sizes = sorted(r["n_sites"] for r in survivors if r["n_sites"] is not None)

    element_hist: dict[str, int] = {}
    for r in survivors:
        for sym, _ in ELEMENT_TOKEN.findall(r["formula_parsed"]):
            if sym:
                element_hist[sym] = element_hist.get(sym, 0) + 1

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/survey_cod.py",
        # random.sample is not guaranteed stable across Python releases:
        # reproducing this row-for-row needs the same interpreter.
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            # pymatgen and pymatgen-core are separate distributions on different
            # release cadences. pymatgen-core is the one that parses the CIFs.
            "pymatgen": dist_version("pymatgen"),
            "pymatgen_core": dist_version("pymatgen-core"),
        },
        "frame": {
            "query": frame.query,
            "built_at": frame.built_at,
            "counts": frame.counts,
            "cod_homepage_total": frame.homepage_total,
            "sampled_from": frame_size,
        },
        "sample": {"size": total, "seed": args.seed, "max_atoms": args.max_atoms},
        "funnel": funnel,
        "independent_pass_counts": independent,
        "size_cap_sweep": size_sweep,
        "survivors": len(survivors),
        "survival_rate": round(rate, 6),
        "survival_pct": round(100 * rate, 2),
        "survival_pct_ci95": [round(100 * low, 2), round(100 * high, 2)],
        "projected_usable_entries": round(rate * frame_size),
        "survivor_sites": {
            "min": sizes[0] if sizes else None,
            "median": sizes[len(sizes) // 2] if sizes else None,
            "max": sizes[-1] if sizes else None,
        },
        "survivors_containing_carbon": sum(1 for r in survivors if r["contains_carbon"]),
        "survivor_element_counts": dict(sorted(element_hist.items(), key=lambda kv: -kv[1])),
    }

    print()
    print(
        f"sampled {total:,} of {frame_size:,} COD entries (seed {args.seed},"
        f" max_atoms {args.max_atoms})"
    )
    print()
    print(f"  {'stage':<28} {'lost':>7} {'left':>8} {'% of sample':>12}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 8} {'-' * 12}")
    print(f"  {'sampled':<28} {'':>7} {total:>8,} {100.00:>11.2f}%")
    for step in funnel:
        print(
            f"  {step['stage']:<28} {step['lost']:>7,} {step['remaining']:>8,}"
            f" {step['pct_of_sample']:>11.2f}%"
        )
    print()
    print(f"  each criterion alone, over the {independent['evaluated_on']:,} that parsed:")
    for key in ("ordered", "elements_within_chgnet", "size_within_limit"):
        got = independent[key]
        pct = 100 * got / independent["evaluated_on"] if independent["evaluated_on"] else 0
        print(f"    {key:<26} {got:>7,} pass  {pct:>6.2f}%")
    print()
    print("  survival by atom-count cap (everything else already passed):")
    print(
        f"    {'cap':>5} {'survivors':>10} {'% sample':>9} {'median':>7} {'p90':>5}"
        f" {'projected':>11}"
    )
    for step in size_sweep:
        print(
            f"    {step['max_atoms']:>5} {step['survivors']:>10,}"
            f" {step['survival_pct']:>8.2f}% {step['median_sites']!s:>7}"
            f" {step['p90_sites']!s:>5} {step['projected_entries']:>11,}"
        )
    print()
    print(
        f"  survival rate  {summary['survival_pct']:.2f}%"
        f"  (95% CI {summary['survival_pct_ci95'][0]:.2f}-"
        f"{summary['survival_pct_ci95'][1]:.2f}%)"
    )
    print(f"  projects to    ~{summary['projected_usable_entries']:,} of {frame_size:,}")
    print()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh-frame", action="store_true")
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "survey_cod.csv")
    ap.add_argument("--summary", type=Path, default=DATA_DIR / "survey_cod_summary.json")
    args = ap.parse_args()

    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print("building sampling frame from COD...")
    frame = build_frame(session, CACHE_DIR / "frame.json", refresh=args.refresh_frame)
    if not frame.ids:
        print("frame is empty -- COD search returned nothing", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n = min(args.sample_size, len(frame.ids))
    sample = sorted(rng.sample(frame.ids, n))

    print(f"fetching {n:,} CIFs with {args.workers} workers...")
    t0 = time.time()
    fetched: dict[int, tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # No byte cap here on purpose: skipping the largest files would bias the
        # funnel, and this survey is the number the README quotes.
        futures = {
            pool.submit(fetch_cif, session, cid, CIF_CACHE, byte_cap=None): cid for cid in sample
        }
        for done, (future, cid) in enumerate(futures.items(), 1):
            fetched[cid] = future.result()
            if done % 500 == 0:
                print(f"  {done:,}/{n:,}  {time.time() - t0:.0f}s")
    print(f"  fetched in {time.time() - t0:.0f}s")

    print("parsing and filtering...")
    t1 = time.time()
    rows: list[dict] = []
    for done, cid in enumerate(sample, 1):
        status, text = fetched[cid]
        if status != 200 or not text.strip():
            # Built from FilterResult so a fetch failure writes exactly the same
            # columns as a filter rejection.
            row = FilterResult(stage_failed="fetched", reject_detail=f"http {status}").as_row()
        else:
            row = evaluate(text, max_atoms=args.max_atoms).as_row()
        rows.append({"cod_id": cid, "http_status": status, **row})
        if done % 500 == 0:
            print(f"  {done:,}/{n:,}  {time.time() - t1:.0f}s")
    print(f"  filtered in {time.time() - t1:.0f}s")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = report(rows, frame, args)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"wrote {rel(args.csv)}")
    print(f"wrote {rel(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
