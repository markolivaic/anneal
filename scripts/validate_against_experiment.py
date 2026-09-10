#!/usr/bin/env python3
"""Measure how far CHGNet's relaxed cell sits from the one the crystallographer measured.

anneal relaxes real COD structures and draws the energy falling, but nothing in
this repo has ever compared a CHGNet result against any reference. The README
says the model is an approximation to DFT and not DFT, which is honest and
unquantified. A measured error is a stronger claim than a caveat, so this script
relaxes a fixed random sample of catalogued structures with the cell free and
compares the relaxed lattice against the deposited one.

What this comparison is, and what it is not
-------------------------------------------
The reference is a diffraction measurement, not a DFT calculation, so the
residual reported here is not the model's error against its own training target.
At least two systematic effects act on it simultaneously and in opposite
directions: a cell measured at finite temperature has expanded relative to its
own 0 K geometry, while a potential fitted to DFT-PBE energies inherits PBE's
lattice-constant bias. Which one dominates is a question for the output of this
script, not for its docstring, so the measurement temperature is parsed out of
each CIF and every statistic is reported per temperature bin as well as overall.

Method
------
1. Draw a fixed-seed random sample from data/catalog.csv, restricted to cells of
   2 to 100 sites so the run finishes in about an hour.
2. Relax each with anneal.relax(relax_cell=True). The cell must be free -- a
   fixed-cell relaxation would be comparing the deposited cell against itself.
3. Record the signed percentage error in volume and in a, b and c.
4. Read `_cell_measurement_temperature`, falling back to
   `_diffrn_ambient_temperature`, and bin on it.

Runs that do not converge inside the step ceiling are written to the CSV and
counted separately in the summary rather than dropped: they are under-relaxed, so
discarding them quietly would pull the headline error toward zero.

Outputs data/validation_vs_experiment.csv, one row per sampled structure
including the ones that were excluded, and data/validation_summary.json.

Usage
-----
    python scripts/validate_against_experiment.py --sample-size 5
    python scripts/validate_against_experiment.py --sample-size 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version as dist_version
from pathlib import Path

from anneal import evaluate, fetch_cif, make_session, relax

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CIF_CACHE = DATA_DIR / "cache" / "cif"

# Both tags carry the measurement temperature in kelvin and a CIF may hold
# either or both. _cell_measurement_temperature wins when both are present: it
# describes the determination of the cell constants, which is the quantity being
# compared here, while _diffrn_ambient_temperature describes the diffraction
# experiment as a whole.
TEMPERATURE_TAGS = ("_cell_measurement_temperature", "_diffrn_ambient_temperature")

# Values outside this are refinement typos or unit confusion, not measurements.
TEMPERATURE_MIN_K = 1.0
TEMPERATURE_MAX_K = 1500.0

TEMPERATURE_BINS = (
    ("lt_150k", "temperature < 150 K"),
    ("150_250k", "150 K <= temperature <= 250 K"),
    ("gt_250k", "temperature > 250 K"),
    ("unknown", "no usable temperature tag in the CIF"),
)

FIELDS = (
    "cod_id",
    "formula",
    "n_sites",
    "temperature_k",
    "temperature_tag",
    "volume_deposited_a3",
    "volume_relaxed_a3",
    "volume_error_pct",
    "a_deposited_a",
    "a_relaxed_a",
    "a_error_pct",
    "b_deposited_a",
    "b_relaxed_a",
    "b_error_pct",
    "c_deposited_a",
    "c_relaxed_a",
    "c_error_pct",
    "steps",
    "converged",
    "outcome",
    "elapsed_s",
)

# Stated in the artifact rather than left for the reader to discover. Nothing
# here is a prediction about the sign or size of the result.
LIMITATIONS = (
    "The reference is a measured cell, not a DFT cell. This is CHGNet's distance "
    "from experiment, which is not the same quantity as its error against the "
    "DFT-PBE data it was trained on.",
    "This measures geometry and nothing else. It is not evidence about the "
    "energies anneal reports, which have no experimental counterpart to be "
    "compared against at all. A cell landing within a percent of the measured "
    "one does not make an energy difference right.",
    "Thermal expansion and DFT-PBE's lattice bias both act on this residual and "
    "point opposite ways. The temperature bins separate them only partially: a "
    "bin is a population of structures, not a controlled variable, so it differs "
    "in chemistry and cell size as well as in temperature. Each bin therefore "
    "reports its carbon-containing fraction and median site count next to its "
    "error, so the confound is visible instead of assumed.",
    "Zero-point motion expands a real crystal even at 0 K. CHGNet predicts a "
    "static lattice, so even the coldest bin is not a like-for-like comparison.",
    "a, b and c are compared vector by vector in the deposited setting. The cell "
    "filter deforms the cell continuously so the axes do not permute, but for a "
    "large or strongly sheared deformation the individual axis errors are less "
    "meaningful than the volume error. Volume is the robust number here.",
    "fmax is a loose convergence criterion and, with the cell free, ASE judges it "
    "over atomic forces and cell stress together. A run marked converged sits in "
    "a shallow basin, not at a tight minimum.",
    "Non-converged runs are under-relaxed, so their errors are biased toward the "
    "deposited cell. They are reported separately for that reason.",
    "The sample is drawn from data/catalog.csv, which is a volume-bounded COD "
    "query already filtered for ordered sites, declared hydrogen and Z <= 94, "
    "and is further cut here to 2-100 sites. These statistics describe that "
    "population, not COD as a whole.",
    "Hydrogen positions from X-ray refinement are poorly determined, and CHGNet "
    "moves them. Structures containing hydrogen are not separated out here.",
)


def rel(path: Path) -> str:
    """Repo-relative path, tolerating paths given on the command line.

    Posix separators even on Windows: this string is written into the summary,
    and an artifact that names its own inputs should read the same everywhere.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def cif_temperature(cif_text: str) -> tuple[float | None, str, str]:
    """Measurement temperature in kelvin, the tag it came from, and why not if absent.

    CIF numeric values carry a parenthesised standard uncertainty -- `293(2)`
    means 293 K with an uncertainty of 2 in the last digit -- which is stripped,
    not parsed, because the uncertainty is far smaller than the binning applied
    to it. `?` and `.` are CIF's two ways of writing "no value".

    A value the sanity check rejects does not condemn the entry: the search falls
    through to the other tag, and the rejection is returned either way so the
    summary can list it instead of hiding it behind a clean-looking number.
    """
    rejected: list[str] = []
    for tag in TEMPERATURE_TAGS:
        match = re.search(
            rf"^{tag}\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))",
            cif_text,
            re.MULTILINE,
        )
        if not match:
            continue
        raw = next((g for g in match.groups() if g is not None), "").strip()
        if not raw or raw in {"?", "."}:
            continue
        try:
            kelvin = float(raw.split("(")[0].strip())
        except ValueError:
            rejected.append(f"unparsable: {tag} = {raw!r}")
            continue
        if not TEMPERATURE_MIN_K <= kelvin <= TEMPERATURE_MAX_K:
            rejected.append(f"out of range: {tag} = {raw!r}")
            continue
        return kelvin, tag, "; ".join(rejected)
    return None, "", "; ".join(rejected) or "absent"


def temperature_bin(kelvin: float | None) -> str:
    if kelvin is None:
        return "unknown"
    if kelvin < 150:
        return "lt_150k"
    if kelvin <= 250:
        return "150_250k"
    return "gt_250k"


def quartiles(values: list[float]) -> tuple[float | None, float | None]:
    """Q1 and Q3, or Nones when there are too few points to interpolate them."""
    if len(values) < 2:
        return None, None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def median_interval(values: list[float]) -> list[float] | None:
    """A 95% confidence interval for the median, from order statistics.

    Distribution free, because these error distributions are skewed and carry
    outliers, so a normal approximation would understate the spread. Built from
    the binomial tail rather than by bootstrapping: a bootstrap would give the
    artifact a random seed of its own, and this way the interval is exact.

    Without it a bin of 30 structures and a bin of 100 print the same way, and
    the reader cannot tell which median is worth anything. Returns None below
    n=6, where no pair of order statistics reaches 95% coverage.

    Coverage of [x_(r), x_(s)] is sum(C(n,i) for i in r..s-1) / 2**n, so the
    interval below is the widest one whose two tails each stay under 2.5%. An
    earlier version was one rank narrower on each side and a simulation put its
    real coverage at 91.5%, which is why the bound is derived here rather than
    written from memory.
    """
    n = len(values)
    if n < 6:
        return None
    ordered = sorted(values)
    total = 2**n
    cumulative = 0
    k = 0
    while k < n // 2 and (cumulative + math.comb(n, k)) / total <= 0.025:
        cumulative += math.comb(n, k)
        k += 1
    if k == 0:
        return None
    return [round(ordered[k - 1], 3), round(ordered[n - k], 3)]


def summarise(records: list[dict]) -> dict:
    """Error statistics over a set of completed relaxations.

    Every field is None rather than 0 when the set is too small to support it, so
    a thin bin reads as thin instead of reading as a measurement.
    """
    if not records:
        return {"n": 0, "note": "no relaxations in this group"}

    signed = sorted(r["volume_error_pct"] for r in records)
    absolute = sorted(abs(v) for v in signed)
    q1, q3 = quartiles(signed)
    abs_q1, abs_q3 = quartiles(absolute)
    temperatures = [r["temperature_k"] for r in records if r["temperature_k"] is not None]

    out = {
        "n": len(records),
        "volume_error_pct": {
            "median_signed": round(statistics.median(signed), 3),
            "median_signed_ci95": median_interval(signed),
            "median_absolute": round(statistics.median(absolute), 3),
            "median_absolute_ci95": median_interval(absolute),
            "mean_signed": round(statistics.fmean(signed), 3),
            "q1": round(q1, 3) if q1 is not None else None,
            "q3": round(q3, 3) if q3 is not None else None,
            "iqr": round(q3 - q1, 3) if q1 is not None else None,
            "absolute_q1": round(abs_q1, 3) if abs_q1 is not None else None,
            "absolute_q3": round(abs_q3, 3) if abs_q3 is not None else None,
            "min": round(signed[0], 3),
            "max": round(signed[-1], 3),
            "n_larger_than_deposited": sum(1 for v in signed if v > 0),
            "n_smaller_than_deposited": sum(1 for v in signed if v < 0),
            # A relaxation that moves the cell this far has not refined the
            # deposited structure, it has found a different one. Counted so the
            # medians can be read knowing how many such cases sit behind them.
            "n_abs_error_over_10pct": sum(1 for v in absolute if v > 10),
            "n_abs_error_over_20pct": sum(1 for v in absolute if v > 20),
        },
        "axis_error_pct": {},
        # What else is different about this group. A temperature bin is a
        # population, not a controlled variable, and these two numbers are the
        # cheapest way to see whether a bin's error could be about its chemistry
        # rather than its temperature.
        "composition": {
            "containing_carbon": sum(1 for r in records if r["contains_carbon"]),
            "median_n_sites": int(statistics.median(r["n_sites"] for r in records)),
            "median_temperature_k": (
                round(statistics.median(temperatures), 1) if temperatures else None
            ),
        },
    }
    for axis in ("a", "b", "c"):
        values = sorted(r[f"{axis}_error_pct"] for r in records)
        out["axis_error_pct"][axis] = {
            "median_signed": round(statistics.median(values), 3),
            "median_absolute": round(statistics.median(abs(v) for v in values), 3),
        }
    return out


def by_bin(records: list[dict]) -> dict:
    grouped = {key: [] for key, _ in TEMPERATURE_BINS}
    for record in records:
        grouped[record["bin"]].append(record)
    return {key: summarise(grouped[key]) for key, _ in TEMPERATURE_BINS}


# A capital C not followed by a lowercase letter, so Ca, Cl, Cu and Cs do not
# read as carbon. Used only to rebuild a resumed row, where the parsed structure
# is long gone and the formula string is all that survived into the CSV.
CARBON_IN_FORMULA = re.compile(r"C(?![a-z])")


def record_from_csv_row(row: dict) -> dict | None:
    """Rebuild an in-memory record for a structure an earlier run already relaxed.

    Two fields the statistics need are not columns in the CSV: the temperature
    bin, which is a pure function of the temperature, and whether the cell holds
    carbon, which the formula string still answers. Everything else is read back
    as written. Returns None for rows that never produced an error figure, which
    are the excluded ones.
    """
    if not row.get("volume_error_pct"):
        return None
    kelvin = float(row["temperature_k"]) if row["temperature_k"] else None
    record = {
        "cod_id": int(row["cod_id"]),
        "formula": row["formula"],
        "n_sites": int(row["n_sites"]),
        "temperature_k": kelvin,
        "temperature_tag": row["temperature_tag"] or "",
        "contains_carbon": bool(CARBON_IN_FORMULA.search(row["formula"])),
        "volume_deposited_a3": float(row["volume_deposited_a3"]),
        "volume_relaxed_a3": float(row["volume_relaxed_a3"]),
        "volume_error_pct": float(row["volume_error_pct"]),
        "steps": int(row["steps"]),
        # Written by csv.DictWriter from a bool, so it comes back as the text
        # "True" or "False" and must be converted, not truthiness-tested: the
        # string "False" is truthy and would mark every resumed run converged.
        "converged": row["converged"] == "True",
        "outcome": row["outcome"],
        "elapsed_s": float(row["elapsed_s"]) if row["elapsed_s"] else 0.0,
        "bin": temperature_bin(kelvin),
    }
    for axis in ("a", "b", "c"):
        record[f"{axis}_error_pct"] = float(row[f"{axis}_error_pct"])
    return record


def csv_row(record: dict) -> dict:
    """One CSV line. Missing numbers are blank, never zero."""
    out = {}
    for field in FIELDS:
        value = record.get(field)
        if isinstance(value, float):
            value = round(value, 4)
        out[field] = "" if value is None else value
    return out


def print_group(label: str, stats: dict) -> None:
    """One table line. The chemistry columns sit next to the error on purpose."""
    if not stats.get("n"):
        print(
            f"  {label:<12} {0:>5}  {'--':>13} {'--':>16} {'--':>13} {'--':>8} {'--':>7} {'--':>8}"
        )
        return
    volume = stats["volume_error_pct"]
    composition = stats["composition"]
    iqr = f"{volume['iqr']:.2f}%" if volume["iqr"] is not None else "--"
    ci = volume["median_signed_ci95"]
    median_t = composition["median_temperature_k"]
    print(
        f"  {label:<12} {stats['n']:>5}  {volume['median_signed']:>+12.2f}%"
        f" {(f'[{ci[0]:+.2f}, {ci[1]:+.2f}]' if ci else '--'):>16}"
        f" {volume['median_absolute']:>12.2f}% {iqr:>8}"
        f" {composition['containing_carbon']:>7}"
        f" {(f'{median_t:.0f} K' if median_t is not None else '--'):>8}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--min-sites", type=int, default=2)
    # 100 rather than the catalogue's 150: relaxation cost grows with the graph,
    # and this bound is what keeps a 200-structure run near an hour.
    ap.add_argument("--max-sites", type=int, default=100)
    ap.add_argument("--fmax", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--catalog", type=Path, default=DATA_DIR / "catalog.csv")
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "validation_vs_experiment.csv")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip cod_ids already in the CSV and append, instead of starting over. "
            "Only valid with the same --sample-size and --seed as the run being "
            "resumed: the sample is drawn as one block, so a different size draws "
            "a different set of structures and the two halves would not belong "
            "to the same sample."
        ),
    )
    ap.add_argument("--summary", type=Path, default=DATA_DIR / "validation_summary.json")
    args = ap.parse_args()

    if not args.catalog.exists():
        print(
            f"no catalogue at {rel(args.catalog)} -- run scripts/build_catalog.py", file=sys.stderr
        )
        return 1

    candidates = [
        row
        for row in csv.DictReader(args.catalog.open(encoding="utf8"))
        if row["n_sites"] and args.min_sites <= int(row["n_sites"]) <= args.max_sites
    ]
    if not candidates:
        print("no catalogue entries in the requested size range", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n = min(args.sample_size, len(candidates))
    sample = rng.sample(candidates, n)
    sample.sort(key=lambda row: int(row["cod_id"]))

    CIF_CACHE.mkdir(parents=True, exist_ok=True)
    session = make_session()

    from anneal.relax import load_model, model_info

    t0 = time.perf_counter()
    load_model()
    info = model_info()
    print(
        f"model: CHGNet pretrained {info['pretrained_version']},"
        f" package {info['package_version']}, loaded in {time.perf_counter() - t0:.1f}s"
    )
    print(
        f"validating {n} of {len(candidates):,} catalogue entries with"
        f" {args.min_sites}-{args.max_sites} sites (seed {args.seed})"
    )
    print(f"relaxing with cell free, fmax {args.fmax} eV/A, ceiling {args.max_steps} steps")
    print()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    excluded: list[dict] = []
    tag_counts: dict[str, int] = {}
    temperature_notes: list[str] = []
    started = time.perf_counter()

    # Resume exists because this run takes hours and died once at 62 of 200 when
    # the session that owned it ended. Rows already on disk are read back rather
    # than recomputed, and the CSV is appended to instead of truncated.
    done_ids: set[int] = set()
    appending = bool(args.resume and args.csv.exists())
    if appending:
        for row in csv.DictReader(args.csv.open(encoding="utf8")):
            done_ids.add(int(row["cod_id"]))
            tag_counts[row["temperature_tag"] or "none"] = (
                tag_counts.get(row["temperature_tag"] or "none", 0) + 1
            )
            rebuilt = record_from_csv_row(row)
            if rebuilt is not None:
                records.append(rebuilt)
            elif row.get("outcome"):
                excluded.append({"cod_id": int(row["cod_id"]), "reason": row["outcome"]})
        print(
            f"resuming: {len(done_ids)} rows already on disk,"
            f" {len(records)} of them usable, {n - len(done_ids)} left to relax"
        )
        print()

    # Written and flushed row by row rather than at the end: the full run takes
    # over an hour, and a partial CSV that can be read while it goes is worth
    # more than a complete one that only exists if nothing crashes.
    with args.csv.open("a" if appending else "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        if not appending:
            writer.writeheader()
        fh.flush()

        for index, row in enumerate(sample, 1):
            cod_id = int(row["cod_id"])
            if cod_id in done_ids:
                continue
            record = {
                "cod_id": cod_id,
                "formula": row["formula_reduced"],
                "n_sites": int(row["n_sites"]),
            }
            prefix = f"  [{index:>3}/{n}] {cod_id:>8} {row['formula_reduced']:<16}"

            status, text = fetch_cif(session, cod_id, CIF_CACHE)
            if status != 200 or not text.strip():
                record["outcome"] = f"excluded: fetch http {status}"
                excluded.append({"cod_id": cod_id, "reason": f"fetch http {status}"})
                writer.writerow(csv_row(record))
                fh.flush()
                print(f"{prefix} {record['outcome']}")
                continue

            kelvin, tag, note = cif_temperature(text)
            record["temperature_k"] = kelvin
            record["temperature_tag"] = tag
            tag_counts[tag or "none"] = tag_counts.get(tag or "none", 0) + 1
            if note and note != "absent":
                temperature_notes.append(f"{cod_id}: {note}")

            # The catalogue was built by this same filter, so a rejection here
            # means the CIF or pymatgen changed since. Re-run rather than trust.
            result = evaluate(text, max_atoms=args.max_sites)
            if not result.survived:
                reason = f"{result.stage_failed}: {result.reject_detail}"
                record["outcome"] = f"excluded: {reason}"
                excluded.append({"cod_id": cod_id, "reason": reason})
                writer.writerow(csv_row(record))
                fh.flush()
                print(f"{prefix} {record['outcome']}")
                continue

            deposited = result.structure.lattice
            try:
                relaxed = relax(
                    result.structure,
                    fmax=args.fmax,
                    max_steps=args.max_steps,
                    relax_cell=True,
                )
            except Exception as exc:  # CHGNet and ASE raise a wide spread of types
                reason = f"relax raised {type(exc).__name__}: {exc}"[:200]
                record["outcome"] = f"error: {reason}"
                excluded.append({"cod_id": cod_id, "reason": reason})
                writer.writerow(csv_row(record))
                fh.flush()
                print(f"{prefix} {record['outcome']}")
                continue

            final = relaxed.final_structure.lattice
            record.update(
                {
                    # Read off the parsed composition, not the catalogue row, so
                    # the per-bin chemistry counts describe the cell that was
                    # actually relaxed. The formula column carries it into the CSV.
                    "contains_carbon": bool(result.contains_carbon),
                    "volume_deposited_a3": deposited.volume,
                    "volume_relaxed_a3": final.volume,
                    "volume_error_pct": 100 * (final.volume - deposited.volume) / deposited.volume,
                    "steps": len(relaxed.steps),
                    "converged": relaxed.converged,
                    "outcome": relaxed.outcome,
                    "elapsed_s": relaxed.elapsed_s,
                }
            )
            for axis, before, after in zip(("a", "b", "c"), deposited.abc, final.abc, strict=True):
                record[f"{axis}_deposited_a"] = before
                record[f"{axis}_relaxed_a"] = after
                record[f"{axis}_error_pct"] = 100 * (after - before) / before

            record["bin"] = temperature_bin(kelvin)
            records.append(record)
            writer.writerow(csv_row(record))
            fh.flush()

            shown = f"{kelvin:.0f} K" if kelvin is not None else "no T"
            print(
                f"{prefix} {record['n_sites']:>3} sites {shown:>7}"
                f"  volume {record['volume_error_pct']:>+7.2f}%"
                f"  {record['steps']:>3} steps  {relaxed.outcome:<13}"
                f" {relaxed.elapsed_s:>6.1f}s"
            )

    elapsed = time.perf_counter() - started
    if not records:
        print("no structure relaxed -- nothing to summarise", file=sys.stderr)
        return 1

    converged = [r for r in records if r["converged"]]
    unconverged = [r for r in records if not r["converged"]]
    with_temperature = [r for r in records if r["temperature_k"] is not None]

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/validate_against_experiment.py",
        "question": (
            "how far does a CHGNet cell relaxation land from the deposited "
            "experimental cell, and does that distance depend on the temperature "
            "the experiment was run at"
        ),
        # random.sample is not guaranteed stable across Python releases:
        # reproducing this row-for-row needs the same interpreter.
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
        "sample": {
            "size": n,
            "seed": args.seed,
            "catalog": rel(args.catalog),
            "candidates_in_size_range": len(candidates),
            "site_range": [args.min_sites, args.max_sites],
        },
        "relaxation": {
            "relax_cell": True,
            "fmax_ev_per_a": args.fmax,
            "max_steps": args.max_steps,
            "wall_clock_s": round(elapsed, 1),
            "median_relax_s": round(statistics.median(r["elapsed_s"] for r in records), 2),
            "median_steps": int(statistics.median(r["steps"] for r in records)),
        },
        "counts": {
            "sampled": n,
            "relaxed": len(records),
            "converged": len(converged),
            "not_converged": len(unconverged),
            "excluded": len(excluded),
        },
        "excluded": {
            "count": len(excluded),
            "entries": excluded,
        },
        "not_converged": {
            "count": len(unconverged),
            "note": (
                "kept in the CSV and reported below; these runs hit the step "
                "ceiling under-relaxed, so their errors are biased toward the "
                "deposited cell"
            ),
            "cod_ids": [r["cod_id"] for r in unconverged],
        },
        "temperature": {
            "tags_searched": list(TEMPERATURE_TAGS),
            "preferred_tag": TEMPERATURE_TAGS[0],
            "sanity_range_k": [TEMPERATURE_MIN_K, TEMPERATURE_MAX_K],
            # Denominator is every CIF read, including any the filter then
            # excluded, so this fraction describes the corpus rather than the
            # subset that happened to relax.
            "cifs_read": sum(tag_counts.values()),
            "tag_used_counts": tag_counts,
            "with_usable_temperature": len(with_temperature),
            "with_usable_temperature_pct": round(100 * len(with_temperature) / len(records), 2),
            "with_usable_temperature_denominator": "relaxed structures",
            "rejected_values": temperature_notes,
            "bin_definitions": dict(TEMPERATURE_BINS),
        },
        "statistics": {
            "basis": "converged relaxations only",
            "overall": summarise(converged),
            "by_temperature_bin": by_bin(converged),
        },
        "statistics_including_non_converged": {
            "basis": "every relaxation that produced a cell, converged or not",
            "overall": summarise(records),
            "by_temperature_bin": by_bin(records),
        },
        "limitations": list(LIMITATIONS),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf8")

    volume = summary["statistics"]["overall"]["volume_error_pct"]
    print()
    print(
        f"{len(records)} relaxed, {len(converged)} converged,"
        f" {len(unconverged)} not converged, {len(excluded)} excluded"
        f"  ({elapsed / 60:.0f} min)"
    )
    print(
        f"{len(with_temperature)} of {len(records)} CIFs carried a usable temperature"
        f" ({summary['temperature']['with_usable_temperature_pct']:.0f}%)"
    )
    print()
    print("volume error vs the deposited cell, converged relaxations only:")
    print(
        f"  {'bin':<12} {'n':>5}  {'median signed':>13} {'95% CI':>16}"
        f" {'median |err|':>13} {'IQR':>8} {'with C':>7} {'median T':>8}"
    )
    print(f"  {'-' * 12} {'-' * 5}  {'-' * 13} {'-' * 16} {'-' * 13} {'-' * 8} {'-' * 7} {'-' * 8}")
    print_group("overall", summary["statistics"]["overall"])
    for key, _ in TEMPERATURE_BINS:
        print_group(key, summary["statistics"]["by_temperature_bin"][key])
    print()
    print(
        f"  larger than deposited {volume['n_larger_than_deposited']},"
        f" smaller {volume['n_smaller_than_deposited']},"
        f" |error| over 10% {volume['n_abs_error_over_10pct']}"
    )
    print()
    print(f"wrote {rel(args.csv)}")
    print(f"wrote {rel(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
