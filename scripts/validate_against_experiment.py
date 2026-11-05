#!/usr/bin/env python3
"""Measure how far CHGNet's relaxed cell lands from the cell measured by diffraction.

A fixed random sample of catalogue structures is relaxed with the cell free, and
the relaxed lattice is compared with the deposited one. The reference is an
experiment, not a DFT calculation, so the residual is not CHGNet's error against
its training data. Thermal expansion and the PBE lattice bias both act on it, so
the measurement temperature is read from each CIF and results are binned by it.

Method
------
1. Draw a fixed-seed sample from data/catalog.csv, cells of 2 to 100 sites.
2. Relax each with the cell free until the largest atomic force is at most
   --fmax and the largest stress component is at most --stress-tol-gpa.
3. Record the signed percentage error in volume and in a, b and c, and the
   stress left on the cell when the run stopped.
4. Read `_cell_measurement_temperature`, falling back to
   `_diffrn_ambient_temperature`, and bin on it.

Runs that hit the step ceiling are kept in the CSV and reported separately.
Structures run in parallel worker processes. Each worker holds about 1.2 GB, so
on a machine short of memory use fewer workers with more threads each. Torch
results can differ in the last bits between thread counts, so the summary
records both.

Outputs data/validation_vs_experiment.csv and data/validation_summary.json.

Usage
-----
    python scripts/validate_against_experiment.py --sample-size 5
    python scripts/validate_against_experiment.py --sample-size 200 --workers 4 --threads 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version as dist_version
from pathlib import Path

from anneal import evaluate, fetch_cif, make_session, relax

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CIF_CACHE = DATA_DIR / "cache" / "cif"

# _cell_measurement_temperature describes how the cell constants were measured,
# which is the quantity compared here, so it wins when both tags are present.
TEMPERATURE_TAGS = ("_cell_measurement_temperature", "_diffrn_ambient_temperature")
TEMPERATURE_MIN_K = 1.0
TEMPERATURE_MAX_K = 1500.0

# Both tags are in kPa. Ambient is about 101 kPa; anything above 1 MPa was
# measured in a pressure cell and is compared here against a zero-pressure
# relaxation.
PRESSURE_TAGS = ("_cell_measurement_pressure", "_diffrn_ambient_pressure")
AMBIENT_MAX_GPA = 0.001

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
    "pressure_gpa",
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
    "final_max_force_ev_per_a",
    "final_stress_gpa",
    "steps",
    "converged",
    "outcome",
    "elapsed_s",
)

LIMITATIONS = (
    "Relaxation is at zero pressure. Deposited cells measured in a pressure cell are "
    "listed under measured_under_pressure and kept in the statistics.",
    "The reference is a measured cell, not a DFT cell. This is CHGNet's distance "
    "from experiment, not its error against the DFT-PBE data it was trained on.",
    "This measures geometry only. It says nothing about the energies anneal "
    "reports, which have no experimental counterpart here.",
    "A temperature bin is a population of structures, not a controlled variable. "
    "Bins differ in chemistry and cell size as well as temperature, so each bin "
    "reports its carbon-containing count and median site count beside its error.",
    "Zero-point motion expands a real crystal even at 0 K. CHGNet predicts a "
    "static lattice, so even the coldest bin is not a like-for-like comparison.",
    "A stress threshold leaves a residual volume strain of roughly threshold over "
    "bulk modulus: about 1% for a molecular crystal near 10 GPa at 0.1 GPa, and a "
    "few tenths of a percent for stiff inorganic solids.",
    "a, b and c are compared vector by vector in the deposited setting. For a "
    "large or strongly sheared deformation the volume error is the robust number.",
    "Runs that hit the step ceiling are under-relaxed, so their errors lean toward "
    "the deposited cell. They are reported separately.",
    "The sample comes from data/catalog.csv, a volume-bounded COD query already "
    "filtered for ordered sites, declared hydrogen and Z <= 94, cut here to 2-100 "
    "sites. The statistics describe that population, not COD as a whole.",
    "Hydrogen positions from X-ray refinement are poorly determined, and CHGNet "
    "moves them. Structures containing hydrogen are not separated out here.",
)

# A capital C not followed by a lowercase letter, so Ca, Cl, Cu and Cs do not count.
CARBON_IN_FORMULA = re.compile(r"C(?![a-z])")


def rel(path: Path) -> str:
    """Repo-relative POSIX path, so the summary reads the same on every OS."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def cif_temperature(cif_text: str) -> tuple[float | None, str, str]:
    """Measurement temperature in kelvin, the tag it came from, and any rejected values.

    The parenthesised uncertainty in values like `293(2)` is stripped. `?` and
    `.` mean no value. A value outside the sanity range falls through to the
    other tag and is reported.
    """
    rejected: list[str] = []
    for tag in TEMPERATURE_TAGS:
        match = re.search(rf"^{tag}\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", cif_text, re.MULTILINE)
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


def cif_pressure_gpa(cif_text: str) -> float | None:
    """Measurement pressure in GPa from the first pressure tag with a value, or None."""
    for tag in PRESSURE_TAGS:
        match = re.search(rf"^{tag}\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", cif_text, re.MULTILINE)
        if not match:
            continue
        raw = next((g for g in match.groups() if g is not None), "").split("(")[0].strip()
        try:
            return float(raw) / 1e6
        except ValueError:
            continue
    return None


def temperature_bin(kelvin: float | None) -> str:
    if kelvin is None:
        return "unknown"
    if kelvin < 150:
        return "lt_150k"
    if kelvin <= 250:
        return "150_250k"
    return "gt_250k"


def quartiles(values: list[float]) -> tuple[float | None, float | None]:
    """Q1 and Q3, or None when there are fewer than two points."""
    if len(values) < 2:
        return None, None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def median_interval(values: list[float]) -> list[float] | None:
    """Distribution-free 95% confidence interval for the median, from order statistics.

    Coverage of [x_(k), x_(n-k+1)] follows the binomial tail, so the interval is
    the widest one whose two tails each stay at or under 2.5%. None below n=6.
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
    """Error statistics over completed relaxations. Thin groups get None, not 0."""
    if not records:
        return {"n": 0, "note": "no relaxations in this group"}

    signed = sorted(r["volume_error_pct"] for r in records)
    absolute = sorted(abs(v) for v in signed)
    q1, q3 = quartiles(signed)
    abs_q1, abs_q3 = quartiles(absolute)
    temperatures = [r["temperature_k"] for r in records if r["temperature_k"] is not None]
    stresses = sorted(r["final_stress_gpa"] for r in records)

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
            "n_abs_error_over_10pct": sum(1 for v in absolute if v > 10),
            "n_abs_error_over_20pct": sum(1 for v in absolute if v > 20),
        },
        "final_stress_gpa": {
            "median": round(statistics.median(stresses), 4),
            "max": round(stresses[-1], 4),
        },
        "axis_error_pct": {},
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


def record_from_csv_row(row: dict) -> dict | None:
    """Rebuild a record from a CSV row written by an earlier run, or None if excluded."""
    if not row.get("volume_error_pct"):
        return None
    kelvin = float(row["temperature_k"]) if row["temperature_k"] else None
    record = {
        "cod_id": int(row["cod_id"]),
        "formula": row["formula"],
        "n_sites": int(row["n_sites"]),
        "temperature_k": kelvin,
        "temperature_tag": row["temperature_tag"] or "",
        "pressure_gpa": float(row["pressure_gpa"]) if row.get("pressure_gpa") else None,
        "contains_carbon": bool(CARBON_IN_FORMULA.search(row["formula"])),
        "volume_deposited_a3": float(row["volume_deposited_a3"]),
        "volume_relaxed_a3": float(row["volume_relaxed_a3"]),
        "volume_error_pct": float(row["volume_error_pct"]),
        "final_max_force_ev_per_a": float(row["final_max_force_ev_per_a"]),
        "final_stress_gpa": float(row["final_stress_gpa"]),
        "steps": int(row["steps"]),
        # csv writes booleans as text, and the string "False" is truthy.
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
    if not stats.get("n"):
        print(f"  {label:<12} {0:>5}  {'--':>13} {'--':>16} {'--':>13} {'--':>8} {'--':>7}")
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


def _init_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from anneal.relax import load_model

    load_model()


def _model_info() -> dict:
    from anneal.relax import model_info

    return model_info()


def relax_one(row: dict, settings: dict) -> dict:
    """Relax one catalogue entry. Returns the record, or an excluded entry with a reason."""
    cod_id = int(row["cod_id"])
    record = {"cod_id": cod_id, "formula": row["formula_reduced"], "n_sites": int(row["n_sites"])}

    status, text = fetch_cif(make_session(), cod_id, CIF_CACHE)
    if status != 200 or not text.strip():
        record["outcome"] = f"excluded: fetch http {status}"
        return {"record": record, "excluded": f"fetch http {status}", "tag": "", "note": ""}

    kelvin, tag, note = cif_temperature(text)
    record["temperature_k"] = kelvin
    record["temperature_tag"] = tag
    record["pressure_gpa"] = cif_pressure_gpa(text)

    result = evaluate(text, max_atoms=settings["max_sites"])
    if not result.survived:
        reason = f"{result.stage_failed}: {result.reject_detail}"
        record["outcome"] = f"excluded: {reason}"
        return {"record": record, "excluded": reason, "tag": tag, "note": note}

    deposited = result.structure.lattice
    try:
        relaxed = relax(
            result.structure,
            fmax=settings["fmax"],
            stress_tol_gpa=settings["stress_tol_gpa"],
            max_steps=settings["max_steps"],
            relax_cell=True,
        )
    except Exception as exc:  # CHGNet and ASE raise a wide spread of types
        reason = f"relax raised {type(exc).__name__}: {exc}"[:200]
        record["outcome"] = f"error: {reason}"
        return {"record": record, "excluded": reason, "tag": tag, "note": note}

    final = relaxed.final_structure.lattice
    last = relaxed.steps[-1]
    record.update(
        {
            "contains_carbon": bool(result.contains_carbon),
            "volume_deposited_a3": deposited.volume,
            "volume_relaxed_a3": final.volume,
            "volume_error_pct": 100 * (final.volume - deposited.volume) / deposited.volume,
            "final_max_force_ev_per_a": last.max_force_ev_per_a,
            "final_stress_gpa": last.max_stress_gpa,
            "steps": len(relaxed.steps),
            "converged": relaxed.converged,
            "outcome": relaxed.outcome,
            "elapsed_s": relaxed.elapsed_s,
            "bin": temperature_bin(kelvin),
        }
    )
    for axis, before, after in zip(("a", "b", "c"), deposited.abc, final.abc, strict=True):
        record[f"{axis}_deposited_a"] = before
        record[f"{axis}_relaxed_a"] = after
        record[f"{axis}_error_pct"] = 100 * (after - before) / before
    return {"record": record, "excluded": None, "tag": tag, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--min-sites", type=int, default=2)
    ap.add_argument("--max-sites", type=int, default=100)
    ap.add_argument("--fmax", type=float, default=0.02, help="largest atomic force, eV/A")
    ap.add_argument(
        "--stress-tol-gpa", type=float, default=0.1, help="largest stress component, GPa"
    )
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--threads", type=int, default=0, help="torch threads per worker; 0 splits the CPUs"
    )
    ap.add_argument("--catalog", type=Path, default=DATA_DIR / "catalog.csv")
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "validation_vs_experiment.csv")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip cod_ids already in the CSV and append. Use the same --sample-size and "
            "--seed as the run being resumed, or the two halves are different samples."
        ),
    )
    ap.add_argument("--summary", type=Path, default=DATA_DIR / "validation_summary.json")
    args = ap.parse_args()
    if args.threads <= 0:
        args.threads = max(1, (os.cpu_count() or 1) // args.workers)

    if not args.catalog.exists():
        print(f"no catalogue at {rel(args.catalog)}; run scripts/build_catalog.py", file=sys.stderr)
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
    print(
        f"validating {n} of {len(candidates):,} catalogue entries with"
        f" {args.min_sites}-{args.max_sites} sites (seed {args.seed})"
    )
    print(
        f"cell free, converged at force <= {args.fmax} eV/A and stress <="
        f" {args.stress_tol_gpa} GPa, ceiling {args.max_steps} steps,"
        f" {args.workers} workers x {args.threads} threads"
    )
    print()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    excluded: list[dict] = []
    tag_counts: dict[str, int] = {}
    temperature_notes: list[str] = []
    started = time.perf_counter()

    done_ids: set[int] = set()
    # Filled for rows from an earlier run, which may predate the pressure column.
    pressure_by_id: dict[int, float | None] = {}
    appending = bool(args.resume and args.csv.exists())
    if appending:
        for row in csv.DictReader(args.csv.open(encoding="utf8")):
            done_ids.add(int(row["cod_id"]))
            tag = row["temperature_tag"] or "none"
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            # The CSV keeps no rejected temperature values; the cached CIF still has them.
            status, text = fetch_cif(make_session(), int(row["cod_id"]), CIF_CACHE)
            if status == 200:
                note = cif_temperature(text)[2]
                if note and note != "absent":
                    temperature_notes.append(f"{row['cod_id']}: {note}")
                pressure_by_id[int(row["cod_id"])] = cif_pressure_gpa(text)
            rebuilt = record_from_csv_row(row)
            if rebuilt is not None:
                rebuilt["pressure_gpa"] = pressure_by_id.get(rebuilt["cod_id"])
                records.append(rebuilt)
            elif row.get("outcome"):
                excluded.append({"cod_id": int(row["cod_id"]), "reason": row["outcome"]})
        print(f"resuming: {len(done_ids)} rows on disk, {n - len(done_ids)} left to relax")
        print()

    settings = {
        "fmax": args.fmax,
        "stress_tol_gpa": args.stress_tol_gpa,
        "max_steps": args.max_steps,
        "max_sites": args.max_sites,
    }
    todo = [row for row in sample if int(row["cod_id"]) not in done_ids]

    # Rows are written as they finish, so an interrupted run resumes instead of restarting.
    with args.csv.open("a" if appending else "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        if not appending:
            writer.writeheader()
        fh.flush()

        # The model is loaded only in the workers; each one holds about 1.2 GB.
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(args.threads,)
        ) as pool:
            info = pool.submit(_model_info).result()
            print(f"model: CHGNet {info['pretrained_version']}, package {info['package_version']}")
            futures = [pool.submit(relax_one, row, settings) for row in todo]
            for done, future in enumerate(as_completed(futures), len(done_ids) + 1):
                out = future.result()
                record = out["record"]
                tag_counts[out["tag"] or "none"] = tag_counts.get(out["tag"] or "none", 0) + 1
                if out["note"] and out["note"] != "absent":
                    temperature_notes.append(f"{record['cod_id']}: {out['note']}")
                writer.writerow(csv_row(record))
                fh.flush()

                prefix = f"  [{done:>3}/{n}] {record['cod_id']:>8} {record['formula']:<16}"
                if out["excluded"] is not None:
                    excluded.append({"cod_id": record["cod_id"], "reason": out["excluded"]})
                    print(f"{prefix} {record['outcome']}", flush=True)
                    continue
                records.append(record)
                kelvin = record["temperature_k"]
                shown = f"{kelvin:.0f} K" if kelvin is not None else "no T"
                print(
                    f"{prefix} {record['n_sites']:>3} sites {shown:>7}"
                    f"  volume {record['volume_error_pct']:>+7.2f}%"
                    f"  stress {record['final_stress_gpa']:>5.2f} GPa"
                    f"  {record['steps']:>4} steps  {record['outcome']:<13}"
                    f" {record['elapsed_s']:>6.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started

    # Completion order depends on the worker count; the committed file should not.
    with args.csv.open(encoding="utf8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda row: int(row["cod_id"]))
    for row in rows:
        if int(row["cod_id"]) in pressure_by_id:
            pressure = pressure_by_id[int(row["cod_id"])]
            row["pressure_gpa"] = "" if pressure is None else round(pressure, 4)
    with args.csv.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    if not records:
        print("no structure relaxed, nothing to summarise", file=sys.stderr)
        return 1

    records.sort(key=lambda r: r["cod_id"])
    converged = [r for r in records if r["converged"]]
    unconverged = [r for r in records if not r["converged"]]
    with_temperature = [r for r in records if r["temperature_k"] is not None]
    pressurised = [r for r in converged if (r.get("pressure_gpa") or 0.0) > AMBIENT_MAX_GPA]
    ambient = [r["volume_error_pct"] for r in converged if r not in pressurised]

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": "scripts/validate_against_experiment.py",
        "question": (
            "how far does a CHGNet cell relaxation land from the deposited "
            "experimental cell, and does that distance depend on the temperature "
            "the experiment was run at"
        ),
        # random.sample is not guaranteed stable across Python releases.
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
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
            "stress_tol_gpa": args.stress_tol_gpa,
            "max_steps": args.max_steps,
            # A resumed run only describes its own part, so the names say so.
            "workers_last_invocation": args.workers,
            "torch_threads_per_worker_last_invocation": args.threads,
            "rows_from_earlier_invocations": len(done_ids),
            "wall_clock_s_last_invocation": round(elapsed, 1),
            "median_relax_s": round(statistics.median(r["elapsed_s"] for r in records), 2),
            "total_relax_s": round(sum(r["elapsed_s"] for r in records), 1),
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
            "entries": sorted(excluded, key=lambda e: e["cod_id"]),
        },
        "not_converged": {
            "count": len(unconverged),
            "note": "hit the step ceiling under-relaxed; errors lean toward the deposited cell",
            "cod_ids": [r["cod_id"] for r in unconverged],
        },
        "temperature": {
            "tags_searched": list(TEMPERATURE_TAGS),
            "preferred_tag": TEMPERATURE_TAGS[0],
            "sanity_range_k": [TEMPERATURE_MIN_K, TEMPERATURE_MAX_K],
            "cifs_read": sum(tag_counts.values()),
            "tag_used_counts": tag_counts,
            "with_usable_temperature": len(with_temperature),
            "with_usable_temperature_pct": round(100 * len(with_temperature) / len(records), 2),
            "with_usable_temperature_denominator": "relaxed structures",
            "rejected_values": sorted(temperature_notes),
            "bin_definitions": dict(TEMPERATURE_BINS),
        },
        "measured_under_pressure": {
            "tags_searched": list(PRESSURE_TAGS),
            "threshold_gpa": AMBIENT_MAX_GPA,
            "note": "deposited cells measured in a pressure cell, compared against a "
            "zero-pressure relaxation; kept in the statistics",
            "entries": [
                {
                    "cod_id": r["cod_id"],
                    "formula": r["formula"],
                    "pressure_gpa": round(r["pressure_gpa"], 4),
                    "volume_error_pct": round(r["volume_error_pct"], 3),
                }
                for r in pressurised
            ],
            "median_signed_without_them": round(statistics.median(ambient), 3),
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
    stress = summary["statistics"]["overall"]["final_stress_gpa"]
    print()
    print(
        f"{len(records)} relaxed, {len(converged)} converged,"
        f" {len(unconverged)} not converged, {len(excluded)} excluded  ({elapsed / 60:.0f} min)"
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
    print_group("overall", summary["statistics"]["overall"])
    for key, _ in TEMPERATURE_BINS:
        print_group(key, summary["statistics"]["by_temperature_bin"][key])
    print()
    print(
        f"  mean {volume['mean_signed']:+.2f}%, max {volume['max']:+.2f}%;"
        f" larger than deposited {volume['n_larger_than_deposited']},"
        f" smaller {volume['n_smaller_than_deposited']},"
        f" |error| over 10% {volume['n_abs_error_over_10pct']}"
    )
    print(f"  residual stress: median {stress['median']:.3f} GPa, max {stress['max']:.3f} GPa")
    print()
    print(f"wrote {rel(args.csv)}")
    print(f"wrote {rel(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
