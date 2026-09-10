#!/usr/bin/env python3
"""Fetch one or more COD entries and run them through anneal's filter.

This is the command-line face of the same path the application takes when a
user pastes a COD id: fetch, filter, and either accept the structure or say
exactly why it was refused.

Usage
-----
    python scripts/fetch_cod.py 1000041
    python scripts/fetch_cod.py 1000041 9016675 2202845 --json
    python scripts/fetch_cod.py --from-file ids.txt --max-atoms 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anneal import (
    DEFAULT_CIF_BYTE_CAP,
    DEFAULT_MAX_ATOMS,
    evaluate,
    fetch_cif,
    make_session,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CIF_CACHE = REPO_ROOT / "data" / "cache" / "cif"


def describe(cod_id: int, status: int, result) -> str:
    if status != 200:
        reason = {413: "file above the size cap", 0: "transport error"}.get(
            status, f"http {status}"
        )
        return f"{cod_id}  REFUSED  fetch: {reason}"
    if not result.survived:
        return f"{cod_id}  REFUSED  {result.stage_failed}: {result.reject_detail}"
    prim = f", {result.n_sites_primitive} primitive" if result.n_sites_primitive is not None else ""
    return (
        f"{cod_id}  OK       {result.formula_parsed}"
        f"  [{result.n_sites} sites{prim}, {result.volume_a3:.1f} A^3]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cod_ids", nargs="*", type=int)
    ap.add_argument("--from-file", type=Path, help="file with one COD id per line")
    ap.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    ap.add_argument("--byte-cap", type=int, default=DEFAULT_CIF_BYTE_CAP)
    ap.add_argument("--cache-dir", type=Path, default=CIF_CACHE)
    ap.add_argument("--json", action="store_true", help="emit one JSON object per entry")
    args = ap.parse_args()

    ids = list(args.cod_ids)
    if args.from_file:
        ids += [int(line) for line in args.from_file.read_text().split() if line.strip()]
    if not ids:
        ap.error("give at least one COD id, or --from-file")

    session = make_session()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    for cod_id in ids:
        status, text = fetch_cif(session, cod_id, args.cache_dir, byte_cap=args.byte_cap)
        result = evaluate(text, max_atoms=args.max_atoms) if status == 200 else None
        if result is not None and result.survived:
            accepted += 1

        if args.json:
            row = {"cod_id": cod_id, "http_status": status}
            row.update(result.as_row() if result is not None else {"survived": False})
            print(json.dumps(row))
        else:
            print(describe(cod_id, status, result))

    if not args.json and len(ids) > 1:
        print(f"\n{accepted}/{len(ids)} accepted at max_atoms={args.max_atoms}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
