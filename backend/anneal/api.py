"""HTTP surface for the viewer.

The relaxation is the animation, not a wait before one, so the central endpoint
streams: `POST /api/relax` opens a Server-Sent Events response and emits one
event per optimizer step as CHGNet computes it. The browser draws the force
trace descending and moves the atoms while the run is still going.

SSE rather than WebSockets because the traffic is one-directional -- steps flow
out, and the only thing flowing in is "stop", which is a separate one-line POST.
A duplex protocol would buy nothing and cost a reconnection story.

The relaxation itself runs on a worker thread. CHGNet is a synchronous torch
call and would otherwise block the event loop for the whole run, which is the
one thing the design says must never happen.
"""

from __future__ import annotations

import asyncio
import csv
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pymatgen.core import Structure

from anneal.cod import DEFAULT_CIF_BYTE_CAP, fetch_cif, make_session
from anneal.filters import DEFAULT_MAX_ATOMS, evaluate
from anneal.mutations import MutationError, check_relaxable, strain, substitute, vacancy
from anneal.relax import (
    DEFAULT_FMAX,
    DEFAULT_MAX_STEPS,
    estimate_relaxation,
    load_benchmark,
    model_info,
    relax,
    single_point,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CIF_CACHE = DATA_DIR / "cache" / "cif"
CATALOG_CSV = DATA_DIR / "catalog.csv"

app = FastAPI(title="anneal", version="0.1.0")

_session = make_session()
_catalog: list[dict] | None = None


# ---------------------------------------------------------------- catalogue


def catalog() -> list[dict]:
    """The browsable catalogue, read once off disk.

    Built by scripts/build_catalog.py from a volume-bounded COD query. Its size
    is a catalogue size, never a survival rate -- that number belongs to
    scripts/survey_cod.py.
    """
    global _catalog
    if _catalog is None:
        if not CATALOG_CSV.exists():
            _catalog = []
        else:
            with CATALOG_CSV.open(encoding="utf8") as fh:
                _catalog = [
                    {
                        **row,
                        "cod_id": int(row["cod_id"]),
                        "n_sites": int(row["n_sites"]),
                        "volume_a3": float(row["volume_a3"]),
                        "spacegroup_number": int(row["spacegroup_number"] or 0) or None,
                    }
                    for row in csv.DictReader(fh)
                ]
    return _catalog


def structure_payload(structure: Structure) -> dict:
    """The geometry the viewer draws, in Cartesian angstroms.

    Fractional coordinates are sent too: the record column shows them, because
    that is what a CIF lists and what a crystallographer reads.
    """
    return {
        "n_sites": len(structure),
        "formula": structure.composition.formula,
        "formula_reduced": structure.composition.reduced_formula,
        "volume_a3": round(structure.volume, 4),
        "lattice": {
            "matrix": [[round(x, 6) for x in row] for row in structure.lattice.matrix],
            "a": round(structure.lattice.a, 5),
            "b": round(structure.lattice.b, 5),
            "c": round(structure.lattice.c, 5),
            "alpha": round(structure.lattice.alpha, 4),
            "beta": round(structure.lattice.beta, 4),
            "gamma": round(structure.lattice.gamma, 4),
        },
        "sites": [
            {
                "index": i,
                "element": site.specie.symbol
                if hasattr(site.specie, "symbol")
                else str(site.specie),
                "z": int(site.specie.Z) if hasattr(site.specie, "Z") else 0,
                "frac": [round(x, 6) for x in site.frac_coords],
                "cart": [round(x, 6) for x in site.coords],
            }
            for i, site in enumerate(structure)
        ],
    }


# ------------------------------------------------------------------ sessions


@dataclass
class Session:
    """One structure a user is working on, plus whatever relaxation is running."""

    original: Structure
    current: Structure
    cod_id: int | None = None
    history: list[dict] = field(default_factory=list)
    stop_flag: threading.Event = field(default_factory=threading.Event)


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"no session {session_id}; load a structure first")
    return session


# ------------------------------------------------------------------- schemas


class LoadRequest(BaseModel):
    cod_id: int = Field(..., description="COD entry id")
    max_atoms: int = DEFAULT_MAX_ATOMS


class MutateRequest(BaseModel):
    session_id: str
    kind: Literal["substitute", "vacancy", "strain"]
    site_index: int | None = None
    element: str | None = None
    strain: float | list[float] | None = None


class RelaxRequest(BaseModel):
    session_id: str
    fmax: float = DEFAULT_FMAX
    max_steps: int = DEFAULT_MAX_STEPS
    relax_cell: bool | None = None


# ------------------------------------------------------------------ endpoints


@app.get("/api/meta")
def meta() -> dict:
    """What is running, so the page never has to hard-code a model fact."""
    benchmark = load_benchmark()
    return {
        "model": model_info(),
        "max_atoms": DEFAULT_MAX_ATOMS,
        "catalog_size": len(catalog()),
        "benchmark": None
        if benchmark is None
        else {
            "generated_at": benchmark.get("generated_at"),
            "processor": benchmark.get("environment", {}).get("processor"),
            "cost_model": benchmark.get("cost_model"),
        },
    }


@app.get("/api/catalog")
def browse(
    q: str = "", element: str = "", system: str = "", limit: int = 60, offset: int = 0
) -> dict:
    rows = catalog()
    if q:
        needle = q.strip().lower()
        rows = [
            r
            for r in rows
            if needle in r["formula_reduced"].lower()
            or needle in str(r["cod_id"])
            or needle in r["spacegroup_symbol"].lower()
        ]
    if element:
        wanted = {e for e in element.split(",") if e}
        rows = [r for r in rows if wanted <= set(r["elements"].split())]
    if system:
        rows = [r for r in rows if r["crystal_system"] == system]
    return {
        "total": len(rows),
        "items": rows[offset : offset + limit],
    }


@app.post("/api/load")
def load(request: LoadRequest) -> dict:
    """Fetch a COD entry, filter it, and open a session if it passes."""
    status, text = fetch_cif(_session, request.cod_id, CIF_CACHE, byte_cap=DEFAULT_CIF_BYTE_CAP)
    if status != 200:
        reason = {413: "file above the size cap", 0: "could not reach COD"}.get(
            status, f"COD returned HTTP {status}"
        )
        raise HTTPException(502 if status == 0 else 404, reason)

    result = evaluate(text, max_atoms=request.max_atoms)
    if not result.survived:
        # A refusal is information, not an error page: say which rule and why.
        raise HTTPException(
            422,
            {
                "stage": result.stage_failed,
                "detail": result.reject_detail,
                "cod_id": request.cod_id,
            },
        )

    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = Session(
            original=result.structure, current=result.structure, cod_id=request.cod_id
        )
    return {
        "session_id": session_id,
        "cod_id": request.cod_id,
        "structure": structure_payload(result.structure),
        "estimate": estimate_relaxation(result.structure),
        "source_url": f"https://www.crystallography.net/cod/{request.cod_id}.html",
    }


@app.post("/api/mutate")
def mutate(request: MutateRequest) -> dict:
    """Apply one edit. Pure geometry -- no model runs here."""
    session = get_session(request.session_id)
    try:
        if request.kind == "substitute":
            if request.site_index is None or not request.element:
                raise MutationError("substitute needs a site and an element")
            structure, mutation = substitute(session.current, request.site_index, request.element)
        elif request.kind == "vacancy":
            if request.site_index is None:
                raise MutationError("vacancy needs a site")
            structure, mutation = vacancy(session.current, request.site_index)
        else:
            if request.strain is None:
                raise MutationError("strain needs a magnitude")
            structure, mutation = strain(session.current, request.strain)

        check_relaxable(structure, DEFAULT_MAX_ATOMS)
    except MutationError as exc:
        raise HTTPException(422, str(exc)) from exc

    session.current = structure
    session.history.append({"kind": mutation.kind, "detail": mutation.detail})

    point = single_point(structure)
    return {
        "structure": structure_payload(structure),
        "mutation": {
            "kind": mutation.kind,
            "detail": mutation.detail,
            "suggested_relax_cell": mutation.suggested_relax_cell,
        },
        "energy": {
            "energy_ev": round(point.energy_ev, 6),
            "energy_per_atom_ev": round(point.energy_per_atom_ev, 6),
            "max_force_ev_per_a": round(point.max_force_ev_per_a, 6),
        },
        "estimate": estimate_relaxation(structure),
        "history": session.history,
    }


@app.post("/api/reset")
def reset(session_id: str) -> dict:
    session = get_session(session_id)
    session.current = session.original
    session.history.clear()
    return {"structure": structure_payload(session.original)}


@app.post("/api/relax/stop")
def stop(session_id: str) -> dict:
    """Ask the running relaxation to stop. It keeps every step already computed."""
    get_session(session_id).stop_flag.set()
    return {"stopping": True}


@app.post("/api/relax")
async def stream_relax(request: RelaxRequest) -> StreamingResponse:
    """Stream one SSE event per optimizer step, as it is computed."""
    session = get_session(request.session_id)
    session.stop_flag.clear()

    relax_cell = request.relax_cell
    if relax_cell is None:
        # Strain is the only edit that wants the cell held; anything else lets
        # the cell respond. See anneal.mutations.Mutation.
        relax_cell = not (session.history and session.history[-1]["kind"] == "strain")

    queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    structure = session.current

    def on_step(step) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("step", step.as_dict()))

    def worker() -> None:
        try:
            result = relax(
                structure,
                fmax=request.fmax,
                max_steps=request.max_steps,
                relax_cell=relax_cell,
                on_step=on_step,
                should_stop=session.stop_flag.is_set,
            )
            session.current = result.final_structure
            payload = {
                "outcome": result.outcome,
                "summary": result.summary(),
                "converged": result.converged,
                "stopped_early": result.stopped_early,
                "hit_step_ceiling": result.hit_step_ceiling,
                "steps": len(result.steps),
                "elapsed_s": round(result.elapsed_s, 3),
                "energy_change_ev": round(result.energy_change_ev, 6),
                "relax_cell": relax_cell,
                "structure": structure_payload(result.final_structure),
            }
            loop.call_soon_threadsafe(queue.put_nowait, ("done", payload))
        except Exception as exc:  # surfaced to the client rather than swallowed
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", {"message": f"{type(exc).__name__}: {exc}"})
            )

    # CHGNet is a blocking torch call. Off the event loop it goes.
    threading.Thread(target=worker, daemon=True).start()

    async def events():
        while True:
            kind, payload = await queue.get()
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if kind in ("done", "error"):
                return

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
