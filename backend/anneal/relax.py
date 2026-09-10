"""Running CHGNet: single-point energies and relaxations with per-step callbacks.

CHGNet ships `StructOptimizer.relax`, which is fine for batch work but builds its
ASE optimizer internally and attaches its own observer, so there is no way to see
a step while it is happening. anneal needs that -- the force-convergence trace is
the point -- so the loop is driven here directly on ASE with the same defaults
CHGNet uses (FIRE, stress_weight = 1 GPa in eV/A^3).

Cell relaxation defaults to ON, and that default was chosen from a measurement,
not from taste. In a centrosymmetric crystal every atom sits at an inversion
centre where forces vanish by symmetry, and a symmetric edit leaves them there:
removing an Na from rocksalt NaCl gives a maximum force of 1.8e-06 eV/A. With
positions alone there is nothing to relax and the screen does not move. Letting
the cell respond gives 13 steps and -0.070 eV for the same edit.

Strain is the exception and passes relax_cell=False, because relaxing the cell
after the user has set it just undoes their edit. anneal.mutations.Mutation
carries the right choice per edit in `suggested_relax_cell`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

# Same optimizer and stress conversion CHGNet's own StructOptimizer defaults to.
DEFAULT_FMAX = 0.1
DEFAULT_RELAX_CELL = True

# A hard ceiling for pathological cases, not a user-experience knob. The
# relaxation streams every step as it computes, so a long run is something you
# watch rather than wait through, and capping it lower would fire on ordinary
# structures -- the median relaxation in data/benchmark.json takes 51 steps.
# When this ceiling does fire, the result reports converged=False and the max
# force it stopped at.
DEFAULT_MAX_STEPS = 200

_model = None
_model_lock = threading.Lock()


def load_model(use_device: str = "cpu"):
    """Load CHGNet once and hand out the same instance afterwards.

    The checkpoint is ~4.9 MB but the first call also pays for importing torch,
    so it is slow in a way the second call is not. scripts/benchmark.py reports
    both numbers separately rather than quoting one as "model load time".
    """
    global _model
    with _model_lock:
        if _model is None:
            from chgnet.model import CHGNet

            _model = CHGNet.load(use_device=use_device, verbose=False)
        return _model


def model_info() -> dict:
    """Parameter counts and versions, read off the loaded model."""
    import chgnet

    model = load_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "package_version": chgnet.__version__,
        "pretrained_version": getattr(model, "version", "unknown"),
        "total_parameters": total,
        "trainable_parameters": trainable,
        # The non-trainable remainder is the composition model's fixed
        # per-element reference energies -- one per element CHGNet covers.
        "element_reference_energies": total - trainable,
    }


def graph_edge_count(structure: Structure) -> int:
    """Directed edges in the graph CHGNet actually convolves over.

    This is the size that governs cost, not the atom count. CHGNet's message
    passing runs per edge, and edge count depends on how densely packed the cell
    is as well as how many atoms it holds -- a 66-atom molecular crystal and an
    88-atom oxide came out at 5,418 and 5,424 edges respectively, and took
    almost the same time per step despite differing by a third in atom count.

    Fitting cost against edges instead of atoms took R^2 from 0.85 to 0.96 on
    the measurements in data/benchmark.json.
    """
    from chgnet.graph import CrystalGraphConverter

    converter = CrystalGraphConverter(on_isolated_atoms="ignore")
    graph = converter(structure, graph_id="estimate")
    return int(graph.atom_graph.shape[0])


def estimate_relaxation(structure: Structure, benchmark: dict | None = None) -> dict | None:
    """Predict how long a relaxation will take, on the machine that benchmarked.

    Returns a range, not a point. Two different quantities are uncertain here and
    both are quoted: cost per step, which the edge fit predicts to within a
    measured residual spread, and the number of steps, which depends on how far
    the edit pushed the structure from its minimum and varied 10 to 88 across the
    benchmark set. Anything narrower than a range would be false precision.

    Returns None when no benchmark has been run -- an estimate from someone
    else's CPU is worse than no estimate.
    """
    if benchmark is None:
        benchmark = load_benchmark()
    if not benchmark:
        return None
    fit = benchmark.get("cost_model")
    if not fit:
        return None

    edges = graph_edge_count(structure)
    ms_per_step = max(fit["floor_ms"], fit["intercept_ms"] + fit["ms_per_edge"] * edges)
    spread = fit.get("rel_spread", 0.3)
    steps_low, steps_high = fit.get("step_range", [10, 90])

    median_steps = fit.get("median_steps", 51)
    return {
        "atoms": len(structure),
        "atom_graph_edges": edges,
        "ms_per_step": round(ms_per_step, 1),
        # The figure to show. Step count is the dominant unknown, so this is the
        # median relaxation length, not a worst case.
        "seconds_typical": round(ms_per_step * median_steps / 1000, 1),
        # The honest outer bounds: cheapest step count at the low end of the
        # cost fit, longest at the high end. Wide on purpose -- an edit that
        # barely disturbs the structure and one that wrecks it are both possible.
        "seconds_low": round(ms_per_step * (1 - spread) * steps_low / 1000, 1),
        "seconds_high": round(ms_per_step * (1 + spread) * steps_high / 1000, 1),
        "median_steps": median_steps,
        "measured_on": benchmark.get("environment", {}).get("processor", "unknown"),
        "benchmark_generated_at": benchmark.get("generated_at"),
    }


def load_benchmark(path: Path | None = None) -> dict | None:
    """Read data/benchmark.json if scripts/benchmark.py has been run here."""
    import json

    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "benchmark.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


@dataclass
class RelaxStep:
    """One optimizer step, in the shape the frontend animates."""

    step: int
    energy_ev: float
    energy_per_atom_ev: float
    max_force_ev_per_a: float
    elapsed_s: float
    positions: list[list[float]] = field(default_factory=list)
    lattice: list[list[float]] = field(default_factory=list)
    # Per-atom force vectors, not just their maximum. The viewer draws these as
    # the accent overlay, so they have to travel with the step -- a scalar max
    # force is enough for the trace and useless for the arrows.
    forces: list[list[float]] = field(default_factory=list)

    def as_dict(self, *, with_geometry: bool = True) -> dict:
        out = {
            "step": self.step,
            "energy_ev": round(self.energy_ev, 6),
            "energy_per_atom_ev": round(self.energy_per_atom_ev, 6),
            "max_force_ev_per_a": round(self.max_force_ev_per_a, 6),
            "elapsed_s": round(self.elapsed_s, 4),
        }
        if with_geometry:
            out["positions"] = [[round(x, 5) for x in p] for p in self.positions]
            out["lattice"] = [[round(x, 5) for x in row] for row in self.lattice]
            out["forces"] = [[round(x, 4) for x in f] for f in self.forces]
        return out


@dataclass
class SinglePoint:
    energy_ev: float
    energy_per_atom_ev: float
    max_force_ev_per_a: float
    forces: list[list[float]]
    elapsed_s: float
    n_atoms: int


class _Stopped(Exception):
    """Raised inside the observer to unwind ASE's loop when the user stops it."""


@dataclass
class RelaxResult:
    final_structure: Structure
    steps: list[RelaxStep]
    converged: bool
    fmax: float
    relax_cell: bool
    elapsed_s: float
    stopped_early: bool = False
    hit_step_ceiling: bool = False

    @property
    def energy_change_ev(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].energy_ev - self.steps[0].energy_ev

    @property
    def outcome(self) -> str:
        """How the run ended, in the words the UI shows."""
        if self.stopped_early:
            return "stopped"
        if self.converged:
            return "converged"
        return "ceiling" if self.hit_step_ceiling else "not converged"

    def summary(self) -> str:
        force = self.steps[-1].max_force_ev_per_a if self.steps else float("nan")
        if self.outcome == "converged":
            cell = " (atoms and cell)" if self.relax_cell else ""
            return f"converged in {len(self.steps)} steps{cell}, max force {force:.3f} eV/A"
        if self.outcome == "stopped":
            return f"stopped at step {len(self.steps)}, max force {force:.3f} eV/A"
        return f"stopped at {len(self.steps)} steps without converging, max force {force:.3f} eV/A"


def _calculator():
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(model=load_model(), use_device="cpu")


def single_point(structure: Structure) -> SinglePoint:
    """Energy and forces for one structure, no geometry change."""
    model = load_model()
    started = time.perf_counter()
    prediction = model.predict_structure(structure)
    elapsed = time.perf_counter() - started

    forces = np.asarray(prediction["f"], dtype=float)
    energy_per_atom = float(prediction["e"])
    n_atoms = len(structure)
    return SinglePoint(
        # CHGNet returns energy per atom; the extensive value is what changes
        # when the user removes an atom, so both are carried.
        energy_ev=energy_per_atom * n_atoms,
        energy_per_atom_ev=energy_per_atom,
        max_force_ev_per_a=float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0,
        forces=forces.tolist(),
        elapsed_s=elapsed,
        n_atoms=n_atoms,
    )


def relax(
    structure: Structure,
    *,
    fmax: float = DEFAULT_FMAX,
    max_steps: int = DEFAULT_MAX_STEPS,
    relax_cell: bool = DEFAULT_RELAX_CELL,
    on_step: Callable[[RelaxStep], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RelaxResult:
    """Relax `structure`, calling `on_step` after every optimizer step.

    The callback fires while the relaxation runs, which is what lets the force
    trace draw itself live instead of appearing all at once at the end.

    `should_stop` is polled once per step. When it returns True the run ends
    where it is and the result keeps every step computed so far -- stopping
    early gives you the state at the stop, never a discarded run.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = _calculator()
    target = FrechetCellFilter(atoms) if relax_cell else atoms

    steps: list[RelaxStep] = []
    started = time.perf_counter()

    def record() -> None:
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        step = RelaxStep(
            step=len(steps),
            energy_ev=energy,
            energy_per_atom_ev=energy / len(atoms),
            max_force_ev_per_a=float(np.linalg.norm(forces, axis=1).max()),
            elapsed_s=time.perf_counter() - started,
            positions=atoms.get_positions().tolist(),
            lattice=atoms.get_cell()[:].tolist(),
            forces=forces.tolist(),
        )
        steps.append(step)
        if on_step is not None:
            on_step(step)
        # Checked after the step is recorded, so a stop keeps the geometry the
        # user was looking at when they pressed it.
        if should_stop is not None and should_stop():
            raise _Stopped

    optimizer = FIRE(target, logfile=None)
    optimizer.attach(record, interval=1)

    stopped = False
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
    except _Stopped:
        stopped = True
    elapsed = time.perf_counter() - started

    if not steps:  # already converged before the first step was taken
        record()

    # Ask the optimizer, do not re-derive it. With relax_cell on, ASE judges
    # convergence over the cell filter -- atomic forces AND cell stress -- so
    # comparing the reported atomic max force against fmax gives the wrong
    # answer. A run can sit at 0.036 eV/A on the atoms and still be working,
    # because the cell has not settled.
    converged = not stopped and bool(optimizer.converged())

    return RelaxResult(
        final_structure=AseAtomsAdaptor.get_structure(atoms),
        steps=steps,
        converged=converged,
        stopped_early=stopped,
        hit_step_ceiling=not stopped and len(steps) > max_steps,
        fmax=fmax,
        relax_cell=relax_cell,
        elapsed_s=elapsed,
    )
