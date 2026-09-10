"""Behavioural tests for the CHGNet wrapper.

These load the real pretrained model -- there is no mock, because the thing
worth testing is that anneal drives CHGNet correctly, and a fake calculator
would test nothing. The model loads once for the whole session and the cells are
small, so the file stays quick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure

from anneal.filters import evaluate
from anneal.mutations import strain, substitute, vacancy
from anneal.relax import model_info, relax, single_point

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NACL_CIF = FIXTURES / "cod-1000041-nacl.cif"
FE3C_CIF = FIXTURES / "cod-9016675-fe3c.cif"


@pytest.fixture(scope="module")
def nacl() -> Structure:
    return evaluate(NACL_CIF.read_text(encoding="utf8")).structure


@pytest.fixture(scope="module")
def fe3c() -> Structure:
    return evaluate(FE3C_CIF.read_text(encoding="utf8")).structure


# ---------------------------------------------------------------- model facts


def test_model_reports_the_parameter_count_the_readme_quotes():
    info = model_info()
    assert info["total_parameters"] == 412_525
    assert info["pretrained_version"] == "0.3.0"


def test_untrainable_parameters_are_the_per_element_references():
    """The 94 fixed parameters are one per element, the same 94 that bounds Z."""
    info = model_info()
    assert info["element_reference_energies"] == 94
    assert info["total_parameters"] - info["trainable_parameters"] == 94


# --------------------------------------------------------------- single point


def test_single_point_energy_is_negative_for_a_bound_crystal(nacl):
    result = single_point(nacl)
    assert result.energy_ev < 0
    assert result.energy_per_atom_ev < 0


def test_single_point_energy_is_extensive(nacl):
    """energy_ev must be the whole cell, energy_per_atom_ev the intensive value."""
    result = single_point(nacl)
    assert result.n_atoms == len(nacl)
    assert result.energy_ev == pytest.approx(result.energy_per_atom_ev * len(nacl), rel=1e-9)


def test_single_point_returns_one_force_vector_per_atom(nacl):
    result = single_point(nacl)
    assert np.asarray(result.forces).shape == (len(nacl), 3)


def test_single_point_is_deterministic(nacl):
    """Same structure, same answer. CHGNet inference has no sampling in it."""
    first = single_point(nacl)
    second = single_point(nacl)
    assert first.energy_ev == pytest.approx(second.energy_ev, rel=1e-12)


def test_doubling_the_cell_doubles_the_energy(nacl):
    """A supercell is the same crystal, so energy per atom must not move."""
    single = single_point(nacl)
    double = single_point(nacl * (2, 1, 1))
    assert double.energy_per_atom_ev == pytest.approx(single.energy_per_atom_ev, abs=2e-3)


# -------------------------------------------------------------------- relaxing


def test_relax_calls_back_once_per_step(fe3c):
    seen: list[int] = []
    result = relax(vacancy(fe3c, 0)[0], max_steps=6, on_step=lambda s: seen.append(s.step))
    assert seen == list(range(len(result.steps)))
    assert len(seen) >= 1


def test_relax_lowers_the_energy(fe3c):
    mutated, mutation = vacancy(fe3c, 0)
    result = relax(mutated, relax_cell=mutation.suggested_relax_cell, max_steps=30)
    assert result.steps[-1].energy_ev <= result.steps[0].energy_ev
    assert result.energy_change_ev <= 0


def test_relax_reduces_the_maximum_force(fe3c):
    mutated, _ = vacancy(fe3c, 0)
    result = relax(mutated, max_steps=30)
    assert result.steps[0].max_force_ev_per_a > 0.1, "vacancy should create forces"
    assert result.steps[-1].max_force_ev_per_a < result.steps[0].max_force_ev_per_a


def test_relax_respects_the_step_ceiling(fe3c):
    mutated, _ = vacancy(fe3c, 0)
    result = relax(mutated, fmax=1e-9, max_steps=3)
    assert len(result.steps) <= 5  # ASE records an extra observation at the end
    assert not result.converged


def test_relax_with_fixed_cell_does_not_move_the_lattice(fe3c):
    mutated, _ = vacancy(fe3c, 0)
    before = mutated.lattice.matrix.copy()
    result = relax(mutated, relax_cell=False, max_steps=10)
    assert np.allclose(result.final_structure.lattice.matrix, before, atol=1e-8)


def test_relax_with_free_cell_does_move_the_lattice(nacl):
    before = nacl.lattice.volume
    result = relax(nacl, relax_cell=True, max_steps=30)
    assert result.final_structure.lattice.volume != pytest.approx(before, rel=1e-9)


def test_relax_preserves_atom_count(fe3c):
    mutated, _ = vacancy(fe3c, 0)
    result = relax(mutated, max_steps=5)
    assert len(result.final_structure) == len(mutated)


def test_every_step_carries_geometry_for_the_animation(fe3c):
    mutated, _ = vacancy(fe3c, 0)
    result = relax(mutated, max_steps=4)
    for step in result.steps:
        assert np.asarray(step.positions).shape == (len(mutated), 3)
        assert np.asarray(step.lattice).shape == (3, 3)
        assert step.elapsed_s >= 0


# --------------------------------------------------- the symmetry consequence


def test_symmetric_edit_of_a_centrosymmetric_crystal_produces_no_forces(nacl):
    """Why relax_cell defaults to True.

    Every site in rocksalt sits at an inversion centre, and removing one leaves
    the survivors at inversion centres too, where forces vanish exactly. With
    positions alone there is nothing to relax and nothing to show.
    """
    mutated, _ = vacancy(nacl, 0)
    assert single_point(mutated).max_force_ev_per_a < 1e-4


def test_the_same_edit_does_move_once_the_cell_is_free(nacl):
    mutated, mutation = vacancy(nacl, 0)
    assert mutation.suggested_relax_cell is True

    result = relax(mutated, relax_cell=True, max_steps=40)
    assert len(result.steps) > 1
    assert result.energy_change_ev < -1e-3


def test_a_low_symmetry_crystal_produces_forces_without_the_cell(fe3c):
    """The contrast case: Pnma has no inversion centre on the Fe sites."""
    mutated, _ = vacancy(fe3c, 0)
    assert single_point(mutated).max_force_ev_per_a > 0.1


# --------------------------------------------------------------- end to end


def test_cif_text_to_relaxed_structure(nacl):
    """The whole path a user takes: file in, filtered, edited, relaxed, out."""
    result = evaluate(NACL_CIF.read_text(encoding="utf8"))
    assert result.survived

    mutated, mutation = substitute(result.structure, 0, "K")
    relaxed = relax(mutated, relax_cell=mutation.suggested_relax_cell, max_steps=30)

    assert (
        relaxed.final_structure.composition.reduced_formula
        != result.structure.composition.reduced_formula
    )
    assert len(relaxed.steps) >= 1
    assert relaxed.steps[-1].energy_per_atom_ev < 0


def test_strained_cell_costs_energy(nacl):
    """Straining away from equilibrium must raise the energy per atom."""
    relaxed = relax(nacl, relax_cell=True, max_steps=40).final_structure
    at_rest = single_point(relaxed).energy_per_atom_ev
    strained, mutation = strain(relaxed, 0.05)

    assert mutation.suggested_relax_cell is False
    assert single_point(strained).energy_per_atom_ev > at_rest
