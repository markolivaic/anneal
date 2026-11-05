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


@pytest.fixture(scope="module")
def displaced(fe3c) -> Structure:
    """Fe3C with every atom moved a little, so a relaxation has work to do."""
    rng = np.random.default_rng(0)
    out = fe3c.copy()
    for index, shift in enumerate(rng.normal(0.0, 0.05, size=(len(out), 3))):
        out.translate_sites([index], shift, frac_coords=False, to_unit_cell=True)
    return out


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


def test_relax_calls_back_once_per_step(displaced):
    seen: list[int] = []
    result = relax(displaced, max_steps=6, on_step=lambda s: seen.append(s.step))
    assert seen == list(range(len(result.steps)))
    assert len(seen) >= 1


def test_relax_lowers_the_energy(displaced):
    result = relax(displaced, max_steps=30)
    assert result.steps[-1].energy_ev <= result.steps[0].energy_ev
    assert result.energy_change_ev <= 0


def test_relax_reduces_the_maximum_force(displaced):
    result = relax(displaced, max_steps=30)
    assert result.steps[0].max_force_ev_per_a > 0.1, "displacement should create forces"
    assert result.steps[-1].max_force_ev_per_a < result.steps[0].max_force_ev_per_a


def test_relax_respects_the_step_ceiling(displaced):
    result = relax(displaced, fmax=1e-9, max_steps=3)
    assert len(result.steps) <= 5  # ASE records an extra observation at the end
    assert not result.converged


def test_relax_with_fixed_cell_does_not_move_the_lattice(displaced):
    before = displaced.lattice.matrix.copy()
    result = relax(displaced, relax_cell=False, max_steps=10)
    assert np.allclose(result.final_structure.lattice.matrix, before, atol=1e-8)


def test_relax_with_free_cell_does_move_the_lattice(nacl):
    before = nacl.lattice.volume
    result = relax(nacl, relax_cell=True, max_steps=30)
    assert result.final_structure.lattice.volume != pytest.approx(before, rel=1e-9)


def test_free_cell_convergence_needs_both_force_and_stress(nacl):
    result = relax(nacl, relax_cell=True, fmax=0.05, stress_tol_gpa=0.1, max_steps=300)
    assert result.converged
    assert result.steps[-1].max_force_ev_per_a <= 0.05
    assert result.final_stress_gpa <= 0.1
    assert all(
        step.max_force_ev_per_a > 0.05 or step.max_stress_gpa > 0.1 for step in result.steps[:-1]
    ), "the run should stop at the first step that meets both"


def test_relax_preserves_atom_count(displaced):
    result = relax(displaced, max_steps=5)
    assert len(result.final_structure) == len(displaced)


def test_every_step_carries_geometry_for_the_animation(displaced):
    result = relax(displaced, max_steps=4)
    for step in result.steps:
        assert np.asarray(step.positions).shape == (len(displaced), 3)
        assert np.asarray(step.lattice).shape == (3, 3)
        assert step.elapsed_s >= 0


# --------------------------------------------------------- an isolated vacancy


def test_a_vacancy_in_the_deposited_cell_is_its_own_image(nacl):
    """Why defects get a supercell.

    In the 8-atom cell the vacancy repeats every 5.6 Å, so the edit describes an
    ordered compound in which every remaining site keeps its inversion centre,
    and no force appears.
    """
    mutated, _ = vacancy(nacl, 0, min_image=0.0)
    assert len(mutated) == 7
    assert single_point(mutated).max_force_ev_per_a < 1e-4


def test_an_isolated_vacancy_pushes_on_its_neighbours(nacl):
    mutated, mutation = vacancy(nacl, 0)
    assert len(mutated) == 63

    origin = mutated.lattice.get_fractional_coords(mutation.removed_cart)
    distances = mutated.lattice.get_all_distances(origin, mutated.frac_coords)[0]
    neighbours = distances < distances.min() + 0.1
    forces = np.linalg.norm(np.asarray(single_point(mutated).forces), axis=1)
    assert neighbours.sum() == 6
    assert forces[neighbours].min() > 0.1


# --------------------------------------------------------------- end to end


def test_cif_text_to_relaxed_structure(nacl):
    """The whole path a user takes: file in, filtered, edited, relaxed, out."""
    result = evaluate(NACL_CIF.read_text(encoding="utf8"))
    assert result.survived

    mutated, _ = substitute(result.structure, 0, "K")
    relaxed = relax(mutated, max_steps=30)

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
    strained, _ = strain(relaxed, 0.05)
    assert single_point(strained).energy_per_atom_ev > at_rest
