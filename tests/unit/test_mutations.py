"""Behavioural tests for the three edits.

These are pure geometry -- no model runs in this file. Every assertion is an
exact statement about a lattice or a site list.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from anneal.filters import CHGNET_MAX_Z
from anneal.mutations import (
    MutationError,
    check_relaxable,
    strain,
    substitute,
    vacancy,
)


def rocksalt(a: float = 5.64) -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


# ----------------------------------------------------------------- substitute


def test_substitute_replaces_species_and_keeps_coordinates():
    original = rocksalt()
    before = original[0].frac_coords.copy()
    out, mutation = substitute(original, 0, "K")

    assert out[0].specie.symbol == "K"
    assert np.allclose(out[0].frac_coords, before)
    assert len(out) == len(original)
    assert mutation.kind == "substitute"


def test_substitute_leaves_the_input_untouched():
    original = rocksalt()
    formula_before = original.composition.formula
    substitute(original, 0, "K")
    assert original.composition.formula == formula_before


def test_substitute_rejects_a_non_element():
    with pytest.raises(MutationError):
        substitute(rocksalt(), 0, "Xx")


def test_substitute_rejects_an_element_chgnet_cannot_embed():
    with pytest.raises(MutationError, match="94"):
        substitute(rocksalt(), 0, "Cf")  # Z=98


def test_substitute_accepts_the_boundary_element():
    out, _ = substitute(rocksalt(), 0, "Pu")  # Z=94
    assert out[0].specie.Z == CHGNET_MAX_Z


def test_substitute_rejects_an_out_of_range_site():
    with pytest.raises(MutationError, match="out of range"):
        substitute(rocksalt(), 99, "K")


def test_substitute_suggests_relaxing_the_cell():
    """Composition changed, so the cell is free to respond."""
    _, mutation = substitute(rocksalt(), 0, "K")
    assert mutation.suggested_relax_cell is True


# -------------------------------------------------------------------- vacancy


def test_vacancy_removes_exactly_one_site():
    original = rocksalt()
    out, mutation = vacancy(original, 0)

    assert len(out) == len(original) - 1
    assert mutation.kind == "vacancy"
    assert len(original) == 8, "input must not be mutated"


def test_vacancy_removes_the_requested_site():
    original = rocksalt()
    removed = original[0]
    out, _ = vacancy(original, 0)
    remaining = [tuple(np.round(s.frac_coords, 6)) for s in out]
    assert tuple(np.round(removed.frac_coords, 6)) not in remaining


def test_vacancy_rejects_emptying_a_single_site_cell():
    one = Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])
    with pytest.raises(MutationError):
        vacancy(one, 0)


def test_vacancy_rejects_an_out_of_range_site():
    with pytest.raises(MutationError, match="out of range"):
        vacancy(rocksalt(), 12)


# --------------------------------------------------------------------- strain


def test_hydrostatic_strain_scales_volume_as_the_cube():
    original = rocksalt()
    out, mutation = strain(original, 0.10)
    assert out.volume == pytest.approx(original.volume * 1.10**3, rel=1e-9)
    assert mutation.kind == "strain"


def test_axial_strain_scales_only_the_named_axes():
    original = rocksalt()
    out, _ = strain(original, [0.05, 0.0, 0.0])
    assert out.lattice.a == pytest.approx(original.lattice.a * 1.05, rel=1e-9)
    assert out.lattice.b == pytest.approx(original.lattice.b, rel=1e-9)
    assert out.lattice.c == pytest.approx(original.lattice.c, rel=1e-9)


def test_strain_keeps_fractional_coordinates():
    """Atoms ride the lattice; they do not stay at fixed cartesian positions."""
    original = rocksalt()
    out, _ = strain(original, 0.10)
    for before, after in zip(original, out, strict=True):
        assert np.allclose(before.frac_coords, after.frac_coords)


def test_compressive_strain_shrinks_the_cell():
    original = rocksalt()
    out, _ = strain(original, -0.02)
    assert out.volume < original.volume


def test_strain_rejects_a_cell_inverting_deformation():
    with pytest.raises(MutationError, match="inverts or flattens"):
        strain(rocksalt(), -1.5)


def test_strain_rejects_a_flattening_deformation():
    with pytest.raises(MutationError, match="inverts or flattens"):
        strain(rocksalt(), [0.0, 0.0, -1.0])


def test_strain_rejects_a_wrongly_shaped_tensor():
    with pytest.raises(MutationError):
        strain(rocksalt(), [0.1, 0.2])


def test_strain_does_not_suggest_relaxing_the_cell():
    """Relaxing the cell after a strain would undo the edit."""
    _, mutation = strain(rocksalt(), 0.02)
    assert mutation.suggested_relax_cell is False


def test_strain_accepts_a_full_tensor():
    out, _ = strain(rocksalt(), [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    assert out.volume > rocksalt().volume


# ------------------------------------------------------------- relaxable gate


def test_check_relaxable_passes_a_normal_structure():
    check_relaxable(rocksalt(), max_atoms=150)


def test_check_relaxable_rejects_a_cell_over_the_cap():
    with pytest.raises(MutationError, match="above the"):
        check_relaxable(rocksalt() * (3, 3, 3), max_atoms=150)


def test_check_relaxable_rejects_atoms_placed_on_top_of_each_other():
    """A substitution can be legal while the geometry it produces is not."""
    overlapping = Structure(Lattice.cubic(6.0), ["Na", "Cl"], [[0, 0, 0], [0.01, 0, 0]])
    with pytest.raises(MutationError, match="apart"):
        check_relaxable(overlapping, max_atoms=150)


def test_check_relaxable_rejects_an_element_chgnet_cannot_embed():
    structure = Structure(Lattice.cubic(5.0), ["Cf", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(MutationError, match="94"):
        check_relaxable(structure, max_atoms=150)


def test_check_relaxable_accepts_a_cell_at_exactly_the_cap():
    structure = rocksalt() * (2, 1, 1)
    check_relaxable(structure, max_atoms=len(structure))
