"""Edits: substitute a site, remove a site, strain the cell.

A point defect in a periodic cell repeats in every periodic image of that cell.
When the images are close, the defects interact and the result is an ordered
compound rather than an isolated defect. Substitutions and vacancies are
therefore made in the smallest supercell whose images are at least
MIN_DEFECT_IMAGE_A apart. Strain acts on the whole cell and needs no supercell.

Each function returns a new Structure and leaves its input unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from anneal.filters import CHGNET_MAX_Z, DEFAULT_MAX_ATOMS, MIN_NN_DISTANCE_A

MIN_DEFECT_IMAGE_A = 10.0

_NEIGHBOUR_SHIFTS = np.array([s for s in itertools.product((-1, 0, 1), repeat=3) if any(s)])


class MutationError(ValueError):
    """The requested edit does not describe a structure CHGNet can be asked about."""


@dataclass(frozen=True)
class Mutation:
    """What an edit did. `supercell` is the integer matrix applied first, if any."""

    kind: str
    detail: str
    supercell: list[list[int]] | None = None
    site_index: int | None = None  # the edited site in the returned structure
    removed_cart: list[float] | None = None  # where a vacancy's atom was, in Angstrom


def shortest_image(lattice: Lattice) -> float:
    """Distance from a point to its nearest periodic image, in Angstrom."""
    reduced = lattice.get_lll_reduced_lattice().matrix
    return float(np.linalg.norm(_NEIGHBOUR_SHIFTS @ reduced, axis=1).min())


def defect_supercell(
    structure: Structure,
    min_image: float = MIN_DEFECT_IMAGE_A,
    max_atoms: int = DEFAULT_MAX_ATOMS,
) -> np.ndarray | None:
    """Integer matrix of the smallest supercell with images at least `min_image` apart.

    Returns the identity when the cell already qualifies and None when no
    supercell within `max_atoms` does. Candidate shapes are as close to FCC as an
    integer matrix allows, because FCC keeps the most distance between lattice
    points for a given volume.
    """
    from ase.build import find_optimal_cell_shape

    if shortest_image(structure.lattice) >= min_image:
        return np.eye(3, dtype=int)

    n = len(structure)
    cell = structure.lattice.matrix
    # FCC needs min_image**3 / sqrt(2) of volume per lattice point, so smaller
    # supercells cannot qualify and need not be searched.
    start = max(2, math.ceil(min_image**3 / math.sqrt(2) / structure.volume))
    for size in range(start, max_atoms // n + 1):
        matrix = find_optimal_cell_shape(cell, size, "fcc", verbose=False)
        if matrix is None:
            continue
        matrix = np.rint(matrix).astype(int)
        if shortest_image(Lattice(matrix @ cell)) >= min_image:
            return matrix
    return None


def _check_site(structure: Structure, site_index: int) -> None:
    if not 0 <= site_index < len(structure):
        raise MutationError(f"site {site_index + 1} out of range for a {len(structure)}-site cell")


def _site_label(structure: Structure, site_index: int) -> str:
    return f"{structure[site_index].specie.symbol}{site_index + 1}"


def _expand_for_defect(
    structure: Structure, site_index: int, min_image: float, max_atoms: int
) -> tuple[Structure, int, np.ndarray]:
    """Supercell for a defect at `site_index`, and the index of that site's image in it."""
    matrix = defect_supercell(structure, min_image, max_atoms)
    if matrix is None:
        raise MutationError(
            f"this cell is too small to hold an isolated defect: no supercell of up to "
            f"{max_atoms} atoms keeps its periodic images {min_image:g} Å apart"
        )
    if (matrix == np.eye(3, dtype=int)).all():
        return structure.copy(), site_index, matrix

    supercell = structure.copy().make_supercell(matrix, in_place=False)
    origin = supercell.lattice.get_fractional_coords(structure[site_index].coords)
    distances = supercell.lattice.get_all_distances(origin, supercell.frac_coords)[0]
    return supercell, int(np.argmin(distances)), matrix


def _supercell_note(matrix: np.ndarray, n_sites: int) -> str:
    if (matrix == np.eye(3, dtype=int)).all():
        return ""
    return f", in a {n_sites}-atom supercell"


def substitute(
    structure: Structure,
    site_index: int,
    element: str,
    *,
    min_image: float = MIN_DEFECT_IMAGE_A,
    max_atoms: int = DEFAULT_MAX_ATOMS,
) -> tuple[Structure, Mutation]:
    """Replace the species on one site, in a supercell large enough to isolate it."""
    _check_site(structure, site_index)
    try:
        new_element = Element(element)
    except Exception as exc:
        raise MutationError(f"{element!r} is not an element symbol") from exc
    if new_element.Z > CHGNET_MAX_Z:
        raise MutationError(
            f"{element} has Z={new_element.Z}; CHGNet covers Z up to {CHGNET_MAX_Z}"
        )

    label = _site_label(structure, site_index)
    out, index, matrix = _expand_for_defect(structure, site_index, min_image, max_atoms)
    out.replace(index, new_element)
    detail = f"{label} -> {element}{_supercell_note(matrix, len(out))}"
    return out, Mutation(
        "substitute", detail, supercell=matrix.tolist(), site_index=index, removed_cart=None
    )


def vacancy(
    structure: Structure,
    site_index: int,
    *,
    min_image: float = MIN_DEFECT_IMAGE_A,
    max_atoms: int = DEFAULT_MAX_ATOMS,
) -> tuple[Structure, Mutation]:
    """Remove one site, in a supercell large enough to isolate the vacancy."""
    _check_site(structure, site_index)
    if len(structure) <= 1:
        raise MutationError("cannot empty a one-site cell")

    label = _site_label(structure, site_index)
    out, index, matrix = _expand_for_defect(structure, site_index, min_image, max_atoms)
    removed = [float(x) for x in out[index].coords]
    detail = f"{label} removed{_supercell_note(matrix, len(out))}"
    out.remove_sites([index])
    return out, Mutation("vacancy", detail, supercell=matrix.tolist(), removed_cart=removed)


# Oxidation-state guessing grows combinatorially with element count; above this
# it can take minutes, and the structures it would skip are molecular crystals.
_CHARGE_CHECK_MAX_ELEMENTS = 4
_CHARGE_CHECK_MAX_SITES = 60


def substitution_breaks_charge_balance(
    structure: Structure, site_index: int, element: str
) -> bool | None:
    """True if `structure` has a charge-balanced set of common oxidation states and the
    swap leaves none. None when there is no ionic picture to break, or the check is
    skipped because the composition is too large to guess quickly."""
    composition = structure.composition.element_composition
    if (
        len(composition.elements) > _CHARGE_CHECK_MAX_ELEMENTS
        or composition.num_atoms > _CHARGE_CHECK_MAX_SITES
    ):
        return None
    if not composition.oxi_state_guesses(max_sites=-1):
        return None
    old = structure[site_index].specie.symbol
    swapped = composition.as_dict()
    swapped[old] -= 1
    swapped[element] = swapped.get(element, 0) + 1
    return not type(composition)(swapped).oxi_state_guesses(max_sites=-1)


def unset_magnetic_elements(structure: Structure) -> list[str]:
    """3d and 4f elements present. anneal sets no initial magnetic moments and CHGNet
    takes none as input, so their spin state cannot be chosen."""
    return sorted(
        el.symbol for el in structure.composition.elements if 21 <= el.Z <= 30 or 57 <= el.Z <= 71
    )


def strain(
    structure: Structure, strain_tensor: float | list[float] | list[list[float]]
) -> tuple[Structure, Mutation]:
    """Deform the cell. Scalar for hydrostatic, three values for axial, or a 3x3.

    Fractional coordinates are held fixed, so the atoms move with the lattice.
    Strain is engineering strain: 0.02 means the axis grows by 2%.
    """
    tensor = np.asarray(strain_tensor, dtype=float)
    if tensor.ndim == 0:
        matrix = np.eye(3) * (1.0 + float(tensor))
        label = f"hydrostatic {float(tensor):+.3%}"
    elif tensor.shape == (3,):
        matrix = np.diag(1.0 + tensor)
        label = "axial " + ", ".join(f"{v:+.3%}" for v in tensor)
    elif tensor.shape == (3, 3):
        matrix = np.eye(3) + tensor
        label = "tensor"
    else:
        raise MutationError(
            f"strain must be a scalar, three values, or a 3x3 matrix; got shape {tensor.shape}"
        )

    if np.linalg.det(matrix) <= 0:
        raise MutationError("that strain inverts or flattens the cell")

    out = structure.copy()
    out.lattice = Lattice(structure.lattice.matrix @ matrix.T)
    return out, Mutation("strain", label)


def check_relaxable(structure: Structure, max_atoms: int) -> None:
    """Raise if an edited structure falls outside the limits the COD filter applies."""
    if len(structure) == 0:
        raise MutationError("empty cell")
    if len(structure) > max_atoms:
        raise MutationError(f"{len(structure)} sites is above the {max_atoms} limit")
    if structure.volume <= 0:
        raise MutationError("non-positive cell volume")

    max_z = max(el.Z for el in structure.composition.elements)
    if max_z > CHGNET_MAX_Z:
        raise MutationError(f"Z={max_z} is above CHGNet's {CHGNET_MAX_Z}")

    if len(structure) > 1:
        closest = float(structure.distance_matrix[np.triu_indices(len(structure), k=1)].min())
        if closest < MIN_NN_DISTANCE_A:
            raise MutationError(
                f"two atoms would sit {closest:.3f} Å apart, below the {MIN_NN_DISTANCE_A} Å floor"
            )
