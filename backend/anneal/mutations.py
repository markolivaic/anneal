"""The three edits anneal offers: substitute a site, remove a site, strain the cell.

All three are pure geometry. They are exact operations on the lattice and site
list, no model involved -- CHGNet only enters when the result is relaxed. That
split matters for honesty: the edit is deterministic and the energy is a
prediction, and the UI labels them differently for that reason.

Every function returns a new Structure and leaves the input untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Element, Structure

from anneal.filters import CHGNET_MAX_Z, MIN_NN_DISTANCE_A


class MutationError(ValueError):
    """The requested edit does not describe a structure CHGNet can be asked about."""


@dataclass(frozen=True)
class Mutation:
    """A record of what was done, so the UI can show provenance of an edit.

    `suggested_relax_cell` is not a preference, it follows from what the edit
    means. Change the *contents* of a cell and the cell is free to respond, so it
    relaxes too. Set the cell *yourself* with strain and it must stay where you
    put it, or the optimizer simply undoes the edit and reports nothing.

    This matters more than it looks. In a centrosymmetric crystal every site sits
    at an inversion centre, and a symmetric edit leaves the survivors at
    inversion centres too, where forces vanish exactly. Removing an Na from
    rocksalt NaCl gives max force 1.8e-06 eV/A: position-only relaxation has
    nothing to do, and the response only appears once the cell can move. The
    textbook structures a teaching tool wants to show are exactly the symmetric
    ones, so this is the common case, not the corner case.
    """

    kind: str
    detail: str
    suggested_relax_cell: bool = True


def _check_site(structure: Structure, site_index: int) -> None:
    if not 0 <= site_index < len(structure):
        raise MutationError(f"site {site_index} out of range for a {len(structure)}-site cell")


def substitute(structure: Structure, site_index: int, element: str) -> tuple[Structure, Mutation]:
    """Replace the species on one site, keeping its coordinates."""
    _check_site(structure, site_index)
    try:
        new_element = Element(element)
    except Exception as exc:
        raise MutationError(f"{element!r} is not an element symbol") from exc

    if new_element.Z > CHGNET_MAX_Z:
        raise MutationError(
            f"{element} has Z={new_element.Z}; CHGNet covers Z up to {CHGNET_MAX_Z}"
        )

    old = structure[site_index].species_string
    out = structure.copy()
    out.replace(site_index, new_element)
    return out, Mutation("substitute", f"site {site_index}: {old} -> {element}")


def vacancy(structure: Structure, site_index: int) -> tuple[Structure, Mutation]:
    """Remove one site, leaving a vacancy."""
    _check_site(structure, site_index)
    if len(structure) <= 1:
        raise MutationError("cannot empty a one-site cell")

    removed = structure[site_index].species_string
    out = structure.copy()
    out.remove_sites([site_index])
    return out, Mutation("vacancy", f"site {site_index}: removed {removed}")


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
    out.lattice = structure.lattice.__class__(structure.lattice.matrix @ matrix.T)
    # Cell held fixed: relaxing it here would just spring back to the unstrained
    # lattice, which is the one thing the user did not ask for.
    return out, Mutation("strain", label, suggested_relax_cell=False)


def check_relaxable(structure: Structure, max_atoms: int) -> None:
    """Raise if a mutated structure is not something CHGNet should be asked about.

    Mutations can produce geometry the COD filter would have refused -- a
    substitution that puts two atoms on top of each other, or a compressive
    strain that collapses the cell. The same limits apply after an edit as before.
    """
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
                f"two atoms would sit {closest:.3f} A apart, below the {MIN_NN_DISTANCE_A} A floor"
            )
