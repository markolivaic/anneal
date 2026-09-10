"""The filter that decides whether a COD entry is usable input for CHGNet.

COD is a dump of experimental refinements. Many entries carry partial site
occupancies, disordered sites, or hydrogen the diffraction never located.
CHGNet does not error on those -- it returns a confident number that means
nothing. Everything anneal offers has passed `evaluate` below.

The stage order is load-bearing. Geometry is checked before size so that a
structure failing only at size has cleared every other criterion; that is what
lets scripts/survey_cod.py sweep alternative atom caps without re-parsing.

Measured behaviour of this filter on a 4,000-structure uniform random sample of
COD lives in data/survey_cod_summary.json. Regenerate with
scripts/survey_cod.py.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Set from the measured size distribution, not from taste. Under the discrete
# edit-then-relax interaction model the cost that matters is total convergence
# time, not per-step latency. See README "Why 150".
DEFAULT_MAX_ATOMS = 150

# chgnet 0.4.2, chgnet/model/encoders.py: AtomEmbedding(max_num_elements=94),
# indexed by atomic_number - 1. Z > 94 is an IndexError, not a bad prediction.
# Note: in a 4,000-structure sample this rejected nothing at all.
CHGNET_MAX_Z = 94

# Two atoms closer than this in the deposited cell means the refinement placed
# alternative positions on top of each other. CHGNet's forces explode there.
MIN_NN_DISTANCE_A = 0.5

# Guards so one pathological entry cannot stall a batch. Both sit above any cap
# this project offers, so neither changes the outcome of the size filter.
PRIMITIVE_CELL_SITE_LIMIT = 200
GEOMETRY_SITE_LIMIT = 400

STAGES = (
    "fetched",
    "parsed",
    "ordered",
    "hydrogen_declared_present",
    "elements_within_chgnet",
    "geometry_sane",
    "size_within_limit",
)

ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)")

_FORMULA_KEYS = (
    "_chemical_formula_sum",
    "_chemical_formula_moiety",
    "_chemical_formula_structural",
)


@dataclass
class FilterResult:
    """What the filter learned about one entry. Serialises straight to a CSV row."""

    stage_failed: str = ""
    reject_detail: str = ""
    n_blocks: int | None = None
    n_sites: int | None = None
    n_sites_primitive: int | None = None
    volume_a3: float | None = None
    is_ordered: bool | None = None
    formula_declared: str = ""
    formula_parsed: str = ""
    max_z: int | None = None
    min_nn_dist_a: float | None = None
    contains_carbon: bool | None = None
    n_parser_warnings: int | None = None

    structure: Structure | None = field(default=None, repr=False, compare=False)

    @property
    def survived(self) -> bool:
        return not self.stage_failed

    def as_row(self) -> dict:
        """CSV-safe view: everything except the structure object."""
        out = {k: v for k, v in self.__dict__.items() if k != "structure"}
        out["survived"] = self.survived
        return out


def declared_elements(cif_text: str) -> tuple[set[str], str]:
    """Element symbols the CIF *claims* the compound contains.

    This is the only handle on hydrogen a refinement never located: the chemist
    writes H in the formula, the diffraction data cannot place it, and the atom
    site loop has none.

    A ratio-based version of this check (declared H vs observed H, scaled by the
    heavy-element ratio) was tried and dropped: on 548 COD entries it caught
    zero cases this rule misses, and its near-threshold examples were false
    positives caused by solvent counted differently in the declared formula.
    """
    for key in _FORMULA_KEYS:
        match = re.search(
            rf"^{key}\s+(?:'([^']*)'|\"([^\"]*)\"|;\s*(.*?)\s*;|(\S+))\s*$",
            cif_text,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            continue
        value = next((g for g in match.groups() if g), "").strip()
        if not value or value in {"?", "."}:
            continue
        symbols = {sym for sym, _ in ELEMENT_TOKEN.findall(value) if sym}
        if symbols:
            # Neutron refinements write D for deuterium. pymatgen has no such
            # element, but chemically it is the hydrogen we are looking for.
            if "D" in symbols:
                symbols.discard("D")
                symbols.add("H")
            return symbols, value
    return set(), ""


def evaluate(cif_text: str, max_atoms: int = DEFAULT_MAX_ATOMS) -> FilterResult:
    """Run one CIF through the filter. `result.survived` is the predicate."""
    result = FilterResult()

    if not cif_text or not cif_text.strip():
        result.stage_failed = "parsed"
        result.reject_detail = "empty file"
        return result

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            structures = CifParser.from_str(cif_text).parse_structures(primitive=False)
        except Exception as exc:  # pymatgen raises a wide spread of types here
            result.stage_failed = "parsed"
            result.reject_detail = f"{type(exc).__name__}: {exc}"[:200]
            result.n_parser_warnings = len(caught)
            return result
        result.n_parser_warnings = len(caught)

    if not structures:
        result.stage_failed = "parsed"
        result.reject_detail = "no structure in file"
        return result

    structure = structures[0]
    result.n_blocks = len(structures)
    result.n_sites = len(structure)
    result.volume_a3 = round(structure.volume, 4)
    result.is_ordered = structure.is_ordered
    result.formula_parsed = structure.composition.formula
    present = {str(el) for el in structure.composition.elements}
    result.contains_carbon = "C" in present
    # Recorded for every parsed structure, not only those reaching the element
    # stage, so independent pass rates are not computed over a subset already
    # thinned by the disorder filter.
    result.max_z = max(el.Z for el in structure.composition.elements)

    if not structure.is_ordered:
        result.stage_failed = "ordered"
        result.reject_detail = "partial or mixed site occupancy"
        return result

    declared, declared_text = declared_elements(cif_text)
    result.formula_declared = declared_text[:120]
    if "H" in declared and "H" not in present:
        result.stage_failed = "hydrogen_declared_present"
        result.reject_detail = "formula declares H, no H site in the cell"
        return result

    if result.max_z > CHGNET_MAX_Z:
        result.stage_failed = "elements_within_chgnet"
        result.reject_detail = f"Z={result.max_z} above CHGNet's {CHGNET_MAX_Z}"
        return result

    if structure.volume <= 0:
        result.stage_failed = "geometry_sane"
        result.reject_detail = "non-positive cell volume"
        return result

    # Cells far above any offered cap are skipped: the distance matrix is O(N^2)
    # and they cannot pass the size stage anyway.
    if 1 < result.n_sites <= GEOMETRY_SITE_LIMIT:
        closest = float(structure.distance_matrix[np.triu_indices(result.n_sites, k=1)].min())
        result.min_nn_dist_a = round(closest, 4)
        if closest < MIN_NN_DISTANCE_A:
            result.stage_failed = "geometry_sane"
            result.reject_detail = f"two atoms {closest:.3f} A apart"
            return result

    if result.n_sites > max_atoms:
        result.stage_failed = "size_within_limit"
        result.reject_detail = f"{result.n_sites} sites > {max_atoms}"
        return result

    if result.n_sites <= PRIMITIVE_CELL_SITE_LIMIT:
        try:
            result.n_sites_primitive = len(structure.get_primitive_structure())
        except Exception:
            result.n_sites_primitive = None

    result.structure = structure
    return result
