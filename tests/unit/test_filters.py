"""Behavioural tests for the COD filter.

The filter decides what anneal will and will not offer, and its survival rate is
quoted in the README, so every stage gets a test that can actually fail.

Nothing here touches the network. Structures are either built in-process or read
from the CC0 CIF fixtures in tests/fixtures/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from anneal.filters import (
    CHGNET_MAX_Z,
    MIN_NN_DISTANCE_A,
    declared_elements,
    evaluate,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NACL_CIF = FIXTURES / "cod-1000041-nacl.cif"
FE3C_CIF = FIXTURES / "cod-9016675-fe3c.cif"


def rocksalt(a: float = 5.64) -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


# --------------------------------------------------------------------- parsing


def test_accepts_a_real_cod_entry():
    result = evaluate(NACL_CIF.read_text(encoding="utf8"))
    assert result.survived
    assert result.stage_failed == ""
    assert result.n_sites == 8
    assert result.is_ordered is True
    assert result.formula_parsed == "Na4 Cl4"


def test_empty_input_is_rejected_not_crashed():
    result = evaluate("")
    assert not result.survived
    assert result.stage_failed == "parsed"


def test_garbage_input_is_rejected_not_crashed():
    result = evaluate("this is not a CIF file at all\nnor is this line")
    assert not result.survived
    assert result.stage_failed == "parsed"


def test_survived_is_false_whenever_a_stage_failed():
    assert evaluate("").survived is False
    assert evaluate(NACL_CIF.read_text(encoding="utf8")).survived is True


# ---------------------------------------------------------- declared elements


def test_declared_elements_reads_formula_sum():
    symbols, text = declared_elements("_chemical_formula_sum 'C19 H23 N O4'")
    assert symbols == {"C", "H", "N", "O"}
    assert text == "C19 H23 N O4"


def test_declared_elements_handles_formula_without_counts():
    symbols, _ = declared_elements("_chemical_formula_sum 'Cl Na'")
    assert symbols == {"Cl", "Na"}


def test_declared_elements_does_not_split_two_letter_symbols():
    """Cl must not read as C + l, and He must not read as H + e."""
    symbols, _ = declared_elements("_chemical_formula_sum 'He Cl Hf Hg'")
    assert symbols == {"He", "Cl", "Hf", "Hg"}
    assert "H" not in symbols


def test_declared_elements_treats_deuterium_as_hydrogen():
    """Neutron refinements write D. pymatgen has no D, but chemically it is H."""
    symbols, _ = declared_elements("_chemical_formula_sum 'C2 D6 O'")
    assert "H" in symbols
    assert "D" not in symbols


def test_declared_elements_falls_back_to_moiety():
    symbols, _ = declared_elements("_chemical_formula_moiety 'H2 O'")
    assert symbols == {"H", "O"}


def test_declared_elements_ignores_placeholder_values():
    assert declared_elements("_chemical_formula_sum ?")[0] == set()
    assert declared_elements("_chemical_formula_sum .")[0] == set()


def test_declared_elements_returns_empty_when_absent():
    assert declared_elements("_cell_length_a 5.64")[0] == set()


# ------------------------------------------------------------------- disorder


def test_partial_occupancy_is_rejected():
    structure = Structure(
        Lattice.cubic(4.0), [{"Na": 0.5}, {"Cl": 1.0}], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    result = evaluate(structure.to(fmt="cif"))
    assert not result.survived
    assert result.stage_failed == "ordered"
    assert result.is_ordered is False


def test_mixed_site_occupancy_is_rejected():
    structure = Structure(
        Lattice.cubic(4.0),
        [{"Na": 0.5, "K": 0.5}, {"Cl": 1.0}],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    result = evaluate(structure.to(fmt="cif"))
    assert not result.survived
    assert result.stage_failed == "ordered"


# ------------------------------------------------------------------- hydrogen


def test_declared_hydrogen_absent_from_sites_is_rejected():
    """The case this whole check exists for: X-ray could not place the H.

    Takes the real NaCl entry, which passes, and changes only its declared
    formula to claim hydrogen. Nothing else about the file moves, so a failure
    here can only be the hydrogen rule.
    """
    cif = NACL_CIF.read_text(encoding="utf8")
    assert evaluate(cif).survived, "fixture must pass before the formula is edited"

    doped = cif.replace(
        "_chemical_formula_sum            'Cl Na'", "_chemical_formula_sum            'Cl H4 Na'", 1
    )
    assert doped != cif, "fixture formula line changed shape; update this test"

    result = evaluate(doped)
    assert not result.survived
    assert result.stage_failed == "hydrogen_declared_present"


def test_formula_sum_wins_over_the_structural_fallback():
    """Several formula keys can disagree. _chemical_formula_sum is authoritative.

    Found by a test that edited the wrong line: breaking _chemical_formula_sum
    silently fell through to _chemical_formula_structural and changed the verdict.
    """
    symbols, text = declared_elements(
        "_chemical_formula_structural     'Na Cl'\n_chemical_formula_sum            'Cl H4 Na'\n"
    )
    assert "H" in symbols
    assert text == "Cl H4 Na"


def test_structure_without_declared_hydrogen_is_not_rejected_for_hydrogen():
    """Carbides genuinely contain no H. They must not be caught by the H rule."""
    result = evaluate(FE3C_CIF.read_text(encoding="utf8"))
    assert result.stage_failed != "hydrogen_declared_present"


# ------------------------------------------------------------------- elements


def test_element_above_chgnet_range_is_rejected():
    """Cf is Z=98, past CHGNet's 94-element embedding."""
    structure = Structure(Lattice.cubic(5.0), ["Cf", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result = evaluate(structure.to(fmt="cif"))
    assert not result.survived
    assert result.stage_failed == "elements_within_chgnet"
    assert result.max_z > CHGNET_MAX_Z


def test_element_at_the_chgnet_boundary_is_accepted():
    """Pu is Z=94, exactly the last element CHGNet covers. Off-by-one guard."""
    structure = Structure(Lattice.cubic(5.0), ["Pu", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result = evaluate(structure.to(fmt="cif"))
    assert result.max_z == CHGNET_MAX_Z
    assert result.stage_failed != "elements_within_chgnet"


# ------------------------------------------------------------------- geometry


def test_overlapping_atoms_are_rejected():
    structure = Structure(Lattice.cubic(6.0), ["Na", "Cl"], [[0, 0, 0], [0.02, 0, 0]])
    result = evaluate(structure.to(fmt="cif"))
    assert not result.survived
    assert result.stage_failed == "geometry_sane"
    assert result.min_nn_dist_a < MIN_NN_DISTANCE_A


# ----------------------------------------------------------------------- size


def test_size_cap_rejects_a_large_cell():
    big = rocksalt() * (3, 3, 3)  # 216 sites
    result = evaluate(big.to(fmt="cif"), max_atoms=150)
    assert not result.survived
    assert result.stage_failed == "size_within_limit"
    assert result.n_sites == 216


def test_size_cap_is_inclusive_at_the_boundary():
    """A cell of exactly max_atoms must be offered, not refused."""
    structure = rocksalt() * (2, 1, 1)  # 16 sites
    assert evaluate(structure.to(fmt="cif"), max_atoms=16).survived
    assert not evaluate(structure.to(fmt="cif"), max_atoms=15).survived


def test_raising_the_cap_admits_a_previously_refused_structure():
    big = rocksalt() * (2, 2, 2)  # 64 sites
    assert not evaluate(big.to(fmt="cif"), max_atoms=32).survived
    assert evaluate(big.to(fmt="cif"), max_atoms=64).survived


# ------------------------------------------------------------- stage ordering


def test_size_is_checked_last_so_the_cap_sweep_stays_exact():
    """A structure failing only on size must have cleared every other stage.

    scripts/survey_cod.py reports survival across alternative caps without
    re-parsing, which is only valid while size is the final stage.
    """
    big = rocksalt() * (3, 3, 3)
    result = evaluate(big.to(fmt="cif"), max_atoms=64)
    assert result.stage_failed == "size_within_limit"
    assert result.is_ordered is True
    assert result.max_z is not None
    assert result.min_nn_dist_a is not None


def test_disorder_is_reported_before_size():
    """A cell that is both disordered and huge is reported as disordered."""
    big = Structure(
        Lattice.cubic(4.0), [{"Na": 0.5}, {"Cl": 1.0}], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ) * (4, 4, 4)
    result = evaluate(big.to(fmt="cif"), max_atoms=8)
    assert result.stage_failed == "ordered"


@pytest.mark.parametrize("cap", [8, 16, 64, 150])
def test_survivors_never_exceed_the_cap(cap):
    for cif in (NACL_CIF, FE3C_CIF):
        result = evaluate(cif.read_text(encoding="utf8"), max_atoms=cap)
        if result.survived:
            assert result.n_sites <= cap
