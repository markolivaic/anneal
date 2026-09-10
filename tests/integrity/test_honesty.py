"""Honesty guards. These are greps, and that is exactly what they are for.

They convert promises into build failures: if someone deletes the disclaimer,
edits a number in the README without rerunning the script behind it, or pastes a
local path into the docs, CI goes red.

They are NOT behavioural tests and are counted separately from tests/unit in the
README. Asserting that a string appears in a file proves nothing about
correctness -- it only proves the file still says what it promised.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DATA = REPO_ROOT / "data"

DOC_FILES = sorted(REPO_ROOT.glob("*.md"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf8")


def flat(path: Path) -> str:
    """File text with every run of whitespace collapsed to one space.

    Prose guards match phrases, and prose gets re-wrapped. A line break landing
    between "not" and "deposited data" silently disarms a literal pattern, which
    happened four separate times while writing these. Matching against flattened
    text removes the whole class of failure instead of sprinkling `\\s+` through
    every regex and hoping none is forgotten.
    """
    return re.sub(r"\s+", " ", read(path))


# ------------------------------------------------------- 10.1 the disclaimer


def test_readme_exists():
    assert README.exists(), "the README is the deliverable, not an afterthought"


def test_readme_declares_what_this_is_not():
    assert re.search(r"^##+ .*What this is NOT", read(README), re.MULTILINE | re.IGNORECASE)


def test_readme_has_a_limitations_section():
    assert re.search(r"^##+ .*Limitations", read(README), re.MULTILINE | re.IGNORECASE)


def test_the_disclaimer_sits_above_the_fold():
    """Within the first screen, not buried at the bottom of a long page.

    The benchmark repo that prompted this rule disclosed honestly on line 208 of
    a 229-line README.
    """
    above_fold = "\n".join(read(README).splitlines()[:45])
    assert re.search(r"What this is NOT", above_fold, re.IGNORECASE), (
        "'What this is NOT' must appear in the first 45 lines"
    )


def test_the_model_is_named_as_an_approximation_above_the_fold():
    above_fold = "\n".join(read(README).splitlines()[:45])
    assert re.search(r"approximation to DFT|not DFT", above_fold, re.IGNORECASE)


def test_readme_states_energies_are_not_publication_grade():
    # Whitespace-tolerant: prose gets re-wrapped, and a line break landing
    # between "not" and "publication-grade" must not silently disarm the guard.
    assert "not publication-grade" in flat(README)


# --------------------------------------------- 10.4 no local absolute paths


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_docs_contain_no_local_absolute_paths(doc):
    """A leaked path is a dead link and it exposes the author's directory tree."""
    text = read(doc)
    assert not re.search(r"/Users/|/home/[a-z]|[A-Z]:\\\\?[A-Za-z]", text), (
        f"{doc.name} contains a local absolute path"
    )


def test_workflow_contains_no_local_absolute_paths():
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert not re.search(r"/Users/|/home/[a-z]|[A-Z]:\\", read(workflow))


# -------------------------------------- 10.3 .env.example matches the code


def test_env_example_exists_only_if_the_code_reads_env_vars():
    """Neither an undocumented variable nor a documented one nothing reads.

    anneal reads no environment variables, so there must be no .env.example
    promising otherwise. If a variable is ever introduced, this fails until the
    example file is written.
    """
    sources = [
        *(REPO_ROOT / "backend").rglob("*.py"),
        *(REPO_ROOT / "scripts").rglob("*.py"),
        *(REPO_ROOT / "web" / "src").rglob("*.js"),
    ]
    pattern = re.compile(
        r"(?:getenv|environ\[|environ\.get|process\.env\.|import\.meta\.env\.)\(?[\"']?(\w+)"
    )
    used = {m.group(1) for path in sources for m in pattern.finditer(read(path))}

    example = REPO_ROOT / ".env.example"
    if not used:
        assert not example.exists(), (
            "no environment variable is read, so .env.example must not exist"
        )
        return

    declared = {
        line.split("=")[0] for line in read(example).splitlines() if re.match(r"^\w+=", line)
    }
    assert not (used - declared), f"read but undocumented: {sorted(used - declared)}"
    assert not (declared - used), f"documented but unused: {sorted(declared - used)}"


# ------------------------------------ every number traces back to its script


def survey() -> dict:
    return json.loads(read(DATA / "survey_cod_summary.json"))


def benchmark() -> dict:
    return json.loads(read(DATA / "benchmark.json"))


def test_survey_artifact_is_committed():
    assert (DATA / "survey_cod_summary.json").exists()
    assert (DATA / "survey_cod.csv").exists()


def test_readme_survival_percentage_matches_the_survey_artifact():
    """The headline claim, checked against the file the script wrote.

    Editing the README number without rerunning scripts/survey_cod.py fails
    here, which is the entire point.
    """
    actual = survey()["survival_pct"]
    text = read(README)
    assert f"{actual:.1f}%" in text, (
        f"survey_cod_summary.json says {actual:.1f}% and the README does not say it"
    )
    # It is the headline claim, so it belongs on the first screen, not in a
    # limitations section the reader may never reach.
    above_fold = "\n".join(text.splitlines()[:45])
    assert f"{actual:.1f}%" in above_fold, (
        f"the survival rate {actual:.1f}% must appear in the first 45 lines"
    )


def test_readme_denominator_matches_the_survey_frame():
    frame = survey()["frame"]["counts"]["default"]
    assert f"{frame:,}" in read(README), (
        f"README must quote the frame size {frame:,} it actually sampled from"
    )


def test_survey_cap_matches_what_the_readme_offers():
    """The artifact and the prose must agree on the atom limit."""
    assert survey()["sample"]["max_atoms"] == 150


def test_readme_survivor_count_matches_the_survey_artifact():
    """Caught a real stale number: the survivor count was still the cap-64 figure
    after the cap moved to 150. Counts derived from the sample rot the moment a
    parameter changes, so the README's copy is checked against the artifact."""
    survivors = survey()["survivors"]
    assert f"{survivors:,}" in read(README), (
        f"README must quote the survivor count {survivors:,} the survey produced"
    )


def test_survivor_count_and_rate_are_consistent():
    data = survey()
    rate = data["survivors"] / data["sample"]["size"]
    assert abs(100 * rate - data["survival_pct"]) < 0.01


def test_readme_parameter_count_matches_the_model():
    assert "412,525" in read(README)
    assert benchmark()["model"]["total_parameters"] == 412_525


def test_readme_catalog_size_matches_the_catalog_file():
    kept = json.loads(read(DATA / "catalog_summary.json"))["kept"]
    assert f"{kept:,}" in read(README), f"README must quote the catalogue size {kept:,}"


def validation() -> dict:
    return json.loads(read(DATA / "validation_summary.json"))


def test_readme_experimental_error_matches_the_validation_artifact():
    """The one number that says how wrong the model is, pinned to its run."""
    overall = validation()["statistics"]["overall"]["volume_error_pct"]
    stated = f"{overall['median_signed']:+.2f}%"
    assert stated in read(README), (
        f"README must quote the measured volume error {stated} from validation_summary.json"
    )


def test_readme_does_not_call_the_experimental_residual_a_dft_error():
    """The reference is a measured cell, not a DFT one.

    Calling this CHGNet's error would claim accuracy against its own training
    target, which this comparison cannot support. The distinction has to survive
    future edits to the prose.
    """
    text = flat(README)
    assert "distance from **experiment**" in text
    assert "not its error against the DFT-PBE data" in text


def test_validation_artifact_states_its_limitations():
    limitations = validation()["limitations"]
    assert len(limitations) >= 5, "a measurement this loaded needs its caveats recorded"
    joined = " ".join(limitations).lower()
    assert "not a dft cell" in joined or "not the same quantity" in joined


def test_readme_element_bound_claim_matches_both_artifacts():
    """The element bound rejected 0 in the survey and 6 in the catalogue build.

    Saying only the first reads as "the Z <= 94 bound never matters", which is
    wider than the evidence. Both numbers move when either run is repeated, so
    both are checked here.
    """
    catalog = json.loads(read(DATA / "catalog_summary.json"))
    rejected = catalog["rejected_by_stage"].get("elements_within_chgnet", 0)
    fetched = catalog["fetched"]
    survey_rejected = next(
        f["lost"] for f in survey()["funnel"] if f["stage"] == "elements_within_chgnet"
    )
    text = read(README)

    if survey_rejected == 0 and rejected > 0:
        # Whitespace-tolerant: prose wraps, and a line break between the count
        # and its denominator must not silently disarm the guard.
        assert re.search(rf"{rejected}\s+of\s+{fetched:,}", text), (
            f"README must say the catalogue build rejected {rejected} of "
            f"{fetched:,} on the element bound, not that nothing ever was"
        )
    assert "Z ≤ 94" in text or "Z <= 94" in text


def test_catalog_summary_labels_itself_as_not_a_survival_rate():
    """The one number most likely to be misquoted, guarded at the source."""
    provenance = json.loads(read(DATA / "catalog_summary.json"))["provenance"]
    assert "not a uniform sample" in provenance
    assert "not a survival rate" in provenance


# ------------------------------------------------------ 10.2 walkthrough media


WALKTHROUGH = REPO_ROOT / "docs" / "walkthrough"
MEDIA = {
    "app-walkthrough.mp4": lambda b: b[4:8] == b"ftyp",
    "app-walkthrough.gif": lambda b: b[:6] in (b"GIF89a", b"GIF87a"),
    "app-walkthrough-poster.jpg": lambda b: b[:3] == b"\xff\xd8\xff",
}


def test_walkthrough_is_either_present_and_real_or_declared_missing():
    """No silent gap between "there is a demo" and "there is not".

    While the recording does not exist, the README must say so where a reader
    looking for it will see it. The moment the files appear, this switches to
    checking they are real media by magic bytes, so a corrupt or renamed file
    fails CI instead of rendering as a broken image.
    """
    present = [name for name in MEDIA if (WALKTHROUGH / name).exists()]

    if not present:
        walkthrough_section = re.search(
            r"^##+ Walkthrough\s*\n(.*?)(?=^##+ |\Z)",
            read(README),
            re.MULTILINE | re.DOTALL,
        )
        assert walkthrough_section, "README has no Walkthrough section"
        assert re.search(
            r"not yet recorded|no recording", walkthrough_section.group(1), re.IGNORECASE
        ), "no walkthrough media exists, so the README must say so under Walkthrough"
        return

    missing = sorted(set(MEDIA) - set(present))
    assert not missing, f"walkthrough is half-there; missing {missing}"
    for name, is_valid in MEDIA.items():
        data = (WALKTHROUGH / name).read_bytes()
        assert is_valid(data), f"{name} is not the format its extension claims"


# ----------------------------------------------- 1.3 no hand-typed badges


def test_no_static_count_badges():
    """Only workflow-status or CI-fed badges. A literal count rots silently."""
    bad = re.findall(
        r"img\.shields\.io/badge/[^)\]]*(?:passing|coverage|\d+%|tests?-\d+)",
        read(README),
        re.IGNORECASE,
    )
    assert not bad, f"static measurement badges found: {bad}"


def test_test_counts_are_reported_separately():
    """Behavioural and integrity counts, never one total."""
    text = read(README)
    assert re.search(r"Behavioural tests?:\s*\d+", text, re.IGNORECASE)
    assert re.search(r"Integrity tests?:\s*\d+", text, re.IGNORECASE)


def collected_count(directory: str) -> int:
    """How many tests pytest actually collects in a directory.

    A subprocess, so this collects without running anything and cannot recurse.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            directory,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    assert match, f"could not read a collected count from pytest:\n{proc.stdout[-800:]}"
    return int(match.group(1))


@pytest.mark.parametrize(
    ("label", "directory"),
    [("Behavioural", "tests/unit"), ("Integrity", "tests/integrity")],
)
def test_readme_test_counts_match_what_pytest_collects(label, directory):
    """The counts in the README are checked, not typed.

    This is the rule that the benchmark repo failed: a badge reading
    "tests-50 passing" that was a hand-written literal nothing verified.
    """
    stated = re.search(rf"{label} tests?:\s*(\d+)", read(README), re.IGNORECASE)
    assert stated, f"README does not state a {label.lower()} test count"
    actual = collected_count(directory)
    assert int(stated.group(1)) == actual, (
        f"README says {label} tests: {stated.group(1)}, pytest collects {actual} in {directory}"
    )


# ------------------------------------------------- third-party attribution


def test_third_party_notices_exist_and_name_every_dependency():
    notices = read(REPO_ROOT / "THIRD_PARTY_NOTICES.md")
    for required in ("CHGNet", "Crystallography Open Database", "pymatgen", "ASE"):
        assert required in notices, f"{required} is unattributed"


def test_third_party_notices_end_with_the_non_claim():
    notices = read(REPO_ROOT / "THIRD_PARTY_NOTICES.md")
    assert re.search(r"What the derived values are not", notices, re.IGNORECASE)
    assert "not DFT results" in flat(REPO_ROOT / "THIRD_PARTY_NOTICES.md")


def test_readme_palette_test_count_matches_the_js_suite():
    """The JS suite is counted too, and separately from the Python ones.

    It exists because the README called assertPaletteDiscipline() a mechanism
    that fails loudly while nothing in CI ran it. A stated guard with no gate
    behind it is exactly what this project is written against.
    """
    suite = REPO_ROOT / "web" / "tests" / "palette.test.mjs"
    assert suite.exists(), "the palette suite the README cites is missing"
    actual = len(re.findall(r"^test\(", read(suite), re.MULTILINE))
    stated = re.search(r"Palette tests \(JS\):\s*(\d+)", read(README))
    assert stated, "README does not state a palette test count"
    assert int(stated.group(1)) == actual, (
        f"README says {stated.group(1)} palette tests, the suite defines {actual}"
    )


def test_ci_runs_the_palette_suite():
    workflow = read(REPO_ROOT / ".github" / "workflows" / "ci.yml")
    assert "npm test" in workflow, "CI must run the JS suite, not only build the bundle"


def test_pinned_requirements_cover_every_declared_dependency():
    """backend/requirements.txt must not drift behind pyproject.toml.

    CI installs from pyproject, so a stale pin file would rot unnoticed while
    still looking like the reproducible manifest the quick start points at.
    """
    import tomllib

    pyproject = tomllib.loads(read(REPO_ROOT / "backend" / "pyproject.toml"))
    project = pyproject["project"]
    declared = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared += list(extra)

    pinned = {
        re.split(r"[=<>\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        for line in read(REPO_ROOT / "backend" / "requirements.txt").splitlines()
        if line.strip() and not line.startswith("#")
    }
    missing = [
        name
        for spec in declared
        if (name := re.split(r"[=<>\[;]", spec, maxsplit=1)[0].strip().lower()) not in pinned
    ]
    assert not missing, f"declared in pyproject but not pinned: {missing}"


def test_licence_file_exists():
    assert (REPO_ROOT / "LICENSE").exists()


def test_cod_fixtures_are_attributed():
    notices = read(REPO_ROOT / "THIRD_PARTY_NOTICES.md")
    for fixture in (REPO_ROOT / "tests" / "fixtures").glob("*.cif"):
        assert fixture.name in notices, f"{fixture.name} is not listed in the notices"


# --------------------------------------------------- the UI tells the truth


def test_the_running_page_carries_the_disclosure_not_just_the_readme():
    """A user who never opens the repository must still learn what is real."""
    page = flat(REPO_ROOT / "web" / "index.html")
    assert "not deposited data" in page
    assert "not publication-grade" in page


def test_the_colour_rule_is_stated_where_it_is_implemented():
    """'Saturation marks provenance' is the honesty mechanism in the design."""
    palette = read(REPO_ROOT / "web" / "src" / "elements.js")
    assert "Saturation marks provenance" in palette
    assert "assertPaletteDiscipline" in palette
