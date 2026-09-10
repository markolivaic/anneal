"""Talking to the Crystallography Open Database.

Everything COD-specific lives here: the two endpoints anneal uses, the on-disk
CIF cache, and the sampling frame.

Two things about COD that are not in its documentation and cost an afternoon to
find:

1. `result?format=lst` returns bare ids, one per line, ~8 bytes each. The whole
   database comes back in one request. The JSON format returns the same rows at
   ~1.5 KB each, which is 800 MB for the same information.
2. There is no "return everything" query. Every search needs at least one
   bounded field, so the frame is built with a volume range no real crystal
   falls outside.

COD content is CC0; the public-domain statement is inside each CIF, not only on
the site.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COD_BASE = "https://www.crystallography.net/cod"

USER_AGENT = (
    "anneal/0.1 (github.com/markolivaic/anneal; "
    "filtering COD entries for CHGNet relaxation; contact via GitHub)"
)

# A volume range no real crystal falls outside. COD's search rejects an
# unbounded query, so this stands in for "everything".
FRAME_VMIN = 0.000001
FRAME_VMAX = 99999999

# COD holds CIFs with embedded structure-factor blocks. The largest seen in a
# 4,000-entry random sample was 48.6 MB against a median of 19.6 KB, and those
# few files dominate wall-clock on any batch fetch. Callers building a catalogue
# pass a cap; the survey deliberately passes None so its funnel stays unbiased.
DEFAULT_CIF_BYTE_CAP = 4_000_000


@dataclass
class Frame:
    """A set of COD ids and the provenance of the counts behind it."""

    ids: list[int] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    homepage_total: int | None = None
    query: str = ""
    built_at: str = ""


def make_session(pool: int = 16) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=pool))
    return session


def search_ids(
    session: requests.Session,
    *,
    vmin: float = FRAME_VMIN,
    vmax: float = FRAME_VMAX,
    extra: str = "",
    timeout: int = 900,
) -> list[int]:
    """Every COD id matching a volume range, as integers."""
    url = f"{COD_BASE}/result?format=lst&vmin={vmin}&vmax={vmax}{extra}"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return [int(x) for x in resp.text.split()]


def homepage_total(session: requests.Session) -> int | None:
    """COD's own entry count, scraped from the sentence it prints on the front page.

    Anchored on "Currently there are<strong>NNNNNN</strong> entries in the COD."
    so it cannot drift onto some other number. Returns None if the wording moves.
    """
    try:
        page = session.get(f"{COD_BASE}/", timeout=60).text
    except requests.RequestException:
        return None
    match = re.search(
        r"there\s+are\s*<strong>\s*([\d,]+)\s*</strong>\s*entries", page, re.IGNORECASE
    )
    return int(match.group(1).replace(",", "")) if match else None


def build_frame(session: requests.Session, cache: Path, *, refresh: bool = False) -> Frame:
    """The full sampling frame, plus what COD's default search leaves out.

    The default search hides duplicates, error-flagged and theoretical entries.
    anneal surveys and offers the default set, so that is the denominator; the
    include_* counts are recorded so the README can state precisely what the
    denominator excludes instead of asserting a bare total.
    """
    if cache.exists() and not refresh:
        return Frame(**json.loads(cache.read_text()))

    variants = {
        "default": "",
        "with_duplicates": "&include_duplicates=1",
        "with_errors": "&include_errors=1",
        "with_theoretical": "&include_theoretical=1",
        "with_all_three": "&include_duplicates=1&include_errors=1&include_theoretical=1",
    }
    counts: dict[str, int] = {}
    ids: list[int] = []
    for label, extra in variants.items():
        got = search_ids(session, extra=extra)
        counts[label] = len(got)
        if label == "default":
            ids = got
        print(f"  {label:<18} {len(got):>7,} ids")

    frame = Frame(
        ids=sorted(set(ids)),
        counts=counts,
        homepage_total=homepage_total(session),
        query=f"format=lst&vmin={FRAME_VMIN}&vmax={FRAME_VMAX}",
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(asdict(frame)))
    return frame


def fetch_cif(
    session: requests.Session,
    cod_id: int,
    cache_dir: Path,
    *,
    byte_cap: int | None = None,
    timeout: int = 120,
) -> tuple[int, str]:
    """Return (http_status, cif_text) for one COD entry, caching on disk.

    Status 0 means a transport error and the message is returned in place of the
    text. Status 413 means the file exceeded `byte_cap` and was abandoned
    mid-download rather than pulled in full.
    """
    path = cache_dir / f"{cod_id}.cif"
    if path.exists():
        return 200, path.read_text(encoding="utf8", errors="replace")

    try:
        resp = session.get(f"{COD_BASE}/{cod_id}.cif", timeout=timeout, stream=True)
    except requests.RequestException as exc:
        return 0, str(exc)

    with resp:
        if resp.status_code != 200:
            return resp.status_code, ""

        if byte_cap is None:
            text = resp.text
        else:
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(65536):
                size += len(chunk)
                if size > byte_cap:
                    return 413, ""
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf8", errors="replace")

    if not text.strip():
        return resp.status_code, ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")
    return 200, text
