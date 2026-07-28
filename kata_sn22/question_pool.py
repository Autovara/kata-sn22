"""The packaged question rows an epoch is built from, and the refusal when they are not real.

Upstream draws its questions from a HuggingFace dataset at run time, and falls back to an LLM call
when that fails. Kata can do neither:

* **No run-time download.** The generator runs inside a sealed room with no egress, and a question
  set fetched at duel time is a question set the two contestants could receive different versions
  of. So the rows are packaged, and their digest goes into the manifest.
* **No LLM fallback.** Upstream's fallback spends the *validator's* money to invent questions.
  Under Kata's funding rule the validator holds no paid credential at all, so that path does not
  merely cost something — it cannot run. A production epoch that cannot find its rows must **fail
  before the duel**, not quietly ask a model to make some up.

**Two kinds of pool, and production accepts only one.** A pool declares itself
``upstream-snapshot`` (rows taken from the pinned upstream dataset by
``tools/snapshot_questions.py``) or ``development`` (rows written by hand so this repository's
tests can run). Production refuses the second by name. Without that distinction the
convenient thing — ship the dev pool, forget to snapshot — is indistinguishable from the
correct one, right up until someone asks where the questions came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

#: Rows taken from the pinned upstream dataset. The only kind a production epoch will use.
KIND_UPSTREAM_SNAPSHOT = "upstream-snapshot"
#: Rows written by hand so tests can run. Refused in production, by name.
KIND_DEVELOPMENT = "development"

#: Upstream's lanes. ``x`` feeds AI search's Twitter tool (advanced search), NOT Basic X search --
#: Basic X queries are generated locally from the seed lists, never drawn from this pool.
WEB_LANES = ("news", "squad", "nq")
X_LANE = "x"

REQUIRED_ROW_FIELDS = ("id", "question")


class PoolError(Exception):
    """The packaged questions are missing, malformed, or not the kind this epoch requires."""


@dataclass(frozen=True)
class QuestionPool:
    """One packaged set of question rows, with the provenance a manifest has to record."""

    name: str
    kind: str
    rows: tuple
    digest: str
    upstream_commit: str = ""
    source: str = ""

    def lane(self, lane: str) -> tuple:
        """Every row for one lane, in file order.

        File order, not shuffled: the caller seeds its own RNG and the manifest records that seed.
        A pool that shuffled on load would make the same seed produce different epochs.
        """
        return tuple(row for row in self.rows if row.get("lane") == lane)

    def web(self) -> tuple:
        return tuple(row for row in self.rows if row.get("lane") in WEB_LANES)

    def as_provenance(self) -> dict:
        """What the manifest records about where the questions came from."""
        return {"pool_name": self.name, "pool_kind": self.kind, "pool_digest": self.digest,
                "pool_rows": len(self.rows), "pool_source": self.source,
                "pool_upstream_commit": self.upstream_commit}

    def require_production_kind(self) -> None:
        """Raise unless these rows are a real upstream snapshot.

        Called before a production epoch is built rather than after, so the refusal happens while
        nobody has spent anything.
        """
        if self.kind != KIND_UPSTREAM_SNAPSHOT:
            raise PoolError(
                f"question pool {self.name!r} is {self.kind!r}, not {KIND_UPSTREAM_SNAPSHOT!r}; a "
                f"production epoch must be built from rows snapshotted out of the pinned upstream "
                f"dataset. Run tools/snapshot_questions.py")


def pool_path(name: str) -> Path:
    return DATASETS_DIR / f"{name}.jsonl"


def meta_path(name: str) -> Path:
    return DATASETS_DIR / f"{name}.meta.json"


def load_pool(name: str) -> QuestionPool:
    """Load a packaged pool and verify it against its recorded digest.

    The digest is checked on every load, not just when it is written. A pool edited in place --
    a row removed to make a test pass, say -- would otherwise change what both contestants are
    asked while the manifest went on reporting the old identity.
    """
    rows_file, metadata_file = pool_path(name), meta_path(name)
    if not rows_file.is_file():
        raise PoolError(
            f"no question pool at {rows_file}. A production epoch cannot be built without one, and "
            f"there is no LLM fallback: upstream's would spend the validator's money, and the "
            f"validator holds no paid credential")
    if not metadata_file.is_file():
        raise PoolError(f"question pool {name!r} has no {metadata_file.name}; its provenance is "
                        f"unknown and a manifest cannot record where the questions came from")

    raw = rows_file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PoolError(f"question pool {name!r} has unreadable metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise PoolError(f"question pool {name!r} metadata is not an object")

    recorded = metadata.get("sha256")
    if recorded != digest:
        raise PoolError(
            f"question pool {name!r} does not match its recorded digest "
            f"(recorded {str(recorded)[:16]}..., actual {digest[:16]}...); it was edited in place")

    kind = metadata.get("pool_kind")
    if kind not in (KIND_UPSTREAM_SNAPSHOT, KIND_DEVELOPMENT):
        raise PoolError(f"question pool {name!r} declares an unknown pool_kind {kind!r}")

    rows = _parse_rows(raw, name)
    return QuestionPool(
        name=name, kind=kind, rows=rows, digest=digest,
        upstream_commit=str(metadata.get("upstream_commit") or ""),
        source=str(metadata.get("source") or ""),
    )


def _parse_rows(raw: bytes, name: str) -> tuple:
    """Upstream's own row contract: an id, a question, and optional dates.

    Rows missing either required field are DROPPED rather than raising, which is upstream's
    behaviour in ``HFQuestionPool._parse_files``. A malformed line is a bad row, not a bad pool.
    """
    rows = []
    seen: set = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not all(row.get(field) for field in REQUIRED_ROW_FIELDS):
            continue
        if row["id"] in seen:
            continue          # upstream dedups by id
        seen.add(row["id"])
        row.setdefault("start_date", None)
        row.setdefault("end_date", None)
        row.setdefault("lane", "news")
        rows.append(row)
    if not rows:
        raise PoolError(f"question pool {name!r} contains no usable rows")
    return tuple(rows)


def load_x_seeds() -> dict:
    """The topic, account and keyword lists upstream's Basic X generator draws from.

    Extracted from the pinned tree rather than retyped, and re-derived by
    ``tests/test_sn22_epoch_manifest.py`` so a divergence is a test failure. They are packaged
    because the room runtime does not read the 2.4 MB vendored tree, and because these three lists
    genuinely are upstream's -- unlike the web questions, which upstream fetches at run time.
    """
    path = DATASETS_DIR / "x_seeds.json"
    if not path.is_file():
        raise PoolError(f"no X seed lists at {path}")
    seeds = json.loads(path.read_text(encoding="utf-8"))
    for field in ("topics", "popular_accounts", "popular_crypto_keywords"):
        if not isinstance(seeds.get(field), list) or not seeds[field]:
            raise PoolError(f"X seed lists are missing {field}")
    return seeds
