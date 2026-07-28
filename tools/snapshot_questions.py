"""Snapshot the pinned upstream question dataset into a packaged pool. Operator-run, once.

Upstream fetches its questions from HuggingFace at run time and falls back to an LLM call when that
fails. A Kata epoch can do neither — the room has no egress, and the validator holds no paid
credential to spend on inventing questions. So the rows are captured here, committed, and their
digest is recorded in every manifest built from them.

    uv run --extra snapshot python tools/snapshot_questions.py --out production

Requires ``huggingface_hub`` and ``datasets``, which are deliberately NOT lane dependencies: this
runs on an operator's machine, never in the room and never on a duel. That is the whole point —
network access at snapshot time is fine, network access at duel time is not.

**Why not just run it in CI.** Because then the question pool would change whenever the upstream
dataset did, silently, between one duel and the next. A snapshot is committed so that "what were the
two contestants asked" has an answer that does not depend on when you look.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kata_sn22.question_pool import (  # noqa: E402
    DATASETS_DIR,
    KIND_UPSTREAM_SNAPSHOT,
    WEB_LANES,
)
from kata_sn22.upstream_snapshot import UPSTREAM_COMMIT  # noqa: E402

#: Read off ``desearch/dataset/hf_dataset.py`` at the pinned commit. Restated rather than imported
#: because importing that module pulls in ``bittensor``.
DATASET_REPO = "desearch/dataset"
SUBSETS = ({"prefix": "questions/", "lane": "news"}, {"prefix": "x/", "lane": "x"})
EXTRA_DATASETS = (
    {"repo": "sentence-transformers/squad", "question_col": "question", "lane": "squad"},
    {"repo": "sentence-transformers/natural-questions", "question_col": "query", "lane": "nq"},
)
DATASET_LIMIT = 5_000

#: A pool smaller than this cannot fill 45 AI tasks without repeating a question many times over,
#: and a repeated question is a task both contestants can answer once and reuse.
MIN_ROWS_PER_LANE = 50


def fetch_desearch_rows() -> list:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    prefixes = tuple(subset["prefix"] for subset in SUBSETS)
    wanted = [name for name in files if name.endswith(".jsonl") and name.startswith(prefixes)]
    if not wanted:
        raise SystemExit(f"ERROR: {DATASET_REPO} exposes no .jsonl question files")

    rows: list = []
    for name in wanted:
        lane = next(s["lane"] for s in SUBSETS if name.startswith(s["prefix"]))
        local = hf_hub_download(DATASET_REPO, name, repo_type="dataset")
        for line in Path(local).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not (row.get("id") and row.get("question")):
                continue
            rows.append({"id": str(row["id"]), "question": str(row["question"]),
                         "start_date": row.get("start_date"), "end_date": row.get("end_date"),
                         "lane": lane})
    return rows


def fetch_extra_rows(limit: int) -> list:
    from datasets import load_dataset

    rows: list = []
    for spec in EXTRA_DATASETS:
        data = load_dataset(spec["repo"], split="train", streaming=True)
        for index, record in enumerate(data):
            if index >= limit:
                break
            question = record.get(spec["question_col"])
            if not question:
                continue
            rows.append({
                "id": f"{spec['lane']}-{index:06d}",
                "question": str(question),
                "start_date": None, "end_date": None, "lane": spec["lane"],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="production",
                        help="pool name to write under kata_sn22/datasets/")
    parser.add_argument("--limit", type=int, default=DATASET_LIMIT,
                        help="rows per auxiliary dataset (upstream's DATASET_LIMIT)")
    args = parser.parse_args()

    rows = fetch_desearch_rows() + fetch_extra_rows(args.limit)

    # Deduplicate by id, exactly as upstream's pool does on load.
    seen: set = set()
    unique: list = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        unique.append(row)

    by_lane: dict = {}
    for row in unique:
        by_lane.setdefault(row["lane"], 0)
        by_lane[row["lane"]] += 1

    # Fail here rather than shipping a pool that cannot fill an epoch. A thin lane means the same
    # question is asked several times in one epoch, which a contestant can answer once and reuse.
    thin = {lane: count for lane, count in by_lane.items() if count < MIN_ROWS_PER_LANE}
    if thin:
        raise SystemExit(f"ERROR: lanes below {MIN_ROWS_PER_LANE} rows: {thin}")
    missing = [lane for lane in (*WEB_LANES, "x") if lane not in by_lane]
    if missing:
        raise SystemExit(f"ERROR: no rows for lane(s) {missing}")

    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in unique)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / f"{args.out}.jsonl").write_text(body, encoding="utf-8")
    metadata = {
        "pool_kind": KIND_UPSTREAM_SNAPSHOT,
        "source": f"{DATASET_REPO} + " + ", ".join(spec["repo"] for spec in EXTRA_DATASETS),
        "upstream_commit": UPSTREAM_COMMIT,
        "row_count": len(unique),
        "rows_per_lane": by_lane,
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    (DATASETS_DIR / f"{args.out}.meta.json").write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(unique)} rows to {DATASETS_DIR / (args.out + '.jsonl')}")
    print(f"  lanes:  {by_lane}")
    print(f"  sha256: {metadata['sha256']}")
    print("Commit both files. Every manifest built from this pool records that digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
