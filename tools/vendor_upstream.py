#!/usr/bin/env python3
"""Regenerate or verify the pinned upstream manifest (SN22-5).

Two subcommands, and the split is the point:

* ``verify`` runs offline against the vendored tree and is what CI and the trusted installer call.
* ``write`` regenerates ``UPSTREAM_MANIFEST.json`` from the tree on disk. It is a DELIBERATE act by
  a reviewer after re-vendoring at a new commit, never something a build does implicitly — a build
  that regenerates its own pin can never detect drift.

Re-vendoring is `git archive` at the audited commit, so the tree is exactly the files tracked there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kata_sn22.upstream_snapshot import (  # noqa: E402
    MANIFEST_NAME,
    UPSTREAM_COMMIT,
    compute_manifest,
    snapshot_root,
    verify_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "write"))
    parser.add_argument("--root", default=None, help="Snapshot root (defaults to ./upstream).")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else snapshot_root()

    if args.action == "write":
        manifest = compute_manifest(root)
        (root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {root / MANIFEST_NAME}")
        print(f"  commit    {UPSTREAM_COMMIT}")
        print(f"  files     {manifest['file_count']}")
        print(f"  tree      {manifest['tree_sha256']}")
        return 0

    verification = verify_snapshot(root)
    print(f"root       {verification.root}")
    print(f"expected   {verification.expected_tree_sha256}")
    print(f"observed   {verification.observed_tree_sha256}")
    for finding in verification.findings:
        print(f"  FINDING  {finding}")
    print("OK" if verification.ok else "DRIFT")
    return 0 if verification.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
