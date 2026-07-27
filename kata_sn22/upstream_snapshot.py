"""Identity and integrity of the pinned upstream snapshot (plan §4, SN22-5).

The lane scores against *a specific commit* of `Desearch-ai/subnet-22`. That claim is only worth
something if it can be checked, so the snapshot is vendored as a complete tree and pinned by a
manifest of per-file digests plus one tree digest over the whole thing.

Three properties, each load-bearing:

* **The tree digest is the upstream's identity.** It goes into the parity evidence and into the
  bundle digest, so changing one upstream byte invalidates both — which is exactly the SN22-5 exit
  gate. There is no way to quietly re-point the adapter at different code.
* **Verification is fail-closed and structural.** A file listed in the manifest but missing, a file
  present but unlisted, a digest that differs, a symlink, or a path that escapes the root are all
  findings. An unlisted file matters as much as a changed one: the adapter imports from this tree,
  so an extra `sitecustomize.py` is code execution.
* **Nothing here executes upstream code.** This module reads bytes. Running the upstream is the
  parity harness's job (`kata_sn22.parity`), and it happens only under the import shim, never in
  the resident runtime.

The pinned tree is produced by ``git archive`` at the audited commit, so it contains exactly the
files tracked at that commit — no build output, no `.git`, nothing local.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

#: The audited upstream (plan §2). Single source of truth: the plugin, the adapter, the parity
#: report and the promotion provenance all import these two rather than restating them.
UPSTREAM_REPO = "https://github.com/Desearch-ai/subnet-22"
UPSTREAM_COMMIT = "bea9712f58a5fc01c57ec441ce279499529d8bf6"

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "UPSTREAM_MANIFEST.json"

#: Never digested or shipped: build artefacts are not upstream source, and including them would make
#: the tree digest depend on whether anyone had run the code.
_EXCLUDED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class SnapshotError(Exception):
    """The pinned snapshot is absent, unreadable, or does not match its manifest."""


def snapshot_root() -> Path:
    """Where the pinned upstream tree lives.

    ``KATA_SN22_UPSTREAM_ROOT`` exists for the installed layout, where the managed upstream tree is
    placed by the trusted installer rather than shipped inside the wheel. It is read once, here, so
    there is exactly one place that decides which tree the lane is scoring against.
    """
    override = os.environ.get("KATA_SN22_UPSTREAM_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "upstream").resolve()


def _iter_files(root: Path):
    """Every regular file in the snapshot, as POSIX-relative paths, in sorted order.

    Symlinks are yielded rather than skipped: the caller must be able to *report* one, because
    silently ignoring a symlink is how a link to `/srv/kata-bot/.env` ends up inside a "verified"
    tree.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDED_DIRS)
        for name in sorted(filenames):
            if name.endswith(_EXCLUDED_SUFFIXES):
                continue
            path = Path(dirpath) / name
            yield path.relative_to(root).as_posix(), path


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(files: dict[str, str]) -> str:
    """One digest over the whole tree: sorted ``path\\0sha256\\n`` lines.

    Path AND content, because a file moved from `neurons/` to `desearch/` is a different tree even
    though every byte is unchanged.
    """
    body = "".join(f"{path}\0{files[path]}\n" for path in sorted(files))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def compute_manifest(root: Path | None = None) -> dict:
    """Build the manifest from what is actually on disk. Used to *generate*, not to verify."""
    root = Path(root).resolve() if root is not None else snapshot_root()
    if not root.is_dir():
        raise SnapshotError(f"pinned upstream snapshot is absent at {root}")
    files: dict[str, str] = {}
    for relative, path in _iter_files(root):
        if relative == MANIFEST_NAME:
            continue   # the manifest cannot contain its own digest
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(f"snapshot entry {relative} is not a regular file")
        files[relative] = _digest_file(path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "file_count": len(files),
        "tree_sha256": tree_digest(files),
        "files": dict(sorted(files.items())),
    }


def load_manifest(root: Path | None = None) -> dict:
    root = Path(root).resolve() if root is not None else snapshot_root()
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise SnapshotError(f"pinned upstream snapshot has no {MANIFEST_NAME} at {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SnapshotError(f"{MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SnapshotError(f"{MANIFEST_NAME} schema {manifest.get('schema_version')!r} "
                            f"is not {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise SnapshotError(
            f"{MANIFEST_NAME} pins {manifest.get('upstream_commit')!r} but this adapter is built "
            f"for {UPSTREAM_COMMIT!r}")
    return manifest


@dataclass(frozen=True)
class SnapshotVerification:
    """The result of checking the tree against its manifest. Empty ``findings`` means intact."""

    root: str
    expected_tree_sha256: str
    observed_tree_sha256: str
    findings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        return {"root": self.root, "expected_tree_sha256": self.expected_tree_sha256,
                "observed_tree_sha256": self.observed_tree_sha256, "ok": self.ok,
                "findings": list(self.findings)}


def verify_snapshot(root: Path | None = None) -> SnapshotVerification:
    """Check the on-disk tree against the manifest. Reports every finding, never raises on drift.

    Returning findings rather than raising on the first one is deliberate: an operator looking at a
    tampered install wants the whole list, and a caller that wants failure can read ``ok``.
    """
    root = Path(root).resolve() if root is not None else snapshot_root()
    manifest = load_manifest(root)
    expected: dict[str, str] = dict(manifest.get("files") or {})
    findings: list[str] = []
    observed: dict[str, str] = {}

    for relative, path in _iter_files(root):
        if relative == MANIFEST_NAME:
            continue
        if path.is_symlink():
            findings.append(f"{relative}: symlink in the pinned snapshot")
            continue
        # A resolved path outside the root means a bind/junction escape; refuse to digest it.
        try:
            path.resolve().relative_to(root)
        except ValueError:
            findings.append(f"{relative}: resolves outside the snapshot root")
            continue
        if not path.is_file():
            findings.append(f"{relative}: not a regular file")
            continue
        observed[relative] = _digest_file(path)

    for relative in sorted(set(expected) - set(observed)):
        findings.append(f"{relative}: listed in the manifest but missing from the tree")
    for relative in sorted(set(observed) - set(expected)):
        findings.append(f"{relative}: present in the tree but not listed in the manifest")
    for relative in sorted(set(expected) & set(observed)):
        if expected[relative] != observed[relative]:
            findings.append(f"{relative}: digest drift "
                            f"({expected[relative][:12]} -> {observed[relative][:12]})")

    return SnapshotVerification(
        root=str(root),
        expected_tree_sha256=str(manifest.get("tree_sha256") or ""),
        observed_tree_sha256=tree_digest(observed),
        findings=tuple(findings),
    )


def require_intact(root: Path | None = None) -> str:
    """Verify and return the tree digest, or raise. For callers that must fail closed."""
    verification = verify_snapshot(root)
    if not verification.ok:
        raise SnapshotError("pinned upstream snapshot failed verification:\n  "
                            + "\n  ".join(verification.findings))
    if verification.observed_tree_sha256 != verification.expected_tree_sha256:
        raise SnapshotError("pinned upstream tree digest does not match the manifest")
    return verification.observed_tree_sha256


def upstream_identity() -> dict:
    """The three fields every report, receipt and provenance record quotes about the upstream."""
    manifest = load_manifest()
    return {"upstream_repo": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tree_sha256": str(manifest.get("tree_sha256") or "")}


__all__ = [
    "MANIFEST_NAME",
    "SnapshotError",
    "SnapshotVerification",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "compute_manifest",
    "load_manifest",
    "require_intact",
    "snapshot_root",
    "tree_digest",
    "upstream_identity",
    "verify_snapshot",
]
