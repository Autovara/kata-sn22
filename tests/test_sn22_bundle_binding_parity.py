"""The bundle binding must match ``kata-tee-runner`` byte for byte, and this file is what says so.

A miner computes this digest on their own machine and seals a credential to it. The room recomputes
it from the extracted bundle and refuses the credential if the two differ. The two implementations
live in different repositories, were written at different times, and neither one's own tests can see
the other.

They *did* differ. Both were internally sensible -- domain separator or not, length-prefixed
framing or nul-separated, content hashed or content inlined -- and both had passing tests. The
disagreement would have surfaced for the first time when a real miner's sealed credential met a real
room, as "sealed miner credential is not bound to this candidate bundle" on every single SN22
submission, with nothing in either repository to explain it.

So the digest is pinned to a literal here. If either side's construction changes, this vector stops
matching and a person has to decide whether both sides were changed. That is the entire point: the
value below is not a convenience, it is the interface.
"""

from __future__ import annotations

import hashlib

import pytest

from kata_sn22.credentials_v2 import SEALED_FILENAME, compute_bundle_binding

#: A fixed bundle and its digest under the authoritative construction (``kata-tee-runner``'s
#: ``room/bundle.py``). Contains a nested path and a dash-named sibling on purpose -- see
#: ``test_ordering_follows_path_parts_not_the_joined_string``.
VECTOR_FILES = {
    "agent.py": b"def agent_main():\n    return {}\n",
    "submission.json": b'{"submission_id":"pinned"}\n',
    "helpers/util.py": b"VALUE = 1\n",
    "helpers-extra.py": b"VALUE = 2\n",
    SEALED_FILENAME: b"deadbeefcafe",
}
#: Produced by running kata-tee-runner's real ``credential_bundle_binding`` against a real
#: directory holding exactly ``VECTOR_FILES``. Mirrored in that repository's
#: ``test_bundle_binding_vector.py`` so a change on EITHER side trips a test.
VECTOR_DIGEST = "e7b9e082a71716f3dab9157797fc476ef4312bc69aa2cb0ea93b66314e524ed5"


def _authoritative(files: dict) -> str:
    """The room's construction, transcribed from ``kata-tee-runner/room/bundle.py``.

    Transcribed rather than imported: the two repositories ship as separate images and kata-sn22's
    runtime is stdlib-only by design (see ``docs/DECISION-bittensor-not-in-the-room.md``), so there
    is no import that could tie them together at runtime. A transcription plus a pinned vector is
    what stands in for that.
    """
    from pathlib import PurePosixPath

    digest = hashlib.sha256(b"kata-miner-credential-bundle-v1\0")
    for relative in sorted(files, key=lambda name: PurePosixPath(name).parts):
        path = PurePosixPath(relative)
        if (
            relative == "sealed_inference_key"
            or path.suffix in {".pyc", ".pyo"}
            or any(part in {".git", "__pycache__"} for part in path.parts)
        ):
            continue
        encoded_path = relative.encode("utf-8")
        content = files[relative]
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def test_this_repos_binding_matches_the_authoritative_construction():
    """THE cross-repository check."""
    assert compute_bundle_binding(VECTOR_FILES) == _authoritative(VECTOR_FILES)


def test_the_pinned_vector_still_holds():
    """A literal, so a change on either side has to be noticed by a person rather than absorbed."""
    assert compute_bundle_binding(VECTOR_FILES) == VECTOR_DIGEST


def test_ordering_follows_path_parts_not_the_joined_string():
    """The room walks a filesystem and sorts ``Path`` objects, which orders by path *parts*.

    ``helpers/util.py`` sorts BEFORE ``helpers-extra.py`` by parts and AFTER it by string, so a
    bundle with both -- entirely ordinary -- would hash differently under a naive string sort.
    This is the sort of detail that is invisible until it costs a miner a duel.
    """
    from pathlib import PurePosixPath

    names = ["helpers/util.py", "helpers-extra.py"]
    assert sorted(names) != sorted(names, key=lambda n: PurePosixPath(n).parts)
    assert sorted(names, key=lambda n: PurePosixPath(n).parts) == [
        "helpers/util.py",
        "helpers-extra.py",
    ]


def test_the_ciphertext_is_excluded_from_its_own_binding():
    without = {name: data for name, data in VECTOR_FILES.items() if name != SEALED_FILENAME}
    assert compute_bundle_binding(without) == VECTOR_DIGEST
    changed = {**VECTOR_FILES, SEALED_FILENAME: b"a-completely-different-ciphertext"}
    assert compute_bundle_binding(changed) == VECTOR_DIGEST


@pytest.mark.parametrize(
    "transient",
    ["__pycache__/agent.cpython-313.pyc", "helper.pyc", "helper.pyo", ".git/HEAD"],
)
def test_transient_local_artifacts_do_not_change_the_binding(transient: str):
    """A miner who ran their agent locally before sealing must not get a credential the room
    rejects. Both sides exclude exactly the same set."""
    assert compute_bundle_binding({**VECTOR_FILES, transient: b"generated"}) == VECTOR_DIGEST


@pytest.mark.parametrize("changed", ["agent.py", "submission.json", "helpers/util.py"])
def test_editing_any_submitted_file_invalidates_the_binding(changed: str):
    """A credential bound to code the miner has since changed is a credential paying for code
    nobody reviewed."""
    tampered = {**VECTOR_FILES, changed: b"tampered\n"}
    assert compute_bundle_binding(tampered) != VECTOR_DIGEST


def test_renaming_a_file_invalidates_the_binding():
    """The path is hashed, not just the content -- otherwise moving code between files would keep
    the seal valid."""
    renamed = {name: data for name, data in VECTOR_FILES.items() if name != "agent.py"}
    renamed["main.py"] = VECTOR_FILES["agent.py"]
    assert compute_bundle_binding(renamed) != VECTOR_DIGEST


def test_content_cannot_be_shifted_between_adjacent_files():
    """Why the lengths are prefixed. Without framing, ``{"ab": b"", "": b"x"}`` and
    ``{"a": b"", "b": b"x"}`` could hash alike, and a bundle could be restructured under a valid
    seal."""
    first = compute_bundle_binding({"ab.py": b"", "c.py": b"x"})
    second = compute_bundle_binding({"a.py": b"", "bc.py": b"x"})
    assert first != second
