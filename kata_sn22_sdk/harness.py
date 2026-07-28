"""The agent image's entry point: load ``/bundle/agent.py``, run one task, frame one answer.

The room starts this, not the submission. That matters for three reasons:

1. **The framing is the lane's.** Every submission's answer reaches the scorer in the same shape,
   produced by one reviewed function. A per-agent serialiser would make "both contestants were asked
   the same thing" depend on two strangers' JSON.
2. **A crash is still an answer.** If a submission raises, times out or returns nonsense, the
   harness writes a well-formed empty answer and a diagnostic on stderr. The alternative is a
   process that exits non-zero and a report the room cannot parse — which reads as a broken room
   rather than a broken agent, and defers a duel that should simply have been lost.
3. **The bundle is read-only and gets no import of ours to subvert.** The submission is imported by
   path, once, with nothing added to it.

The harness holds no credential and no scorer. It can reach the broker and stdout, and that is all.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import time
import traceback

from kata_sn22_sdk.agent import Agent, Emit
from kata_sn22_sdk.models import (
    PROTOCOL_VERSION,
    AiSearchResult,
    AiSearchSynapse,
    ScraperTextRole,
    SdkError,
    XSearchResult,
    synapse_from_input,
)

DEFAULT_BUNDLE_AGENT = "/bundle/agent.py"

#: Ceilings on what a submission may return. Not a scoring rule -- a memory bound on the room, which
#: is shared with the contestant that runs next.
MAX_RESULTS = 1_000
MAX_FIELD_CHARS = 100_000
#: Quoted spans per source. Every one must be found, in order, in the validator's own copy of the
#: page, so more is not better -- each extra is another chance to fail evidence entirely.
MAX_HIGHLIGHTS = 32


def load_agent(path: str) -> Agent:
    """Import the submission and instantiate its Agent subclass.

    Exactly one subclass, named or not. Two would make "which one runs" a question answered by
    definition order, and a submission whose behaviour depends on that is a submission nobody can
    review.
    """
    # The bundle is mounted read-only in the room, so bytecode could never be written there
    # anyway. Off explicitly because the harness also runs outside the room -- in a miner's
    # calibration and in this repository's tests -- and a __pycache__ deposited in a submission
    # directory is a file the miner did not put there and did not seal.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("kata_submission", path)
    if spec is None or spec.loader is None:
        raise SdkError(f"cannot load a submission from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a submission that imports itself does not re-execute.
    sys.modules["kata_submission"] = module
    spec.loader.exec_module(module)

    found = [
        value for value in vars(module).values()
        if inspect.isclass(value) and issubclass(value, Agent) and value is not Agent
    ]
    if not found:
        raise SdkError(
            "the submission defines no Agent subclass; see `from kata_sn22_sdk import Agent`")
    if len(found) > 1:
        raise SdkError(
            f"the submission defines {len(found)} Agent subclasses "
            f"({', '.join(sorted(cls.__name__ for cls in found))}); exactly one must be defined")
    return found[0]()


def _strings(value, limit: int = MAX_RESULTS) -> list:
    return [item for item in (value or []) if isinstance(item, dict) and item][:limit]


def _bounded(value: object) -> object:
    """Trim a returned string so one enormous field cannot be used against the room."""
    return value[:MAX_FIELD_CHARS] if isinstance(value, str) else value


def _bounded_objects(items: list) -> list:
    """Bound every value, including inside a list.

    The list case matters: ``highlights`` is a list of quoted spans, and bounding only top-level
    strings would leave an unbounded one reachable through it.
    """
    return [
        {str(key): _bounded_value(item[key]) for key in list(item)[:64]}
        for item in items
    ]


def _bounded_value(value: object) -> object:
    if isinstance(value, list):
        return [_bounded(entry) for entry in value[:MAX_HIGHLIGHTS]]
    return _bounded(value)


def frame_ai_answer(synapse: AiSearchSynapse, result: AiSearchResult, emit: Emit) -> dict:
    """The AI-search answer, in exactly the fields the pinned scorer reads.

    ``completion`` and ``texts`` are derived from what was emitted unless the submission set them
    explicitly. Deriving is the default because the two must agree: a completion that says something
    the streamed chunks did not is a completion the streaming penalty was computed against something
    else.
    """
    if not isinstance(result, AiSearchResult):
        raise SdkError("smart_scraper must return an AiSearchResult")
    texts = result.texts if isinstance(result.texts, dict) else emit.texts()
    completion = result.completion if isinstance(result.completion, str) else emit.completion()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": synapse.task_id,
        "completion": _bounded(completion),
        "texts": {str(role): _bounded(text) for role, text in texts.items()},
        "chunks": emit.chunks,
        "search_results": _bounded_objects(_strings(result.search_results)),
        "miner_tweets": _bounded_objects(_strings(result.miner_tweets)),
        "text_chunks": {role: [_bounded(text) for text in texts]
                        for role, texts in emit.text_chunks().items()},
    }


def frame_x_answer(synapse, result: XSearchResult) -> dict:
    if not isinstance(result, XSearchResult):
        raise SdkError("twitter_search must return an XSearchResult")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": synapse.task_id,
        "results": _bounded_objects(_strings(result.results)),
    }


def empty_answer(synapse) -> dict:
    """A well-formed answer that scores nothing.

    Written whenever a submission fails, so a bad agent loses its duel instead of breaking the room.
    """
    if isinstance(synapse, AiSearchSynapse):
        return {"protocol_version": PROTOCOL_VERSION, "task_id": synapse.task_id,
                "completion": "", "texts": {}, "chunks": [], "search_results": [],
                "miner_tweets": [], "text_chunks": {}}
    return {"protocol_version": PROTOCOL_VERSION, "task_id": synapse.task_id, "results": []}


async def _run_once(agent: Agent, synapse, *, timeout: float) -> dict:
    emit = Emit()
    if isinstance(synapse, AiSearchSynapse):
        coroutine = agent.smart_scraper(synapse, emit)
        result = await asyncio.wait_for(coroutine, timeout=timeout)
        return frame_ai_answer(synapse, result, emit)
    coroutine = agent.twitter_search(synapse)
    result = await asyncio.wait_for(coroutine, timeout=timeout)
    return frame_x_answer(synapse, result)


def run(document: dict, *, agent_path: str = DEFAULT_BUNDLE_AGENT,
        stderr=None) -> dict:
    """Run one task end to end and return the answer document.

    Never raises for a submission's own fault. The room reads the answer on stdout and the stderr
    tail is kept for the operator; a contestant that could influence its score by what it printed to
    stderr would be scoring itself, so nothing here is read back into scoring.
    """
    stderr = stderr if stderr is not None else sys.stderr
    synapse = synapse_from_input(document)
    # Upstream's own serving budget for the mode. Answering late is penalised and answering very
    # late is worth less than not answering, so the harness stops rather than overrunning.
    timeout = float(getattr(synapse.limits, "max_execution_time", 30))
    started = time.monotonic()
    try:
        agent = load_agent(agent_path)
    except Exception:  # noqa: BLE001 - a submission's import-time failure is its own
        traceback.print_exc(file=stderr)
        return empty_answer(synapse)
    try:
        return asyncio.run(_run_once(agent, synapse, timeout=timeout))
    except NotImplementedError:
        # The submission declined this task family. An empty answer scores badly in one pool; a
        # crash would be an invalid run and cost more than the pool was worth.
        print(f"submission does not implement {synapse.search_type.value}", file=stderr)
        return empty_answer(synapse)
    except TimeoutError:
        print(f"submission exceeded max_execution_time={timeout}s", file=stderr)
        return empty_answer(synapse)
    except Exception:  # noqa: BLE001 - every submission fault ends the same way
        traceback.print_exc(file=stderr)
        return empty_answer(synapse)
    finally:
        print(f"elapsed={time.monotonic() - started:.3f}s", file=stderr)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    agent_path = argv[0] if argv else DEFAULT_BUNDLE_AGENT
    try:
        document = json.loads(sys.stdin.read())
    except ValueError as exc:
        print(f"task is not valid JSON: {exc}", file=sys.stderr)
        return 2
    try:
        answer = run(document, agent_path=agent_path)
    except SdkError as exc:
        # The TASK was unusable, which is the room's fault rather than the submission's. Exiting
        # non-zero is right here: there is no answer to frame, and pretending otherwise would score
        # a contestant for a question it was never asked.
        print(f"unusable task: {exc}", file=sys.stderr)
        return 2
    json.dump(answer, sys.stdout, separators=(",", ":"))
    sys.stdout.flush()
    return 0


__all__ = ["Emit", "ScraperTextRole", "empty_answer", "frame_ai_answer", "frame_x_answer",
           "load_agent", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
