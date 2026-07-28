"""The base class a submission subclasses, and the streaming recorder it is handed.

```python
from kata_sn22_sdk import Agent


class Submission(Agent):
    async def smart_scraper(self, synapse, emit):
        ...

    async def twitter_search(self, synapse):
        ...
```

**Why two methods rather than one.** They are upstream's two task families and they are scored by
different code — AI search on quality, structure, groundedness and evidence; Basic X search on
whether the tweets you returned are real and correctly ordered. A single entry point would make
every submission begin by branching on something the harness already knows.

**Why ``emit``.** Upstream's miners stream their completion, and the streaming penalty counts tokens
per emitted chunk. A submission that computed a final answer and returned it in one piece would take
that penalty for a difference that has nothing to do with answer quality. ``emit`` records what a
streamed answer would have streamed, and the harness derives ``completion``, ``texts`` and
``text_chunks`` from it — so the natural way to write an agent is also the way that scores.

**Async because the work is I/O.** A submission spends its time waiting on searches. ``asyncio`` is
in the standard library and the harness runs the event loop; nothing here requires an agent to
understand it beyond writing ``async def`` and ``await``.
"""

from __future__ import annotations

from kata_sn22_sdk.broker import BrokerClient
from kata_sn22_sdk.models import (
    AiSearchResult,
    AiSearchSynapse,
    ScraperTextRole,
    SdkError,
    XSearchResult,
    XSearchSynapse,
)


class Emit:
    """Records the chunks a streamed answer would have streamed, in order.

    Call it as ``emit(role, text)``. Everything the scorer reads about your prose is derived from
    what you emitted:

    * ``chunks`` — every call, in order. What the streaming penalty counts.
    * ``text_chunks[role]`` — the texts for one role, in order.
    * ``texts[role]`` — those texts joined. ``texts["summary"]`` is what the groundedness judge
      reads, so that is the one worth getting right.
    * ``completion`` — every chunk joined, in emission order.

    Emitting nothing is allowed and is the right answer for an ``ONLY_LINKS`` task: there is no
    summary to judge, and writing one spends tokens on something nobody grades.
    """

    #: A submission that streams without bound would fill the room's memory before its timeout.
    MAX_CHUNKS = 4_096
    MAX_TOTAL_CHARS = 2_000_000

    def __init__(self) -> None:
        self._chunks: list[tuple[str, str]] = []
        self._total_chars = 0

    def __call__(self, role, text: str) -> None:
        role_value = role.value if isinstance(role, ScraperTextRole) else str(role)
        try:
            ScraperTextRole(role_value)
        except ValueError as exc:
            raise SdkError(
                f"unknown text role {role_value!r}; the scorer reads a fixed set and a chunk it "
                f"cannot place is a chunk it does not count") from exc
        if not isinstance(text, str):
            raise SdkError("emitted text must be a string")
        if len(self._chunks) >= self.MAX_CHUNKS:
            raise SdkError(f"emitted more than {self.MAX_CHUNKS} chunks")
        self._total_chars += len(text)
        if self._total_chars > self.MAX_TOTAL_CHARS:
            raise SdkError(f"emitted more than {self.MAX_TOTAL_CHARS} characters in total")
        self._chunks.append((role_value, text))

    @property
    def chunks(self) -> list:
        return [{"role": role, "text": text} for role, text in self._chunks]

    def text_chunks(self) -> dict:
        grouped: dict = {}
        for role, text in self._chunks:
            grouped.setdefault(role, []).append(text)
        return grouped

    def texts(self) -> dict:
        return {role: "".join(texts) for role, texts in self.text_chunks().items()}

    def completion(self) -> str:
        return "".join(text for _role, text in self._chunks)


class Agent:
    """Subclass this. Implement the method for each task family you intend to answer.

    A submission that does not implement one still runs: the harness returns an empty answer for
    that family rather than crashing. That is deliberate — an empty answer scores badly in one pool,
    while a crash is an invalid run and costs more than the pool was worth. But note that every pool
    is weighted, so declining one is a decision, not a saving.
    """

    def __init__(self) -> None:
        #: The only way out. See :mod:`kata_sn22_sdk.broker` -- there is no credential on it.
        self.broker = BrokerClient()

    async def smart_scraper(self, synapse: AiSearchSynapse, emit: Emit) -> AiSearchResult:
        """Answer one AI-search task.

        Return an :class:`~kata_sn22_sdk.models.AiSearchResult` carrying the sources you found, and
        stream your prose through ``emit``.
        """
        raise NotImplementedError

    async def twitter_search(self, synapse: XSearchSynapse) -> XSearchResult:
        """Answer one Basic X-search task.

        Return an :class:`~kata_sn22_sdk.models.XSearchResult` carrying raw tweet objects, unedited.
        """
        raise NotImplementedError
