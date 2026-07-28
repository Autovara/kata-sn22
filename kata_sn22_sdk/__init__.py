"""The SN22 agent SDK: everything a submission is allowed to see, and nothing else.

```python
from kata_sn22_sdk import Agent, AiSearchResult, ScraperTextRole, XSearchResult


class Submission(Agent):
    async def smart_scraper(self, synapse, emit):
        results = self.broker.web_search(synapse.prompt, count=synapse.count)
        emit(ScraperTextRole.FINAL_SUMMARY, "...")
        return AiSearchResult(search_results=results)

    async def twitter_search(self, synapse):
        return XSearchResult(results=self.broker.x_search(synapse.query, count=synapse.count))
```

**What is deliberately absent.** No provider key and no way to ask for one; no scorer, no judge and
no scoring prompt; no wallet, chain client or deployment secret; no package installer. A submission
runs on the standard library plus this package, and the image ships nothing else to reach for.

**What the image ships.** Python, this package, and a harness that imports ``/bundle/agent.py``. The
bundle is mounted read-only, so anything a submission needs at run time has to be here at build time
-- which is why the surface is small and reviewed rather than convenient.
"""

from kata_sn22_sdk.agent import Agent, Emit
from kata_sn22_sdk.broker import AGENT_OPERATIONS, BrokerClient, BrokerError, in_sealed_room
from kata_sn22_sdk.models import (
    PROTOCOL_VERSION,
    AiSearchResult,
    AiSearchSynapse,
    Limits,
    ResultType,
    ScraperTextRole,
    SdkError,
    SearchMode,
    SearchType,
    Synapse,
    XSearchResult,
    XSearchSynapse,
    synapse_from_input,
)

__all__ = [
    "AGENT_OPERATIONS",
    "PROTOCOL_VERSION",
    "Agent",
    "AiSearchResult",
    "AiSearchSynapse",
    "BrokerClient",
    "BrokerError",
    "Emit",
    "Limits",
    "ResultType",
    "ScraperTextRole",
    "SdkError",
    "SearchMode",
    "SearchType",
    "Synapse",
    "XSearchResult",
    "XSearchSynapse",
    "in_sealed_room",
    "synapse_from_input",
]
