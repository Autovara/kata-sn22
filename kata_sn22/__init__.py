"""The SN22 (Desearch) subnet package.

Two layers live here, and importing one must not drag in the other:

* **The evaluation protocol** (``protocol``, ``manifests``, ``scoring``, ``providers``,
  ``fixtures``) — SN22-2. Self-contained and independent of the Kata core, so the contract
  can be reviewed and calibrated before any lane exists to run it.
* **The plugin** (``plugin``) — the adapter the core resolves by evaluator id, rewritten in SN22-3
  against the current ``kata.plugins`` ABI on top of that protocol.

The plugin is still imported LAZILY, so the protocol layer stays importable on a host that has no
Kata core installed at all — which is what makes it reviewable and calibratable on its own.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "SN22_DESEARCH_PLUGIN",
    "Sn22DesearchPlugin",
    "Sn22Problems",
    "Sn22RawRun",
]


def __getattr__(name: str) -> Any:
    """Resolve plugin symbols on first use, so the protocol layer imports on any core."""
    if name in __all__:
        from kata_sn22 import plugin as _plugin

        if name == "SN22_DESEARCH_PLUGIN":
            # The entry point the engine loads. It MUST arrive with its verification transports
            # wired: scoring fetches pages, asks a judge and re-scrapes tweets, and a plugin built
            # without them raises on the first attempt it scores. It did, until this was added --
            # every production round would have failed at scoring while the whole test suite passed,
            # because the tests construct the plugin themselves and pass cassettes.
            import os

            mode = os.environ.get("KATA_SN22_VERIFICATION_MODE", "live").strip().lower()
            if mode == "live":
                from kata_sn22.providers import transports_from_env

                transports = transports_from_env()
                singleton = _plugin.Sn22DesearchPlugin(
                    page_transport=transports.page_transport,
                    judge_client=transports.judge_client,
                    tweet_scraper=transports.tweet_scraper,
                )
            elif mode == "recorded":
                from kata_sn22 import fixtures
                from kata_sn22.fetch import RecordedPages

                tweets = fixtures.recorded_tweets()
                singleton = _plugin.Sn22DesearchPlugin(
                    page_transport=RecordedPages(records=fixtures.recorded_pages()),
                    judge_client=fixtures.scripted_judge(),
                    tweet_scraper=lambda ids: {
                        tweet_id: tweets[tweet_id] for tweet_id in ids if tweet_id in tweets
                    },
                    search_provider=fixtures.search_provider(),
                )
            else:
                raise RuntimeError(
                    "KATA_SN22_VERIFICATION_MODE must be 'live' or 'recorded'; "
                    f"got {mode!r}")
            globals()["SN22_DESEARCH_PLUGIN"] = singleton
            return singleton
        return getattr(_plugin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
