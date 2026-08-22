"""Shared LLM builder so multiple agents construct the model the same way.

Kept in its own module to avoid a circular import (orchestrator and advisor both
need a model, and agent_tools imports the advisor).
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import OPENROUTER_BASE_URL, active_model, require_llm_key


# One request may take a minute on a free model under load. Measured: a propose
# call averages ~23 s and a critique ~22 s, with the tail well past that.
REQUEST_TIMEOUT_SECONDS = 60

# ONE retry, not the SDK's default of two — and this is the setting that matters.
#
# `timeout` bounds a single HTTP request; `max_retries` multiplies it. Left at
# the default, a provider that accepts a connection and then goes quiet costs
# 60 s x 3 = THREE MINUTES before anything reaches the user, with no signal in
# between. That was observed live: a high-risk question sat at "thinking…" for
# 2:02 while the second model turn silently burned through its retries.
#
# One retry still absorbs a transient blip, which is the case retries exist for,
# and caps the worst case at about two minutes. Dropping to zero would make a
# single dropped packet fail a turn the user would rather have waited for.
#
# The evaluation suite does NOT depend on this: it retries at the case level with
# its own backoff, so a tighter budget here costs it nothing.
MAX_RETRIES = 1


def build_llm(model: str | None = None, temperature: float = 0.0,
              trace: bool = True, timeout: int = REQUEST_TIMEOUT_SECONDS,
              max_retries: int = MAX_RETRIES) -> ChatOpenAI:
    """Return an OpenRouter-backed chat model (raises a clear error if no key).

    Resolves the model through `active_model()` rather than a module-level
    constant so a demo/full switch takes effect on the next build, not the next
    process restart.

    Sub-agent calls are traced by default. Without this the Logs tab showed only
    the orchestrator's own turns, because the tracer was attached in
    `stream_answer` and nowhere else — so the Advisor's beam search, which is
    most of the wall-clock time on a DIY question, was invisible. "Where did the
    30 seconds go" is unanswerable if the expensive part does not log, and the
    beam is now four calls rather than one.

    Pass `trace=False` for throwaway calls (spikes, one-off scripts) that should
    not clutter the event log.
    """
    api_key = require_llm_key()
    callbacks = []
    if trace:
        # Imported here, not at module scope: agents.tracing pulls in LangChain's
        # callback machinery, and this module exists specifically to stay light
        # enough to break the orchestrator/advisor import cycle.
        try:
            from agents.tracing import TelemetryCallbackHandler

            callbacks.append(TelemetryCallbackHandler())
        except Exception:
            pass  # tracing is observability; it must never stop a model building
    return ChatOpenAI(
        model=model or active_model(),
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        callbacks=callbacks or None,
    )
