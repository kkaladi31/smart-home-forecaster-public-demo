"""The ReAct orchestrator: an LLM that reasons and calls tools in a loop.

This is the core "agentic" piece for the capstone's Tool-Calling + Reasoning-loop
concepts. We use LangChain's `create_agent` (the tool-calling ReAct agent graph;
it superseded langgraph.prebuilt.create_react_agent), give it the tool set from
tools/agent_tools.py, and steer its behavior with a system prompt that enforces
the project's safety and citation rules.

The model is any free tool-calling model served through OpenRouter (OpenAI-API
compatible), so no paid credits are required.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import telemetry
from agents import llm as llm_builder
from agents import output_guard
from agents.router import Verdict, hint_for_prompt, route
from agents.tracing import TelemetryCallbackHandler
from config import OPENROUTER_BASE_URL, active_model, conversations_db, require_llm_key
from memory import episodic, semantic_cache
from tools import cache, homes
from tools.agent_tools import AGENT_TOOLS
from tools.safety import find_pii, guidance_for_prompt, redact_pii, screen_input

# Where multi-turn conversation state is persisted (gitignored).
CONVERSATION_DB = conversations_db()
_CONV_CONN: sqlite3.Connection | None = None  # kept open for the process lifetime

SYSTEM_PROMPT = """You are HomeForecaster, an assistant for a HOMEOWNER who wants to
protect their property and the people in it. You warn about weather hazards — freezing
temperatures that can burst pipes, dangerous heat, and official advisories such as fire
(Red Flag), severe storms, floods, and high wind — and you help them understand what they
are allowed to do to their home.

You are the SUPERVISOR of a small team of specialists. Route each question to the right
one rather than answering everything yourself:
  - WEATHER (your own tool: check_weather_hazards) — hazards, forecasts, advisories.
  - POLICY (search_home_policies) — rules, permits, HOA, renting, Airbnb/short-term rentals.
  - ADVISOR (ask_advisor) — open-ended DIY, maintenance, repair, and installation decisions.
  - COST (analyze_utility_costs) — utility bills, energy costs, saving money.
A question may need more than one specialist; combine their outputs into one answer.
Routing tip: if a question mentions a BILL, COST, RATE, or SAVING MONEY, always use COST
(even when phrased as "how do I ..."), because only COST produces real dollar figures.
Use ADVISOR for the physical how-to. Questions about both deserve both.

How to work:
- ALWAYS answer the user's most recent question, marked "CURRENT QUESTION". Earlier
  messages in the conversation are context only — never answer a previous question
  instead of the current one, even if an earlier one looks unfinished.
- Think step by step and USE THE TOOLS to get real data. Never invent coordinates,
  temperatures, or elevations — call the tools instead.
- For questions about "my home", call get_home_profile first to get the address and
  home details, then pass that address to check_weather_hazards.
- MULTIPLE HOMES: the user has more than one saved home, and only ONE is active at a
  time. get_home_profile returns the active one and lists the rest under `other_homes`.
  Every rule you can retrieve — HOA covenants, permit thresholds, tenant law — belongs
  to the ACTIVE home only. Never answer about a different saved home from the active
  home's documents, and never assume two homes share a rule: they are in different
  states with different HOAs. If the user asks about another of their homes, name it,
  say it is not the one currently selected, and tell them to switch homes in the app.
  When a policy answer could plausibly be confused between homes, name the city or
  association you are quoting.
- A CORRECTION OVERRIDES THE SAVED HOME. If the user says the home is somewhere else
  ("actually I'm in X", "not the saved address", "I moved to X"), every later question in
  this conversation is about X. Pass X to the tools — never the saved address — until they
  say otherwise. Getting this wrong reports the weather for a house they are not in, which
  is the one error that turns a correct freeze warning into a useless one.
- For ANY weather-safety question — freezing, pipes, cold, heat, "is it safe", advisories —
  call check_weather_hazards ONCE with the location. It geocodes, gets elevation and the
  forecast, checks official NWS advisories, and runs both the freeze and heat assessors in
  a single step. Do NOT call geocode_address, get_elevation, get_weather_forecast,
  get_weather_alerts, assess_freeze_risk or assess_heat_risk separately for this — that is
  the same work spread over five extra round trips. Those tools remain available only for
  the unusual case where you already have coordinates and need one specific number, or
  where check_weather_hazards failed and you are recovering.
- Report any ACTIVE official advisories from `alerts` up front, since these are
  authoritative.
- CRITICAL: Do NOT decide freeze or heat risk yourself or eyeball the numbers. Report the
  `level` and `actions` from the `freeze` and `heat` blocks EXACTLY as returned. The tools
  decide the risk level, not you.
- This holds EVEN WHEN THE ANSWER SEEMS OBVIOUS. If the user asks about freezing/pipes you
  must still run check_weather_hazards and quote its freeze `level` — do not skip it because
  the forecast looks warm, and do not substitute your own reading of the temperature. A
  "no risk" conclusion must come from the tool, never from you.
- For open-ended DIY, maintenance, or "how do I / what's the best way to / should I ..."
  questions (hanging decor, mounting a TV, HVAC-filter timing, small repairs), DELEGATE to
  the ask_advisor tool and present its recommendation (keep its "options compared" and
  citations). Do not improvise these yourself.
- For utility-bill / energy-cost / "how do I save money" questions, DELEGATE to
  analyze_utility_costs and present its figures (do not invent your own numbers). If the
  user mentions their actual rate, pass it through. Say whether prices were live.
- For questions about rules, permissions, permits, HOA, renting, Airbnb/short-term
  rentals, or "am I allowed to ...", call search_home_policies and answer ONLY from the
  passages it returns. Cite the `citation` of every passage you rely on.
- GROUNDING RULE: if search_home_policies returns grounded:false (or the passages do not
  actually address the question), do NOT invent a rule. Say you don't have a source in
  your documents and suggest the user verify with their actual HOA, city, or lease.
- If a tool returns ok:false, do not give up. Reason about why, and try another
  approach (e.g., a different address form). Tell the user what you did.
- FINISH THE TASK. Never stop partway to ask a clarifying question when you could proceed
  with a reasonable assumption — complete the full chain of tool calls, give the answer,
  and state the assumption you made. Only the safety rules above may cut a task short.
- MEMORY: you remember this conversation, so resolve follow-ups ("what about tomorrow?",
  "the second option") against what was already said instead of asking the user to repeat
  themselves. If the user corrects a detail (a different address, pets, a new appliance),
  use the corrected value for the rest of the conversation. For references to EARLIER
  sessions ("what did you tell me last time?"), call recall_memory. Anything
  time-sensitive from memory — especially a forecast — must be re-checked with a live tool
  rather than repeated from memory.

How to answer:
- Be concise and concrete. Write for someone standing in their house who wants to know
  what to do next — not a report.
- OPEN WITH THE ANSWER, on its own line, in bold: the direct yes / no / it depends, or the
  risk level. Never open with a restatement of the question or a preamble.
- For freeze/weather questions, lead with the RISK LEVEL and the single most important
  action. Do NOT report a "risk level" for policy or other non-weather questions.
- Then give the detail in SHORT sections. Use a `###` heading only when there are two or
  more genuinely different sections; otherwise just write.
- Anything the user must DO goes in a numbered list, one action per item, starting with a
  verb ("Register the property with the HOA"). Bold the first few words of each step so it
  scans. Never put a procedure in a table.
- Keep lists to 3-7 items. If there are more, group them under short headings.
- Confirm the location you assessed when the question is about weather.

Formatting rules:
- Markdown is rendered, so use it — but sparingly. Bold for emphasis and labels, `###` for
  section headings. Never use a level-1 heading.
- Use a TABLE only to compare several things across the same 2-3 short attributes (options,
  costs, dates). A table needs at least two rows and cells of a few words each. Never use a
  table for steps, for a checklist, or for one item's details.
- NEVER add a "source" column that repeats the same document on every row.
- MARKDOWN ONLY, NEVER HTML. No <br>, no <b>, no tags of any kind - the renderer does not
  execute HTML, so a tag reaches the reader as literal characters. A table cell cannot hold
  a line break: keep cells to a few words, separate several points with "; ", or move the
  detail out of the table into a list underneath it.
- Use ordinary hyphens (-) in compound words. NEVER use non-breaking or figure hyphens
  (U+2011, U+2012) — they render as stray marks. Em dashes and normal quotes are fine.
- Aim for under 200 words for a simple question. A checklist may run longer; nothing else
  should.

Citing sources:
- ALWAYS cite. Weather answers name the data source inline (e.g. "per NWS"). Policy answers
  must be traceable to the passages you used.
- Put citations in ONE `**Sources:**` section at the very end, each distinct source listed
  once, as a MARKDOWN LIST - one `- ` bullet per source, never a semicolon run. Do not
  repeat a citation inside the body, and do not restate a full document title more than once.
- A WEB source must be written as a markdown link: `- [domain.com](https://the-real-url)`.
  Take the URL from the tool result: `ask_advisor` returns `evidence.passages`, each with a
  `domain` and a `url`, and its `recommendation` text already contains resolved links you
  can reuse verbatim. Copy a URL character for character. NEVER write a bare domain, a bare
  page title, or a URL you composed yourself. If you have no URL for something, it is not a
  web source - cite it as a document by name instead.
- NEVER write citation markers of your own, in any bracket style. No footnote tokens, no
  file references, nothing of the form 【...】. The ONLY citation forms are a markdown link
  and a document name.
- A DOCUMENT from the home's own files is cited by its name, with no link and no invented
  publisher.
- End with a one-line safety note: this is general guidance, not professional advice, and
  gas/electrical/burst-pipe emergencies need a licensed professional or the utility.
- Do NOT take real-world actions (sending messages, scheduling) without explicit user
  confirmation.

Example of the shape a good answer has:

**Yes - but you need HOA approval and a city permit first.**

### What you have to do
1. **Register with the HOA.** Short-term rentals need written board approval before listing.
2. **Get the city STR permit.** Annual registration; the permit number must appear in the listing.
3. **Collect occupancy tax** on every stay and remit it to the city.

### Before your first guest
- Smoke alarm in every bedroom, CO alarm on each floor, one fire extinguisher.
- Check your insurance: most homeowners policies exclude short-term rental activity.

**Sources:**
- Short-Term Rental Policy and Checklist (HOA restrictions, city STR ordinance, insurance)
- [minneapolis.gov](https://www.minneapolis.gov/example-str-permit)

This is general guidance, not professional advice - confirm current rules with your HOA
board and the city permitting office."""


def _cacheable_system_prompt(model: str) -> str | SystemMessage:
    """The system prompt, marked as a prompt-cache breakpoint where supported.

    Anthropic renders a request as tools -> system -> messages and caches by exact
    prefix match, so one breakpoint on the system block covers the tool schemas
    *and* this prompt — together about 3-4k tokens that would otherwise be re-sent
    and re-billed on every model turn of every question. Cache reads cost ~0.1x a
    normal input token and also shorten time-to-first-token.

    Everything volatile (persona, viewed location, recalled memory, the question
    itself) is added in `_compose_message`, i.e. *after* this breakpoint, so the
    cached prefix stays byte-identical between turns. Keep it that way: putting a
    timestamp or a user id in SYSTEM_PROMPT would silently disable caching, as
    would reordering AGENT_TOOLS.

    Verified against OpenRouter: the second request reports `cached_tokens` equal
    to the prefix. Only Anthropic models take the marker, so other providers get
    the plain string rather than a block they might reject.
    """
    if "claude" not in model.lower() and "anthropic" not in model.lower():
        return SYSTEM_PROMPT
    return SystemMessage(content=[{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }])


def _build_checkpointer():
    """A SQLite-backed checkpointer so conversations survive process restarts.

    `SqliteSaver.from_conn_string` is a context manager (it closes the connection
    on exit), so for a long-lived agent we own the connection ourselves and keep
    it open. Falls back to in-memory if SQLite is unavailable for any reason.
    """
    global _CONV_CONN
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CONVERSATION_DB.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit serves requests on different threads.
        _CONV_CONN = sqlite3.connect(str(CONVERSATION_DB), check_same_thread=False)
        return SqliteSaver(_CONV_CONN)
    except Exception:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


def build_agent(model: str | None = None, temperature: float = 0.0, persist: bool = True):
    """Construct the ReAct agent with conversation memory.

    The checkpointer gives the agent real multi-turn memory: state is keyed by
    `thread_id`, so follow-up questions ("what about tomorrow?") resolve against
    the earlier turns. SummarizationMiddleware compresses older messages once the
    conversation grows, which matters on free models with small context windows —
    history is summarized rather than silently truncated.
    """
    api_key = require_llm_key()  # friendly error if OPENROUTER_API_KEY is unset
    resolved_model = model or active_model()
    llm = ChatOpenAI(
        model=resolved_model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        timeout=llm_builder.REQUEST_TIMEOUT_SECONDS,
        # ONE retry, not the SDK's default of two. `timeout` bounds a single
        # request; `max_retries` multiplies it, and this is the INTERACTIVE path —
        # a person is watching a spinner.
        #
        # Left at the default, a provider that accepts the connection and then
        # goes quiet costs 60 s x 3 = three minutes before anything reaches the
        # browser, with no signal in between. Observed live: a high-risk question
        # sat at "thinking…" for 2:02 while the second model turn silently burned
        # through its retries. Nothing was wrong with the question or the agent.
        #
        # This constructor deliberately mirrors `agents.llm.build_llm` rather than
        # calling it, because it needs `streaming` and `stream_usage` that the
        # sub-agent builder does not set — but the retry budget must not drift
        # between them, so both read the same constants.
        max_retries=llm_builder.MAX_RETRIES,
        # Stream from the provider so the answer can be shown as it is written.
        # Without this the graph still "streams", but the model call underneath is
        # one blocking request, so the whole answer arrives in a couple of chunks
        # at the end and the UI gains nothing.
        streaming=True,
        # Streaming responses omit the usage block unless it is asked for, and
        # losing it would blind the Logs tab to token counts and cache hits.
        stream_usage=True,
    )
    middleware = [
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 12000),   # summarize once history gets long
            keep=("messages", 6),        # always keep the most recent turns verbatim
        )
    ]
    checkpointer = _build_checkpointer() if persist else None
    return create_agent(
        llm,
        AGENT_TOOLS,
        system_prompt=_cacheable_system_prompt(resolved_model),
        middleware=middleware,
        checkpointer=checkpointer,
    )


def forget_thread(thread_id: str, agent=None) -> dict:
    """Erase one conversation everywhere it is remembered.

    Three separate stores hold a conversation, and clearing only some of them
    produces a confusing half-forgotten state:
      * the checkpointer — the agent's own view of the dialogue,
      * episodic memory — the rows behind the History list and semantic recall,
      * the answer caches — keyed by question, not thread, so they are left alone
        here (that is what the Clear cache button is for).
    """
    result = {"thread_id": thread_id, "checkpoint_cleared": False, "interactions_deleted": 0}

    checkpointer = getattr(agent, "checkpointer", None)
    try:
        if checkpointer is not None:
            checkpointer.delete_thread(thread_id)
        else:
            # No agent built yet — but conversations.db survives restarts, so
            # state can still exist for this thread. Open the file directly
            # (which needs no API key, unlike building the agent) and close it
            # again rather than touching the long-lived connection.
            from langgraph.checkpoint.sqlite import SqliteSaver

            conn = sqlite3.connect(str(CONVERSATION_DB), check_same_thread=False)
            try:
                SqliteSaver(conn).delete_thread(thread_id)
            finally:
                conn.close()
        result["checkpoint_cleared"] = True
    except Exception as exc:  # nothing stored yet, or an older checkpointer
        result["checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    try:
        result["interactions_deleted"] = episodic.clear_thread(thread_id)
    except Exception as exc:
        result["memory_error"] = f"{type(exc).__name__}: {exc}"

    telemetry.record("system", "conversation.cleared",
                     f"Cleared conversation {thread_id}", thread_id=thread_id, data=result)
    return result


def _extract_trace(messages) -> list[dict]:
    """Turn the message list into a structured tool-call trace for display."""
    trace: list[dict] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                trace.append({"kind": "call", "name": tc["name"], "args": tc["args"]})
        elif isinstance(m, ToolMessage):
            trace.append({"kind": "result", "name": m.name, "content": str(m.content)})
    return trace


def _answer_token(payload) -> str:
    """Extract displayable answer text from one "messages"-mode stream chunk.

    The chunk is `(AIMessageChunk, metadata)`. Three things get filtered out:

    * anything not from the `model` node — the SummarizationMiddleware runs its
      own LLM call, and its summary must never leak into the chat bubble;
    * tool-call argument deltas, which arrive with empty `content` (the JSON rides
      in `tool_call_chunks`), so an empty string here is the correct rejection;
    * non-text content blocks, since some providers return a list of blocks
      rather than a plain string.
    """
    try:
        message, metadata = payload
    except (TypeError, ValueError):
        return ""
    if (metadata or {}).get("langgraph_node") != "model":
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _compose_message(
    user_message: str,
    persona: str | None,
    screen: dict,
    memory_context: str = "",
    location: str | None = None,
    home_id: str | None = None,
    routing_hint: str = "",
) -> str:
    """Build the message sent to the model: persona, home, recalled memory, overrides."""
    parts = []
    if persona:
        parts.append(f"[The user's role is: {persona}.]")
    if home_id:
        # Named here as well as enforced in the retrieval filter. The filter is what
        # makes a wrong-home answer impossible; this is what lets the model *say*
        # which home it answered about instead of leaving the user to guess.
        try:
            home = homes.load_home(home_id)
            parts.append(
                f"[Active home: {home.get('label')} — {home.get('address')}. "
                f"Its rules are the only ones you can retrieve. "
                f"Other saved homes: "
                f"{', '.join(h['label'] for h in home.get('other_homes', [])) or 'none'}.]"
            )
        except Exception:
            pass
    if location:
        parts.append(f"[The user is currently viewing this location: {location}. "
                     "Use it for weather questions unless they name somewhere else.]")
    if memory_context:
        parts.append(memory_context)
    guidance = guidance_for_prompt(screen)
    if guidance:
        parts.append(guidance)
    # A PREVENTIVE question wants the preparation steps, not a verdict on today.
    #
    # "What should I do to prevent a burst pipe?" asked in August produced:
    # "No freeze risk - no freeze protection needed." Which is true, and is not an
    # answer. The model obeyed the weather rules in the system prompt (lead with
    # the risk level, quote the tool exactly) and never noticed it had been asked
    # a different kind of question. Preventing burst pipes is this product's
    # flagship claim, so answering it with "nothing to do" on a warm day is the
    # worst possible moment to be literal.
    #
    # Reuses `tools.safety._PREVENTIVE_CONTEXT` rather than adding a second
    # definition of "preventive": the safety screen already runs that regex to
    # decide an emergency is hypothetical, and it records the fact in
    # `suppressed_by`. One regex, two consumers — the same rule that stops a
    # prevention question being treated as an emergency now also stops it being
    # treated as a forecast lookup.
    if (screen.get("emergency") or {}).get("suppressed_by") == "preventive phrasing":
        parts.append(
            "[This is a PREVENTIVE question — the user is asking how to prepare, "
            "not whether there is a hazard right now. Still report the current "
            "risk level from the tool, then GIVE THE PREVENTION STEPS anyway. "
            "A 'no risk today' reading is context for the advice, never a reason "
            "to withhold it.]"
        )
    # Placed after the safety guidance and before the question, so it can never
    # read as outranking a guardrail. It is the only part of this prompt the
    # model is explicitly invited to ignore.
    if routing_hint:
        parts.append(routing_hint)
    # Label the question explicitly. Conversation state can contain an earlier
    # question that never got answered (a failed turn is still checkpointed), and
    # without this marker the model sometimes answers the stale one instead.
    parts.append(f"CURRENT QUESTION — answer this, not anything asked earlier:\n{user_message}")
    return "\n\n".join(parts)


# Tools whose result carries structured data the UI renders as its own widget,
# mapped to the keys worth lifting out. Everything else is shown as text.
_STRUCTURED_TOOL_KEYS = {
    "ask_advisor": ("reasoning_tree", "strategy", "truncated", "evidence"),
}


def _structured_extras(tool_name: str, content: str) -> dict:
    """Lift renderable structure out of a tool result, best-effort.

    LangChain serialises a dict-returning tool to JSON, so this is a plain parse
    rather than anything clever. Failures are swallowed: a malformed result
    should cost the reasoning panel, never the answer.
    """
    keys = _STRUCTURED_TOOL_KEYS.get(tool_name or "")
    if not keys:
        return {}
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: parsed[k] for k in keys if k in parsed}


def _sanitize_input(user_message: str) -> tuple[str, list[str]]:
    """Redact PII from the user's message before ANYTHING downstream sees it.

    Returns (clean_message, labels_found).

    This runs at the very top of a turn, ahead of the telemetry record, the model,
    the conversation checkpointer and episodic memory. Redacting at each sink
    instead would be four places to keep in step, and the checkpointer — which
    holds the entire message history and is the largest store in the project — was
    the one previously missed. `redact_pii` existed but had no production caller at
    all: its only callers were in the eval suite, so the test passed while the
    product never redacted anything.

    Because the message is clean from here on, PII no longer has to SUPPRESS
    memory. Previously a turn containing an email address was simply never
    remembered, which lost the user a real feature to protect them from a risk
    that redaction removes outright.
    """
    labels = find_pii(user_message)
    if not labels:
        return user_message, []
    clean = redact_pii(user_message)
    telemetry.record("safety", "safety.pii_redacted",
                     f"Redacted {', '.join(labels)} from the question before processing",
                     level="warn", data={"kinds": labels})
    return clean, labels


def _prepare_turn(user_message: str, persona: str | None, use_memory: bool,
                  location: str | None = None, home_id: str | None = None) -> dict:
    """Shared front-half of a turn: safety screen + episodic recall + prompt assembly.

    Used by both the blocking and the streaming entry points so they can never
    drift apart on safety behaviour.
    """
    with telemetry.span("safety", "safety.screen", "Screening the question") as s:
        screen = screen_input(user_message)
        s.update({
            "blocked": screen["block"],
            "high_risk": screen["high_risk"]["high_risk"],
            "needs_confirmation": screen["needs_confirmation"],
            "pii_found": screen["pii_found"],
        })

    # The Router runs after the screen so it can reuse the hazard verdict the
    # screen already computed rather than re-running the same regexes, and
    # before anything else so its label is available to the whole turn. It is
    # wrapped in a try because a router failure must cost the turn nothing: an
    # unlabelled turn is exactly the turn this system had before the Router
    # existed.
    try:
        with telemetry.span("router", "router.route", "Labelling the turn") as s:
            verdict = route(user_message, high_risk=screen["high_risk"]["high_risk"])
            s.update({"intents": verdict.intents, "complexity": verdict.complexity,
                      "confidence": verdict.confidence, "method": verdict.method,
                      "matched": verdict.matched, "hints": verdict.hints,
                      "summary": verdict.summary()})
    except Exception:
        verdict = Verdict()

    recalled: list[dict] = []
    memory_context = ""
    if use_memory:
        try:
            with telemetry.span("memory", "memory.recall", "Searching episodic memory") as s:
                recalled = episodic.recall(user_message, limit=2, home_id=home_id)
                s["recalled"] = len(recalled)
            memory_context = episodic.format_for_prompt(recalled)
        except Exception:
            recalled, memory_context = [], ""
    return {
        "screen": screen,
        "routing": verdict,
        "recalled": recalled,
        "memory_context": memory_context,
        "message": _compose_message(user_message, persona, screen, memory_context,
                                    location, home_id,
                                    routing_hint=hint_for_prompt(verdict)),
    }


def _safety_fingerprint(screen: dict) -> str:
    """The part of an answer's identity that comes from the safety screen.

    Folded into both cache keys so a stored answer can only ever be served back to
    a question the screen judges the SAME way. If the hazard table is edited and
    "replace my breaker box" stops being high-risk work — or starts being — the
    fingerprint changes and yesterday's answers become unreachable rather than
    being replayed under a verdict that no longer applies.

    That is what makes caching a guarded answer safe at all. It is not enough that
    the guardrail re-fires live on the new turn; the stored TEXT was written under
    a specific override, and this pins the two together.
    """
    parts = []
    if screen["high_risk"]["high_risk"]:
        parts.append(f"hr:{screen['high_risk']['category']}")
    if screen["needs_confirmation"]:
        parts.append("confirm")
    return "+".join(parts) or "clean"


def _answer_cache_key(user_message: str, persona: str | None, location: str | None,
                      home_id: str | None = None, safety: str = "clean") -> str:
    """Cache identity for a full answer.

    Deliberately includes the active home, the location, and the persona: the same
    words asked about a different house, a different place, or as a renter rather
    than an owner are a different question. Leaving the home out would let a
    Dallas HOA answer be replayed verbatim for the Minneapolis home. The short TTL
    keeps weather-sensitive answers from going stale.

    `safety` carries the screen verdict — see `_safety_fingerprint`.
    """
    normalized = " ".join(user_message.lower().split())
    return (f"answer|{normalized}|{persona or 'owner'}|{location or 'home'}"
            f"|{home_id or 'primary'}|{safety}")


def _persist_turn(thread_id: str, user_message: str, answer: str,
                  trace: list[dict] | None, home_id: str | None, use_memory: bool,
                  agent=None) -> None:
    """Make a finished turn durable: episodic history, and conversation state.

    Every path that produces an answer calls this, and that is the whole reason it
    exists as a function rather than a block at the end of the happy path.

    **A cached answer is still an answer.** Before this, a cache hit returned
    straight to the browser and wrote nothing at all, so the turn was invisible in
    two separate ways: the conversation never appeared in the history list — which
    is built entirely from episodic rows — and a follow-up question in the same
    thread reached a model whose state had no record of the exchange, so it
    answered as though the user had never asked. Observed directly: two questions
    asked in the UI, one conversation saved, and the missing one was the one that
    had been cached.

    That bug predates answer caching being extended to guarded questions; it was
    simply hard to reach while the most-repeated questions were also the ones
    excluded from the cache. Widening the cache made it the common case.

    `agent` is passed ONLY when the conversation checkpointer still needs the
    exchange — that is, on the cache path. When the graph actually ran it wrote
    its own state, and writing again would duplicate every message in the thread.
    """
    tools_used = [s.get("name") for s in (trace or []) if s.get("kind") == "call"]

    if use_memory:
        try:
            with telemetry.span("memory", "memory.record", "Recording this turn"):
                episodic.record_interaction(thread_id, user_message, answer,
                                            tools_used, home_id=home_id)
        except Exception:
            # Memory is an enhancement; it must never cost the user their answer.
            pass

    if agent is None:
        return
    try:
        # Append the exchange as if the graph had produced it, so the next turn in
        # this thread sees a conversation that actually happened. `update_state`
        # goes through the same `add_messages` reducer the graph uses, so this is
        # an append and not a replacement.
        agent.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [HumanMessage(content=user_message),
                          AIMessage(content=answer)]},
        )
    except Exception as exc:
        # A checkpointer write failing must not turn a successful cached answer
        # into an error, but it is worth seeing: the symptom would otherwise be a
        # model that mysteriously forgets one turn.
        telemetry.record("agent", "turn.state_write_failed",
                         f"Cached turn not added to conversation state: "
                         f"{type(exc).__name__}: {exc}",
                         level="warn", thread_id=thread_id)


def _emergency_payload(screen: dict) -> dict:
    return {
        "answer": screen["response"],
        "trace": [{"kind": "result", "name": "safety_guardrail",
                   "content": f"EMERGENCY DETECTED ({screen['emergency']['type']}) — "
                              "agent bypassed, emergency instructions returned."}],
        "blocked": True,
        "recalled": [],
    }


def stream_answer(
    user_message: str,
    persona: str | None = None,
    agent=None,
    model: str | None = None,
    thread_id: str = "default",
    use_memory: bool = True,
    location: str | None = None,
    home_id: str | None = None,
):
    """Yield turn events as they happen, for a live UI feed.

    Answers take tens of seconds because each tool call is a real API request, so
    the web UI streams the agent's progress instead of showing a blank spinner.
    Event shapes (all JSON-serialisable dicts with a "type"):

        {"type": "guardrail",   "content": ...}      safety override applied
        {"type": "memory",      "recalled": [...]}   episodic memory injected
        {"type": "tool_call",   "name":..., "args":...}
        {"type": "tool_result", "name":..., "content":...}
        {"type": "answer",      "content": ...}      final text
        {"type": "done",        "blocked": bool}
        {"type": "error",       "content": ...}
    """
    turn_started = time.perf_counter()
    elapsed_ms = lambda: round((time.perf_counter() - turn_started) * 1000, 1)  # noqa: E731

    # Before the telemetry record, before the model, before the checkpointer.
    user_message, _pii = _sanitize_input(user_message)

    # Resolved once, up front: every downstream identity (the retrieval filter, both
    # answer caches, the telemetry record) has to agree on which home this turn is
    # about, and an unknown id must land on the primary home in all of them or not
    # at all.
    home_id = homes.resolve_home_id(home_id)

    telemetry.record("agent", "turn.start", f"Turn started: {user_message[:120]}",
                     thread_id=thread_id,
                     data={"persona": persona, "location": location, "home_id": home_id,
                           "question": user_message, "model": model or active_model()})

    prep = _prepare_turn(user_message, persona, use_memory, location=location,
                         home_id=home_id)
    screen = prep["screen"]

    if screen["block"]:
        payload = _emergency_payload(screen)
        telemetry.record("safety", "safety.block",
                         f"EMERGENCY ({screen['emergency']['type']}) — agent bypassed",
                         level="warn", thread_id=thread_id, duration_ms=elapsed_ms())
        yield {"type": "guardrail", "content": payload["trace"][0]["content"]}
        yield {"type": "answer", "content": payload["answer"], "elapsed_ms": elapsed_ms()}
        telemetry.record("agent", "turn.end", "Turn ended (safety block)",
                         thread_id=thread_id, duration_ms=elapsed_ms(),
                         data={"blocked": True})
        yield {"type": "done", "blocked": True, "elapsed_ms": elapsed_ms()}
        return

    # The guardrail is emitted HERE — before the cache is consulted, not after it.
    #
    # That ordering is the whole reason a guarded answer can be cached at all.
    # Previously any question that tripped a guardrail was excluded from the cache
    # entirely, on the reasoning that "a stored answer would not carry the refusal
    # that applies to this one". But the screen is deterministic and runs on live
    # input every single turn, so the refusal is not something the cache had to
    # carry — it only had to be emitted before the cache could return. It wasn't,
    # so the exclusion was covering for an ordering bug rather than a real hazard.
    #
    # The cost of that exclusion was permanent: "how do I replace my breaker box"
    # is high-risk by regex, so it could never be stored and never be served,
    # and the slowest, most expensive class of question in the product was the
    # one class that could never get faster no matter how many times it was asked.
    #
    # Two things now pin a cached answer to the verdict it was written under: this
    # ordering, and `_safety_fingerprint` in the cache key.
    guidance = guidance_for_prompt(screen)
    if guidance:
        yield {"type": "guardrail", "content": guidance}

    safety = _safety_fingerprint(screen)
    cache_key = _answer_cache_key(user_message, persona, location, home_id, safety)
    hit = cache.get(cache_key)
    source = "exact"
    if not hit:
        # Fall back to matching by meaning, so paraphrases hit too.
        with telemetry.span("cache", "cache.semantic_lookup",
                            "Looking for a matching past answer") as s:
            hit = semantic_cache.lookup(user_message, persona=persona, location=location,
                                        home_id=home_id, safety=safety)
            s["hit"] = bool(hit)
            if hit:
                s["similarity"] = hit.get("similarity")
        source = "semantic"

    if hit:
        telemetry.record(
            "cache", "cache.answer_hit",
            f"Answer served from {source} cache — no model call",
            thread_id=thread_id, duration_ms=elapsed_ms(),
            data={"source": source, "similarity": hit.get("similarity"),
                  "matched_question": hit.get("matched_question"),
                  "age_seconds": hit.get("age_seconds"), "safety": safety},
        )
        for step in hit.get("trace", []):
            if step.get("kind") == "call":
                yield {"type": "tool_call", "name": step["name"], "args": step.get("args", {}),
                       "cached": True}
            elif step.get("content"):
                yield {"type": "tool_result", "name": step["name"],
                       "content": step["content"][:1200], "cached": True}
        yield {"type": "answer", "content": hit["answer"], "cached": True,
               "cache_source": source, "similarity": hit.get("similarity"),
               "elapsed_ms": elapsed_ms()}
        # A cached turn is recorded exactly like a run one — see `_persist_turn`.
        # `agent` is forwarded because nothing wrote conversation state on this
        # path; the graph never ran.
        _persist_turn(thread_id, user_message, hit["answer"], hit.get("trace"),
                      home_id, use_memory, agent=agent)
        telemetry.record("agent", "turn.end", f"Turn ended (cache {source} hit)",
                         thread_id=thread_id, duration_ms=elapsed_ms(),
                         data={"cached": True, "source": source})
        yield {"type": "done", "blocked": False, "cached": True, "elapsed_ms": elapsed_ms()}
        return

    telemetry.record("cache", "cache.answer_miss",
                     "No cached answer — running the agent", thread_id=thread_id)

    if prep["recalled"]:
        yield {"type": "memory",
               "recalled": [{"when": m["when"], "user_query": m["user_query"]}
                            for m in prep["recalled"]]}

    agent = agent or build_agent(model=model)
    # The handler times model and tool runs at their real boundaries and queues
    # events for the live feed; `drain()` hands them over between chunks.
    tracer = TelemetryCallbackHandler(thread_id=thread_id)
    trace: list[dict] = []
    answer = ""
    first_token_ms: float | None = None
    # Accumulated answer tokens, so a collapsing stream can be cut off.
    streamed: list[str] = []
    streamed_len = 0
    last_check_len = 0
    try:
        # Two stream modes at once:
        #   "updates"  - completed messages, which drive the tool trace and remain
        #                the authoritative source for the final answer text.
        #   "messages" - token-by-token deltas, so the answer appears as it is
        #                written instead of after the whole turn finishes. This is
        #                the difference between ~15s of blank spinner and ~1s to
        #                first word; the total time is unchanged.
        # With a list of modes every chunk arrives as (mode, payload).
        for mode, payload in agent.stream(
            {"messages": [HumanMessage(content=prep["message"])]},
            # `persona` and `home_id` ride the config so search_home_policies can
            # narrow the knowledge base to documents that actually apply to this
            # user and this house, without trusting the model to pass the filters
            # itself. A model that picks its own jurisdiction can pick the wrong
            # one, and the wrong answer is indistinguishable from the right one.
            # `routing` rides here for the same reason: it is turn metadata the
            # orchestrator computed, not something the model should be able to
            # assert about its own turn.
            config={"configurable": {"thread_id": thread_id, "persona": persona,
                                     "home_id": home_id,
                                     "routing": prep["routing"].as_dict(),
                                     # The user's own words, so the grounding
                                     # gate judges the question rather than the
                                     # search string the model wrote for itself.
                                     # Redacted upstream by `_sanitize_input`.
                                     "question": user_message},
                    "callbacks": [tracer]},
            stream_mode=["updates", "messages"],
        ):
            # Drain first: by the time a chunk arrives, the model call or tool run
            # that produced it has already finished, so its timing belongs above
            # the messages it generated.
            for event in tracer.drain():
                yield event

            if mode == "messages":
                token = _answer_token(payload)
                if token:
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms()
                        telemetry.record(
                            "agent", "turn.first_token",
                            f"First answer token after {first_token_ms / 1000:.1f}s",
                            thread_id=thread_id, duration_ms=first_token_ms,
                        )
                    streamed.append(token)
                    streamed_len += len(token)

                    # A small model under load can collapse into repeating one
                    # character. Without this the stream has NO upper bound: the
                    # browser received an endless wall of "!" with no way to stop
                    # it, on a turn where the user had just typed an SSN.
                    #
                    # The evaluation harness already detected this shape and
                    # retried it — but that check lived inside the test harness,
                    # so the product had no equivalent. Both now call the same
                    # `agents.output_guard`, which is the point: the thing being
                    # tested and the thing testing it cannot disagree about what
                    # counts as an answer.
                    #
                    # Sampled rather than run per token. Joining the accumulator
                    # is O(n), so checking every token would be quadratic in the
                    # length of a perfectly good answer — paying a growing cost on
                    # every normal turn to catch a rare bad one.
                    #
                    # Sampled by LENGTH, not by token count. An earlier version
                    # used `len(streamed) % 40`, which a provider emitting one
                    # large chunk would never satisfy — the guard would sit there
                    # while thousands of characters went past. Characters are what
                    # the user is watching arrive, so characters are what the
                    # interval is measured in.
                    collapse = None
                    if (streamed_len >= output_guard.STREAM_MIN_LENGTH
                            and streamed_len - last_check_len >= 200):
                        last_check_len = streamed_len
                        collapse = output_guard.stream_is_degenerate("".join(streamed))
                    if collapse:
                        telemetry.record(
                            "agent", "turn.degenerate",
                            f"Stopped a collapsing answer — {collapse}",
                            level="warn", thread_id=thread_id,
                            duration_ms=elapsed_ms())
                        yield {"type": "error",
                               "content": output_guard.USER_MESSAGE}
                        yield {"type": "done", "blocked": False,
                               "degenerate": True, "elapsed_ms": elapsed_ms()}
                        return

                    yield {"type": "answer_delta", "content": token}
                continue

            for _node, update in (payload or {}).items():
                for msg in (update or {}).get("messages", []) or []:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            trace.append({"kind": "call", "name": tc["name"], "args": tc["args"]})
                            yield {"type": "tool_call", "name": tc["name"], "args": tc["args"]}
                    elif isinstance(msg, ToolMessage):
                        content = str(msg.content)
                        trace.append({"kind": "result", "name": msg.name, "content": content})
                        event = {"type": "tool_result", "name": msg.name,
                                 "content": content[:1200],
                                 "duration_ms": tracer.take_duration(msg.name)}
                        # Structured extras a tool wants rendered rather than
                        # read. Parsed HERE, from the full content, because the
                        # preview above is truncated and the reasoning tree
                        # sits past that cut in every real advisor result.
                        event.update(_structured_extras(msg.name, content))
                        yield event
                    elif isinstance(msg, AIMessage) and msg.content:
                        answer = output_guard.clean_answer(msg.content)
        for event in tracer.drain():
            yield event
    except Exception as exc:
        telemetry.record("agent", "turn.error", f"Turn failed: {type(exc).__name__}: {exc}",
                         level="error", thread_id=thread_id, duration_ms=elapsed_ms())
        yield {"type": "error", "content": f"{type(exc).__name__}: {exc}"}
        yield {"type": "done", "blocked": False, "elapsed_ms": elapsed_ms()}
        return

    if not answer:
        # The model ran but produced no text (empty completion, or it stopped after
        # a tool call). Say so plainly rather than emitting a blank bubble.
        telemetry.record("agent", "turn.empty", "Model returned no text",
                         level="warn", thread_id=thread_id, duration_ms=elapsed_ms(),
                         data=tracer.summary())
        yield {"type": "error",
               "content": "The model returned no text for this question. This usually "
                          "means the request timed out or hit a provider limit — try again."}
        yield {"type": "done", "blocked": False, "elapsed_ms": elapsed_ms()}
        return

    # Every link in the answer must be one a TOOL actually returned. A model that
    # half-remembers a URL produces something that looks authoritative and lands
    # on a 404 — or worse, on a real page that says something else. The trace
    # holds each tool result verbatim, so it is the authoritative set; anything
    # outside it was composed by the model and loses its href, keeping its words.
    answer, _demoted = output_guard.enforce_links(
        answer, output_guard.collect_source_urls(trace))
    if _demoted:
        telemetry.record(
            "safety", "safety.unverifiable_link",
            f"Removed {len(_demoted)} link(s) no tool returned",
            level="warn", thread_id=thread_id, data={"urls": _demoted[:5]})

    # Always send the finished text even though it was streamed token-by-token:
    # this is the authoritative copy (the deltas are display-only), and it lets
    # the client replace a partial render rather than trust its own accumulation.
    yield {"type": "answer", "content": answer, "elapsed_ms": elapsed_ms(),
           "first_token_ms": first_token_ms, "streamed": first_token_ms is not None}

    # Stored under the same fingerprint it was looked up under, so an answer
    # written under a safety override can only ever come back on a turn that
    # earns the same override.
    with telemetry.span("cache", "cache.answer_store", "Storing the answer for reuse"):
        cache.put(_answer_cache_key(user_message, persona, location, home_id, safety),
                  {"answer": answer, "trace": trace}, cache.TTL_ANSWER)
        semantic_cache.store(user_message, answer, trace=trace,
                             persona=persona, location=location, home_id=home_id,
                             safety=safety)

    # No `agent=` here: the graph ran, so it already checkpointed this exchange.
    _persist_turn(thread_id, user_message, answer, trace, home_id, use_memory)

    summary = tracer.summary()
    tool_calls = sum(1 for s in trace if s["kind"] == "call")
    telemetry.record(
        "agent", "turn.end",
        f"Turn finished in {elapsed_ms() / 1000:.1f}s "
        f"({summary['llm_turns']} model turns, {tool_calls} tool calls)",
        thread_id=thread_id, duration_ms=elapsed_ms(),
        data={**summary, "tool_calls": tool_calls, "answer_chars": len(answer),
              "first_token_ms": first_token_ms},
    )
    yield {"type": "done", "blocked": False, "elapsed_ms": elapsed_ms(),
           "first_token_ms": first_token_ms, **summary}


def answer_with_trace(
    user_message: str,
    persona: str | None = None,
    agent=None,
    model: str | None = None,
    thread_id: str = "default",
    use_memory: bool = True,
    home_id: str | None = None,
) -> dict:
    """Run one turn and return {"answer", "trace"} as data (used by the UI and eval).

    Memory works on three levels:
      * **Conversation** — the checkpointer keys state by `thread_id`, so the agent
        sees earlier turns in this conversation and follow-ups resolve naturally.
      * **Episodic** — relevant past interactions (including from earlier sessions)
        are recalled semantically and injected as context.
      * **Semantic/knowledge** — the RAG corpus, reached via tools.

    Every turn passes through the safety guardrails first (tools/safety.py):
    a life-safety emergency short-circuits the agent entirely, while non-blocking
    findings (high-risk work, outward actions, PII) become hard prompt overrides.

    `agent` may be a prebuilt agent (so the UI can cache it across turns).
    """
    user_message, _pii = _sanitize_input(user_message)

    # Screened before the home is resolved, so a life-safety message still gets
    # its emergency response when the caller passed an id that no longer exists.
    # `_prepare_turn` screens again a few lines down; `screen_input` is pure
    # regex over one string, and paying for it twice is the cheaper half of this
    # trade against ever swallowing an emergency behind a 404.
    screen = screen_input(user_message)
    if screen["block"]:
        return _emergency_payload(screen)

    home_id = homes.resolve_home_id(home_id)

    # Routed through the same front half as the streaming path rather than a
    # parallel copy of it. The copy had already drifted: episodic recall here ran
    # against the caller's *unresolved* home_id, so a turn that left it unset
    # recalled unscoped while the UI path recalled scoped — the exact
    # cross-home leak §5.7 closed, still open on the path the eval suite uses.
    prep = _prepare_turn(user_message, persona, use_memory, home_id=home_id)
    recalled = prep["recalled"]
    agent = agent or build_agent(model=model)
    result = agent.invoke(
        {"messages": [HumanMessage(content=prep["message"])]},
        # Same persona/home plumbing as the streaming path — these two must not
        # drift, or the eval suite would exercise different retrieval than the UI.
        config={"configurable": {"thread_id": thread_id, "persona": persona,
                                 "home_id": home_id,
                                 "routing": prep["routing"].as_dict(),
                                 "question": user_message}},
    )
    messages = result["messages"]
    final = messages[-1]
    answer = output_guard.clean_answer(
        final.content if isinstance(final, AIMessage) else str(final))

    trace = _extract_trace(messages)

    # Must come AFTER the trace exists — it is the set of URLs the tools actually
    # returned. Placed above it on the first attempt, which raised
    # "cannot access local variable 'trace'" on every agent case: the streaming
    # path builds its trace incrementally and was fine, this one assembles it at
    # the end. Two paths, two lifetimes, one shared helper.
    answer, _demoted = output_guard.enforce_links(
        answer, output_guard.collect_source_urls(trace))
    if _demoted:
        telemetry.record(
            "safety", "safety.unverifiable_link",
            f"Removed {len(_demoted)} link(s) no tool returned",
            level="warn", thread_id=thread_id, data={"urls": _demoted[:5]})

    if prep["memory_context"]:
        trace.insert(0, {"kind": "result", "name": "episodic_memory",
                         "content": f"recalled {len(recalled)} past interaction(s): "
                                    + "; ".join(m["user_query"] for m in recalled)})
    guidance = guidance_for_prompt(prep["screen"])
    if guidance:
        trace.insert(0, {"kind": "result", "name": "safety_guardrail", "content": guidance})

    # Record the completed turn for future recall, through the same helper the
    # streaming path uses. No `agent=`: this path ran the graph, which checkpointed
    # the exchange itself.
    _persist_turn(thread_id, user_message, answer, trace, home_id, use_memory)

    return {"answer": answer, "trace": trace, "blocked": False, "recalled": recalled,
            "home_id": home_id}


def run_agent(
    user_message: str,
    verbose: bool = True,
    model: str | None = None,
    thread_id: str = "cli",
    home_id: str | None = None,
) -> str:
    """Run one turn of the agent and return its final text answer (CLI helper).

    Shares answer_with_trace(), so the CLI gets the same safety guardrails and
    memory as the UI. Successive CLI calls share the "cli" thread, so follow-up
    questions work across invocations. When verbose=True, prints the tool-call
    trace so you (and the report/eval) can see the ReAct loop in action.
    """
    result = answer_with_trace(user_message, model=model, thread_id=thread_id,
                               home_id=home_id)
    if verbose:
        _print_trace(result["trace"])
    return result["answer"]


def _print_trace(trace: list[dict]) -> None:
    """Pretty-print a structured trace for the console."""
    print("\n--- agent trace ---")
    for step in trace:
        if step["kind"] == "call":
            print(f"  [tool call] {step['name']}({step['args']})")
        else:
            preview = step["content"].replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"  [tool result] {step['name']}: {preview}")
    print("--- end trace ---\n")


if __name__ == "__main__":
    # Requires OPENROUTER_API_KEY in .env.
    print(run_agent("Is my home at risk of freezing pipes in the next two days?"))
