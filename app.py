"""Streamlit demo UI for the Smart-Home Forecaster agent (Phase 2).

Run it from the project root:
    streamlit run app.py

**The React app in `web/` is the primary interface** — it is what the capstone
video shows, and the only one with live token streaming, the dashboard, and the
Logs tab. This Streamlit UI is kept as the simpler fallback and as a record of
what the project looked like before that front-end existed.

It shows a chat with the agent, a sidebar with the home profile and an
Owner/Renter toggle, and an expandable "Agent trace" panel revealing every tool
call — the visible proof of tool-calling and the ReAct reasoning loop.

Deliberately still blocking: this calls `answer_with_trace()`, the single-result
entry point also used by `main.py` and the evaluation suite, rather than the
streaming `stream_answer()` the React app consumes over SSE. Keeping a real UI on
that path means the eval suite exercises code a human also runs, instead of
becoming its only consumer. The Owner/Renter toggle is passed through, so this UI
does get the same persona-filtered retrieval as the React app.
"""
from __future__ import annotations

import uuid

import streamlit as st

import config
from tools.homes import list_homes, load_home

st.set_page_config(page_title="Smart-Home Forecaster", page_icon="🏠", layout="wide")


# --- cached resources ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_agent():
    """Build the LangGraph agent once and reuse it across reruns."""
    from agents.orchestrator import build_agent

    return build_agent()


@st.cache_data(show_spinner=False)
def _load_home_profile(home_id: str) -> dict:
    return load_home(home_id)


@st.cache_data(show_spinner=False)
def _home_options() -> list[dict]:
    return list_homes()


def _kb_count() -> int:
    from memory.rag_store import count

    return count()


EXAMPLES = [
    "Are my pipes at risk of freezing in the next two days?",
    "Am I allowed to replace my front lawn with gravel?",
    "I want to list my house on Airbnb — am I allowed to, and what do I need to do?",
    "Do I need a permit to replace my water heater?",
]


# --- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("🏠 Smart-Home Forecaster")
    st.caption("Forecaster capstone · proactive home-risk agent")

    persona = st.radio("I am the...", ["owner", "renter"], index=0,
                       format_func=str.capitalize, horizontal=True)

    options = _home_options()
    home_id = st.selectbox(
        "Home", [h["home_id"] for h in options],
        format_func=lambda hid: next(
            f"{h['label']}{' (primary)' if h['is_primary'] else ''}"
            for h in options if h["home_id"] == hid
        ),
        help="Each home has its own HOA, city rules, and contractors. Switching "
             "here switches which documents the agent can read.",
    )

    profile = _load_home_profile(home_id)
    with st.expander("Home profile", expanded=False):
        st.write(f"**Address:** {profile['address']}")
        st.write(f"**Type:** {profile['dwelling_type'].replace('_', ' ')} · built {profile['year_built']}")
        st.write(f"**Climate zone:** {profile['climate_zone']}")
        st.write(f"**HVAC filter:** {profile['systems']['hvac']['filter_size']} "
                 f"(every {profile['systems']['hvac']['filter_interval_days']} days)")
        st.write(f"**HOA:** {profile['hoa']['association_name']}")

    st.divider()
    kb = _kb_count()
    if kb:
        st.success(f"Knowledge base: {kb} passages")
    else:
        st.warning("Knowledge base is empty.")
    if st.button("Rebuild knowledge base", use_container_width=True):
        from memory.rag_store import ingest

        with st.spinner("Ingesting corpus..."):
            stats = ingest(verbose=False)
        st.success(f"Ingested {stats['chunks']} passages.")
        st.rerun()

    st.divider()
    st.subheader("Memory")
    try:
        from memory.episodic import count as _episodic_count

        st.caption(f"🧠 {_episodic_count()} past interactions remembered")
    except Exception:
        st.caption("🧠 episodic memory unavailable")
    st.caption(f"Conversation: `{st.session_state.get('thread_id', '—')}`")
    if st.button("Start new conversation", use_container_width=True,
                 help="Clears this chat and starts a fresh conversation thread. "
                      "Long-term memory of past interactions is kept."):
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
        st.rerun()

    st.divider()
    st.caption(f"Model: `{config.active_model()}`"
               + ("  ·  demo mode (free/open services)" if config.demo_mode() else ""))
    if config.OPENROUTER_API_KEY:
        st.caption("OpenRouter key: ✅ loaded")
    else:
        st.caption("OpenRouter key: ❌ missing")


# --- main pane ----------------------------------------------------------------
st.title("Smart-Home Forecaster")
st.caption("Ask about weather risks to your home, or what you're allowed to do to it. "
           "Every answer is grounded in live data or cited documents.")

if not config.OPENROUTER_API_KEY:
    st.error(
        "No OpenRouter API key found. Copy `.env.example` to `.env`, add your free key "
        "from https://openrouter.ai/keys, then restart the app. "
        "(The `python main.py --demo` freeze check still works without a key.)"
    )
    st.stop()

if _kb_count() == 0:
    st.info("The policy knowledge base isn't built yet — click **Rebuild knowledge base** "
            "in the sidebar to enable the 'Am I allowed to…?' answers.")

# chat history lives in session state (for rendering). The agent's own memory is
# keyed by thread_id — that is what actually gives it multi-turn continuity.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"


def _render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander("🔧 Agent trace (tool calls)", expanded=False):
        for step in trace:
            if step["kind"] == "call":
                st.markdown(f"**→ called `{step['name']}`**")
                st.code(json.dumps(step["args"], indent=2), language="json")
            else:
                content = step["content"]
                preview = content if len(content) < 800 else content[:800] + " …(truncated)"
                st.markdown(f"**← `{step['name']}` returned**")
                st.code(preview, language="json")


# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_trace(msg.get("trace", []))

# example prompts (only before the first question, to keep things tidy)
prompt = None
if not st.session_state.messages:
    st.markdown("**Try an example:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            prompt = ex

# chat input always available
typed = st.chat_input("Ask about freeze risk, HOA rules, permits, Airbnb…")
if typed:
    prompt = typed

# handle a new question
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (calling tools)…"):
            from agents.orchestrator import answer_with_trace

            try:
                result = answer_with_trace(
                    prompt, persona=persona, agent=_get_agent(),
                    thread_id=st.session_state.thread_id, home_id=home_id,
                )
                answer, trace = result["answer"], result["trace"]
            except Exception as exc:  # keep the UI alive on model/network hiccups
                answer, trace = f"⚠️ Something went wrong: {exc}", []
        st.markdown(answer)
        _render_trace(trace)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
