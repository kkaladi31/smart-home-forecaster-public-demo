import { useCallback, useEffect, useRef, useState } from 'react'
import {
  clearAllMemory, clearCaches, clearChatThread, getAdminStats,
  getConversation, getConversations, streamChat,
} from '../api'
import { log } from '../logbus'
import { fmtDuration } from '../utils/duration'
import Markdown from './Markdown'
import ReasoningTree from './ReasoningTree'

const EXAMPLES = [
  'Are my pipes at risk of freezing in the next two days?',
  'Am I allowed to replace my front lawn with gravel?',
  'I want to list my house on Airbnb — what do I need to do?',
  'How can I lower my utility bills?',
]

// Friendly names for the live feed — raw tool names are accurate but noisy.
const TOOL_LABELS = {
  get_home_profile: 'Reading your home profile',
  geocode_address: 'Locating the address',
  get_elevation: 'Checking elevation',
  get_weather_forecast: 'Fetching the forecast',
  get_weather_alerts: 'Checking official advisories',
  assess_freeze_risk: 'Assessing freeze risk',
  assess_heat_risk: 'Assessing heat risk',
  search_home_policies: 'Searching your home documents',
  ask_advisor: 'Consulting the DIY advisor',
  analyze_utility_costs: 'Analyzing utility costs',
  recall_memory: 'Recalling past conversations',
  find_licensed_pros: 'Checking the contractor registry',
  safety_guardrail: 'Safety guardrail',
  episodic_memory: 'Recalled memory',
}

export default function Chat({
  persona, threadId, homeId, location, prefill, onPrefillConsumed,
  onNewThread, onResumeThread, onBusyChange,
}) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [live, setLive] = useState(null)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [history, setHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [showMenu, setShowMenu] = useState(false)
  const abortRef = useRef(null)
  const endRef = useRef(null)
  const inputRef = useRef(null)

  const resumingRef = useRef(false)
  const startedRef = useRef(0)

  // A hazard notification can hand a question over. It is placed in the box and
  // FOCUSED, never sent: the notification decided something is worth asking
  // about, but only the person decides to ask. Auto-sending would also fire a
  // model call from a component whose whole point is that it costs nothing.
  //
  // Refused while a turn is in flight — overwriting the box mid-answer would
  // destroy whatever the user was typing next.
  useEffect(() => {
    if (!prefill || busy) return
    setInput(prefill)
    inputRef.current?.focus()
    onPrefillConsumed?.()
  }, [prefill, busy, onPrefillConsumed])

  // The running timer. Answers take tens of seconds, and a spinner with no
  // number reads as "stuck"; a moving clock reads as "working". 100ms keeps the
  // tenths place honest without repainting more than the eye can use.
  useEffect(() => {
    onBusyChange?.(busy)
    if (!busy) return undefined
    const id = setInterval(() => setElapsed(performance.now() - startedRef.current), 100)
    return () => clearInterval(id)
  }, [busy, onBusyChange])

  // A new thread means a new conversation — clear the transcript. Without this
  // the "New conversation" button changed the thread id but left the old
  // messages on screen, so it looked like nothing happened.
  // Resuming a past conversation also changes threadId, so it sets a flag first
  // to stop this effect from wiping the transcript it just loaded.
  useEffect(() => {
    if (resumingRef.current) {
      resumingRef.current = false
      return
    }
    setMessages([])
    setLive(null)
    abortRef.current?.()
    setBusy(false)
  }, [threadId])

  // Only scroll when there is something to scroll to. Without the guard,
  // clearing the transcript for a new conversation still fired a scroll, which
  // yanked the whole page down.
  useEffect(() => {
    if (messages.length === 0 && !live) return
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, live])

  const refreshHistory = useCallback(() => {
    getConversations()
      .then((d) => setHistory(d.conversations ?? []))
      .catch(() => {})
  }, [])

  useEffect(() => { refreshHistory() }, [refreshHistory])

  async function resume(threadIdToLoad) {
    setShowHistory(false)
    log('ui', 'chat.resume', `Resumed conversation ${threadIdToLoad}`)
    try {
      const { messages: past } = await getConversation(threadIdToLoad)
      // Point the agent at the same thread so its own memory lines up with what
      // the user can see, then replay the transcript.
      resumingRef.current = true
      onResumeThread?.(threadIdToLoad)
      setMessages(past ?? [])
    } catch {
      /* ignore */
    }
  }

  function send(text) {
    const message = (text ?? input).trim()
    if (!message || busy) return

    log('ui', 'chat.send', `Sent: ${message.slice(0, 80)}`, { data: { chars: message.length } })
    setMessages((m) => [...m, { role: 'user', content: message }])
    setInput('')
    setBusy(true)
    startedRef.current = performance.now()
    setElapsed(0)
    setLive({ steps: [], answer: '' })

    // Plain closure variables: `answer` and `done` can arrive in the same tick,
    // and React state updates are not guaranteed to flush between them.
    const steps = []
    let answerText = ''
    let errorText = ''
    let meta = {}
    // Answer tokens can arrive in a tight burst, and every repaint re-parses the
    // whole markdown string. Repainting on each one is wasted work the eye cannot
    // use, so deltas are coalesced to ~15fps. Every other event still paints
    // immediately, and the authoritative 'answer' event always flushes, so a
    // skipped frame can never lose text.
    const PAINT_INTERVAL_MS = 66
    let lastPaint = 0

    abortRef.current = streamChat({ message, threadId, persona, location, homeId }, (ev) => {
      if (ev.type === 'tool_call') {
        steps.push({ kind: 'call', name: ev.name, args: ev.args, cached: ev.cached })
        // Any text streamed before a tool call was the model narrating what it was
        // about to do, not the answer. Drop it so the bubble ends up holding the
        // final answer alone rather than a preamble stacked on top of it.
        answerText = ''
      } else if (ev.type === 'answer_delta') {
        // The answer as it is being written. Shown immediately; the authoritative
        // copy still arrives in the 'answer' event below and replaces this.
        answerText += ev.content
      } else if (ev.type === 'tool_result') {
        const pending = [...steps].reverse().find((s) => s.kind === 'call' && s.name === ev.name)
        if (pending) {
          pending.done = true
          pending.durationMs = ev.duration_ms
          // Structured extras the server lifted out of the full result (the
          // `content` above is only a preview). Today: the Advisor's search tree.
          if (ev.reasoning_tree) {
            pending.tree = ev.reasoning_tree
            pending.strategy = ev.strategy
            pending.truncated = ev.truncated
          }
        }
      } else if (ev.type === 'llm_turn') {
        // The model's own thinking time — usually the largest single slice.
        steps.push({
          kind: 'llm', name: 'llm', done: true, durationMs: ev.duration_ms,
          detail: ev.output_tokens ? `${ev.output_tokens} tokens out` : undefined,
          turn: ev.turn,
        })
      } else if (ev.type === 'guardrail' || ev.type === 'memory') {
        steps.push({
          kind: ev.type,
          name: ev.type === 'guardrail' ? 'safety_guardrail' : 'episodic_memory',
          detail: ev.content ?? (ev.recalled ?? []).map((r) => r.user_query).join('; '),
          done: true,
        })
      } else if (ev.type === 'answer') {
        answerText = ev.content
        meta = {
          cached: ev.cached,
          cacheSource: ev.cache_source,
          similarity: ev.similarity,
          firstTokenMs: ev.first_token_ms,
        }
      } else if (ev.type === 'error') {
        // Keep the reason visible instead of burying it in a collapsed trace.
        errorText = ev.content
        steps.push({ kind: 'error', detail: ev.content, done: true })
      } else if (ev.type === 'done') {
        // Trust the server's own measurement when it sent one; it excludes the
        // browser's render lag and is what the logs will agree with.
        const durationMs = ev.elapsed_ms ?? performance.now() - startedRef.current
        setMessages((m) => [
          ...m,
          {
            role: 'assistant',
            content: answerText || '',
            error: answerText ? '' : (errorText || 'No answer was returned.'),
            steps: [...steps],
            durationMs,
            llmTurns: ev.llm_turns,
            ...meta,
          },
        ])
        setLive(null)
        setBusy(false)
        refreshHistory()
        return
      }

      if (ev.type === 'answer_delta') {
        const now = performance.now()
        if (now - lastPaint < PAINT_INTERVAL_MS) return
        lastPaint = now
      }
      setLive({ steps: [...steps], answer: answerText })
    })
  }

  function stop() {
    log('ui', 'chat.stop', 'Stopped the answer', { level: 'warn' })
    abortRef.current?.()
    setBusy(false)
    setLive(null)
  }

  async function handleClearChat() {
    abortRef.current?.()
    setBusy(false)
    setLive(null)
    setMessages([])
    try {
      const result = await clearChatThread(threadId)
      log('ui', 'chat.cleared', 'Cleared this conversation', { data: result })
    } catch (e) {
      log('ui', 'chat.clear_failed', e.message, { level: 'error' })
    }
    onNewThread?.()      // start a fresh thread so nothing carries over
    refreshHistory()
  }

  return (
    <div className="card flex flex-col h-full min-h-0">
      <div
        className="flex items-center justify-between px-4 py-3 relative"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <h2 className="text-sm font-semibold">Ask the agent</h2>
        <div className="flex items-center gap-3">
          {/* Opening history must never wait on the network — the list is already
              loaded, so it opens instantly even mid-response. */}
          <button
            onClick={() => { setShowMenu(false); setShowHistory((s) => !s) }}
            className="text-xs muted hover:underline"
          >
            History ({history.length})
          </button>
          <button onClick={onNewThread} className="text-xs hover:underline"
            style={{ color: 'var(--series-1)' }}>
            + New
          </button>
          <button
            onClick={() => { setShowHistory(false); setShowMenu((s) => !s) }}
            className="text-xs muted px-1 leading-none"
            title="Clear chat, memory, or cache"
            aria-label="Maintenance"
          >
            ⋯
          </button>
        </div>

        {showHistory && (
          <div
            className="absolute right-3 top-12 z-50 card w-[300px] max-h-[320px] overflow-y-auto scroll-thin"
            style={{ boxShadow: '0 8px 24px rgba(0,0,0,0.18)' }}
          >
            {history.length === 0 ? (
              <p className="text-xs muted p-3">No past conversations yet.</p>
            ) : (
              history.map((c) => (
                <button
                  key={c.thread_id}
                  onClick={() => resume(c.thread_id)}
                  className="w-full text-left px-3 py-2 hover:opacity-80"
                  style={{
                    borderBottom: '1px solid var(--border)',
                    background: c.thread_id === threadId ? 'var(--surface-page)' : 'transparent',
                  }}
                >
                  <div className="text-xs truncate">{c.title}</div>
                  <div className="text-[11px] muted tabular">
                    {c.turns} turn{c.turns === 1 ? '' : 's'} · {when(c.last_at)}
                  </div>
                </button>
              ))
            )}
          </div>
        )}

        {showMenu && (
          <MaintenanceMenu
            onClose={() => setShowMenu(false)}
            onClearChat={handleClearChat}
            onCleared={refreshHistory}
          />
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin px-4 py-3 space-y-4">
        {messages.length === 0 && !live && (
          <div>
            <p className="text-xs muted mb-2">Try one of these:</p>
            <div className="grid gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => send(ex)}
                  className="text-left text-sm px-3 py-2 rounded-lg"
                  style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => <Message key={i} message={m} />)}

        {live && (
          <div className="space-y-2">
            <TraceFeed steps={live.steps} active elapsed={elapsed} />
            {live.answer && <Bubble role="assistant" content={live.answer} />}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="p-3" style={{ borderTop: '1px solid var(--border)' }}>
        <form onSubmit={(e) => { e.preventDefault(); send() }} className="flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about weather risk, HOA rules, repairs, bills…"
            disabled={busy}
            className="flex-1 px-3 py-2 rounded-lg text-sm outline-none disabled:opacity-60"
            style={{
              background: 'var(--surface-page)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
            }}
          />
          {busy ? (
            <button type="button" onClick={stop}
              className="px-3 py-2 rounded-lg text-sm font-medium tabular"
              style={{ border: '1px solid var(--border)' }}>
              Stop · {fmtDuration(elapsed)}
            </button>
          ) : (
            <button type="submit"
              className="px-4 py-2 rounded-lg text-sm font-medium text-white"
              style={{ background: 'var(--series-1)' }}>
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  )
}

/** Clear chat / memory / cache. Each action confirms in place before running. */
function MaintenanceMenu({ onClose, onClearChat, onCleared }) {
  const [stats, setStats] = useState(null)
  const [confirming, setConfirming] = useState(null)
  const [result, setResult] = useState('')
  const [working, setWorking] = useState(false)

  useEffect(() => {
    getAdminStats().then(setStats).catch(() => {})
  }, [])

  async function run(key, label, fn) {
    // Two clicks, not a browser confirm(): these wipe real state, and a native
    // dialog steals focus from a stream that may still be running.
    if (confirming !== key) {
      setConfirming(key)
      setResult('')
      return
    }
    setWorking(true)
    log('ui', 'maintenance.run', label, { level: 'warn' })
    try {
      const out = await fn()
      setResult(summarize(key, out))
      log('ui', 'maintenance.done', `${label} — done`, { data: out })
      getAdminStats().then(setStats).catch(() => {})
      onCleared?.()
    } catch (e) {
      setResult(`Failed: ${e.message}`)
      log('ui', 'maintenance.failed', `${label} — ${e.message}`, { level: 'error' })
    } finally {
      setWorking(false)
      setConfirming(null)
    }
  }

  const cacheEntries = (stats?.tool_cache?.entries ?? 0) + (stats?.semantic_cache_entries ?? 0)

  const actions = [
    {
      key: 'chat',
      label: 'Clear this conversation',
      hint: 'Wipes the transcript and the agent\'s memory of this thread, then starts a new one. Other conversations are kept.',
      run: async () => { await onClearChat(); return {} },
    },
    {
      key: 'memory',
      label: 'Clear all memory',
      hint: `Deletes every remembered interaction${stats ? ` (${stats.remembered_interactions})` : ''} across all conversations. The knowledge base is not touched.`,
      danger: true,
      run: clearAllMemory,
    },
    {
      key: 'cache',
      label: 'Clear cache',
      hint: `Drops cached answers and tool results${stats ? ` (${cacheEntries})` : ''}. The next question runs at full speed — slower, but live.`,
      run: clearCaches,
    },
  ]

  return (
    <div
      className="absolute right-3 top-12 z-50 card w-[320px] p-2"
      style={{ boxShadow: '0 8px 24px rgba(0,0,0,0.18)' }}
    >
      {actions.map((a) => (
        <div key={a.key} className="px-2 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
          <button
            disabled={working}
            onClick={() => run(a.key, a.label, a.run)}
            className="text-xs font-medium hover:underline disabled:opacity-50"
            style={{ color: confirming === a.key ? 'var(--status-critical)' : a.danger ? 'var(--status-critical)' : 'var(--text-primary)' }}
          >
            {confirming === a.key ? `Click again to confirm — ${a.label.toLowerCase()}` : a.label}
          </button>
          <p className="text-[11px] muted mt-0.5 leading-snug">{a.hint}</p>
        </div>
      ))}
      <div className="flex items-center justify-between px-2 pt-2">
        <span className="text-[11px] muted">{result || (working ? 'Working…' : '')}</span>
        <button onClick={onClose} className="text-[11px] muted hover:underline">Close</button>
      </div>
    </div>
  )
}

function summarize(key, out) {
  if (key === 'chat') return 'Conversation cleared.'
  if (key === 'memory') return `Deleted ${out.deleted ?? 0} remembered interactions.`
  if (key === 'cache') {
    return `Dropped ${out.entries ?? 0} cached entries + ${out.semantic_entries ?? 0} semantic.`
  }
  return 'Done.'
}

function Message({ message }) {
  const [showTrace, setShowTrace] = useState(false)
  if (message.role === 'user') return <Bubble role="user" content={message.content} />

  const callCount = (message.steps ?? []).filter((s) => s.kind === 'call').length
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        {message.steps?.length > 0 && (
          <button onClick={() => setShowTrace((s) => !s)} className="text-xs muted hover:underline">
            {showTrace ? '▾' : '▸'} {callCount} tool call{callCount === 1 ? '' : 's'}
          </button>
        )}
        {message.durationMs != null && (
          <span className="text-xs muted tabular" title="Time from sending to the finished answer">
            {fmtDuration(message.durationMs)}
          </span>
        )}
        {message.firstTokenMs != null && (
          <span
            className="text-xs muted tabular"
            title="Time until the first word of the answer appeared"
          >
            first word {fmtDuration(message.firstTokenMs)}
          </span>
        )}
        {message.cached && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: 'var(--surface-page)', border: '1px solid var(--border)',
              color: 'var(--status-good)' }}
            title={message.similarity ? `Matched a past question (similarity ${message.similarity})` : 'Served from cache'}
          >
            cached{message.cacheSource === 'semantic' ? ' · similar question' : ''}
          </span>
        )}
      </div>
      {showTrace && message.steps?.length > 0 && <TraceFeed steps={message.steps} />}
      {message.content && <Bubble role="assistant" content={message.content} />}
      {message.error && (
        <div
          className="rounded-xl px-3 py-2 text-sm"
          style={{
            border: '1px solid var(--status-critical)',
            color: 'var(--status-critical)',
            background: 'var(--surface-page)',
          }}
        >
          ⚠️ {message.error}
        </div>
      )}
    </div>
  )
}

function Bubble({ role, content }) {
  const isUser = role === 'user'
  // The user's own words go through verbatim; the agent's answer is markdown and
  // is rendered as such — printing it raw is what made answers unreadable.
  return (
    <div className={isUser ? 'flex justify-end' : ''}>
      <div
        className={`rounded-xl px-3 py-2 text-sm max-w-[90%] ${isUser ? 'whitespace-pre-wrap' : ''}`}
        style={
          isUser
            ? { background: 'var(--series-1)', color: '#fff' }
            : { background: 'var(--surface-page)', border: '1px solid var(--border)' }
        }
      >
        {isUser ? content : <Markdown>{content}</Markdown>}
      </div>
    </div>
  )
}

/** What the spinner should say, given what has already finished.
 *
 * A plain "thinking…" made the wait look like it belonged to whatever was named
 * last. Reported from a live session: "it takes FOREVER checking the contractor
 * registry" — on a turn where that lookup took 4 ms and the model then spent 114
 * seconds composing. The registry was simply the last thing with a name on it.
 *
 * Nothing can be streamed during that gap. The model is emitting reasoning
 * tokens, not content, so the server has no chunk to send and no opportunity to
 * drain a queued event until the first real token arrives. The client is the only
 * place that can say what is happening, so it says it here.
 */
function waitingLabel(steps, elapsed) {
  const lastTool = [...steps].reverse().find((s) => s.kind === 'call')
  const base = lastTool?.done
    ? 'reading the results and writing the answer…'
    : lastTool
      ? `${TOOL_LABELS[lastTool.name] ?? lastTool.name}…`
      : 'thinking…'
  // Past a minute the honest thing is to name the cause, so a long first ask
  // reads as the free model being slow rather than the app being stuck.
  return elapsed > 60000 ? `${base} (the free model is slow on a first ask)` : base
}

/** The live feed of what the agent is doing — the visible ReAct loop. */
function TraceFeed({ steps, active, elapsed }) {
  return (
    <div className="rounded-lg px-3 py-2 text-xs space-y-1"
      style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}>
      {steps.map((s, i) => {
        const label = s.kind === 'llm'
          ? `Thinking (model turn ${s.turn})`
          : TOOL_LABELS[s.name] ?? s.name ?? 'Working'
        const color =
          s.kind === 'guardrail' || s.kind === 'error' ? 'var(--status-critical)'
            : s.kind === 'memory' ? 'var(--series-2)'
              : s.kind === 'llm' ? 'var(--text-muted)'
                : 'var(--series-1)'
        return (
          <div key={i} className="flex items-start gap-2">
            <span style={{ color }} className={!s.done && active ? 'pulse' : ''}>
              {s.done ? '●' : '○'}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="secondary">{s.kind === 'error' ? 'Error' : label}</span>
                {s.durationMs != null && (
                  <span className="muted tabular shrink-0">{fmtDuration(s.durationMs)}</span>
                )}
              </div>
              {s.detail && <div className="muted break-words">{s.detail}</div>}
              {s.tree && (
                <div className="mt-1">
                  <ReasoningTree tree={s.tree} strategy={s.strategy} truncated={s.truncated} />
                </div>
              )}
            </div>
          </div>
        )
      })}
      {active && (
        <div className="flex items-center justify-between gap-2 muted">
          <span><span className="pulse">○</span> {waitingLabel(steps, elapsed)}</span>
          <span className="tabular">{fmtDuration(elapsed)}</span>
        </div>
      )}
    </div>
  )
}

function when(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const days = Math.floor((Date.now() - d.getTime()) / 86400000)
  if (days === 0) return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  if (days === 1) return 'yesterday'
  if (days < 7) return `${days}d ago`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}
