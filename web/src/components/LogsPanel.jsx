import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { clearBackendLogs, getBackendLogs } from '../api'
import * as logbus from '../logbus'
import { fmtDuration } from '../utils/duration'

/**
 * The Logs tab: everything the system did, on both sides of the wire.
 *
 * Frontend and backend keep separate buffers on purpose — they answer different
 * questions ("did the browser send it?" vs "what did the agent do with it?") —
 * so the source selector is the primary control, with a combined timeline for
 * when the interesting part is the handoff between them.
 *
 * Backend events are pulled by sequence number rather than timestamp, so tailing
 * can never skip or duplicate a row.
 */

const SOURCES = [
  { key: 'backend', label: 'Backend' },
  { key: 'frontend', label: 'Frontend' },
  { key: 'combined', label: 'Combined' },
]

// What each group means, shown as a tooltip on its chip. The vocabulary is the
// point of the tab: a reader should not have to guess what "external" covers.
const GROUP_HELP = {
  http: 'Inbound API requests and chat streams',
  agent: 'One agent turn, start to finish',
  router: 'Deterministic turn labelling: intent, complexity, risk (advisory only)',
  llm: 'Model round-trips and token counts',
  tool: 'Agent tool calls',
  external: 'Outbound third-party APIs (weather, geocode, energy)',
  cache: 'Answer and tool-result caches',
  memory: 'Episodic recall and recording',
  rag: 'Knowledge-base searches',
  research: 'Web research: provider searches, evidence ranking, dropped passages',
  safety: 'Guardrail screens',
  system: 'Startup and maintenance actions',
  ui: 'User actions in the browser',
  api: 'Fetch calls from the browser',
  stream: 'Server-sent events received',
  render: 'React lifecycle and render errors',
  error: 'Uncaught errors and rejected promises',
}

const LEVEL_COLOR = {
  error: 'var(--status-critical)',
  warn: 'var(--status-warning)',
  info: 'var(--series-1)',
  debug: 'var(--text-muted)',
}

export default function LogsPanel() {
  const [source, setSource] = useState('backend')
  const [backend, setBackend] = useState([])
  const [frontend, setFrontend] = useState(() => logbus.snapshot())
  const [group, setGroup] = useState(null)
  const [query, setQuery] = useState('')
  const [showDebug, setShowDebug] = useState(false)
  const [tailing, setTailing] = useState(true)
  const [meta, setMeta] = useState({ dropped: 0, capacity: 0 })
  const [error, setError] = useState('')
  const sinceRef = useRef(0)
  const bottomRef = useRef(null)

  // --- frontend: push-based, already in memory --------------------------
  useEffect(() => logbus.subscribe(() => setFrontend(logbus.snapshot())), [])

  // --- backend: poll by sequence number ---------------------------------
  const poll = useCallback(async () => {
    try {
      const data = await getBackendLogs({ since: sinceRef.current, limit: 1000 })
      setMeta({ dropped: data.dropped, capacity: data.capacity })
      setError('')

      // A latest_seq behind our cursor means the server restarted and its
      // counter began again at 1. Waiting for a sequence number that will never
      // arrive would freeze the tail, so start the cursor over.
      if (data.latest_seq < sinceRef.current) {
        sinceRef.current = 0
        setBackend(data.events)
        return
      }
      if (data.events.length) {
        sinceRef.current = data.latest_seq
        setBackend((prev) => [...prev, ...data.events].slice(-2000))
      }
    } catch (e) {
      setError(e.message)
    }
  }, [])

  useEffect(() => {
    poll()
    if (!tailing) return undefined
    const id = setInterval(poll, 1000)
    return () => clearInterval(id)
  }, [poll, tailing])

  const rows = useMemo(() => {
    let list =
      source === 'backend' ? backend
        : source === 'frontend' ? frontend
          : [...backend, ...frontend].sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0))

    if (group) list = list.filter((e) => e.group === group)
    if (!showDebug) list = list.filter((e) => e.level !== 'debug')
    if (query.trim()) {
      const q = query.toLowerCase()
      list = list.filter(
        (e) => e.message.toLowerCase().includes(q)
          || e.event.toLowerCase().includes(q)
          || JSON.stringify(e.data ?? '').toLowerCase().includes(q),
      )
    }
    return list
  }, [source, backend, frontend, group, query, showDebug])

  // Group counts are computed before the group filter, so the chips keep showing
  // what else is available instead of collapsing to the current selection.
  const counts = useMemo(() => {
    const base =
      source === 'backend' ? backend : source === 'frontend' ? frontend : [...backend, ...frontend]
    const visible = showDebug ? base : base.filter((e) => e.level !== 'debug')
    const out = {}
    for (const e of visible) out[e.group] = (out[e.group] ?? 0) + 1
    return out
  }, [source, backend, frontend, showDebug])

  // Where the time actually goes: the slowest measured operations on screen.
  const slowest = useMemo(
    () => rows.filter((e) => e.duration_ms != null)
      .sort((a, b) => b.duration_ms - a.duration_ms)
      .slice(0, 5),
    [rows],
  )

  useEffect(() => {
    if (tailing) bottomRef.current?.scrollIntoView({ block: 'nearest' })
  }, [rows, tailing])

  async function handleClear() {
    if (source !== 'frontend') {
      try {
        await clearBackendLogs()
      } catch { /* surfaced by the next poll */ }
      sinceRef.current = 0
      setBackend([])
    }
    if (source !== 'backend') {
      logbus.clear()
      setFrontend(logbus.snapshot())
    }
  }

  function handleExport() {
    // Whatever is on screen, in the shape it is stored — so an exported run can
    // be attached to a bug report or the capstone write-up as-is.
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `forecaster-logs-${source}-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`
    a.click()
    URL.revokeObjectURL(url)
    logbus.log('ui', 'logs.exported', `Exported ${rows.length} events`)
  }

  const errors = rows.filter((e) => e.level === 'error').length

  return (
    <div className="card flex flex-col min-h-0" style={{ height: 'calc(100vh - 9rem)' }}>
      {/* Controls */}
      <div className="px-4 py-3 space-y-3" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {SOURCES.map((s) => (
              <button
                key={s.key}
                onClick={() => { setSource(s.key); setGroup(null) }}
                className="px-3 py-1.5 text-xs font-medium"
                style={{
                  background: source === s.key ? 'var(--series-1)' : 'transparent',
                  color: source === s.key ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {s.label}
              </button>
            ))}
          </div>

          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter events…"
            className="px-3 py-1.5 rounded-lg text-xs outline-none flex-1 min-w-[140px]"
            style={{
              background: 'var(--surface-page)', border: '1px solid var(--border)',
              color: 'var(--text-primary)',
            }}
          />

          <label className="flex items-center gap-1.5 text-xs muted cursor-pointer">
            <input type="checkbox" checked={showDebug}
              onChange={(e) => setShowDebug(e.target.checked)} />
            Debug
          </label>
          <label className="flex items-center gap-1.5 text-xs muted cursor-pointer"
            title="Poll the backend every second and stick to the newest row">
            <input type="checkbox" checked={tailing}
              onChange={(e) => setTailing(e.target.checked)} />
            Live
          </label>

          <button onClick={handleExport} className="text-xs px-2 py-1.5 rounded-lg"
            style={{ border: '1px solid var(--border)' }}>
            Export
          </button>
          <button onClick={handleClear} className="text-xs px-2 py-1.5 rounded-lg"
            style={{ border: '1px solid var(--border)', color: 'var(--status-critical)' }}>
            Clear
          </button>
        </div>

        {/* Subgroups */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <Chip label="All" count={rows.length} active={!group} onClick={() => setGroup(null)} />
          {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([g, n]) => (
            <Chip
              key={g}
              label={g}
              count={n}
              title={GROUP_HELP[g]}
              active={group === g}
              onClick={() => setGroup(group === g ? null : g)}
            />
          ))}
        </div>

        <div className="flex items-center gap-3 text-[11px] muted flex-wrap">
          <span>{rows.length} shown</span>
          {errors > 0 && (
            <span style={{ color: 'var(--status-critical)' }}>{errors} error{errors === 1 ? '' : 's'}</span>
          )}
          {meta.dropped > 0 && <span>{meta.dropped} older backend events dropped (buffer holds {meta.capacity})</span>}
          {error && <span style={{ color: 'var(--status-critical)' }}>Log fetch failed: {error}</span>}
          {slowest.length > 0 && (
            <span className="tabular">
              Slowest: {slowest.map((e) => `${shortName(e)} ${fmtDuration(e.duration_ms)}`).join(' · ')}
            </span>
          )}
        </div>
      </div>

      {/* Rows */}
      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin">
        {rows.length === 0 ? (
          <p className="text-xs muted p-4">
            No events match. Ask the agent something, then come back — a single answer
            produces a few dozen.
          </p>
        ) : (
          rows.map((e) => <Row key={`${e.source}-${e.seq}`} event={e} showSource={source === 'combined'} />)
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function Chip({ label, count, active, onClick, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="text-[11px] px-2 py-0.5 rounded-full tabular"
      style={{
        border: '1px solid var(--border)',
        background: active ? 'var(--series-1)' : 'var(--surface-page)',
        color: active ? '#fff' : 'var(--text-secondary)',
      }}
    >
      {label} {count}
    </button>
  )
}

function Row({ event, showSource }) {
  const [open, setOpen] = useState(false)
  const hasData = event.data && Object.keys(event.data).length > 0

  return (
    <div
      className="px-4 py-1.5 text-xs"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div className="flex items-baseline gap-2">
        <span className="muted tabular shrink-0" title={event.ts}>{clock(event.ts)}</span>
        {showSource && (
          <span className="shrink-0 text-[10px] px-1 rounded"
            style={{
              border: '1px solid var(--border)',
              color: event.source === 'frontend' ? 'var(--series-2)' : 'var(--series-1)',
            }}>
            {event.source === 'frontend' ? 'FE' : 'BE'}
          </span>
        )}
        <span className="shrink-0 tabular" style={{ color: LEVEL_COLOR[event.level] }}>
          {event.group}
        </span>
        <button
          onClick={() => hasData && setOpen((o) => !o)}
          className={`text-left flex-1 min-w-0 ${hasData ? 'hover:underline' : 'cursor-default'}`}
          style={{ color: event.level === 'error' ? 'var(--status-critical)' : 'var(--text-primary)' }}
        >
          <span className="muted">{event.event}</span>{' '}
          <span className="break-words">{event.message}</span>
        </button>
        {event.duration_ms != null && (
          <span className="tabular shrink-0"
            style={{ color: event.duration_ms > 3000 ? 'var(--status-warning)' : 'var(--text-muted)' }}>
            {fmtDuration(event.duration_ms)}
          </span>
        )}
      </div>
      {open && hasData && (
        <pre className="mt-1 ml-4 p-2 rounded overflow-x-auto scroll-thin text-[11px]"
          style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}>
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  )
}

/** Local wall-clock time with milliseconds — enough to line up two sources. */
function clock(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '--:--:--'
  const pad = (n, w = 2) => String(n).padStart(w, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`
}

function shortName(event) {
  return event.data?.tool ?? event.data?.function?.split('.').pop() ?? event.event
}
