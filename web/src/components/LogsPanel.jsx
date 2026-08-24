import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { clearBackendLogs, getBackendLogs } from '../api'
import * as logbus from '../logbus'
import { fmtDuration } from '../utils/duration'

/**
 * The Logs tab: everything the system did, on both sides of the wire.
 *
 * **Source is a grouping, not a mode.** It used to be a three-way selector, which
 * meant the two halves of a request could only be read one at a time — and the
 * interesting part of a slow turn is usually the handoff between them. Both
 * buffers are now always loaded and the tree opens
 * `source → level → step → rows`, so "the browser sent it, the backend took nine
 * seconds" is one screen instead of two.
 *
 * **Filtering is by dimension.** Four independent dropdowns — source, level, step,
 * time — plus a keyword search over the message, the event name and the payload.
 * They compose, so "backend warnings from the retrieval step in the last five
 * minutes" is four selections rather than a scroll.
 *
 * Backend events are pulled by sequence number rather than timestamp, so tailing
 * can never skip or duplicate a row.
 */

// What each step means, shown as a tooltip. The vocabulary is the point of the
// tab: a reader should not have to guess what "external" covers.
const STEP_HELP = {
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

// Worst first — the order someone scanning for a problem reads in.
const LEVEL_ORDER = ['error', 'warn', 'info', 'debug']
const SOURCE_ORDER = ['frontend', 'backend']
const SOURCE_LABEL = { frontend: 'Frontend', backend: 'Backend' }

const TIME_WINDOWS = [
  { key: 'all', label: 'Any time', ms: null },
  { key: '5m', label: 'Last 5 min', ms: 5 * 60_000 },
  { key: '15m', label: 'Last 15 min', ms: 15 * 60_000 },
  { key: '1h', label: 'Last hour', ms: 60 * 60_000 },
  { key: 'today', label: 'Today', ms: null },
]

export default function LogsPanel() {
  const [backend, setBackend] = useState([])
  const [frontend, setFrontend] = useState(() => logbus.snapshot())

  const [sourceFilter, setSourceFilter] = useState('all')
  const [levelFilter, setLevelFilter] = useState('all')
  const [stepFilter, setStepFilter] = useState('all')
  const [timeFilter, setTimeFilter] = useState('all')
  const [showDebug, setShowDebug] = useState(false)
  const [query, setQuery] = useState('')

  const [view, setView] = useState('grouped')
  // Groups open unless closed; leaf steps closed unless opened. The two levels
  // have opposite defaults, so they need separate sets — folding both into one
  // "collapsed" set made expand-all ambiguous.
  const [closed, setClosed] = useState(() => new Set())
  const [opened, setOpened] = useState(() => new Set())

  const [tailing, setTailing] = useState(true)
  const [meta, setMeta] = useState({ dropped: 0, capacity: 0 })
  const [error, setError] = useState('')
  const sinceRef = useRef(0)
  const bottomRef = useRef(null)

  useEffect(() => logbus.subscribe(() => setFrontend(logbus.snapshot())), [])

  const poll = useCallback(async () => {
    try {
      const data = await getBackendLogs({ since: sinceRef.current, limit: 1000 })
      setMeta({ dropped: data.dropped, capacity: data.capacity })
      setError('')
      // A latest_seq behind our cursor means the server restarted and its counter
      // began again at 1. Waiting for a sequence that will never arrive would
      // freeze the tail, so start the cursor over.
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

  // Both buffers, always. Source is a filter and a grouping, never a fetch mode.
  const all = useMemo(
    () => [...backend, ...frontend].sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0)),
    [backend, frontend],
  )

  const cutoff = useMemo(() => {
    const w = TIME_WINDOWS.find((t) => t.key === timeFilter)
    if (!w || timeFilter === 'all') return null
    if (timeFilter === 'today') {
      const d = new Date()
      d.setHours(0, 0, 0, 0)
      return d.getTime()
    }
    return Date.now() - w.ms
  }, [timeFilter])

  // ONE derivation. The count shown to the user and the list rendered below it
  // come from this same array, so they cannot disagree about what is on screen.
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return all.filter((e) => {
      if (!showDebug && e.level === 'debug') return false
      if (sourceFilter !== 'all' && e.source !== sourceFilter) return false
      if (levelFilter !== 'all' && e.level !== levelFilter) return false
      if (stepFilter !== 'all' && e.group !== stepFilter) return false
      if (cutoff != null && new Date(e.ts).getTime() < cutoff) return false
      if (q) {
        const hay = `${e.message} ${e.event} ${JSON.stringify(e.data ?? '')}`.toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [all, showDebug, sourceFilter, levelFilter, stepFilter, cutoff, query])

  // Steps offered in the dropdown, with counts, from everything the other
  // filters admit — so the list narrows with context instead of going stale.
  const steps = useMemo(() => {
    const out = new Map()
    for (const e of all) {
      if (!showDebug && e.level === 'debug') continue
      if (sourceFilter !== 'all' && e.source !== sourceFilter) continue
      out.set(e.group, (out.get(e.group) ?? 0) + 1)
    }
    return [...out.entries()].sort((a, b) => b[1] - a[1])
  }, [all, showDebug, sourceFilter])

  // source -> level -> step -> rows, first-seen order preserved at every level so
  // the folded view still reads chronologically rather than alphabetically.
  const tree = useMemo(() => {
    const bySource = new Map()
    for (const e of rows) {
      if (!bySource.has(e.source)) bySource.set(e.source, new Map())
      const byLevel = bySource.get(e.source)
      if (!byLevel.has(e.level)) byLevel.set(e.level, new Map())
      const byStep = byLevel.get(e.level)
      if (!byStep.has(e.group)) byStep.set(e.group, [])
      byStep.get(e.group).push(e)
    }
    // Stable ordering: known sources first, then levels worst-first.
    return [...bySource.entries()]
      .sort((a, b) => SOURCE_ORDER.indexOf(a[0]) - SOURCE_ORDER.indexOf(b[0]))
      .map(([src, byLevel]) => [
        src,
        [...byLevel.entries()].sort(
          (a, b) => LEVEL_ORDER.indexOf(a[0]) - LEVEL_ORDER.indexOf(b[0]),
        ),
      ])
  }, [rows])

  const slowest = useMemo(
    () => rows.filter((e) => e.duration_ms != null)
      .sort((a, b) => b.duration_ms - a.duration_ms).slice(0, 5),
    [rows],
  )

  useEffect(() => {
    if (tailing && view === 'timeline') bottomRef.current?.scrollIntoView({ block: 'nearest' })
  }, [rows, tailing, view])

  const dirty = sourceFilter !== 'all' || levelFilter !== 'all' || stepFilter !== 'all'
    || timeFilter !== 'all' || query.trim() !== '' || showDebug

  function resetFilters() {
    setSourceFilter('all'); setLevelFilter('all'); setStepFilter('all')
    setTimeFilter('all'); setQuery(''); setShowDebug(false)
  }

  const keys = useMemo(() => {
    const groupKeys = []
    const leafKeys = []
    for (const [src, levels] of tree) {
      groupKeys.push(`s:${src}`)
      for (const [lvl, byStep] of levels) {
        groupKeys.push(`l:${src}/${lvl}`)
        for (const step of byStep.keys()) leafKeys.push(`${src}/${lvl}/${step}`)
      }
    }
    return { groupKeys, leafKeys }
  }, [tree])

  async function handleClear() {
    try { await clearBackendLogs() } catch { /* surfaced by the next poll */ }
    sinceRef.current = 0
    setBackend([])
    logbus.clear()
    setFrontend(logbus.snapshot())
  }

  function handleExport() {
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `forecaster-logs-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`
    a.click()
    URL.revokeObjectURL(url)
    logbus.log('ui', 'logs.exported', `Exported ${rows.length} events`)
  }

  const errors = rows.filter((e) => e.level === 'error').length

  return (
    <div className="card flex flex-col min-h-0" style={{ height: 'calc(100vh - 9rem)' }}>
      <div className="px-4 py-3 space-y-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
        {/* Search + view + actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search message, event name or payload…"
            className="px-3 py-1.5 rounded-md text-xs outline-none flex-1 min-w-[180px]"
            style={{
              background: 'var(--surface-page)', border: '1px solid var(--border)',
              color: 'var(--text-primary)',
            }}
          />
          <Segmented
            options={[{ key: 'grouped', label: 'Grouped' }, { key: 'timeline', label: 'Timeline' }]}
            value={view}
            onChange={setView}
          />
          <label className="flex items-center gap-1.5 text-xs muted cursor-pointer"
            title="Poll the backend every second and stick to the newest row">
            <input type="checkbox" checked={tailing}
              onChange={(e) => setTailing(e.target.checked)} />
            Live
          </label>
          <Btn onClick={handleExport}>Export</Btn>
          <Btn onClick={handleClear} danger>Clear</Btn>
        </div>

        {/* Filter dimensions — independent, and they compose */}
        <div className="flex items-center gap-2 flex-wrap">
          <Select label="Source" value={sourceFilter} onChange={setSourceFilter}
            options={[
              { key: 'all', label: `All sources (${all.length})` },
              ...SOURCE_ORDER.map((s) => ({
                key: s,
                label: `${SOURCE_LABEL[s]} (${all.filter((e) => e.source === s).length})`,
              })),
            ]} />

          <Select label="Level" value={levelFilter} onChange={setLevelFilter}
            options={[
              { key: 'all', label: 'All levels' },
              ...LEVEL_ORDER.map((l) => ({
                key: l,
                label: `${l} (${rowsAtLevel(all, l, showDebug, sourceFilter)})`,
              })),
            ]} />

          <Select label="Step" value={stepFilter} onChange={setStepFilter}
            title={STEP_HELP[stepFilter]}
            options={[
              { key: 'all', label: 'All steps' },
              ...steps.map(([g, n]) => ({ key: g, label: `${g} (${n})` })),
            ]} />

          <Select label="Time" value={timeFilter} onChange={setTimeFilter}
            options={TIME_WINDOWS.map((t) => ({ key: t.key, label: t.label }))} />

          <label className="flex items-center gap-1.5 text-xs muted cursor-pointer"
            title="Debug is the only level that is routinely noise, so it is opt-in">
            <input type="checkbox" checked={showDebug}
              onChange={(e) => setShowDebug(e.target.checked)} />
            Debug
          </label>

          {dirty && <Btn onClick={resetFilters} dashed>Reset filters</Btn>}
        </div>

        <div className="flex items-center gap-3 text-[11px] muted flex-wrap">
          <span><b>{rows.length}</b> shown of {all.length}</span>
          {errors > 0 && (
            <span style={{ color: 'var(--status-critical)' }}>{errors} error{errors === 1 ? '' : 's'}</span>
          )}
          {view === 'grouped' && (
            <>
              <button className="underline"
                onClick={() => { setClosed(new Set()); setOpened(new Set(keys.leafKeys)) }}>
                expand all
              </button>
              <button className="underline"
                onClick={() => { setClosed(new Set(keys.groupKeys)); setOpened(new Set()) }}>
                collapse all
              </button>
            </>
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

      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin">
        {rows.length === 0 ? (
          <p className="text-xs muted p-4">
            {all.length === 0
              ? 'No events yet. Ask the agent something, then come back — a single answer produces a few dozen.'
              : 'No events match these filters. Widen a dimension, or reset them.'}
          </p>
        ) : view === 'timeline' ? (
          rows.map((e) => <Row key={`${e.source}-${e.seq}`} event={e} showSource showStep />)
        ) : (
          tree.map(([src, levels]) => (
            <Section
              key={src}
              title={SOURCE_LABEL[src] ?? src}
              count={levels.reduce((n, [, byStep]) => n + [...byStep.values()].flat().length, 0)}
              tone="var(--text-primary)"
              open={!closed.has(`s:${src}`)}
              onToggle={() => setClosed((s) => toggleIn(s, `s:${src}`))}
              depth={0}
            >
              {levels.map(([lvl, byStep]) => (
                <Section
                  key={lvl}
                  title={lvl}
                  count={[...byStep.values()].flat().length}
                  tone={LEVEL_COLOR[lvl]}
                  open={!closed.has(`l:${src}/${lvl}`)}
                  onToggle={() => setClosed((s) => toggleIn(s, `l:${src}/${lvl}`))}
                  depth={1}
                >
                  {[...byStep.entries()].map(([step, events]) => (
                    <Section
                      key={step}
                      title={step}
                      help={STEP_HELP[step]}
                      count={events.length}
                      ms={totalMs(events)}
                      tone="var(--text-secondary)"
                      open={opened.has(`${src}/${lvl}/${step}`)}
                      onToggle={() => setOpened((s) => toggleIn(s, `${src}/${lvl}/${step}`))}
                      depth={2}
                    >
                      {events.map((e) => (
                        <Row key={`${e.source}-${e.seq}`} event={e} indent />
                      ))}
                    </Section>
                  ))}
                </Section>
              ))}
            </Section>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function rowsAtLevel(all, level, showDebug, sourceFilter) {
  return all.filter((e) => e.level === level
    && (showDebug || e.level !== 'debug')
    && (sourceFilter === 'all' || e.source === sourceFilter)).length
}

function toggleIn(set, value) {
  const next = new Set(set)
  if (next.has(value)) next.delete(value)
  else next.add(value)
  return next
}

function totalMs(events) {
  const timed = events.filter((e) => e.duration_ms != null)
  return timed.length ? timed.reduce((s, e) => s + e.duration_ms, 0) : null
}

function Section({ title, help, count, ms, tone, open, onToggle, depth, children }) {
  const pad = ['pl-4', 'pl-8', 'pl-12'][depth] ?? 'pl-4'
  return (
    <div style={depth === 0 ? { borderBottom: '1px solid var(--border)' } : undefined}>
      <button
        onClick={onToggle}
        title={help}
        className={`w-full text-left ${pad} pr-4 py-1.5 text-xs flex items-baseline gap-2`}
        style={depth === 0 ? { background: 'var(--surface-page)' } : undefined}
      >
        <span className="muted shrink-0">{open ? '▾' : '▸'}</span>
        <span className={depth === 0 ? 'font-semibold' : depth === 1 ? 'font-medium' : ''}
          style={{ color: tone }}>
          {title}
        </span>
        <span className="muted tabular">{count}</span>
        <span className="flex-1" />
        {ms != null && <span className="muted tabular">{fmtDuration(ms)}</span>}
      </button>
      {open && children}
    </div>
  )
}

function Segmented({ options, value, onChange }) {
  return (
    <div className="flex rounded-md overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      {options.map((o) => (
        <button
          key={o.key}
          onClick={() => onChange(o.key)}
          className="px-3 py-1.5 text-xs font-medium"
          style={{
            background: value === o.key ? 'var(--series-1)' : 'transparent',
            color: value === o.key ? '#fff' : 'var(--text-secondary)',
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function Select({ label, value, onChange, options, title }) {
  return (
    <label className="flex items-center gap-1.5 text-[11px] muted" title={title}>
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="px-2 py-1 rounded-md text-xs outline-none"
        style={{
          background: 'var(--surface-page)', border: '1px solid var(--border)',
          color: 'var(--text-primary)',
        }}
      >
        {options.map((o) => (
          <option key={o.key} value={o.key}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

function Btn({ children, onClick, danger, dashed }) {
  return (
    <button
      onClick={onClick}
      className="text-xs px-2.5 py-1.5 rounded-md"
      style={{
        border: `1px ${dashed ? 'dashed' : 'solid'} var(--border)`,
        color: danger ? 'var(--status-critical)' : 'var(--text-secondary)',
      }}
    >
      {children}
    </button>
  )
}

function Row({ event, showSource, showStep, indent }) {
  const [open, setOpen] = useState(false)
  const hasData = event.data && Object.keys(event.data).length > 0

  return (
    <div
      className={`${indent ? 'pl-16 pr-4' : 'px-4'} py-1.5 text-xs`}
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div className="flex items-baseline gap-2">
        <span className="muted tabular shrink-0" title={event.ts}>{clock(event.ts)}</span>
        {showSource && (
          <span className="shrink-0 text-[10px] px-1 rounded-md"
            style={{
              border: '1px solid var(--border)',
              color: event.source === 'frontend' ? 'var(--series-2)' : 'var(--series-1)',
            }}>
            {event.source === 'frontend' ? 'FE' : 'BE'}
          </span>
        )}
        {showStep && (
          <span className="shrink-0 tabular" style={{ color: LEVEL_COLOR[event.level] }}>
            {event.group}
          </span>
        )}
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
        <pre className="mt-1 ml-4 p-2 rounded-md overflow-x-auto scroll-thin text-[11px]"
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
