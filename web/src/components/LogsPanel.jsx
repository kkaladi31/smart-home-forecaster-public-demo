import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { clearBackendLogs, getBackendLogs } from '../api'
import * as logbus from '../logbus'
import { fmtDuration } from '../utils/duration'

/**
 * The Logs tab: everything the system did, on both sides of the wire.
 *
 * **Two views at once, not a toggle.** A summary tree rolls the run up by
 * `source → level → event type`, and a filterable table underneath holds every
 * individual row. Those answer different questions — *"what went wrong and how
 * often"* versus *"show me that exact moment"* — and making them alternatives
 * meant switching back and forth to hold one answer against the other.
 *
 * The summary is an **aggregation**, not a fold: one line per distinct event
 * type, carrying how many times it fired, which step it belongs to, the total
 * time it accounts for, and one representative message. Thirty `http.request`
 * rows become one line that says 30.
 *
 * Backend events are pulled by sequence number rather than timestamp, so tailing
 * can never skip or duplicate a row.
 */

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

// One colour per step. Scanning a run is a colour task before it is a reading
// task — the eye finds "the pink one" faster than it reads "research". Hues are
// spread and mid-toned so they hold up on both themes; the app's own palette has
// two series colours, which cannot separate twelve steps.
const STEP_COLOR = {
  http: '#3b82f6', agent: '#8b5cf6', router: '#0ea5e9', llm: '#d946ef',
  tool: '#10b981', external: '#f59e0b', cache: '#64748b', memory: '#14b8a6',
  rag: '#6366f1', research: '#ec4899', safety: '#ef4444', system: '#94a3b8',
  ui: '#22c55e', api: '#2dd4bf', stream: '#a855f7', render: '#f97316',
  error: '#dc2626',
}
const stepColor = (g) => STEP_COLOR[g] ?? '#94a3b8'

const LEVEL_META = {
  error: { color: '#ef4444', label: 'Error', blurb: 'Failed, needs attention' },
  warn: { color: '#f59e0b', label: 'Warning', blurb: 'Recovered or retried' },
  info: { color: '#3b82f6', label: 'Info', blurb: 'Normal operation' },
  debug: { color: '#64748b', label: 'Debug', blurb: 'Diagnostic detail' },
}
const SOURCE_META = {
  backend: { color: '#2dd4bf', label: 'Backend', blurb: 'Agent turns, tools and providers' },
  frontend: { color: '#a855f7', label: 'Frontend', blurb: 'Browser actions and console errors' },
}

const LEVEL_ORDER = ['error', 'warn', 'info', 'debug']
const SOURCE_ORDER = ['backend', 'frontend']

const TIME_WINDOWS = [
  { key: 'all', label: 'Any time' },
  { key: '5m', label: 'Last 5 min', ms: 5 * 60_000 },
  { key: '15m', label: 'Last 15 min', ms: 15 * 60_000 },
  { key: '1h', label: 'Last hour', ms: 60 * 60_000 },
  { key: 'today', label: 'Today' },
  { key: 'custom', label: 'Custom range…' },
]

const COLUMNS = [
  { key: 'ts', label: 'Time', width: '150px' },
  { key: 'level', label: 'Level', width: '78px' },
  { key: 'source', label: 'Source', width: '92px' },
  { key: 'group', label: 'Step', width: '104px' },
  { key: 'event', label: 'Event', width: '190px' },
  { key: 'duration_ms', label: 'Took', width: '74px' },
  { key: 'message', label: 'Message', width: 'minmax(220px,1fr)' },
]

function toLocalInput(d) {
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    + `T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export default function LogsPanel() {
  const [backend, setBackend] = useState([])
  const [frontend, setFrontend] = useState(() => logbus.snapshot())

  const [query, setQuery] = useState('')
  const [eventQuery, setEventQuery] = useState('')
  const [levelFilter, setLevelFilter] = useState('all')
  const [sourceFilter, setSourceFilter] = useState('all')
  const [stepFilter, setStepFilter] = useState('all')
  const [timeFilter, setTimeFilter] = useState('all')
  const [rangeFrom, setRangeFrom] = useState('')
  const [rangeTo, setRangeTo] = useState('')
  const [showDebug, setShowDebug] = useState(false)

  const [sort, setSort] = useState({ key: 'ts', dir: 'desc' })
  const [closed, setClosed] = useState(() => new Set())

  const [tailing, setTailing] = useState(true)
  const [meta, setMeta] = useState({ dropped: 0, capacity: 0 })
  const [error, setError] = useState('')
  const sinceRef = useRef(0)
  const inFlightRef = useRef(false)

  useEffect(() => logbus.subscribe(() => setFrontend(logbus.snapshot())), [])

  const poll = useCallback(async () => {
    // Two polls must never overlap. React mounts effects twice in development,
    // so the immediate call and the interval both started with `since = 0` and
    // both appended the same batch. Duplicated events mean duplicated React
    // keys, and a list with colliding keys stops reconciling: rows persist
    // after a filter changes, and the expand-state inside a row attaches to
    // whichever node React happened to reuse. Both of those looked like filter
    // bugs and were really this.
    if (inFlightRef.current) return
    inFlightRef.current = true
    try {
      const data = await getBackendLogs({ since: sinceRef.current, limit: 1000 })
      setMeta({ dropped: data.dropped, capacity: data.capacity })
      setError('')
      if (data.latest_seq < sinceRef.current) {
        // The server restarted and its counter began again at 1. Waiting for a
        // sequence that will never arrive would freeze the tail.
        sinceRef.current = 0
        setBackend(dedupeBySeq(data.events))
        return
      }
      if (data.events.length) {
        sinceRef.current = data.latest_seq
        // Deduplicate on merge as well as guarding the poll. The endpoint
        // returns a WINDOW of the most recent `limit` events rather than a
        // strict tail after `since`, so overlap is a property of the API, not
        // only of a racing caller.
        setBackend((prev) => dedupeBySeq([...prev, ...data.events]).slice(-2000))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      inFlightRef.current = false
    }
  }, [])

  useEffect(() => {
    poll()
    if (!tailing) return undefined
    const id = setInterval(poll, 1000)
    return () => clearInterval(id)
  }, [poll, tailing])

  const all = useMemo(
    () => [...backend, ...frontend].sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0)),
    [backend, frontend],
  )

  const range = useMemo(() => {
    if (timeFilter === 'all') return null
    if (timeFilter === 'custom') {
      return {
        from: rangeFrom ? new Date(rangeFrom).getTime() : null,
        to: rangeTo ? new Date(rangeTo).getTime() : null,
      }
    }
    if (timeFilter === 'today') {
      const d = new Date(); d.setHours(0, 0, 0, 0)
      return { from: d.getTime(), to: null }
    }
    const w = TIME_WINDOWS.find((x) => x.key === timeFilter)
    return w?.ms ? { from: Date.now() - w.ms, to: null } : null
  }, [timeFilter, rangeFrom, rangeTo])

  // ONE derivation. Every count on screen and every row in the table come from
  // this array, so the summary and the detail can never disagree.
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const eq = eventQuery.trim().toLowerCase()
    return all.filter((e) => {
      if (!showDebug && e.level === 'debug') return false
      if (levelFilter !== 'all' && e.level !== levelFilter) return false
      if (sourceFilter !== 'all' && e.source !== sourceFilter) return false
      if (stepFilter !== 'all' && e.group !== stepFilter) return false
      if (range) {
        const at = new Date(e.ts).getTime()
        if (range.from != null && at < range.from) return false
        if (range.to != null && at > range.to) return false
      }
      if (eq && !e.event.toLowerCase().includes(eq)) return false
      if (q) {
        const hay = `${e.message} ${e.event} ${JSON.stringify(e.data ?? '')}`.toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [all, showDebug, levelFilter, sourceFilter, stepFilter, range, query, eventQuery])

  const steps = useMemo(() => {
    const out = new Map()
    for (const e of all) {
      if (!showDebug && e.level === 'debug') continue
      out.set(e.group, (out.get(e.group) ?? 0) + 1)
    }
    return [...out.entries()].sort((a, b) => b[1] - a[1])
  }, [all, showDebug])

  // The summary: source -> level -> one aggregated line per distinct event type.
  const summary = useMemo(() => {
    const bySource = new Map()
    for (const e of rows) {
      if (!bySource.has(e.source)) bySource.set(e.source, new Map())
      const byLevel = bySource.get(e.source)
      if (!byLevel.has(e.level)) byLevel.set(e.level, new Map())
      const byEvent = byLevel.get(e.level)
      const prev = byEvent.get(e.event)
      if (prev) {
        prev.count += 1
        prev.ms += e.duration_ms ?? 0
        prev.steps.add(e.group)
      } else {
        byEvent.set(e.event, {
          event: e.event, count: 1, ms: e.duration_ms ?? 0,
          steps: new Set([e.group]), sample: e.message, level: e.level,
        })
      }
    }
    return SOURCE_ORDER
      .filter((s) => bySource.has(s))
      .map((s) => [
        s,
        LEVEL_ORDER
          .filter((l) => bySource.get(s).has(l))
          .map((l) => [
            l,
            [...bySource.get(s).get(l).values()].sort((a, b) => b.count - a.count),
          ]),
      ])
  }, [rows])

  const sorted = useMemo(() => {
    const list = [...rows]
    const { key, dir } = sort
    list.sort((a, b) => {
      let x = a[key], y = b[key]
      if (key === 'duration_ms') { x = x ?? -1; y = y ?? -1 }
      if (key === 'level') { x = LEVEL_ORDER.indexOf(a.level); y = LEVEL_ORDER.indexOf(b.level) }
      if (x === y) return 0
      const cmp = x < y ? -1 : 1
      return dir === 'asc' ? cmp : -cmp
    })
    return list
  }, [rows, sort])

  const dirty = levelFilter !== 'all' || sourceFilter !== 'all' || stepFilter !== 'all'
    || timeFilter !== 'all' || query.trim() !== '' || eventQuery.trim() !== '' || showDebug

  function resetFilters() {
    setLevelFilter('all'); setSourceFilter('all'); setStepFilter('all')
    setTimeFilter('all'); setQuery(''); setEventQuery(''); setShowDebug(false)
    setRangeFrom(''); setRangeTo('')
  }

  async function handleClear() {
    try { await clearBackendLogs() } catch { /* surfaced by the next poll */ }
    sinceRef.current = 0
    setBackend([])
    logbus.clear()
    setFrontend(logbus.snapshot())
  }

  function handleExport() {
    const blob = new Blob([JSON.stringify(sorted, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `forecaster-logs-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`
    a.click()
    URL.revokeObjectURL(url)
    logbus.log('ui', 'logs.exported', `Exported ${sorted.length} events`)
  }

  const grid = COLUMNS.map((c) => c.width).join(' ')

  return (
    <div className="card flex flex-col min-h-0" style={{ height: 'calc(100vh - 9rem)' }}>
      {/* ---------- Summary ---------- */}
      <div className="flex-1 min-h-0 overflow-y-auto scroll-thin"
        style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="px-4 pt-3 pb-1 flex items-baseline justify-between">
          <span className="text-[11px] font-semibold tracking-wide muted">
            ACTIVITY BY SOURCE
          </span>
          <span className="text-[11px] muted">
            {rows.length} of {all.length} events
          </span>
        </div>

        {summary.length === 0 ? (
          <p className="text-xs muted px-4 py-6">
            {all.length === 0
              ? 'No events yet. Ask the agent something — a single answer produces a few dozen.'
              : 'Nothing matches these filters.'}
          </p>
        ) : summary.map(([src, levels]) => (
          <Fold
            key={src}
            open={!closed.has(`s:${src}`)}
            onToggle={() => setClosed((s) => flip(s, `s:${src}`))}
            chip={SOURCE_META[src]?.label ?? src}
            chipColor={SOURCE_META[src]?.color ?? '#94a3b8'}
            blurb={SOURCE_META[src]?.blurb}
            indent={0}
          >
            {levels.map(([lvl, events]) => (
              <Fold
                key={lvl}
                open={!closed.has(`l:${src}/${lvl}`)}
                onToggle={() => setClosed((s) => flip(s, `l:${src}/${lvl}`))}
                chip={LEVEL_META[lvl]?.label ?? lvl}
                chipColor={LEVEL_META[lvl]?.color ?? '#94a3b8'}
                blurb={LEVEL_META[lvl]?.blurb}
                indent={1}
              >
                {events.map((row) => (
                  <SummaryRow key={row.event} row={row}
                    onPick={() => { setEventQuery(row.event); setLevelFilter(row.level) }} />
                ))}
              </Fold>
            ))}
          </Fold>
        ))}
      </div>

      {/* ---------- Filters ---------- */}
      <div className="px-4 py-2.5 flex items-center gap-2 flex-wrap"
        style={{ borderBottom: '1px solid var(--border)' }}>
        <Text value={query} onChange={setQuery} placeholder="Search message…" width="180px" />
        <Text value={eventQuery} onChange={setEventQuery} placeholder="Event name…" width="150px" />
        <Drop value={levelFilter} onChange={setLevelFilter}
          options={[{ k: 'all', v: 'Any level' },
            ...LEVEL_ORDER.map((l) => ({ k: l, v: LEVEL_META[l].label }))]} />
        <Drop value={sourceFilter} onChange={setSourceFilter}
          options={[{ k: 'all', v: 'Any source' },
            ...SOURCE_ORDER.map((s) => ({ k: s, v: SOURCE_META[s].label }))]} />
        <Drop value={stepFilter} onChange={setStepFilter} title={STEP_HELP[stepFilter]}
          options={[{ k: 'all', v: 'Any step' },
            ...steps.map(([g, n]) => ({ k: g, v: `${g} (${n})` }))]} />
        <Drop value={timeFilter} onChange={setTimeFilter}
          options={TIME_WINDOWS.map((t) => ({ k: t.key, v: t.label }))} />

        {timeFilter === 'custom' && (
          <>
            <DateInput value={rangeFrom} onChange={setRangeFrom} />
            <span className="text-[11px] muted">to</span>
            <DateInput value={rangeTo} onChange={setRangeTo} />
            <button className="text-[11px] underline muted"
              title="Fill the range from the events currently loaded"
              onClick={() => {
                if (!all.length) return
                setRangeFrom(toLocalInput(new Date(all[0].ts)))
                setRangeTo(toLocalInput(new Date(all[all.length - 1].ts)))
              }}>fit to data</button>
          </>
        )}

        <label className="flex items-center gap-1.5 text-[11px] muted cursor-pointer"
          title="Debug is the only level that is routinely noise, so it is opt-in">
          <input type="checkbox" checked={showDebug}
            onChange={(e) => setShowDebug(e.target.checked)} />
          Debug
        </label>

        {dirty && <Btn onClick={resetFilters} dashed>Reset</Btn>}

        <span className="flex-1" />
        <span className="flex items-center gap-1.5 text-[11px] muted">
          <span style={{
            width: 7, height: 7, borderRadius: 2, display: 'inline-block',
            background: tailing ? '#22c55e' : 'var(--text-muted)',
          }} />
          <label className="cursor-pointer">
            <input type="checkbox" className="hidden" checked={tailing}
              onChange={(e) => setTailing(e.target.checked)} />
            Live
          </label>
          · {sorted.length} entries
        </span>
        <Btn onClick={handleExport}>Export</Btn>
        <Btn onClick={handleClear} danger>Clear</Btn>
      </div>

      {/* ---------- Detail table ---------- */}
      <div className="flex-1 min-h-0 overflow-auto scroll-thin">
        <div className="text-[11px] font-medium sticky top-0 z-10"
          style={{
            display: 'grid', gridTemplateColumns: grid,
            background: 'var(--surface-page)', borderBottom: '1px solid var(--border)',
          }}>
          {COLUMNS.map((c) => (
            <button key={c.key}
              onClick={() => setSort((s) => ({
                key: c.key,
                dir: s.key === c.key && s.dir === 'desc' ? 'asc' : 'desc',
              }))}
              className="text-left px-2 py-1.5 flex items-center gap-1"
              style={{ color: 'var(--text-secondary)' }}>
              {c.label}
              <span className="muted">
                {sort.key === c.key ? (sort.dir === 'desc' ? '▼' : '▲') : '↕'}
              </span>
            </button>
          ))}
        </div>

        {sorted.length === 0 ? (
          <p className="text-xs muted px-4 py-6">Nothing matches these filters.</p>
        ) : sorted.map((e) => (
          <TableRow key={`${e.source}-${e.seq}-${e.ts}`} event={e} grid={grid} />
        ))}
      </div>

      {(meta.dropped > 0 || error) && (
        <div className="px-4 py-1.5 text-[11px] muted"
          style={{ borderTop: '1px solid var(--border)' }}>
          {meta.dropped > 0 && <span>{meta.dropped} older backend events dropped (buffer holds {meta.capacity}). </span>}
          {error && <span style={{ color: 'var(--status-critical)' }}>Log fetch failed: {error}</span>}
        </div>
      )}
    </div>
  )
}

/** Last write wins, ordered by sequence. Identity is the sequence number. */
function dedupeBySeq(events) {
  const bySeq = new Map()
  for (const e of events) bySeq.set(e.seq, e)
  return [...bySeq.values()].sort((a, b) => a.seq - b.seq)
}

function flip(set, value) {
  const next = new Set(set)
  if (next.has(value)) next.delete(value)
  else next.add(value)
  return next
}

/** A collapsible section headed by a coloured chip and a plain-language blurb. */
function Fold({ open, onToggle, chip, chipColor, blurb, indent, children }) {
  return (
    <div className={indent === 0 ? 'mt-1' : ''}>
      <button onClick={onToggle}
        className={`w-full text-left flex items-center gap-2 py-1.5 ${indent ? 'pl-8' : 'pl-4'} pr-4`}>
        <span className="muted text-[10px] shrink-0">{open ? '▼' : '▶'}</span>
        <Pill color={chipColor}>{chip}</Pill>
        {blurb && <span className="text-[11px] muted">{blurb}</span>}
      </button>
      {open && <div className={indent === 0 ? 'pl-2' : 'pl-6'}>{children}</div>}
    </div>
  )
}

/** One aggregated event type: how often, which step, how long, what it said. */
function SummaryRow({ row, onPick }) {
  const step = [...row.steps][0]
  const color = LEVEL_META[row.level]?.color ?? '#94a3b8'
  return (
    <button onClick={onPick}
      title="Filter the table below to this event"
      className="w-full text-left flex items-center gap-2 pl-8 pr-4 py-1 rounded-md"
      style={{ borderLeft: `2px solid ${stepColor(step)}` }}>
      <span className="shrink-0 text-[11px] font-semibold tabular px-1.5 py-0.5 rounded-md"
        style={{ background: color, color: '#fff', minWidth: 28, textAlign: 'center' }}>
        {row.count}
      </span>
      <span className="shrink-0 text-xs font-semibold font-mono" style={{ color }}>
        {row.event}
      </span>
      {row.steps.size === 1 && (
        <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded-md"
          style={{ color: stepColor(step), border: `1px solid ${stepColor(step)}` }}>
          {step}
        </span>
      )}
      {row.ms > 0 && (
        <span className="shrink-0 text-[11px] muted tabular">{fmtDuration(row.ms)}</span>
      )}
      <span className="text-[11px] muted truncate">{row.sample}</span>
    </button>
  )
}

function TableRow({ event, grid }) {
  const [open, setOpen] = useState(false)
  const hasData = event.data && Object.keys(event.data).length > 0
  const lvl = LEVEL_META[event.level] ?? {}
  const src = SOURCE_META[event.source] ?? {}

  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="text-xs items-center" style={{ display: 'grid', gridTemplateColumns: grid }}>
        <span className="px-2 py-1.5 muted tabular truncate" title={event.ts}>{clock(event.ts)}</span>
        <span className="px-2 py-1.5"><Pill color={lvl.color} small>{event.level}</Pill></span>
        <span className="px-2 py-1.5"><Pill color={src.color} small>{event.source}</Pill></span>
        <span className="px-2 py-1.5">
          <span className="text-[10px] px-1.5 py-0.5 rounded-md"
            style={{ color: stepColor(event.group), border: `1px solid ${stepColor(event.group)}` }}>
            {event.group}
          </span>
        </span>
        <span className="px-2 py-1.5 font-mono text-[11px] truncate"
          style={{ color: lvl.color }} title={event.event}>{event.event}</span>
        <span className="px-2 py-1.5 tabular muted"
          style={{ color: event.duration_ms > 3000 ? 'var(--status-warning)' : undefined }}>
          {event.duration_ms != null ? fmtDuration(event.duration_ms) : ''}
        </span>
        <button onClick={() => hasData && setOpen((o) => !o)}
          className={`px-2 py-1.5 text-left truncate ${hasData ? 'hover:underline' : 'cursor-default'}`}
          style={{ color: 'var(--text-primary)' }}
          title={event.message}>
          {event.message}
        </button>
      </div>
      {open && hasData && (
        <pre className="mx-2 mb-2 p-2 rounded-md overflow-x-auto scroll-thin text-[11px]"
          style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}>
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  )
}

function Pill({ color, small, children }) {
  return (
    <span className={`inline-block rounded-md font-medium ${small ? 'text-[10px] px-1.5 py-0.5' : 'text-[11px] px-2 py-0.5'}`}
      style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}>
      {children}
    </span>
  )
}

function Text({ value, onChange, placeholder, width }) {
  return (
    <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      className="px-2.5 py-1.5 rounded-md text-xs outline-none"
      style={{
        width, background: 'var(--surface-page)', border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }} />
  )
}

function Drop({ value, onChange, options, title }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} title={title}
      className="px-2 py-1.5 rounded-md text-xs outline-none"
      style={{
        background: 'var(--surface-page)', border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }}>
      {options.map((o) => <option key={o.k} value={o.k}>{o.v}</option>)}
    </select>
  )
}

function DateInput({ value, onChange }) {
  return (
    <input type="datetime-local" value={value} onChange={(e) => onChange(e.target.value)}
      className="px-2 py-1.5 rounded-md text-xs outline-none"
      style={{
        background: 'var(--surface-page)', border: '1px solid var(--border)',
        color: 'var(--text-primary)', colorScheme: 'light dark',
      }} />
  )
}

function Btn({ children, onClick, danger, dashed }) {
  return (
    <button onClick={onClick} className="text-xs px-2.5 py-1.5 rounded-md"
      style={{
        border: `1px ${dashed ? 'dashed' : 'solid'} var(--border)`,
        color: danger ? 'var(--status-critical)' : 'var(--text-secondary)',
      }}>
      {children}
    </button>
  )
}

/** Local wall-clock time with milliseconds — enough to line up two sources. */
function clock(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '--:--:--'
  const pad = (n, w = 2) => String(n).padStart(w, '0')
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:`
    + `${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`
}
