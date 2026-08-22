/**
 * Browser-side event log — the frontend half of the Logs tab.
 *
 * The backend has its own recorder (telemetry.py). This one deliberately stays
 * in the browser rather than posting every click to the API: the interesting
 * frontend events happen *while* a 30-second answer is streaming, and adding a
 * request per event would both distort what it is measuring and compete with the
 * stream for connections. The Logs tab shows the two sources side by side and
 * merges them for the combined timeline.
 *
 * Event shape matches telemetry.py exactly, so merged rows need no translation:
 *   { seq, ts, t_ms, source, group, event, level, message, duration_ms?, data? }
 */

export const MAX_EVENTS = 1000

// Same groups the backend advertises for the frontend source.
export const FRONTEND_GROUPS = ['ui', 'api', 'stream', 'render', 'error']

const events = []
const listeners = new Set()
let seq = 0
let dropped = 0
const start = performance.now()

/** Truncate long strings so one big payload can't dominate the buffer. */
function shrink(data, limit = 600) {
  if (!data) return undefined
  const out = {}
  for (const [key, value] of Object.entries(data).slice(0, 24)) {
    out[key] =
      typeof value === 'string' && value.length > limit
        ? `${value.slice(0, limit)}… (+${value.length - limit} chars)`
        : value
  }
  return out
}

/**
 * Record one frontend event.
 * @param {string} group one of FRONTEND_GROUPS
 * @param {string} event dotted name, e.g. 'api.response'
 * @param {string} message the human-readable line shown in the log
 */
export function log(group, event, message, { level = 'info', durationMs, data } = {}) {
  const entry = {
    seq: ++seq,
    ts: new Date().toISOString(),
    t_ms: Math.round((performance.now() - start) * 10) / 10,
    source: 'frontend',
    group,
    event,
    level,
    message,
  }
  if (durationMs != null) entry.duration_ms = Math.round(durationMs * 10) / 10
  const trimmed = shrink(data)
  if (trimmed) entry.data = trimmed

  events.push(entry)
  if (events.length > MAX_EVENTS) {
    events.shift()
    dropped += 1
  }
  for (const fn of listeners) {
    try {
      fn(entry)
    } catch {
      /* a broken listener must not break logging */
    }
  }
  return entry
}

/** Time an async operation, logging start and end (or the failure). */
export async function timed(group, event, message, fn, data) {
  const t0 = performance.now()
  try {
    const result = await fn()
    log(group, `${event}.end`, message, { durationMs: performance.now() - t0, data })
    return result
  } catch (err) {
    log(group, `${event}.error`, `${message} failed: ${err.message}`, {
      level: 'error',
      durationMs: performance.now() - t0,
      data: { ...data, error: String(err) },
    })
    throw err
  }
}

export const snapshot = () => events.slice()
export const stats = () => ({ total: events.length, dropped, capacity: MAX_EVENTS })

export function clear() {
  const removed = events.length
  events.length = 0
  dropped = 0
  seq = 0
  log('ui', 'logs.cleared', `Frontend log cleared (${removed} events)`)
  return removed
}

/** Subscribe to new events. Returns an unsubscribe function. */
export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

/**
 * Catch what React never sees: uncaught exceptions and rejected promises.
 * Installed once from main.jsx.
 */
export function installGlobalHandlers() {
  window.addEventListener('error', (e) => {
    log('error', 'window.error', e.message || 'Uncaught error', {
      level: 'error',
      data: { source: e.filename, line: e.lineno, column: e.colno },
    })
  })
  window.addEventListener('unhandledrejection', (e) => {
    log('error', 'promise.rejected', String(e.reason?.message ?? e.reason), {
      level: 'error',
    })
  })
  log('render', 'app.start', 'Frontend started', {
    data: { url: window.location.href, userAgent: navigator.userAgent },
  })
}
