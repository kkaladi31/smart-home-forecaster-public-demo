// Thin client for the FastAPI backend.
// The demo token lives in sessionStorage (not localStorage) so it disappears when
// the tab closes — the auth flow is a demo, and nothing personal is persisted.
//
// Every call is timed into the frontend log (logbus.js) so the Logs tab can show
// how much of a slow response was the network and how much was the agent.

import { log } from './logbus'

// Talk to uvicorn directly instead of through Vite's dev proxy.
//
// Measured on this machine: ~20% of requests sent through the proxy hung
// indefinitely (60s+, no error), while 20/20 of the same calls made straight to
// :8000 returned in 28-63ms. That proxy stall — not the agent, not the network —
// was the "app takes forever to load" bug. The API sets CORS for the dev origins,
// so a direct call needs no other change.
//
// Override with VITE_API_BASE (set it to "" when the built bundle is served by
// FastAPI itself, which makes every call same-origin).
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
export const apiUrl = (path) => `${API_BASE}${path}`

const TOKEN_KEY = 'shf_token'

export const getToken = () => sessionStorage.getItem(TOKEN_KEY)
export const setToken = (t) => sessionStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => sessionStorage.removeItem(TOKEN_KEY)

function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

// Polling the log endpoint must not itself be logged, or the Logs tab would fill
// with the noise of its own refreshes.
const isSelfPoll = (path) => path.startsWith('/api/logs')

async function request(path, options = {}) {
  const method = options.method ?? 'GET'
  const t0 = performance.now()
  if (!isSelfPoll(path)) log('api', 'api.request', `${method} ${path}`, { level: 'debug' })

  let res
  try {
    res = await fetch(apiUrl(path), { ...options, headers: authHeaders(options.headers) })
  } catch (err) {
    log('api', 'api.error', `${method} ${path} — network error`, {
      level: 'error', durationMs: performance.now() - t0, data: { error: String(err) },
    })
    throw new Error(`Could not reach the API (${err.message}). Is uvicorn running?`)
  }

  const durationMs = performance.now() - t0
  if (!isSelfPoll(path)) {
    log('api', 'api.response', `${method} ${path} → ${res.status}`, {
      level: res.ok ? 'info' : 'error', durationMs, data: { status: res.status },
    })
  }

  if (res.status === 401) {
    clearToken()
    throw new Error('unauthenticated')
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

const get = (path) => request(path)
const post = (path, body) =>
  request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })

export const getStatus = () => get('/api/status')
export const getClientConfig = () => get('/api/client-config')
export const getHomes = () => get('/api/homes')
export function getProfile(homeId) {
  return get(`/api/profile${homeId ? `?home_id=${encodeURIComponent(homeId)}` : ''}`)
}
export const getMode = () => get('/api/mode')
export const setMode = (demo) => post('/api/mode', { demo })
export const getConversations = () => get('/api/conversations')
export const getConversation = (threadId) =>
  get(`/api/conversations/${encodeURIComponent(threadId)}`)
// Returns { trade, trade_label, source, area, notes, results, withheld }.
// `withheld` carries the professionals the licence gate refused, each with the
// reason — showing them is the point, not a leak. `trade` accepts a trade name
// or a whole natural-language need ("my water heater is leaking").
export function getContractors(trade, homeId) {
  const params = new URLSearchParams()
  if (trade) params.set('trade', trade)
  if (homeId) params.set('home_id', homeId)
  const qs = params.toString()
  return get(`/api/contractors${qs ? `?${qs}` : ''}`)
}
export function suggestPlaces(q, near) {
  const params = new URLSearchParams({ q })
  if (near?.lat != null && near?.lon != null) {
    params.set('lat', near.lat)
    params.set('lon', near.lon)
  }
  return get(`/api/geocode/suggest?${params.toString()}`)
}
export const reverseGeocode = (lat, lon) =>
  get(`/api/reverse-geocode?lat=${lat}&lon=${lon}`)

// --- logs and maintenance ---------------------------------------------------
export function getBackendLogs({ since = 0, group, level, limit = 500 } = {}) {
  const params = new URLSearchParams({ since, limit })
  if (group) params.set('group', group)
  if (level) params.set('level', level)
  return get(`/api/logs?${params.toString()}`)
}

export const getLogStats = () => get('/api/logs/stats')
export const clearBackendLogs = () => request('/api/logs', { method: 'DELETE' })

export const getAdminStats = () => get('/api/admin/stats')
export const clearChatThread = (threadId) =>
  post('/api/admin/clear-chat', { thread_id: threadId })
export const clearAllMemory = () => post('/api/admin/clear-memory')
export const clearCaches = () => post('/api/admin/clear-cache')

export function getWeather({ lat, lon, address, homeId }) {
  const params = new URLSearchParams()
  if (lat != null && lon != null) {
    params.set('lat', lat)
    params.set('lon', lon)
  } else if (address) {
    params.set('address', address)
  } else if (homeId) {
    // No coordinates and no address: fall back to the selected home rather than
    // the primary one, or switching homes would leave the panel on the old city.
    params.set('home_id', homeId)
  }
  const qs = params.toString()
  return get(`/api/weather${qs ? `?${qs}` : ''}`)
}

export function getDashboard({ lat, lon, address, hours = 48, homeId }) {
  const params = new URLSearchParams()
  if (lat != null && lon != null) {
    params.set('lat', lat)
    params.set('lon', lon)
  } else if (address) {
    params.set('address', address)
  } else if (homeId) {
    params.set('home_id', homeId)
  }
  params.set('hours', hours)
  return get(`/api/dashboard?${params.toString()}`)
}

// Kept off `request()` on purpose: a 401 here means "wrong password", not "your
// session expired", and it must not clear a token or surface as 'unauthenticated'.
export async function login(username, password) {
  const t0 = performance.now()
  const res = await fetch(apiUrl('/api/auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  log('api', 'auth.login', `Sign-in ${res.ok ? 'succeeded' : 'rejected'}`, {
    level: res.ok ? 'info' : 'warn', durationMs: performance.now() - t0,
    data: { status: res.status },
  })
  if (!res.ok) throw new Error('Invalid credentials')
  const data = await res.json()
  setToken(data.token)
  return data
}

export async function logout() {
  try {
    await fetch(apiUrl('/api/auth/logout'), { method: 'POST', headers: authHeaders() })
    log('api', 'auth.logout', 'Signed out')
  } finally {
    clearToken()
  }
}

/**
 * Stream one agent turn. `onEvent` is called for every server event as it
 * arrives (guardrail / memory / tool_call / tool_result / answer / done), which
 * is what lets the UI show the agent working instead of a blank spinner.
 * Returns an abort function.
 */
export function streamChat({ message, threadId, persona, location, homeId }, onEvent) {
  const controller = new AbortController()
  const t0 = performance.now()
  const since = () => performance.now() - t0
  let firstByteAt = null

  log('stream', 'chat.start', `Asking: ${message.slice(0, 80)}`, {
    data: { thread_id: threadId, persona, location, home_id: homeId, chars: message.length },
  })

  ;(async () => {
    let count = 0
    try {
      const res = await fetch(apiUrl('/api/chat/stream'), {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          message, thread_id: threadId, persona, location, home_id: homeId,
        }),
        signal: controller.signal,
      })
      log('stream', 'chat.connected', `Stream open (${res.status})`, {
        durationMs: since(), data: { status: res.status },
      })
      if (!res.ok || !res.body) throw new Error(`Stream failed (${res.status})`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE frames are separated by a blank line.
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''
        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data: '))
          if (!line) continue
          let ev
          try {
            ev = JSON.parse(line.slice(6))
          } catch {
            log('stream', 'frame.malformed', 'Dropped an unparseable SSE frame',
              { level: 'warn' })
            continue
          }
          count += 1
          if (firstByteAt === null) {
            firstByteAt = since()
            log('stream', 'chat.first_event', `First event after ${(firstByteAt / 1000).toFixed(1)}s`,
              { durationMs: firstByteAt, data: { type: ev.type } })
          }
          // Every server event, timestamped from the moment we asked — this is
          // the record that shows which step ate the wall-clock time.
          log('stream', `event.${ev.type}`, describe(ev), {
            level: ev.type === 'error' ? 'error' : 'debug',
            durationMs: since(),
            data: ev,
          })
          onEvent(ev)
        }
      }
      log('stream', 'chat.end', `Answer complete in ${(since() / 1000).toFixed(1)}s`, {
        durationMs: since(), data: { events: count },
      })
    } catch (err) {
      if (err.name === 'AbortError') {
        log('stream', 'chat.aborted', 'Stopped by the user', { level: 'warn', durationMs: since() })
        return
      }
      log('stream', 'chat.error', `Stream failed: ${err.message}`, {
        level: 'error', durationMs: since(),
      })
      onEvent({ type: 'error', content: err.message })
      onEvent({ type: 'done' })
    }
  })()

  return () => controller.abort()
}

/** One-line description of a server event, for the log row. */
function describe(ev) {
  switch (ev.type) {
    case 'tool_call': return `Tool call: ${ev.name}`
    case 'tool_result': return `Tool result: ${ev.name}`
    case 'llm_turn': return `Model turn ${ev.turn}`
    case 'guardrail': return 'Safety guardrail applied'
    case 'memory': return `Recalled ${(ev.recalled ?? []).length} past interaction(s)`
    case 'answer': return ev.cached ? `Answer (${ev.cache_source} cache hit)` : 'Answer received'
    case 'error': return `Error: ${ev.content}`
    case 'done': return ev.cached ? 'Done (served from cache)' : 'Done'
    default: return ev.type
  }
}
