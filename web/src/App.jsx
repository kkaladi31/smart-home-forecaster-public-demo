import { useCallback, useEffect, useRef, useState } from 'react'
import {
  clearToken, getClientConfig, getHomes, getMode, getProfile, getStatus, getToken, getWeather,
  logout, setMode,
} from './api'
import { log } from './logbus'
import Login from './components/Login'
import Sidebar from './components/Sidebar'
import HazardAlert from './components/HazardAlert'
import LocationBar from './components/LocationBar'
import MapPanel from './components/MapPanel'
import MetricTiles from './components/MetricTiles'
import CurrentConditions from './components/CurrentConditions'
import WeatherPanel from './components/WeatherPanel'
import AirQualityCard from './components/AirQualityCard'
import HazardPanel from './components/HazardPanel'
import ProsPanel from './components/ProsPanel'
import Chat from './components/Chat'
import LogsPanel from './components/LogsPanel'
import ErrorBoundary from './components/ErrorBoundary'
import { fmtDuration } from './utils/duration'

const newThreadId = () => `web-${Math.random().toString(36).slice(2, 12)}`

const TABS = [
  { key: 'app', label: 'Dashboard' },
  { key: 'logs', label: 'Logs' },
]

export default function App() {
  const [authed, setAuthed] = useState(Boolean(getToken()))
  const [status, setStatus] = useState(null)
  const [homes, setHomes] = useState([])
  const [homeId, setHomeId] = useState(null)
  const [profile, setProfile] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [persona, setPersona] = useState('owner')
  const [threadId, setThreadId] = useState(newThreadId)
  const [mapsKey, setMapsKey] = useState(null)
  const [mode, setModeState] = useState(null)
  const [modeBusy, setModeBusy] = useState(false)
  const [tab, setTab] = useState('app')
  const [chatBusy, setChatBusy] = useState(false)
  const [chatElapsed, setChatElapsed] = useState(0)
  const chatStartedRef = useRef(0)
  // Set by a hazard notification's "Ask the assistant" button. The chat consumes
  // it into its input rather than sending it, so the user still chooses to ask.
  const [chatPrefill, setChatPrefill] = useState('')

  // Looking somewhere OTHER than the active home is deliberately opt-in.
  //
  // The dashboard and the policy answers are two halves of one screen, and only
  // one of them follows this toggle: retrieval stays bound to the active home no
  // matter where the map is pointed. Letting a stray map click silently repoint
  // the weather would put a forecast for one place beside covenants for another,
  // with nothing on screen admitting it — which is R7 (assesses the wrong
  // property) arriving through the UI instead of through the geocoder.
  //
  // Applies to owners and renters alike; being away from home is not a tenure
  // question.
  const [awayMode, setAwayMode] = useState(false)

  // Mirror the chat's timer in the header so the wait is still visible from the
  // Logs tab — which is exactly where someone goes to find out why it is slow.
  useEffect(() => {
    if (!chatBusy) return undefined
    chatStartedRef.current = performance.now()
    setChatElapsed(0)
    const id = setInterval(() => setChatElapsed(performance.now() - chatStartedRef.current), 200)
    return () => clearInterval(id)
  }, [chatBusy])

  useEffect(() => {
    getStatus().then(setStatus).catch(() => {})
  }, [])

  // A dead token must never fail silently. The API keeps its sessions in memory,
  // so every restart of uvicorn — including every --reload triggered by an edit —
  // invalidates the token this tab is holding. Swallowing that 401 left the app
  // apparently signed in but permanently empty, with no way back except a manual
  // refresh. Anything that can 401 goes through here.
  const guard = useCallback((promise) => promise.catch((e) => {
    if (e.message === 'unauthenticated') setAuthed(false)
    return null
  }), [])

  // Only the most recent dashboard request may write to state. Two requests are
  // in flight whenever the home is switched while the previous one is still
  // loading, and without this the slower (older) response lands last and leaves
  // the panel showing the home you just navigated away from.
  const reqSeq = useRef(0)

  // The home selected right now, readable from inside an async callback that
  // captured an older value.
  const currentHomeRef = useRef(null)
  currentHomeRef.current = homeId

  // `homeId` is in the dependency list so that a bare loadDashboard() — no
  // coordinates, no address — falls back to the *selected* home. Without it the
  // backend would default to the primary home and switching to the Texas house
  // would leave the dashboard sitting on the primary one.
  const loadDashboard = useCallback(async (target) => {
    const seq = ++reqSeq.current
    const stale = () => seq !== reqSeq.current
    setLoading(true)
    setError('')
    try {
      const data = await getWeather({ homeId, ...(target ?? {}) })
      if (stale()) return
      setDashboard(data)
    } catch (e) {
      if (stale()) return
      if (e.message === 'unauthenticated') setAuthed(false)
      else setError(e.message)
    } finally {
      if (!stale()) setLoading(false)
    }
  }, [homeId])

  // On sign-in, load the list of saved homes and select the primary one.
  useEffect(() => {
    if (!authed) return
    guard(getHomes().then(({ homes: list }) => {
      setHomes(list)
      setHomeId((current) => current ?? list.find((h) => h.is_primary)?.home_id ?? list[0]?.home_id)
    }))
    guard(getClientConfig().then((c) => setMapsKey(c.google_maps_key)))
    guard(getMode().then(setModeState))
  }, [authed, guard])

  // Flip between the full stack and the free/open one. Everything downstream of
  // the switch has to be refetched: the maps key is withheld in demo mode, the
  // dashboard was rendered by a provider that is no longer active, and the
  // status card names the model that just changed.
  async function switchMode(demo) {
    if (modeBusy || demo === mode?.demo) return
    setModeBusy(true)
    try {
      const next = await setMode(demo)
      setModeState(next)
      log('ui', 'mode.switch', `Switched to ${next.mode} mode`, { data: next })
      const cfg = await guard(getClientConfig())
      if (cfg) setMapsKey(cfg.google_maps_key)
      guard(getStatus().then(setStatus))
    } catch (e) {
      setError(`Could not switch mode: ${e.message}`)
      setModeBusy(false)
      return
    }
    // The switch itself is complete once the mode and the maps key are updated,
    // so release the toggle here. Awaiting the dashboard refetch as well kept the
    // buttons disabled and the card reading "Switching…" for several more
    // seconds, which read as a frozen app.
    setModeBusy(false)
    // Deliberately NOT clearing the dashboard here. Unlike a home switch, the
    // location has not changed — only which provider supplies it — so the
    // readings on screen are still true. Blanking them made the switch look
    // like a stall while the new provider was fetched cold.
    loadDashboard()
  }

  // Whenever the selected home changes, reload its profile and re-centre the
  // dashboard on it. Both are the home's, not the user's, so neither survives a switch.
  useEffect(() => {
    if (!authed || !homeId) return
    // Drop the previous home's readings before fetching the new ones. Leaving
    // them on screen made a switch look like it had silently failed: the sidebar
    // highlighted the new home while every tile still showed the old one's
    // weather, with nothing to say a request was in progress.
    setDashboard(null)
    // The effect below re-centres on the new home, so leaving the toggle On
    // would leave the control contradicting the screen.
    setAwayMode(false)
    guard(getProfile(homeId).then((p) => {
      // Ignore a profile that arrives after the user has moved on again.
      if (homeId === currentHomeRef.current) setProfile(p)
    }))
    loadDashboard()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authed, homeId])

  function switchHome(nextId) {
    if (nextId === homeId) return
    const next = homes.find((h) => h.home_id === nextId)
    log('ui', 'home.switch', `Switched to ${next?.label ?? nextId}`,
      { data: { home_id: nextId } })
    // A new thread as well: the agent's conversation memory holds the previous
    // home's answers, and a follow-up like "what about the fence?" resolved
    // against those would silently answer for the wrong house.
    setThreadId(newThreadId())
    setHomeId(nextId)
  }

  async function handleLogout() {
    await logout()
    clearToken()
    setAuthed(false)
    setDashboard(null)
    setProfile(null)
    setHomes([])
    setHomeId(null)
  }

  function switchTab(key) {
    setTab(key)
    log('ui', 'tab.switch', `Switched to the ${key} tab`)
  }

  // Turning the toggle OFF must actually go home, not merely stop accepting new
  // picks — otherwise the dashboard sits on the last place you looked while the
  // control says you are home, which is worse than never having the toggle.
  function toggleAway(next) {
    setAwayMode(next)
    log('ui', 'location.away', next
      ? 'Away-from-home enabled — the dashboard can be pointed elsewhere'
      : 'Away-from-home disabled — dashboard returned to the active home')
    if (!next) loadDashboard()
  }

  // A hazard notification hands its question to the chat. The tab switch is part
  // of the action, not a nicety: the chat column is `hidden` on the Logs tab, so
  // prefilling without switching would drop the question somewhere invisible.
  function askAboutHazard(question) {
    setChatPrefill(question)
    setTab('app')
    log('ui', 'hazard.ask', `Hazard notification handed a question to the chat: ${question}`)
  }

  if (!authed) {
    return <Login status={status} onSuccess={() => setAuthed(true)} />
  }

  const loc = dashboard?.location
  const onApp = tab === 'app'

  return (
    <div className="min-h-full p-4 lg:p-6">
      {/* Outside the tab columns on purpose. A hazard notification that is
          hidden because you happened to be on the Logs tab is not a
          notification. */}
      <ErrorBoundary label="Hazard notifications">
        <HazardAlert dashboard={dashboard} onAsk={askAboutHazard} />
      </ErrorBoundary>
      <div className="mx-auto w-full">
        {/* Tab bar */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => switchTab(t.key)}
                className="px-4 py-1.5 text-sm font-medium"
                style={{
                  background: tab === t.key ? 'var(--series-1)' : 'var(--surface-1)',
                  color: tab === t.key ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
          {chatBusy && (
            <button
              onClick={() => switchTab('app')}
              className="text-xs px-2 py-1 rounded-lg tabular"
              style={{ border: '1px solid var(--border)', color: 'var(--series-1)' }}
              title="An answer is being generated — click to go back to the chat"
            >
              <span className="pulse">●</span> answering… {fmtDuration(chatElapsed)}
            </button>
          )}
        </div>

        <div
          className={`grid gap-4 ${onApp
            // Both content columns are FRACTIONS, so the layout scales with the
            // monitor instead of pinning chat to a fixed 360-420px and dumping
            // every extra pixel into the dashboard.
            //
            // A first attempt gave chat `minmax(720px,840px)` — twice the old
            // fixed width — which was the wrong instrument: a hard 720px floor
            // crushed the dashboard to ~380px on a 1400px window. Doubling a
            // MINIMUM is not the same as doubling a SHARE. Ratios cannot crush
            // either side, because both shrink together.
            //
            // 2 : 1 gives chat a third of the content area — ~740px on a 2560px
            // monitor, still well over the old fixed 420px. It was 1.4 : 1 (a
            // 41.7% share) and is now 33.3%, which is that share reduced by
            // exactly a fifth.
            //
            // The 380px floor is deliberately NOT scaled with it: it is a
            // readability limit, not a proportion — below it a cited answer
            // containing a table stops being readable at any window size.
            ? 'lg:grid-cols-[260px_minmax(0,2fr)_minmax(380px,1fr)]'
            : 'lg:grid-cols-[260px_minmax(0,1fr)]'}`}
        >
          <Sidebar
            profile={profile}
            status={status}
            persona={persona}
            onPersona={setPersona}
            homes={homes}
            homeId={homeId}
            onHome={switchHome}
            onLogout={handleLogout}
            mode={mode}
            modeBusy={modeBusy}
            onMode={switchMode}
          />

          {/* Dashboard column — hidden rather than unmounted, so switching tabs
              does not refetch the weather or lose the map's position. */}
          <main className={`space-y-4 min-w-0 ${onApp ? '' : 'hidden'}`}>
            {/* Search and "use my location" are ALWAYS available. Typing an
                address is a deliberate act; clicking a map is something you do
                by accident while panning. Only the accidental one is gated. */}
            <LocationBar
              busy={loading}
              label={loc?.label}
              near={loc ? { lat: loc.latitude, lon: loc.longitude } : null}
              onPick={(target) => loadDashboard(target)}
            />

            <div className="card p-3 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <label htmlFor="away-toggle" className="text-sm font-medium">
                  Away from current home location
                </label>
                <p className="text-xs muted mt-0.5">
                  {awayMode
                    ? 'You can pick anywhere on the map. Weather follows it — home rules and documents still come from the active home.'
                    : 'Map clicks will not move the dashboard. Search and “use my location” still work.'}
                </p>
              </div>
              <button
                id="away-toggle"
                type="button"
                role="switch"
                aria-checked={awayMode}
                onClick={() => toggleAway(!awayMode)}
                className="shrink-0 text-xs px-2.5 py-1 rounded"
                style={{
                  border: '1px solid var(--border)',
                  background: awayMode ? 'var(--series-1)' : 'var(--surface-1)',
                  color: awayMode ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {awayMode ? 'On' : 'Off'}
              </button>
            </div>

            {error && (
              <div className="card p-3 text-sm" style={{ color: 'var(--status-critical)' }}>
                {error}
              </div>
            )}

            {loading && !dashboard && <Skeleton />}

            {dashboard && (
              <>
                <ErrorBoundary label="Current conditions">
                  <CurrentConditions weather={dashboard} />
                </ErrorBoundary>
                <ErrorBoundary label="Metrics">
                  <MetricTiles weather={dashboard} />
                </ErrorBoundary>
                <ErrorBoundary label="The forecast">
                  <WeatherPanel weather={dashboard} />
                </ErrorBoundary>
                <ErrorBoundary label="Hazards">
                  <HazardPanel dashboard={dashboard} />
                </ErrorBoundary>
                <ErrorBoundary label="Air quality">
                  <AirQualityCard weather={dashboard} />
                </ErrorBoundary>
                {/* Directly after the hazards and air quality, because "what is
                    the risk" and "who is licensed to fix it" are one thought.
                    Keyed on homeId so switching homes refetches rather than
                    leaving another house's contractors on screen. */}
                <ErrorBoundary label="Licensed professionals">
                  <ProsPanel key={homeId} homeId={homeId} />
                </ErrorBoundary>
                <ErrorBoundary label="The map">
                  <MapPanel
                    lat={loc?.latitude}
                    lon={loc?.longitude}
                    apiKey={mapsKey}
                    // The map stays visible and usable when home; what the
                    // toggle gates is its ability to REPOINT the dashboard.
                    // Passed as a FLAG rather than by withholding the callback —
                    // swapping onPick to undefined crashed the Leaflet handler
                    // and left it holding a stale closure afterwards.
                    onPick={(target) => loadDashboard(target)}
                    canPick={awayMode}
                  />
                </ErrorBoundary>
                {/* `source`, not `provider`. The API has always returned `source`
                    (e.g. "NWS (api.weather.gov)"); this read an undefined key, so
                    the provenance line rendered as a bare "Data:" with nothing
                    after it. Found by running the app and diffing the payload
                    against what the UI reads — no test asserts on rendered text,
                    and a blank is not an error anywhere. It matters more here
                    than it looks: naming the source is the whole provenance
                    claim this project makes about live data. */}
                <p className="text-xs muted px-1">
                  Data: {dashboard.source}
                  {dashboard.alerts_available ? ' · advisories from NWS' : ''}. This is general
                  guidance, not professional advice — for gas, electrical, flooding, or medical
                  emergencies contact 911, your utility, or a licensed professional.
                </p>
              </>
            )}
          </main>

          {/* Logs column */}
          {!onApp && (
            <div className="min-w-0">
              <ErrorBoundary label="The logs">
                <LogsPanel />
              </ErrorBoundary>
            </div>
          )}

          {/* Chat column — also hidden rather than unmounted: unmounting mid-answer
              would abort the stream and throw away a 30-second response. */}
          <div className={`lg:sticky lg:top-6 h-[70vh] lg:h-[calc(100vh-6rem)] ${onApp ? '' : 'hidden'}`}>
            <ErrorBoundary label="The chat">
              <Chat
                persona={persona}
                threadId={threadId}
                homeId={homeId}
                location={loc?.label}
                prefill={chatPrefill}
                onPrefillConsumed={() => setChatPrefill('')}
                onBusyChange={setChatBusy}
                onNewThread={() => setThreadId(newThreadId())}
                onResumeThread={(id) => setThreadId(id)}
              />
            </ErrorBoundary>
          </div>
        </div>
      </div>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => <div key={i} className="card h-20 pulse" />)}
      </div>
      <div className="card h-64 pulse" />
    </div>
  )
}
