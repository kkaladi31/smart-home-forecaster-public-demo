# Smart-Home Forecaster — web front-end

React 19 + Vite 8 + Tailwind 4. This is the demo UI: a live weather dashboard beside a chat
panel that streams the agent's tool calls as it works.

```bash
npm install     # first time only
npm run dev     # http://localhost:5173
```

The API must be running separately (`python -m uvicorn api.main:app --port 8000` from the
project root). On Windows PowerShell run the two in separate windows — `&&` is a parser
error in PowerShell 5.1, so `cd web && npm run dev` silently never starts.

Sign in with `demo` / `forecaster`.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with HMR on :5173. Use while editing. |
| `npm run build` | Production bundle into `dist/`. |
| `npm run preview` | Serves the built bundle on :4173. Loads faster than dev, but it is a **static snapshot** — rebuild after any change or you are looking at old code. |
| `npm run lint` | Oxlint. |

## The API is called directly, not through the dev proxy

`src/api.js` sends every request to `apiUrl(path)`, where the base comes from
`VITE_API_BASE` (default `http://127.0.0.1:8000`). FastAPI sets CORS for the :5173 and :4173
origins, so this works with no extra configuration.

This is deliberate and worth knowing before you "simplify" it back to a relative path.
Vite's `/api` proxy is the conventional arrangement and it *was* used originally, but on the
development machine **~20% of requests through the Vite 8 proxy hung indefinitely** — no
error, no retry, no timeout — while 20/20 of the identical calls made straight to uvicorn
returned in 28–63 ms. It presented as "the app randomly takes minutes to load", and it was
hard to trace because a hung request produces no server-side log line at all (FastAPI's
timing middleware records *after* the handler returns, so a hang looks like a request that
never arrived).

`vite.config.js` still defines the proxy for both `server` and `preview`, with keep-alive
disabled and explicit timeouts, so the relative-path setup still works if you want it: set
`VITE_API_BASE=""` in a `.env` file. That is also the right setting if the built bundle is
ever served by FastAPI itself, since everything is then same-origin.

## Layout

```
src/
  App.jsx              tabs, auth, home switching, run-mode switching, dashboard state
  api.js               fetch + SSE client; apiUrl() base lives here
  logbus.js            client-side event log mirroring the backend's event shape
  components/
    Sidebar.jsx        home switcher, persona, Run mode card, profile, floor plan
    Chat.jsx           streamed answers + live tool trace
    LogsPanel.jsx      Logs tab (frontend + backend events)
    WeatherPanel.jsx   forecast, TempChart, hazards, air quality, map
```

Two pieces of state handling are load-bearing and should not be removed casually:

- **`guard()` in `App.jsx`** routes any 401 to the login screen. The API keeps sessions in
  memory, so a server restart invalidates the tab's token; the original `.catch(() => {})`
  swallowed that and left the app signed in but permanently empty.
- **`reqSeq` in `App.jsx`** ensures only the newest dashboard response may write state.
  Switching homes while a request is in flight otherwise lets the older response land last
  and leaves the panel showing the home you just navigated away from.
