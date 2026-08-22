import { useEffect, useRef, useState } from 'react'
import { suggestPlaces } from '../api'

/**
 * Three ways to set the location, in the order people reach for them:
 *   1. "Use my location" — browser geolocation, resolved to a real address.
 *   2. Type anything — street address, city, or landmark, with results biased
 *      toward wherever the map currently is (that bias is what makes an
 *      autocomplete feel right; without it "Maple St" matches another state).
 *   3. Click the map (see the map panel).
 */
export default function LocationBar({ onPick, busy, label, near }) {
  const [query, setQuery] = useState('')
  const [options, setOptions] = useState([])
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const [geoError, setGeoError] = useState('')
  const [locating, setLocating] = useState(false)
  const boxRef = useRef(null)

  // Debounced lookup so we aren't calling out on every keystroke.
  useEffect(() => {
    const text = query.trim()
    if (text.length < 3) {
      setOptions([])
      setOpen(false)
      return
    }
    const id = setTimeout(async () => {
      try {
        const { results } = await suggestPlaces(text, near)
        setOptions(results ?? [])
        setHighlight(0)
        setOpen((results ?? []).length > 0)
      } catch {
        setOptions([])
      }
    }, 250)
    return () => clearTimeout(id)
  }, [query, near?.lat, near?.lon])

  useEffect(() => {
    const onDocClick = (e) => {
      if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  function useMyLocation() {
    setGeoError('')
    if (!navigator.geolocation) {
      setGeoError('This browser does not support location access.')
      return
    }
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false)
        setQuery('')
        // Pass coordinates straight through — the backend reverse-geocodes them
        // to a street address for display.
        onPick({ lat: pos.coords.latitude, lon: pos.coords.longitude })
      },
      (err) => {
        setLocating(false)
        setGeoError(
          err.code === err.PERMISSION_DENIED
            ? 'Location blocked. Allow it in the address-bar icon, or search below.'
            : err.code === err.TIMEOUT
              ? 'Location timed out — try again or search below.'
              : 'Could not get your location — search below instead.',
        )
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    )
  }

  function choose(opt) {
    setQuery(opt.label)
    setOpen(false)
    // Autocomplete gives text, not coordinates; the backend geocodes it.
    onPick({ address: opt.label })
  }

  function onKeyDown(e) {
    if (!open || !options.length) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlight((h) => (h + 1) % options.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlight((h) => (h - 1 + options.length) % options.length)
    } else if (e.key === 'Enter') {
      e.preventDefault()
      choose(options[highlight])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  function submit(e) {
    e.preventDefault()
    if (open && options.length) choose(options[highlight])
    else if (query.trim()) onPick({ address: query.trim() })
  }

  return (
    <div className="card p-3">
      <div className="flex flex-col sm:flex-row gap-2">
        <button
          onClick={useMyLocation}
          disabled={busy || locating}
          className="shrink-0 px-3 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-50"
          style={{ background: 'var(--series-1)' }}
        >
          {locating ? 'Locating…' : '📍 Use my location'}
        </button>

        <form onSubmit={submit} className="relative flex-1" ref={boxRef}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            onFocus={() => options.length && setOpen(true)}
            placeholder="Search an address, city, or place…"
            autoComplete="off"
            className="w-full px-3 py-2 rounded-lg text-sm outline-none"
            style={{
              background: 'var(--surface-page)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
            }}
          />
          {open && options.length > 0 && (
            <ul
              className="absolute z-[1000] left-0 right-0 mt-1 card overflow-hidden"
              style={{ boxShadow: '0 8px 24px rgba(0,0,0,0.16)' }}
            >
              {options.map((o, i) => (
                <li key={o.place_id ?? o.label}>
                  <button
                    type="button"
                    onMouseEnter={() => setHighlight(i)}
                    onClick={() => choose(o)}
                    className="w-full text-left px-3 py-2"
                    style={{
                      background: i === highlight ? 'var(--surface-page)' : 'transparent',
                      borderBottom: i < options.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                  >
                    <span className="text-sm block truncate">📍 {o.primary ?? o.label}</span>
                    {o.secondary && (
                      <span className="text-xs muted block truncate">{o.secondary}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </form>
      </div>

      <div className="mt-2 text-xs">
        {geoError ? (
          <span style={{ color: 'var(--status-critical)' }}>{geoError}</span>
        ) : (
          <span className="muted">
            Showing: <span className="secondary">{label ?? '—'}</span>
          </span>
        )}
      </div>
    </div>
  )
}
