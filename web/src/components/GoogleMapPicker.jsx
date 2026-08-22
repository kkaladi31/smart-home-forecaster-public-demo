import { useEffect, useRef, useState } from 'react'
// v2 of the loader removed the `Loader` class in favour of this functional API.
import { importLibrary, setOptions } from '@googlemaps/js-api-loader'

const RADAR_INDEX = 'https://api.rainviewer.com/public/weather-maps.json'

/**
 * Google Dynamic Maps with a precipitation-radar overlay.
 *
 * Radar tiles come from RainViewer (free, no key) and are mounted as a Google
 * `ImageMapType` — the Leaflet TileLayer equivalent — so switching basemaps did
 * not cost us the radar.
 *
 * The API key is fetched from the backend at runtime rather than baked into the
 * bundle. It still reaches the browser (unavoidable for the Maps JS API), so the
 * key must carry an HTTP-referrer restriction in the Google Cloud console.
 */
export default function GoogleMapPicker({ lat, lon, apiKey, onPick }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const markerRef = useRef(null)
  const radarRef = useRef(null)
  const onPickRef = useRef(onPick)

  const [showRadar, setShowRadar] = useState(true)
  const [mapType, setMapType] = useState('roadmap')
  const [radarFrame, setRadarFrame] = useState(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)

  // Keep the latest click handler without re-creating the map.
  useEffect(() => { onPickRef.current = onPick }, [onPick])

  // Latest radar frame.
  useEffect(() => {
    let cancelled = false
    fetch(RADAR_INDEX)
      .then((r) => r.json())
      .then((d) => {
        const past = d?.radar?.past ?? []
        const latest = past[past.length - 1]
        if (!cancelled && latest) setRadarFrame(`${d.host}${latest.path}`)
      })
      .catch(() => { /* radar is optional */ })
    return () => { cancelled = true }
  }, [])

  // Create the map once.
  useEffect(() => {
    if (!apiKey || !containerRef.current || mapRef.current) return
    let cancelled = false

    setOptions({ key: apiKey, v: 'weekly' })
    importLibrary('maps')
      .then(({ Map }) => {
        if (cancelled || !containerRef.current) return
        const map = new Map(containerRef.current, {
          center: { lat, lng: lon },
          zoom: 9,
          mapTypeControl: false,
          streetViewControl: true,
          fullscreenControl: true,
          clickableIcons: false,
        })
        map.addListener('click', (e) => {
          if (e.latLng) onPickRef.current?.({ lat: e.latLng.lat(), lon: e.latLng.lng() })
        })
        mapRef.current = map
        setReady(true)
      })
      .catch((e) => setError(e?.message ?? 'Google Maps failed to load'))

    return () => { cancelled = true }
  }, [apiKey, lat, lon])

  // Follow the selected location.
  useEffect(() => {
    const map = mapRef.current
    if (!map || lat == null || lon == null) return
    const position = { lat, lng: lon }
    map.panTo(position)
    if (markerRef.current) {
      markerRef.current.setPosition(position)
    } else if (window.google?.maps) {
      markerRef.current = new window.google.maps.Marker({ map, position })
    }
  }, [lat, lon, ready])

  // Mount / unmount the radar overlay.
  useEffect(() => {
    const map = mapRef.current
    const google = window.google
    if (!map || !google?.maps) return

    if (radarRef.current) {
      const index = map.overlayMapTypes.getArray().indexOf(radarRef.current)
      if (index > -1) map.overlayMapTypes.removeAt(index)
      radarRef.current = null
    }

    if (showRadar && radarFrame) {
      const layer = new google.maps.ImageMapType({
        getTileUrl: (coord, zoom) => `${radarFrame}/256/${zoom}/${coord.x}/${coord.y}/2/1_1.png`,
        tileSize: new google.maps.Size(256, 256),
        opacity: 0.6,
        name: 'Radar',
      })
      map.overlayMapTypes.push(layer)
      radarRef.current = layer
    }
  }, [showRadar, radarFrame, ready])

  useEffect(() => {
    if (mapRef.current) mapRef.current.setMapTypeId(mapType)
  }, [mapType, ready])

  if (error) {
    return (
      <div className="card p-4">
        <h3 className="text-sm font-semibold mb-1">Map unavailable</h3>
        <p className="text-xs secondary">{error}</p>
        <p className="text-xs muted mt-1">
          Check that the <strong>Maps JavaScript API</strong> is enabled for this key and that
          your referrer restriction allows <code>localhost:5173</code>.
        </p>
      </div>
    )
  }

  return (
    <div className="card p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <h3 className="text-sm font-semibold">Location &amp; radar</h3>
        <div className="flex items-center gap-3">
          <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {[['roadmap', 'Map'], ['hybrid', 'Satellite'], ['terrain', 'Terrain']].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setMapType(id)}
                className="px-2 py-1 text-xs font-medium"
                style={
                  mapType === id
                    ? { background: 'var(--series-1)', color: '#fff' }
                    : { background: 'transparent', color: 'var(--text-secondary)' }
                }
              >
                {label}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1.5 text-xs secondary cursor-pointer">
            <input type="checkbox" checked={showRadar} onChange={(e) => setShowRadar(e.target.checked)} />
            Radar
          </label>
        </div>
      </div>

      <div ref={containerRef} style={{ height: 320, borderRadius: 10, overflow: 'hidden' }} />

      <p className="text-xs muted mt-2">
        Click the map to assess a different spot
        {showRadar && radarFrame ? ' · radar © RainViewer' : ''}
      </p>
    </div>
  )
}
