import { useEffect, useState } from 'react'
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Leaflet's default marker icons are resolved relative to the CSS by default,
// which breaks under a bundler. Point them at the CDN-free bundled assets.
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

// Leaflet derives its icon URLs from the CSS path unless this getter is removed
// first — without the delete, mergeOptions is ignored and the marker renders as
// a broken image under a bundler.
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

function ClickHandler({ onPick }) {
  useMapEvents({
    click(e) {
      onPick({ lat: e.latlng.lat, lon: e.latlng.lng })
    },
  })
  return null
}

function Recenter({ lat, lon }) {
  const map = useMap()
  useEffect(() => {
    if (lat != null && lon != null) map.setView([lat, lon], map.getZoom(), { animate: true })
  }, [lat, lon, map])
  return null
}

/**
 * Precipitation radar from RainViewer — free, no API key, no billing account.
 * Their index endpoint lists recent radar frames; we overlay the most recent one.
 */
function useRadarFrame(enabled) {
  const [frame, setFrame] = useState(null)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('https://api.rainviewer.com/public/weather-maps.json')
        const data = await res.json()
        const past = data?.radar?.past ?? []
        const latest = past[past.length - 1]
        if (!cancelled && latest) setFrame(`${data.host}${latest.path}`)
      } catch {
        /* radar is optional — the map still works without it */
      }
    })()
    return () => { cancelled = true }
  }, [enabled])

  return frame
}

/** Click anywhere to assess that spot. Coordinates feed the same tools the agent uses. */
export default function MapPicker({ lat, lon, onPick }) {
  const [showRadar, setShowRadar] = useState(true)
  const radarFrame = useRadarFrame(showRadar)

  if (lat == null || lon == null) return null

  return (
    <div className="card p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
        <h3 className="text-sm font-semibold">Location &amp; radar</h3>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs secondary cursor-pointer">
            <input
              type="checkbox"
              checked={showRadar}
              onChange={(e) => setShowRadar(e.target.checked)}
            />
            Precipitation radar
          </label>
          <span className="text-xs muted">Click the map to assess a spot</span>
        </div>
      </div>

      <div style={{ height: 300 }}>
        <MapContainer center={[lat, lon]} zoom={9} scrollWheelZoom style={{ height: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {showRadar && radarFrame && (
            <TileLayer
              // 256px tiles, colour scheme 2, smoothed + snow shown.
              url={`${radarFrame}/256/{z}/{x}/{y}/2/1_1.png`}
              opacity={0.6}
              attribution='Radar &copy; <a href="https://www.rainviewer.com/">RainViewer</a>'
            />
          )}
          <Marker position={[lat, lon]} />
          <ClickHandler onPick={onPick} />
          <Recenter lat={lat} lon={lon} />
        </MapContainer>
      </div>

      {showRadar && !radarFrame && (
        <p className="text-xs muted mt-2">Loading radar…</p>
      )}
    </div>
  )
}
