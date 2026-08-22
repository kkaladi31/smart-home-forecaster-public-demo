import GoogleMapPicker from './GoogleMapPicker'
import MapPicker from './MapPicker'

/**
 * Picks the map implementation at runtime.
 *
 * With a Google Maps key configured we use Google Dynamic Maps (satellite,
 * terrain, Street View). Without one we fall back to Leaflet/OpenStreetMap,
 * which needs no key and no billing account — so the project stays runnable by
 * anyone who clones the repo. Both carry the same RainViewer radar overlay.
 */
export default function MapPanel({ lat, lon, apiKey, onPick }) {
  if (apiKey) {
    return <GoogleMapPicker lat={lat} lon={lon} apiKey={apiKey} onPick={onPick} />
  }
  return <MapPicker lat={lat} lon={lon} onPick={onPick} />
}
