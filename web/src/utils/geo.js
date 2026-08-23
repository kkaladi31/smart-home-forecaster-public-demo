/**
 * Coordinate normalisation, in one place because two map providers need it.
 *
 * Both Leaflet and Google keep "world copies" — pan east past the date line and
 * the map hands you longitude 200, or -267.43. Those are perfectly sensible
 * numbers for a scrolling canvas and meaningless to a weather API: Open-Meteo
 * answers `400 Bad Request` and the dashboard renders the raw failing URL at the
 * user, which is how this was found.
 *
 * Longitude WRAPS (it is cyclic — 190°E is 170°W, the same meridian).
 * Latitude CLAMPS (it is not — there is nothing north of the pole, and a map
 * click above the top edge should resolve to the pole rather than fold over onto
 * the other side of the planet).
 */
export function wrapLongitude(lon) {
  if (!Number.isFinite(lon)) return null
  // ((x % 360) + 360) % 360 avoids JS's negative-modulo result before shifting
  // the range to [-180, 180).
  const wrapped = (((lon + 180) % 360) + 360) % 360 - 180
  return wrapped
}

export function clampLatitude(lat) {
  if (!Number.isFinite(lat)) return null
  return Math.min(90, Math.max(-90, lat))
}

/** `{lat, lon}` guaranteed to be inside the ranges every provider accepts. */
export function normalizeLatLon(lat, lon) {
  const la = clampLatitude(lat)
  const lo = wrapLongitude(lon)
  if (la == null || lo == null) return null
  return { lat: la, lon: lo }
}
