/**
 * Air quality and pollen.
 *
 * AQI is always reported on the **US EPA scale** (higher = worse) regardless of
 * which provider answered — Google's default Universal AQI is inverted, so the
 * backend explicitly requests the EPA index to keep one consistent meaning.
 *
 * Pollen is shown only where the provider actually has readings. Many US
 * locations return none, and saying so is better than implying "zero pollen".
 */
const STATUS_COLOR = {
  good: 'var(--status-good)',
  moderate: 'var(--status-warning)',
  high: 'var(--status-serious)',
  severe: 'var(--status-critical)',
  none: 'var(--text-muted)',
}

export default function AirQualityCard({ weather }) {
  const air = weather?.air_quality
  const pollen = weather?.pollen
  if (!air) return null

  if (!air.available) {
    return (
      <div className="card p-4">
        <h3 className="text-sm font-semibold mb-1">Air quality</h3>
        <p className="text-xs muted">Air-quality data isn’t available for this location.</p>
      </div>
    )
  }

  const status = air.category?.status ?? 'none'
  const pollutants = normalisePollutants(air)

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-semibold">Air quality</h3>
        <span className="text-xs muted">US EPA AQI</span>
      </div>

      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className="grid place-items-center rounded-full text-sm font-bold text-white shrink-0"
          style={{ width: 44, height: 44, background: STATUS_COLOR[status] }}
        >
          {air.aqi ?? '—'}
        </span>
        <div className="min-w-0">
          {/* Text label carries the meaning; colour only reinforces it. */}
          <div className="text-sm font-semibold">{air.category?.label ?? 'Unknown'}</div>
          {air.dominant_pollutant && (
            <div className="text-xs muted">Dominant: {air.dominant_pollutant.toUpperCase()}</div>
          )}
        </div>
      </div>

      {air.advice && <p className="text-xs secondary mt-2 line-clamp-3">{air.advice}</p>}

      {pollutants.length > 0 && (
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1">
          {pollutants.map((p) => (
            <div key={p.name} className="text-xs">
              <span className="muted">{p.name}</span>{' '}
              <span className="secondary tabular">{p.value}</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
        <h4 className="text-xs font-semibold mb-1">Pollen</h4>
        {pollen?.available ? (
          <div className="grid grid-cols-3 gap-2">
            {pollen.types.map((t) => (
              <div key={t.code} className="text-xs">
                <div className="muted">{t.name}</div>
                <div className="secondary">{t.category ?? '—'}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs muted">
            {pollen?.note ?? weather?.air_quality?.pollen_note ??
              'Pollen readings aren’t published for this location.'}
          </p>
        )}
      </div>
    </div>
  )
}

/** Both providers report pollutants; flatten either shape to name/value pairs. */
function normalisePollutants(air) {
  if (air.pollutants) {
    return Object.values(air.pollutants)
      .filter((p) => p.value != null)
      .map((p) => ({ name: p.name, value: `${round(p.value)} ${short(p.units)}` }))
  }
  return [
    ['PM2.5', air.pm2_5, 'µg/m³'],
    ['PM10', air.pm10, 'µg/m³'],
    ['O3', air.ozone, 'µg/m³'],
    ['NO2', air.no2, 'µg/m³'],
    ['CO', air.co, 'µg/m³'],
  ]
    .filter(([, v]) => v != null)
    .map(([name, v, unit]) => ({ name, value: `${round(v)} ${unit}` }))
}

const round = (v) => (typeof v === 'number' ? Math.round(v * 10) / 10 : v)
const short = (units) =>
  units === 'PARTS_PER_BILLION' ? 'ppb'
    : units === 'MICROGRAMS_PER_CUBIC_METER' ? 'µg/m³'
      : ''
