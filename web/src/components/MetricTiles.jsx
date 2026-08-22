/**
 * Stat tiles — a single headline number each, so no chart is warranted.
 * Values use proportional figures at display size; units stay recessive.
 */
export default function MetricTiles({ weather }) {
  if (!weather?.current) return null
  const { current, location } = weather

  const tiles = [
    { label: 'Humidity', value: fmt(current.humidity), unit: '%' },
    {
      label: 'Wind',
      value: fmt(current.wind_mph),
      unit: 'mph',
      hint: [current.wind_dir, current.wind_gust_mph ? `gusts ${fmt(current.wind_gust_mph)}` : null]
        .filter(Boolean).join(' · '),
    },
    { label: 'Dew point', value: fmt(current.dew_point_f), unit: '°F' },
    { label: 'Pressure', value: fmt(current.pressure_hpa), unit: 'hPa' },
    { label: 'Visibility', value: fmt(current.visibility_mi), unit: 'mi' },
    { label: 'UV index', value: fmt(current.uv), unit: '' },
    { label: 'Elevation', value: fmt(location?.elevation_ft), unit: 'ft' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3">
      {tiles.map((t) => (
        <div key={t.label} className="card p-3">
          <div className="text-xs muted">{t.label}</div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-semibold leading-none">{t.value}</span>
            {t.unit && <span className="text-xs muted">{t.unit}</span>}
          </div>
          {t.hint && <div className="text-xs muted mt-1 truncate">{t.hint}</div>}
        </div>
      ))}
    </div>
  )
}

const fmt = (v) => (v == null ? '—' : Math.round(v).toString())
