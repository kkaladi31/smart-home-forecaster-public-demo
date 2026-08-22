import {
  Area, AreaChart, CartesianGrid, Legend, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

/**
 * 48-hour temperature outlook.
 *
 * Form: change-over-time for two continuous series -> line/area chart.
 * Colour: categorical slots 1 (actual) and 2 (feels-like) from the validated
 * palette; identity is never colour-alone because a legend is always present.
 * Marks stay thin (2px strokes), the grid is recessive, and a 32F reference line
 * marks freezing so the freeze story is readable straight off the chart.
 */
export default function TempChart({ hourly, freezeLevel, title = '48-hour temperature outlook' }) {
  if (!hourly?.length) return null

  const data = hourly.map((h) => {
    const d = new Date(h.time)
    return {
      ...h,
      date: d,
      label: d.toLocaleString([], { weekday: 'short', hour: 'numeric' }),
    }
  })

  // Tick density has to scale with the window, or a 7-day view renders ~30
  // overlapping labels. Pick a step in hours, then emit explicit ticks so the
  // spacing is exact rather than whatever `interval` happens to produce.
  const span = data.length
  const step = span <= 24 ? 3 : span <= 48 ? 6 : span <= 72 ? 12 : 24
  const perDay = step >= 24

  const ticks = data
    .filter((d) => {
      const hour = d.date.getHours()
      // For multi-day views anchor ticks at midday so each label sits under its
      // own day; for shorter windows step evenly from midnight.
      return perDay ? hour === 12 : hour % step === 0
    })
    .map((d) => d.time)

  const formatTick = (value) => {
    const d = new Date(value)
    if (perDay) return d.toLocaleDateString([], { weekday: 'short' })
    if (span > 24) {
      // Mark the start of each day so a 48h view doesn't read as one long day.
      return d.getHours() === 0
        ? d.toLocaleDateString([], { weekday: 'short' })
        : d.toLocaleTimeString([], { hour: 'numeric' })
    }
    return d.toLocaleTimeString([], { hour: 'numeric' })
  }

  // Subtle midnight separators help the eye group multi-day data.
  const dayBoundaries = span > 30
    ? data.filter((d) => d.date.getHours() === 0).map((d) => d.time)
    : []

  const temps = data.flatMap((d) => [d.temp_f, d.feels_like_f]).filter((v) => v != null)
  const min = Math.min(...temps)
  const max = Math.max(...temps)
  const showFreezing = min <= 40 // only draw the 32F marker when it is in play

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-1">
        <h2 className="text-sm font-semibold">{title}</h2>
        <span className="text-xs muted tabular">
          low {Math.round(min)}° · high {Math.round(max)}°
        </span>
      </div>
      <p className="text-xs muted mb-3">
        Air temperature and what it feels like, hour by hour.
        {showFreezing && ' The dashed line marks freezing (32°F).'}
      </p>

      <div style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="tempFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--series-1)" stopOpacity={0.22} />
                <stop offset="100%" stopColor="var(--series-1)" stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid stroke="var(--gridline)" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="time"
              ticks={ticks}
              tickFormatter={formatTick}
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--baseline)' }}
              minTickGap={12}
              className="tabular"
            />
            <YAxis
              width={52}
              domain={[(dv) => Math.floor(dv - 4), (dv) => Math.ceil(dv + 4)]}
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${Math.round(v)}°`}
              className="tabular"
            />

            {dayBoundaries.map((t) => (
              <ReferenceLine key={t} x={t} stroke="var(--gridline)" strokeWidth={1} />
            ))}

            {showFreezing && (
              <ReferenceLine
                y={32}
                stroke="var(--status-critical)"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                label={{
                  value: 'Freezing 32°',
                  position: 'insideTopLeft',
                  fill: 'var(--text-secondary)',
                  fontSize: 11,
                }}
              />
            )}

            <Tooltip content={<TempTooltip />} cursor={{ stroke: 'var(--baseline)', strokeWidth: 1 }} />
            <Legend
              verticalAlign="top"
              align="right"
              height={28}
              iconType="plainline"
              wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
            />

            <Area
              type="monotone"
              dataKey="temp_f"
              name="Actual"
              stroke="var(--series-1)"
              strokeWidth={2}
              fill="url(#tempFill)"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
            />
            <Area
              type="monotone"
              dataKey="feels_like_f"
              name="Feels like"
              stroke="var(--series-2)"
              strokeWidth={2}
              strokeDasharray="5 3"
              fill="none"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {freezeLevel && freezeLevel !== 'none' && (
        <p className="text-xs mt-2 secondary">
          Freeze risk in this window: <strong>{freezeLevel}</strong>
        </p>
      )}
    </div>
  )
}

function TempTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload
  if (!point) return null
  // `condition` is an object ({code,label,icon}) in the weather payload — never
  // render it directly; an object as a React child throws and blanks the app.
  const conditionText =
    typeof point.condition === 'string' ? point.condition : point.condition?.label
  return (
    <div
      className="card px-3 py-2 text-xs"
      style={{ boxShadow: '0 4px 14px rgba(0,0,0,0.12)' }}
    >
      <div className="font-semibold mb-1">{point.label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 tabular">
          <span
            style={{
              width: 8, height: 8, borderRadius: 2,
              background: p.stroke, display: 'inline-block',
            }}
          />
          <span className="secondary">{p.name}</span>
          <span className="ml-auto font-medium">{Math.round(p.value)}°F</span>
        </div>
      ))}
      {point.humidity != null && (
        <div className="muted mt-1 tabular">
          Humidity {point.humidity}% · wind {Math.round(point.wind_mph ?? 0)} mph
        </div>
      )}
      {conditionText && <div className="muted">{conditionText}</div>}
    </div>
  )
}
