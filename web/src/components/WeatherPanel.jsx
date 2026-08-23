import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import TempChart from './TempChart'

const RANGES = [
  { id: '24h', label: '24 hours', hours: 24 },
  { id: '48h', label: '48 hours', hours: 48 },
  { id: '7d', label: '7 days', hours: 24 * 7 },
]

/**
 * Forecast explorer: a range toggle over the hourly chart, plus a strip of days
 * you can click to scope the chart and open that day's full detail.
 */
/**
 * Summarise a day into blocks of `size` hours, each represented by its most
 * significant hour.
 *
 * The first version took every Nth hour — six spot readings — and that put a
 * visible contradiction on screen: a Thursday whose daily icon was drizzle
 * showed six suns, because the drizzle fell at 06:00, 22:00 and 23:00 and the
 * samples landed at 00/04/08/12/16/20. Both readings were accurate and they
 * disagreed, which to anyone looking at it means the app is broken.
 *
 * The cause was two different rules for one question. Open-Meteo's DAILY code
 * reports the day's most significant weather, not its most common — 3 hours of
 * drizzle outrank 12 hours of clear, and rightly so, because the drizzle is the
 * part that changes what you do. Sampling instead reports whatever happened to
 * be true on the hour. Blocks now use the daily rule, so the strip is a summary
 * of the whole day rather than six readings that ignore eighteen hours of it.
 *
 * "Most significant" is `max(code)`, which is what the WMO code ordering already
 * encodes: 0 clear, 1-3 cloud, 45-48 fog, 51-67 drizzle and rain, 71-77 snow,
 * 80-86 showers, 95-99 thunderstorm. Severity rises with the number, so no
 * separate ranking table is needed — and one that drifted from the codes would
 * be worse than none.
 */
function blocksOf(hours, size) {
  const out = []
  for (let i = 0; i < hours.length; i += size) {
    const block = hours.slice(i, i + size)
    if (!block.length) continue
    const worst = block.reduce(
      (a, b) => ((b.condition?.code ?? 0) > (a.condition?.code ?? 0) ? b : a),
      block[0],
    )
    // Labelled by when the block STARTS, but described by its worst hour: the
    // label answers "when", the glyph answers "what to expect around then".
    // Past only when the whole block is past, so a block still under way is not
    // greyed out while it is happening.
    out.push({ ...worst, time: block[0].time, is_past: block.every((h) => h.is_past) })
  }
  return out
}

export default function WeatherPanel({ weather }) {
  const [rangeId, setRangeId] = useState('48h')
  const [selectedDate, setSelectedDate] = useState(null)

  const range = RANGES.find((r) => r.id === rangeId) ?? RANGES[1]
  const days = useMemo(
    () => (weather?.daily ?? []).filter((d) => !d.is_yesterday),
    [weather],
  )
  const selectedDay = days.find((d) => d.date === selectedDate) ?? null

  // How much room each day card actually has, measured rather than inferred
  // from the viewport.
  //
  // The first version gated the intra-day strip on a viewport breakpoint, which
  // is the wrong quantity: this panel shares the width with the sidebar and the
  // chat column, so a 1280px window can still leave each of eight cards at its
  // 86px minimum. The strip then overflowed its own card and the last slot was
  // clipped mid-glyph — visible at exactly the middle sizes a breakpoint cannot
  // see. A card knows its width; the window does not know the card's.
  const stripRef = useRef(null)
  const [perCard, setPerCard] = useState(0)

  useLayoutEffect(() => {
    const el = stripRef.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const measure = () => setPerCard(el.clientWidth / Math.max(days.length, 1))
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [days.length])

  // Degrades in steps instead of appearing and vanishing: six glyphs when there
  // is room, three when there is less, none when the card can only carry the
  // day summary. A cramped strip is worse than no strip — it reads as broken,
  // where its absence just reads as a compact card.
  //
  // Thresholds come from the glyph budget, not round numbers:
  //   one slot  = max(emoji ~14px, "00" at 9px tabular ~11px) = 14px
  //   six slots = 6*14 + 5 gaps*2 = 94px, + px-2 padding 16px = 110px needed
  //   three     = 3*14 + 2 gaps*2 = 46px, +               16px =  62px needed
  // with ~18px of slack each so a wider font never clips the last glyph.
  const slotStep = perCard >= 128 ? 4 : perCard >= 96 ? 8 : 0

  // Selecting a day scopes the chart to that day; otherwise show the range window.
  const chartHours = useMemo(() => {
    if (!weather?.hourly) return []
    if (selectedDay) return selectedDay.hours ?? []
    const future = weather.hourly.filter((h) => !h.is_past)
    return future.slice(0, range.hours)
  }, [weather, selectedDay, range])

  if (!weather?.daily?.length) return null

  return (
    <div className="space-y-3">
      <div className="card p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">Forecast</h2>
          <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {RANGES.map((r) => (
              <button
                key={r.id}
                onClick={() => { setRangeId(r.id); setSelectedDate(null) }}
                className="px-3 py-1.5 text-xs font-medium"
                style={
                  r.id === rangeId && !selectedDay
                    ? { background: 'var(--series-1)', color: '#fff' }
                    : { background: 'transparent', color: 'var(--text-secondary)' }
                }
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {/* Day strip — click for detail */}
        <div ref={stripRef} className="mt-3 flex gap-2 overflow-x-auto scroll-thin pb-1">
          {days.map((d) => {
            const active = d.date === selectedDate
            return (
              <button
                key={d.date}
                onClick={() => setSelectedDate(active ? null : d.date)}
                // `flex-1` so the row FILLS the panel instead of leaving the
                // days bunched at the left of a wide monitor, and `min-w` so it
                // still scrolls rather than crushing them on a narrow one.
                className="flex-1 min-w-[86px] rounded-lg px-2 py-2 text-center"
                style={{
                  border: `1px solid ${active ? 'var(--series-1)' : 'var(--border)'}`,
                  background: active ? 'color-mix(in oklab, var(--series-1) 12%, transparent)' : 'var(--surface-page)',
                }}
              >
                <div className="text-xs muted">{d.is_today ? 'Today' : weekday(d.date)}</div>
                <div className="text-xl leading-tight my-0.5">{d.condition?.icon}</div>
                <div className="text-xs tabular">
                  <span className="font-semibold">{round(d.high_f)}°</span>{' '}
                  <span className="muted">{round(d.low_f)}°</span>
                </div>
                {d.precip_chance > 10 && (
                  <div className="text-[10px] tabular" style={{ color: 'var(--series-1)' }}>
                    💧{d.precip_chance}%
                  </div>
                )}

                {/* How the day actually unfolds, every 4 hours.
                    A single icon per day hides the thing people want from a
                    forecast: "clear" for a day that clouds over at four is not
                    wrong, it is just not the answer to what they asked.
                    The data was already in the payload (`daily[].hours`) from
                    Open-Meteo, so this is free and identical in both builds.
                    Shown only from `xl` up, where the card is wide enough to
                    read six glyphs; below that the day summary stands alone.
                    aria-hidden because the button already announces the day,
                    its icon and its temperatures — this is texture, not new
                    information, and reading twelve more tokens per day would
                    make the strip hostile to a screen reader. */}
                {slotStep > 0 && d.hours?.length > 0 && (
                  <div
                    aria-hidden="true"
                    className="flex items-center justify-between gap-0.5 mt-1.5 pt-1.5 overflow-hidden"
                    style={{ borderTop: '1px solid var(--border)' }}
                  >
                    {blocksOf(d.hours, slotStep).map((h) => (
                      <div
                        key={h.time}
                        className="flex flex-col items-center"
                        // Past hours dimmed rather than dropped, so today's card
                        // stays the same shape as every other day's.
                        style={{ opacity: h.is_past ? 0.35 : 1 }}
                        title={`${h.time.slice(11, 16)} ${h.condition?.label ?? ''}`}
                      >
                        <span className="text-[11px] leading-none">{h.condition?.icon}</span>
                        <span className="text-[9px] muted tabular leading-none mt-0.5">
                          {h.time.slice(11, 13)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </button>
            )
          })}
        </div>
        {selectedDay && (
          <p className="text-xs muted mt-2">
            Showing {formatDate(selectedDay.date)} — click the day again to return to the {range.label} view.
          </p>
        )}
      </div>

      <TempChart
        hourly={chartHours}
        freezeLevel={weather.freeze?.level}
        title={selectedDay ? `${formatDate(selectedDay.date)} — hourly` : `Next ${range.label}`}
      />

      {selectedDay && <DayDetail day={selectedDay} />}
    </div>
  )
}

/** Everything about one day, in the spirit of a phone weather app. */
function DayDetail({ day }) {
  const stats = [
    { label: 'High / Low', value: `${round(day.high_f)}° / ${round(day.low_f)}°` },
    { label: 'Feels like', value: `${round(day.feels_high_f)}° / ${round(day.feels_low_f)}°` },
    { label: 'Sunrise', value: time(day.sunrise), icon: '🌅' },
    { label: 'Sunset', value: time(day.sunset), icon: '🌇' },
    { label: 'UV index', value: uvLabel(day.uv_max), icon: '☀️' },
    { label: 'Rain chance', value: day.precip_chance != null ? `${day.precip_chance}%` : '—', icon: '💧' },
    { label: 'Precipitation', value: day.precip_in != null ? `${day.precip_in}"` : '—' },
    { label: 'Max wind', value: day.wind_max_mph != null ? `${round(day.wind_max_mph)} mph` : '—', icon: '💨' },
    { label: 'Moon', value: `${day.moon?.name} · ${day.moon?.illumination_pct}%`, icon: day.moon?.icon },
  ]

  // Hourly detail rows — the numbers a phone app shows when you scroll a day.
  const hours = (day.hours ?? []).filter((_, i) => i % 3 === 0)

  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-2xl">{day.condition?.icon}</span>
        <div>
          <h3 className="text-sm font-semibold">{formatDate(day.date)}</h3>
          <p className="text-xs muted">{day.condition?.label}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2">
        {stats.map((s) => (
          <div key={s.label}>
            <div className="text-xs muted">{s.icon ? `${s.icon} ` : ''}{s.label}</div>
            <div className="text-sm secondary tabular">{s.value}</div>
          </div>
        ))}
      </div>

      {hours.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold mb-2">Through the day</h4>
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full text-xs tabular">
              <thead>
                <tr className="muted text-left">
                  <th className="pr-3 font-normal">Time</th>
                  <th className="pr-3 font-normal">Temp</th>
                  <th className="pr-3 font-normal">Feels</th>
                  <th className="pr-3 font-normal">Rain</th>
                  <th className="pr-3 font-normal">Humidity</th>
                  <th className="pr-3 font-normal">Dew pt</th>
                  <th className="pr-3 font-normal">Wind</th>
                  <th className="pr-3 font-normal">Pressure</th>
                  <th className="font-normal">Visibility</th>
                </tr>
              </thead>
              <tbody className="secondary">
                {hours.map((h) => (
                  <tr key={h.time} style={{ borderTop: '1px solid var(--border)' }}>
                    <td className="py-1 pr-3">{time(h.time)}</td>
                    <td className="pr-3">{round(h.temp_f)}°</td>
                    <td className="pr-3">{round(h.feels_like_f)}°</td>
                    <td className="pr-3">{h.precip_chance ?? 0}%</td>
                    <td className="pr-3">{h.humidity ?? '—'}%</td>
                    <td className="pr-3">{round(h.dew_point_f)}°</td>
                    <td className="pr-3">{round(h.wind_mph)} mph</td>
                    <td className="pr-3">{round(h.pressure_hpa, 0)} hPa</td>
                    <td>{h.visibility_mi != null ? `${h.visibility_mi} mi` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

const round = (v, d = 0) => (v == null ? '—' : d ? v.toFixed(d) : Math.round(v))
const time = (iso) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '—'
const weekday = (d) => new Date(`${d}T12:00:00`).toLocaleDateString([], { weekday: 'short' })
const formatDate = (d) =>
  new Date(`${d}T12:00:00`).toLocaleDateString([], { weekday: 'long', month: 'short', day: 'numeric' })

function uvLabel(uv) {
  if (uv == null) return '—'
  const v = Math.round(uv)
  const band = v <= 2 ? 'Low' : v <= 5 ? 'Moderate' : v <= 7 ? 'High' : v <= 10 ? 'Very high' : 'Extreme'
  return `${v} · ${band}`
}
