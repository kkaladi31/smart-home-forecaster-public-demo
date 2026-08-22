/**
 * The headline: one hero number plus the context that makes it actionable.
 * Deliberately not a chart — a single current value is a stat, not a series.
 */
export default function CurrentConditions({ weather }) {
  if (!weather?.current) return null
  const { current, comparison, running, tomorrow } = weather
  const today = weather.daily?.find((d) => d.is_today)

  return (
    <div className="card p-4">
      <div className="flex flex-wrap items-start gap-4">
        {/* Hero */}
        <div className="flex items-center gap-3">
          {current.condition?.icon_url ? (
            <img src={current.condition.icon_url} alt="" width={56} height={56} />
          ) : (
            <span className="text-5xl leading-none">{current.condition?.icon ?? '🌡️'}</span>
          )}
          <div>
            <div className="flex items-start gap-1">
              <span className="text-5xl font-semibold leading-none">
                {round(current.temp_f)}
              </span>
              <span className="text-lg muted mt-1">°F</span>
            </div>
            <div className="text-sm secondary mt-1">
              {current.condition?.label ?? '—'} · feels {round(current.feels_like_f)}°
            </div>
          </div>
        </div>

        {/* Context */}
        <div className="flex-1 min-w-[200px] grid grid-cols-2 gap-x-4 gap-y-1 text-xs self-center">
          {today && (
            <Fact label="Today" value={`${round(today.high_f)}° / ${round(today.low_f)}°`} />
          )}
          {comparison && <Fact label="vs yesterday" value={comparison.summary} />}
          {tomorrow && (
            <Fact
              label="Tomorrow"
              value={`${tomorrow.condition?.icon ?? ''} ${round(tomorrow.high_f)}° / ${round(tomorrow.low_f)}°`}
            />
          )}
          {today?.sunrise && (
            <Fact label="Sun" value={`↑ ${time(today.sunrise)}  ↓ ${time(today.sunset)}`} />
          )}
        </div>

        {/* Running conditions — a derived index, computed deterministically */}
        {running?.score != null && (
          <div
            className="rounded-lg px-3 py-2 min-w-[150px]"
            style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}
            title={running.reasons?.join(' ')}
          >
            <div className="text-xs muted">Good for a run?</div>
            <div className="flex items-baseline gap-2">
              <span className="text-lg font-semibold" style={{ color: runColor(running.score) }}>
                {running.verdict}
              </span>
              <span className="text-xs muted tabular">{running.score}/100</span>
            </div>
            {running.reasons?.[0] && (
              <div className="text-xs muted mt-0.5 line-clamp-2">{running.reasons[0]}</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Fact({ label, value }) {
  return (
    <>
      <span className="muted">{label}</span>
      <span className="secondary text-right">{value}</span>
    </>
  )
}

const round = (v) => (v == null ? '—' : Math.round(v))
const time = (iso) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '—'

function runColor(score) {
  if (score >= 80) return 'var(--status-good)'
  if (score >= 65) return 'var(--status-good)'
  if (score >= 45) return 'var(--status-warning)'
  return 'var(--status-critical)'
}
