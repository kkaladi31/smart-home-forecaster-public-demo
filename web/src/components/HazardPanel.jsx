/**
 * Freeze / heat risk and any official NWS advisories.
 *
 * Status colours are the reserved status palette and are ALWAYS paired with an
 * icon and a text label — colour never carries the meaning on its own. On the
 * light surface `warning` and `serious` are deliberately sub-3:1, which is
 * exactly why the icon + label pairing is mandatory rather than decorative.
 */
const LEVEL_STYLE = {
  none: { color: 'var(--status-good)', icon: '✓', word: 'No risk' },
  low: { color: 'var(--status-good)', icon: '✓', word: 'Low' },
  moderate: { color: 'var(--status-warning)', icon: '!', word: 'Moderate' },
  high: { color: 'var(--status-serious)', icon: '▲', word: 'High' },
  severe: { color: 'var(--status-critical)', icon: '⚠', word: 'Severe' },
}

const SEVERITY_STYLE = {
  Extreme: 'var(--status-critical)',
  Severe: 'var(--status-critical)',
  Moderate: 'var(--status-serious)',
  Minor: 'var(--status-warning)',
}

export default function HazardPanel({ dashboard }) {
  if (!dashboard) return null
  const { freeze, heat, alerts, alerts_available: alertsAvailable } = dashboard

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <RiskCard title="Freeze risk" risk={freeze} />
      <RiskCard title="Heat risk" risk={heat} extra={
        heat?.heat_index_f != null ? `Heat index ${Math.round(heat.heat_index_f)}°F` : null
      } />

      <div className="card p-4 md:col-span-2">
        <h3 className="text-sm font-semibold mb-2">Official advisories</h3>
        {!alertsAvailable ? (
          <p className="text-xs muted">
            Official alerts are US-only (National Weather Service) and unavailable for this
            location — this is not the same as “no hazards”.
          </p>
        ) : alerts.length === 0 ? (
          <p className="text-xs secondary">
            <span style={{ color: 'var(--status-good)' }}>✓</span>{' '}
            No active advisories for this location.
          </p>
        ) : (
          <ul className="space-y-2">
            {alerts.map((a, i) => (
              <li key={i} className="flex gap-2">
                <span
                  aria-hidden
                  style={{ color: SEVERITY_STYLE[a.severity] ?? 'var(--status-warning)' }}
                >
                  ⚠
                </span>
                <div className="min-w-0">
                  <div className="text-sm font-medium">
                    {a.event}{' '}
                    <span className="text-xs muted">({a.severity})</span>
                  </div>
                  {a.headline && (
                    <div className="text-xs secondary line-clamp-2">{a.headline}</div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function RiskCard({ title, risk, extra }) {
  const level = risk?.level ?? 'none'
  const style = LEVEL_STYLE[level] ?? LEVEL_STYLE.none

  return (
    <div className="card p-4">
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className="grid place-items-center rounded-full text-xs font-bold"
          style={{
            width: 22, height: 22, color: '#fff', background: style.color,
          }}
        >
          {style.icon}
        </span>
        {/* Text label, not colour alone, carries the meaning. */}
        <span className="text-base font-semibold capitalize">{style.word}</span>
      </div>
      {risk?.headline && <p className="text-xs secondary mt-2">{risk.headline}</p>}
      {extra && <p className="text-xs muted mt-1 tabular">{extra}</p>}
      {risk?.actions?.length > 0 && level !== 'none' && (
        <ul className="mt-2 space-y-1">
          {risk.actions.slice(0, 3).map((a, i) => (
            <li key={i} className="text-xs secondary flex gap-1.5">
              <span className="muted">•</span>
              <span>{a}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
