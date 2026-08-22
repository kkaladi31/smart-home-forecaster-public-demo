/**
 * Licensed professionals — and, deliberately, the ones that were refused.
 *
 * The withheld list is the reason this panel exists. The gate in `tools/pros`
 * hides refused contractors from the Advisor, because a professional the system
 * must not recommend has no business in a model's prompt. A person reading the
 * screen is owed the opposite: "three roofers nearby, all with lapsed
 * registrations" is far more useful than an empty list, and it is the honest
 * answer. So the two consumers get different views on purpose.
 *
 * Two rules the visual design has to keep:
 *
 * 1. **A withheld row must never read as a weak recommendation.** It is not
 *    ranked below the others — it is in a separate, labelled section, with its
 *    reason in text rather than implied by colour. The highest-rated business
 *    here is often the refused one, so anything that looks like a ranking would
 *    actively mislead.
 * 2. **Colour never carries meaning alone**, matching HazardPanel: every status
 *    colour is paired with an icon and a word.
 */
import { useCallback, useEffect, useState } from 'react'
import { getContractors } from '../api'
import { splitQualifier } from '../utils/provenance'

// A short list rather than every trade in the table: this is a browsing
// affordance, not a search form. The agent handles the long tail through
// find_licensed_pros, where the user can just describe the problem.
const TRADES = [
  { key: '', label: 'All' },
  { key: 'plumbing', label: 'Plumbing' },
  { key: 'electrical', label: 'Electrical' },
  { key: 'hvac', label: 'Heating & cooling' },
  { key: 'roofing', label: 'Roofing' },
  { key: 'handyman', label: 'Handyman' },
]

// `unassessed` means no trade was requested, so there is no verdict to report —
// distinct from `unknown`, which means we asked and the registration does not
// cover it. Showing the second when the truth is the first labels every properly
// registered plumber as not registered for plumbing.
const MATCH_STYLE = {
  specialist: { color: 'var(--status-good)', icon: '✓', word: 'Registered for this trade' },
  general: { color: 'var(--status-warning)', icon: '~', word: 'General contractor' },
  unknown: { color: 'var(--text-muted)', icon: '·', word: 'Registration does not name this trade' },
  unassessed: { color: 'var(--text-muted)', icon: '·', word: null },
}

function money(value) {
  if (value == null) return null
  return `$${Number(value).toLocaleString()}`
}

export default function ProsPanel({ homeId }) {
  const [trade, setTrade] = useState('')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await getContractors(trade || null, homeId))
    } catch (e) {
      setError(e?.message || 'Could not reach the contractor registry.')
    } finally {
      setLoading(false)
    }
  }, [trade, homeId])

  useEffect(() => { load() }, [load])

  const eligible = data?.results ?? []
  const withheld = data?.withheld ?? []

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between gap-2 flex-wrap">
        <h3 className="text-sm font-semibold">Licensed professionals</h3>
        {data?.area && <span className="text-xs muted">{data.area}</span>}
      </div>

      <div className="flex flex-wrap gap-1.5 mt-3">
        {TRADES.map((t) => (
          <button
            key={t.key || 'all'}
            onClick={() => setTrade(t.key)}
            className="text-xs px-2 py-1 rounded-lg"
            style={
              trade === t.key
                ? { background: 'var(--series-1)', color: '#fff' }
                : { border: '1px solid var(--border)', color: 'var(--text-secondary)' }
            }
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && (
        <p className="text-xs mt-3" style={{ color: 'var(--status-critical)' }}>{error}</p>
      )}
      {loading && !data && <p className="text-xs muted mt-3">Checking the registry…</p>}

      {data && !loading && eligible.length === 0 && withheld.length === 0 && (
        <p className="text-xs secondary mt-3">
          No registered professionals for this trade in the home’s service area.
        </p>
      )}

      {eligible.length > 0 && (
        <ul className="mt-3 space-y-2">
          {eligible.map((p, i) => <ProRow key={`${p.name}-${i}`} pro={p} />)}
        </ul>
      )}

      {/* The refused list. Separate section, explicit heading, reason in words. */}
      {withheld.length > 0 && (
        <div className="mt-4 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
          <h4 className="text-xs font-semibold" style={{ color: 'var(--status-warning)' }}>
            ⚠ Not recommended — {withheld.length} nearby{' '}
            {withheld.length === 1 ? 'business' : 'businesses'}
          </h4>
          <p className="text-xs muted mt-0.5">
            These matched the trade but hold no current registration. They are shown so the
            list is honest, not as alternatives.
          </p>
          <ul className="mt-2 space-y-1.5">
            {withheld.map((p, i) => (
              <li key={`${p.name}-${i}`} className="text-xs">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>
                    {p.name}
                  </span>
                  {p.rating != null && <span className="muted">{p.rating}★</span>}
                  {p.license_status && (
                    <span
                      className="px-1.5 py-0.5 rounded"
                      style={{
                        border: '1px solid var(--status-warning)',
                        color: 'var(--status-warning)',
                      }}
                    >
                      {p.license_status}
                    </span>
                  )}
                </div>
                <div className="muted">Withheld — {p.withheld_reason}</div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {data?.notes?.length > 0 && (
        <ul className="mt-3 space-y-1">
          {data.notes.map((note, i) => (
            <li key={i} className="text-xs muted">{note}</li>
          ))}
        </ul>
      )}

      {/* Provenance rides in the parenthetical and is moved to a hover title,
          the same treatment the home profile gets. The label stays honest
          without the word "synthetic" appearing in the interface. */}
      {data?.source && (() => {
        const { value, note } = splitQualifier(data.source)
        return (
          <p className="text-xs muted mt-3" title={note || undefined}>
            Registration status from{' '}
            <span style={note ? { borderBottom: '1px dotted var(--border)' } : undefined}>
              {value}
            </span>
            . A current registration is not a recommendation — always confirm
            scope and price directly.
          </p>
        )
      })()}
    </div>
  )
}

function ProRow({ pro }) {
  const match = MATCH_STYLE[pro.match] ?? MATCH_STYLE.unknown
  return (
    <li className="text-xs" style={{ borderLeft: `2px solid ${match.color}`, paddingLeft: 8 }}>
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-sm font-medium">{pro.name}</span>
        {pro.rating != null && (
          <span className="muted">
            {pro.rating}★{pro.reviews != null ? ` (${pro.reviews})` : ''}
          </span>
        )}
        {pro.hourly_rate_usd != null && (
          <span className="muted">{money(pro.hourly_rate_usd)}/hr</span>
        )}
      </div>

      <div style={{ color: match.color }}>
        <span aria-hidden>{match.icon}</span>{' '}
        {match.word
          ? `${match.word}${pro.specialty ? ` — ${pro.specialty}` : ''}`
          : (pro.specialty || pro.license_type || 'Registered contractor')}
      </div>

      <div className="muted">
        {pro.license_status}
        {pro.license_number ? ` · ${pro.license_number}` : ''}
        {pro.license_expires ? ` · valid to ${String(pro.license_expires).slice(0, 10)}` : ''}
        {pro.bond_usd ? ` · bond ${money(pro.bond_usd)}` : ''}
      </div>

      {(pro.phone || pro.availability) && (
        <div className="muted">
          {pro.phone}
          {pro.phone && pro.availability ? ' · ' : ''}
          {pro.availability}
        </div>
      )}
    </li>
  )
}
