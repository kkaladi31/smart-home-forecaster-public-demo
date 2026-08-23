import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

/**
 * Proactive hazard notifications — the "forecaster" half of the product.
 *
 * Everything else here answers a question that was asked. A forecaster has to
 * speak first: a freeze warning is worthless if you only see it because you
 * thought to look.
 *
 * WHY THE TEXT IS NOT WRITTEN BY THE MODEL. Levels and actions come straight
 * from `assess_freeze_risk` / `assess_heat_risk`, and advisory home-actions from
 * `tools/home_precautions.py` — the same deterministic sources the agent itself
 * is forbidden to second-guess. That is what makes it safe to pop up unprompted:
 *
 *   - it appears instantly, where a model call on the free tier measured
 *     6-100 s. A warning that arrives after you have moved on is not a warning.
 *   - it cannot invent a precaution. This is the one surface that speaks without
 *     being asked, so it is the worst possible place for a hallucination.
 *   - it costs nothing, so it can run on every dashboard load forever.
 *
 * The tailored answer is one click away instead: "Ask the assistant" hands the
 * hazard to the chat, where latency is expected and a person is watching.
 *
 * DISMISSING IS NOT SILENCING. A dismissed hazard collapses to a small pill that
 * says how many are still active, so the way back is always visible. Anything
 * else would make a single stray click hide a live freeze warning until the
 * level happened to change.
 *
 * RENDERED THROUGH A PORTAL, and that is not stylistic. Leaflet builds its own
 * stacking context and puts panes and controls between z-index 200 and 1000, so
 * a warning sitting at Tailwind's `z-50` — which is literally `z-index: 50` —
 * rendered UNDERNEATH the map. Bidding a bigger number would fix today's map and
 * lose to the next component that has its own opinion, and it cannot fix the
 * other half of the problem at all: any ancestor with a transform, filter or
 * `will-change` creates a containing block that traps a fixed-position child
 * regardless of z-index. Portalling to `document.body` leaves both arguments,
 * and the z-index below then only has to beat things that are also on body.
 */

const LEVEL_STYLE = {
  moderate: { color: 'var(--status-warning)', icon: '!', word: 'Moderate' },
  high: { color: 'var(--status-serious)', icon: '▲', word: 'High' },
  severe: { color: 'var(--status-critical)', icon: '⚠', word: 'Severe' },
}

// Below `moderate` nothing is shown. A notification for "low risk" teaches
// people to dismiss notifications unread, which costs you the one that mattered.
const RANK = { moderate: 1, high: 2, severe: 3 }

function hazardsFrom(dashboard) {
  if (!dashboard) return []
  const out = []

  for (const alert of dashboard.alerts ?? []) {
    out.push({
      key: `alert|${alert.event}|${alert.severity ?? ''}`,
      title: alert.event || 'Official advisory',
      // An advisory outranks any computed rating: it is a human authority
      // speaking about this specific place, right now.
      rank: 99,
      color: 'var(--status-critical)',
      icon: '⚠',
      word: alert.severity || 'Advisory',
      headline: alert.headline || '',
      // The NWS instruction is about keeping PEOPLE safe; `home_actions` is what
      // it means for the building. Both are shown, labelled, and never merged —
      // one is quoted from an authority, the other is ours.
      official: alert.instruction || '',
      actions: alert.home_actions ?? [],
      ask: `There is an active ${alert.event} advisory for my area. What should I do to protect my home?`,
    })
  }

  for (const [kind, risk, label] of [
    ['freeze', dashboard.freeze, 'Freeze risk'],
    ['heat', dashboard.heat, 'Heat risk'],
  ]) {
    const level = risk?.level
    if (!level || !RANK[level]) continue
    out.push({
      key: `${kind}|${level}`,
      title: label,
      rank: RANK[level],
      ...LEVEL_STYLE[level],
      headline: risk.headline || '',
      official: '',
      actions: risk.actions ?? [],
      ask: kind === 'freeze'
        ? 'There is a freeze risk at my home. What should I do to protect the pipes?'
        : 'There is a heat risk at my home. What should I do to keep the house cool and safe?',
    })
  }

  return out.sort((a, b) => b.rank - a.rank)
}

export default function HazardAlert({ dashboard, onAsk }) {
  // Keyed by the CONDITION, not by the notification. Dismissing "freeze:
  // moderate" must not also dismiss the "freeze: severe" that arrives an hour
  // later — silencing an escalation is the one thing this must never do.
  const [dismissed, setDismissed] = useState(() => new Set())
  const dialogRef = useRef(null)
  const restoreFocusRef = useRef(null)

  const place = dashboard?.location?.label ?? ''
  const hazards = useMemo(() => hazardsFrom(dashboard), [dashboard])
  const open = hazards.filter((h) => !dismissed.has(`${place}|${h.key}`))
  const hidden = hazards.length - open.length
  const isOpen = open.length > 0

  const closeAll = () =>
    setDismissed((prev) => {
      const next = new Set(prev)
      for (const h of hazards) next.add(`${place}|${h.key}`)
      return next
    })

  // Focus moves into the dialog on open and back out on close, so a keyboard
  // user is not left tabbing behind an overlay they cannot see.
  useEffect(() => {
    if (!isOpen) return undefined
    restoreFocusRef.current = document.activeElement
    dialogRef.current?.focus()
    const onKey = (e) => { if (e.key === 'Escape') closeAll() }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      if (restoreFocusRef.current instanceof HTMLElement) restoreFocusRef.current.focus()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen])

  // Nothing active at all — no pop-up, and nothing to offer a way back to.
  if (!hazards.length) return null

  if (!isOpen) {
    return createPortal(
      <button
        type="button"
        onClick={() => setDismissed(new Set())}
        className="fixed text-xs px-3 py-2 rounded-full shadow-lg"
        style={{
          bottom: 16, right: 16, zIndex: 10000,
          background: 'var(--surface-1)',
          border: '1px solid var(--status-critical)',
          color: 'var(--text-primary)',
        }}
        aria-label={`Show ${hidden} active hazard ${hidden === 1 ? 'notification' : 'notifications'}`}
      >
        <span aria-hidden="true" style={{ color: 'var(--status-critical)' }}>⚠</span>{' '}
        {hidden} active {hidden === 1 ? 'alert' : 'alerts'}
      </button>,
      document.body,
    )
  }

  return createPortal(
    <div
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.45)', zIndex: 10000 }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) closeAll() }}
    >
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="hazard-dialog-title"
        tabIndex={-1}
        className="card w-full overflow-y-auto outline-none"
        style={{ maxWidth: 560, maxHeight: '85vh', background: 'var(--surface-1)' }}
      >
        <div
          className="flex items-center justify-between gap-3 px-4 py-3 sticky top-0"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface-1)' }}
        >
          <h2 id="hazard-dialog-title" className="text-sm font-semibold">
            {open.length === 1 ? 'Weather alert' : `${open.length} weather alerts`}
            {place && <span className="muted font-normal"> · {place}</span>}
          </h2>
          <button type="button" onClick={closeAll} className="text-sm muted px-1"
            aria-label="Dismiss all alerts">✕</button>
        </div>

        <div className="p-4 space-y-4">
          {open.map((h) => (
            <section key={h.key} style={{ borderLeft: `3px solid ${h.color}`, paddingLeft: 12 }}>
              <div className="flex items-center gap-2 flex-wrap">
                {/* Icon AND word — colour never carries the meaning alone. */}
                <span aria-hidden="true" style={{ color: h.color }}>{h.icon}</span>
                <span className="text-sm font-semibold">{h.title}</span>
                <span
                  className="text-xs px-1.5 py-0.5 rounded"
                  style={{ color: h.color, border: `1px solid ${h.color}` }}
                >
                  {h.word}
                </span>
              </div>

              {h.headline && <p className="text-xs mt-1.5 secondary">{h.headline}</p>}

              {h.official && (
                <div className="mt-2">
                  <div className="text-xs font-medium">Official guidance</div>
                  <p className="text-xs mt-0.5 secondary">{h.official}</p>
                </div>
              )}

              {h.actions.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs font-medium">For your home</div>
                  <ul className="text-xs mt-1 space-y-1">
                    {h.actions.map((a, i) => (
                      <li key={i} className="flex gap-1.5">
                        <span aria-hidden="true" className="muted">·</span>
                        <span>{a}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {onAsk && (
                <button
                  type="button"
                  className="text-xs mt-2"
                  style={{ color: 'var(--accent)' }}
                  onClick={() => { closeAll(); onAsk(h.ask) }}
                >
                  Ask the assistant about this →
                </button>
              )}
            </section>
          ))}
        </div>

        <p className="text-xs muted px-4 pb-4">
          General guidance, not professional advice. For gas, electrical, flooding or
          medical emergencies contact 911, your utility, or a licensed professional.
        </p>
      </div>
    </div>,
    document.body,
  )
}
