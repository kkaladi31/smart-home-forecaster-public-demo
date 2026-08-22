import { useState } from 'react'
import { apiUrl } from '../api'
import { splitQualifier } from '../utils/provenance'

/** Home switcher, profile, persona toggle, run mode, knowledge status, floor plan. */
export default function Sidebar({
  profile, status, persona, onPersona, homes = [], homeId, onHome, onLogout,
  mode, modeBusy, onMode,
}) {
  const [showPlan, setShowPlan] = useState(false)
  const home = profile?.home
  // `floorplans` is already scoped to this home by /api/profile and its entries
  // are "<home_id>/<file>". No cross-home fallback: a home with no plan of its
  // own shows nothing, because showing another house's layout is worse than
  // showing none.
  const plan = profile?.floorplans?.[0] ?? null

  return (
    <aside className="space-y-3">
      <div className="card p-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-base font-semibold leading-tight">
              Smart-Home Forecaster
              {mode && <ModeBadge demo={mode.demo} />}
            </h1>
            <p className="text-xs muted">Proactive home-risk agent</p>
          </div>
          <button onClick={onLogout} className="text-xs muted hover:underline">Sign out</button>
        </div>

        {homes.length > 1 && (
          <div className="mt-4">
            <span className="text-xs secondary">Home</span>
            <div className="mt-1 space-y-1">
              {homes.map((h) => {
                const active = h.home_id === homeId
                // The HOA name carries its provenance inline — "Maple Grove HOA
                // (synthetic)". Split it so the button reads like a real
                // association and the qualifier lives on hover, the same
                // treatment every other value in this panel gets via `Row`.
                // This render site was the one that never got wired up, so the
                // word appeared on screen in the home switcher.
                const { value: hoa, note } = splitQualifier(h.hoa)
                return (
                  <button
                    key={h.home_id}
                    onClick={() => onHome?.(h.home_id)}
                    className="w-full text-left px-2 py-1.5 rounded-lg text-xs"
                    title={[h.address, note].filter(Boolean).join(' — ')}
                    style={
                      active
                        ? { background: 'var(--series-1)', color: '#fff' }
                        : { border: '1px solid var(--border)', color: 'var(--text-secondary)' }
                    }
                  >
                    <span className="font-medium">{h.label}</span>
                    {h.is_primary && (
                      <span className="ml-1.5 opacity-70">primary</span>
                    )}
                    <span className="block opacity-70 truncate">{hoa}</span>
                  </button>
                )
              })}
            </div>
            <p className="mt-1.5 text-[11px] muted">
              Each home has its own HOA, city rules, and contractors. Switching starts a
              new conversation.
            </p>
          </div>
        )}

        <div className="mt-4">
          <span className="text-xs secondary">I am the…</span>
          <div className="mt-1 flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {['owner', 'renter'].map((p) => (
              <button
                key={p}
                onClick={() => onPersona(p)}
                className="flex-1 py-1.5 text-xs font-medium capitalize"
                style={
                  persona === p
                    ? { background: 'var(--series-1)', color: '#fff' }
                    : { background: 'transparent', color: 'var(--text-secondary)' }
                }
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {mode && <ModeCard mode={mode} busy={modeBusy} onMode={onMode} />}

      {home && (
        <div className="card p-4">
          <h2 className="text-sm font-semibold mb-2">Your home</h2>
          <dl className="text-xs space-y-1">
            <Row label="Address" value={home.address} />
            <Row label="Type" value={`${home.dwelling_type?.replace('_', ' ')} · ${home.year_built}`} />
            <Row label="Builder" value={home.builder} />
            <Row label="Size" value={`${home.square_feet?.toLocaleString()} sq ft · ${home.bedrooms} bd / ${home.bathrooms} ba`} />
            <Row label="Climate" value={home.climate_zone} />
            <Row label="HVAC filter" value={`${home.systems?.hvac?.filter_size} every ${home.systems?.hvac?.filter_interval_days} days`} />
            <Row label="HOA" value={home.hoa?.association_name} />
            <Row label="Permits" value={home.jurisdiction?.building_department} />
          </dl>

          {plan && (
            <>
              <button
                onClick={() => setShowPlan((s) => !s)}
                className="mt-3 text-xs hover:underline"
                style={{ color: 'var(--series-1)' }}
              >
                {showPlan ? 'Hide floor plan' : 'View floor plan'}
              </button>
              {showPlan && (
                <div className="mt-2 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)', background: '#fff' }}>
                  <img src={apiUrl(`/api/floorplan/${plan}`)} alt="Floor plan" className="w-full" />
                </div>
              )}
            </>
          )}
        </div>
      )}

      {status && (
        <div className="card p-4">
          <h2 className="text-sm font-semibold mb-2">System</h2>
          <dl className="text-xs space-y-1">
            <Row label="Model" value={status.model} mono />
            <Row label="Knowledge" value={`${status.knowledge_passages} passages`} />
            <Row label="Memory" value={`${status.remembered_interactions} interactions`} />
            <Row label="Live prices" value={status.has_eia_key ? 'EIA connected' : 'using averages'} />
          </dl>
        </div>
      )}
    </aside>
  )
}

/** Small chip in the header showing which stack is live. */
function ModeBadge({ demo }) {
  return (
    <span
      className="ml-2 align-middle px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide"
      title={demo
        ? 'Demo mode — free and open services only'
        : 'Full mode — includes metered services'}
      style={demo
        ? { background: 'var(--status-good)', color: '#fff' }
        : { border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
    >
      {demo ? 'Demo' : 'Full'}
    </span>
  )
}

// How each cost tier reads in the service list. "gov" is called out separately
// from "free" because a free government API is the strongest provenance the
// project has — it stays on in both modes, and that is worth showing.
const TIERS = {
  gov: { label: 'gov', color: 'var(--series-1)' },
  free: { label: 'free', color: 'var(--status-good)' },
  billed: { label: 'billed', color: 'var(--status-warning)' },
}

/** Run-mode card. Shows the switch ONLY on a build that has something to switch to.
 *
 * `mode.switchable` is present only in a full build. A demo build's payload has
 * no concept of another mode at all — not a disabled button, not a `locked`
 * flag, nothing. Before this, the published demo rendered a "Full — best
 * quality, uses metered services" button that did nothing when clicked:
 * advertising a paid stack the artifact does not have, via a visibly broken
 * control, in the thing a reviewer is looking at.
 *
 * The service list still renders in both, and it is the more valuable half —
 * in a demo build no row is `billed`, which is a claim the reader can check on
 * screen rather than take on trust.
 */
function ModeCard({ mode, busy, onMode }) {
  const [open, setOpen] = useState(false)
  const billed = mode.services.filter((s) => s.tier === 'billed').length

  return (
    <div className="card p-4">
      <h2 className="text-sm font-semibold mb-2">
        {mode.switchable ? 'Run mode' : 'Services'}
      </h2>

      {mode.switchable && (
        <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          {[
            { demo: false, label: 'Full', hint: 'Best quality — uses metered services' },
            { demo: true, label: 'Demo', hint: 'Free and open services only' },
          ].map((opt) => (
            <button
              key={opt.label}
              onClick={() => onMode?.(opt.demo)}
              disabled={busy}
              title={opt.hint}
              className="flex-1 py-1.5 text-xs font-medium disabled:opacity-60"
              style={
                mode.demo === opt.demo
                  ? { background: 'var(--series-1)', color: '#fff' }
                  : { background: 'transparent', color: 'var(--text-secondary)' }
              }
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      <p className={`text-[11px] muted ${mode.switchable ? 'mt-1.5' : ''}`}>
        {busy
          ? 'Switching…'
          : mode.demo
            ? 'Free and open services only. Government APIs stay live.'
            : `${billed} service${billed === 1 ? '' : 's'} on a metered account.`}
      </p>

      {mode.switchable && !mode.demo && !mode.has_google_key && (
        <p className="mt-1 text-[11px] muted">
          No Google key configured, so the free providers are already in use.
        </p>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        className="mt-2 w-full flex items-center gap-1 text-[11px] hover:underline"
        style={{ color: 'var(--series-1)' }}
        aria-expanded={open}
      >
        <span aria-hidden="true">{open ? '▾' : '▸'}</span>
        {open ? 'Hide services' : `Show services (${mode.services.length})`}
      </button>

      {open && (
        <ul className="mt-2 space-y-2">
          {mode.services.map((s) => {
            const tier = TIERS[s.tier] ?? TIERS.free
            return (
              // Provider on its own line under the service name. Side-by-side
              // columns forced long values like "Google Weather + Air Quality"
              // to wrap into the tier chip in a 260px sidebar.
              <li key={s.service} className="leading-tight">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] muted">{s.service}</span>
                  <span
                    className="ml-auto shrink-0 px-1 rounded text-[9px] font-medium tracking-wide"
                    style={{ color: tier.color, background: 'color-mix(in srgb, currentColor 12%, transparent)' }}
                  >
                    {tier.label}
                  </span>
                </div>
                <div className="text-[11px] secondary truncate" title={s.provider}>
                  {s.provider}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function Row({ label, value, mono }) {
  if (!value) return null
  // Provenance ("synthetic", "illustrative") is kept in the data but shown only
  // on hover, so the panel reads like real home details.
  const { value: shown, note } = splitQualifier(value)
  return (
    <div className="flex gap-2">
      <dt className="muted shrink-0">{label}</dt>
      <dd
        title={note ?? undefined}
        className={`ml-auto text-right secondary ${mono ? 'font-mono text-[11px]' : ''}`}
        style={note ? { textDecoration: 'underline dotted', textUnderlineOffset: 3, cursor: 'help' } : undefined}
      >
        {shown}
      </dd>
    </div>
  )
}
