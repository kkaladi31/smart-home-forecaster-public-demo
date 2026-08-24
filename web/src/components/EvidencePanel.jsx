/**
 * The web evidence behind an answer — what was used, how far it was trusted, and
 * what was thrown away.
 *
 * The counterpart to ReasoningTree, and it exists for the same reason. That panel
 * shows options considered and ruled out; this one shows *sources* considered and
 * ruled out. Before it existed the researcher was the only agent whose work was
 * completely invisible: citations appeared inline in the prose, and everything
 * else — which queries ran, which pages were screened, why the pack was sometimes
 * empty — reached no surface at all except the Logs tab.
 *
 * The drops carry the weight here. Three of them are claims the system makes about
 * itself and could not otherwise evidence:
 *
 *   - a page that tried to issue instructions was DROPPED, not merely ranked low;
 *   - a pack that came back empty did so because nothing was relevant, which from
 *     outside is indistinguishable from a search that simply failed;
 *   - no single site may own the pack, so apparent corroboration is real.
 *
 * Authority is shown as the label, never the number. "manufacturer" is a claim a
 * reader can dispute; "0.90" invites them to trust a weight they have no way to
 * check.
 */

/** Screened-out passages, grouped so six drops from one cause read as one line. */
function groupDrops(dropped) {
  const groups = new Map()
  for (const d of dropped ?? []) {
    const key = d.reason ?? ''
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(d.domain || 'unknown')
  }
  return [...groups.entries()].map(([reason, domains]) => ({ reason, domains }))
}

export default function EvidencePanel({ evidence }) {
  const passages = evidence?.passages ?? []
  const dropped = evidence?.dropped ?? []
  const queries = evidence?.queries ?? []
  if (!passages.length && !dropped.length) return null

  const groups = groupDrops(dropped)

  return (
    <div
      className="rounded-lg px-3 py-2 text-xs space-y-2"
      style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="secondary">Web evidence</span>
        {evidence?.provider && (
          <span className="muted tabular shrink-0">{evidence.provider}</span>
        )}
      </div>

      {passages.length > 0 ? (
        <div className="space-y-1">
          {passages.map((p) => (
            <div key={p.ref} className="flex items-start gap-2">
              <span className="muted tabular shrink-0">{p.ref}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="secondary break-all"
                  >
                    {p.domain}
                  </a>
                  <span className="muted tabular shrink-0">
                    {p.score != null ? p.score.toFixed(2) : ''}
                  </span>
                </div>
                {p.authority && p.authority !== 'unrated' && (
                  <div className="muted">{p.authority}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        // The empty pack is a RESULT, not a gap in the panel. Saying so is the
        // whole point of the relevance floor: the alternative was citing six
        // sources that scored far below the threshold retrieval refuses at.
        <div className="muted">
          Nothing met the relevance threshold, so the answer cites no web source.
        </div>
      )}

      {groups.length > 0 && (
        <div className="space-y-1">
          <div className="muted">Screened out</div>
          {groups.map((g) => (
            <div key={g.reason} className="flex items-start gap-2">
              <span className="muted shrink-0">×</span>
              <div className="min-w-0 flex-1">
                <span className="muted break-words" style={{ textDecoration: 'line-through' }}>
                  {g.domains.join(', ')}
                </span>
                <div className="muted break-words">{g.reason}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {queries.length > 1 && (
        <details>
          <summary className="muted cursor-pointer">{queries.length} searches</summary>
          <div className="mt-1 space-y-0.5">
            {queries.map((q) => (
              <div key={q} className="muted break-words">{q}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
