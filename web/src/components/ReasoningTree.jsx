/**
 * The Advisor's beam search, drawn.
 *
 * This is the one place a user can see that options were *considered and ruled
 * out*, not merely that one was produced. Pruned branches are shown struck
 * through with the reason beside them, which matters for more than presentation:
 * the risk this reasoning design cannot self-detect is a good branch discarded by
 * a weak evaluation signal, and a branch discarded invisibly is a branch nobody
 * can question. Rendering the whole tree, losers included, is the mitigation.
 */
export default function ReasoningTree({ tree, strategy, truncated }) {
  if (!tree?.length) return null

  const depths = [...new Set(tree.map((n) => n.depth))].sort()
  const best = Math.max(...tree.filter((n) => n.score != null).map((n) => n.score), 0)

  return (
    <div
      className="rounded-lg px-3 py-2 text-xs space-y-2"
      style={{ background: 'var(--surface-page)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="secondary">Options considered</span>
        {strategy && <span className="muted tabular shrink-0">{strategy}</span>}
      </div>

      {depths.map((depth) => (
        <div key={depth} className="space-y-1">
          {depths.length > 1 && (
            <div className="muted">
              {depth === 1 ? 'Approaches' : 'Detailed plans'}
            </div>
          )}
          {tree
            .filter((n) => n.depth === depth)
            .map((n, i) => (
              <Branch key={`${depth}-${i}`} node={n} best={best} />
            ))}
        </div>
      ))}

      {truncated && (
        <div className="muted">
          Search stopped early at its time or call budget — the comparison above is
          what it managed to score.
        </div>
      )}
    </div>
  )
}

function Branch({ node, best }) {
  const pruned = node.status === 'pruned'
  const superseded = node.status === 'superseded'
  // A gated node has no score by design: it was removed by a rule, not out-argued
  // on points. Showing 0 would read as "considered and poor" rather than "not
  // allowed", which are different things a homeowner needs to tell apart.
  const ruledOut = node.score == null
  const winner = !pruned && !superseded && node.score != null && node.score >= best

  const colour = pruned
    ? 'var(--text-muted)'
    : winner
      ? 'var(--series-1)'
      : 'var(--text-secondary)'

  return (
    <div className="flex items-start gap-2">
      <span style={{ color: colour }}>{pruned ? '×' : winner ? '●' : '○'}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span
            className="secondary break-words"
            style={{
              textDecoration: pruned ? 'line-through' : 'none',
              opacity: pruned ? 0.7 : 1,
              fontWeight: winner ? 600 : 400,
            }}
          >
            {node.name}
          </span>
          <span className="muted tabular shrink-0">
            {ruledOut ? 'ruled out' : node.score.toFixed(1)}
          </span>
        </div>
        {pruned && node.reason && (
          <div className="muted break-words">{node.reason}</div>
        )}
      </div>
    </div>
  )
}
