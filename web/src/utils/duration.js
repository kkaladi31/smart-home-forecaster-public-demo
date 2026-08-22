/**
 * One way to write a duration, shared by the chat timer, the trace feed, and the
 * Logs tab — so the same 12.4 seconds never reads as "12400ms" in one place and
 * "12s" in another.
 *
 * Milliseconds below a second (tool calls are often that fast), one decimal of
 * seconds up to a minute (the resolution that matters while waiting), and m:ss
 * beyond that.
 */
export function fmtDuration(ms) {
  if (ms == null) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}
