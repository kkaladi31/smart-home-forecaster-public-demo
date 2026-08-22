/**
 * Data in the home profile carries its provenance inline, e.g.
 * "Maple Grove HOA (synthetic)" or "Carrier (illustrative)". Keeping that in the
 * source data matters — the project's data rule requires synthetic content to be
 * labelled — but it shouldn't shout from the UI.
 *
 * This splits the trailing qualifier off so the interface can show a clean value
 * and reveal the provenance on hover instead.
 */
const QUALIFIER = /^(.*?)\s*\(([^()]+)\)\s*$/

// Only these read as provenance; a genuine parenthetical stays part of the value.
//
// Anchored at the START, not both ends, so a qualifier may carry a descriptor:
// the corpus says "(synthetic stand-in)" and "(synthetic values)" as often as
// bare "(synthetic)". Requiring an exact single word meant those five values
// rendered the word "synthetic" as literal on-screen text — the exact thing this
// module exists to prevent — while the one-word cases were quietly handled and
// looked like proof it worked.
const PROVENANCE_WORDS =
  /^(synthetic|illustrative|sample|demo|example|fictional|placeholder|invented)\b/i

export function splitQualifier(text) {
  const raw = text == null ? '' : String(text)
  const match = QUALIFIER.exec(raw)
  if (!match) return { value: raw, note: null }

  const [, value, qualifier] = match
  if (!PROVENANCE_WORDS.test(qualifier.trim())) return { value: raw, note: null }

  const q = qualifier.trim()
  return {
    value: value.trim(),
    note: `${q.charAt(0).toUpperCase()}${q.slice(1)} — created for this demo, not a real record.`,
  }
}
