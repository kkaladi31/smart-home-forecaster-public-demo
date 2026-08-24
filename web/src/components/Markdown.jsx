import { Children, isValidElement, cloneElement } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * Renders an agent answer as formatted text.
 *
 * The model has always written markdown — headings, bold labels, tables, ordered
 * steps — but the chat bubble printed it as plain text, so users saw literal
 * `**asterisks**` and pipe-delimited table rows. That is the whole reason
 * answers looked unreadable. GFM is enabled because tables and strikethrough are
 * not in core markdown.
 *
 * Every element is styled against the app's CSS variables so answers match the
 * dashboard in both light and dark themes, and tables scroll inside their own
 * box rather than pushing the chat column sideways.
 */

// The model writes `<br>` inside table cells, because a GFM cell cannot contain a
// real newline and that is the only line break HTML offers it. Those tags were
// rendering as literal text — "• No drilling needed <br>• Minimal wall damage" —
// which is worse than no break at all.
//
// The fix is NOT `rehype-raw`. Answers carry model-authored prose and text
// retrieved from the open web, so turning on raw HTML here would hand a fetched
// page a way to inject markup into the interface — the exact thing five layers of
// R14 handling exist to prevent. Instead the tags are swapped for a sentinel that
// survives markdown parsing as ordinary text, and the renderer turns that
// sentinel into real <br> ELEMENTS it constructs itself. Nothing from the model
// is ever interpreted as markup.
const BR_TAG = /<br\s*\/?>/gi
const SENTINEL = '' // private-use area: cannot occur in prose or a URL

/** Split text children on the sentinel and interleave real line breaks. */
function withLineBreaks(children) {
  return Children.map(children, (child) => {
    if (typeof child === 'string') {
      if (!child.includes(SENTINEL)) return child
      const parts = child.split(SENTINEL)
      return parts.flatMap((part, i) =>
        i === 0 ? [part] : [<br key={`br-${i}`} />, part])
    }
    // Recurse, so a break inside **bold** or a link still renders.
    if (isValidElement(child) && child.props?.children) {
      return cloneElement(child, {
        ...child.props,
        children: withLineBreaks(child.props.children),
      })
    }
    return child
  })
}

/** A component that renders its children with sentinels expanded. */
const breaking = (Tag) =>
  function Breaking({ node: _node, children, ...props }) {
    return <Tag {...props}>{withLineBreaks(children)}</Tag>
  }

export default function Markdown({ children }) {
  const source =
    typeof children === 'string' ? children.replace(BR_TAG, SENTINEL) : children

  return (
    <div className="md text-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // `node` is react-markdown's AST node; it is not a DOM attribute and
          // must be dropped rather than spread onto the element.
          table: ({ node: _node, ...props }) => (
            // The wrapper is what scrolls. Without it a wide table stretches the
            // flex column and the whole page gets a horizontal scrollbar.
            <div className="md-table-wrap scroll-thin">
              <table {...props} />
            </div>
          ),
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
          // Every element a line break can legitimately appear inside.
          td: breaking('td'),
          th: breaking('th'),
          p: breaking('p'),
          li: breaking('li'),
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  )
}
