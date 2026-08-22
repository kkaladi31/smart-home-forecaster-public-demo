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
export default function Markdown({ children }) {
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
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
