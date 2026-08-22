import { Component } from 'react'
import { log } from '../logbus'

/**
 * Without a boundary, any render-time exception unmounts the whole tree and the
 * user just sees a white page — which is also useless for diagnosis. This keeps
 * the rest of the app alive and shows what actually failed.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Surfaced in the browser console for debugging, and in the Logs tab so it
    // survives the console being closed.
    console.error('[Smart-Home Forecaster] render error:', error, info)
    log('render', 'render.error', `${this.props.label ?? 'A panel'} failed to render: ${error.message}`, {
      level: 'error',
      data: { label: this.props.label, stack: info?.componentStack },
    })
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <div className="card p-4">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--status-critical)' }}>
          {this.props.label ?? 'This panel'} could not be displayed
        </h3>
        <p className="text-xs secondary mt-1">{String(this.state.error?.message ?? this.state.error)}</p>
        <button
          onClick={() => this.setState({ error: null })}
          className="mt-2 text-xs px-2 py-1 rounded"
          style={{ border: '1px solid var(--border)' }}
        >
          Try again
        </button>
      </div>
    )
  }
}
