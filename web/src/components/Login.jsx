import { useState } from 'react'
import { login } from '../api'

/**
 * Demo login. This exists to show an authentication flow — it deliberately keeps
 * no user records and no personal data, because the capstone forbids putting
 * private information in the project (see docs/safety.md).
 */
export default function Login({ onSuccess, status }) {
  const [username, setUsername] = useState('demo')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      const user = await login(username, password)
      onSuccess(user)
    } catch {
      setError('Invalid credentials. Try demo / forecaster.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-full grid place-items-center p-6">
      <div className="card p-6 w-full max-w-sm">
        <h1 className="text-lg font-semibold">Smart-Home Forecaster</h1>
        <p className="text-xs muted mt-1">
          Proactive home-risk agent — weather hazards, home rules, repairs, and bills.
        </p>

        <form onSubmit={submit} className="mt-5 space-y-3">
          <Field label="Username" value={username} onChange={setUsername} />
          <Field label="Password" value={password} onChange={setPassword} type="password" />
          {error && (
            <p className="text-xs" style={{ color: 'var(--status-critical)' }}>{error}</p>
          )}
          <button
            type="submit"
            disabled={busy}
            className="w-full py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
            style={{ background: 'var(--series-1)' }}
          >
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <div className="mt-4 text-xs muted space-y-1">
          <p>Demo credentials: <code className="secondary">demo / forecaster</code></p>
          {status && (
            <p>
              Model <code className="secondary">{status.model}</code> ·{' '}
              {status.knowledge_passages} knowledge passages
            </p>
          )}
          <p>No personal data is stored — this login is for demonstration only.</p>
        </div>
      </div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text' }) {
  return (
    <label className="block">
      <span className="text-xs secondary">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none"
        style={{
          background: 'var(--surface-page)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
        }}
      />
    </label>
  )
}
