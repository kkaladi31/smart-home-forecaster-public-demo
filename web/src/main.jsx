import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { installGlobalHandlers } from './logbus'

// Catch errors React never sees (uncaught exceptions, rejected promises) so they
// land in the Logs tab instead of only the browser console.
installGlobalHandlers()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
