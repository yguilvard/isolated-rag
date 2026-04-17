import { useState } from 'react'
import Login from './pages/Login'
import Upload from './pages/Upload'

export default function App() {
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('token'),
  )

  function handleLogin(t: string): void {
    localStorage.setItem('token', t)
    setToken(t)
  }

  function handleLogout(): void {
    localStorage.removeItem('token')
    setToken(null)
  }

  if (token) {
    return <Upload onLogout={handleLogout} onSessionExpired={handleLogout} />
  }
  return <Login onLogin={handleLogin} />
}
