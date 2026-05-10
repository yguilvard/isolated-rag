import { useState } from 'react'
import Login from './pages/Login'
import Upload from './pages/Upload'
import DocumentDetail from './pages/DocumentDetail'
import type { DocumentSummary } from './api/client'

export default function App() {
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('token'),
  )
  const [selectedDoc, setSelectedDoc] = useState<DocumentSummary | null>(null)

  function handleLogin(t: string): void {
    localStorage.setItem('token', t)
    setToken(t)
  }

  function handleLogout(): void {
    localStorage.removeItem('token')
    setToken(null)
    setSelectedDoc(null)
  }

  if (!token) {
    return <Login onLogin={handleLogin} />
  }

  if (selectedDoc) {
    return (
      <DocumentDetail
        doc={selectedDoc}
        onBack={() => setSelectedDoc(null)}
        onSessionExpired={handleLogout}
      />
    )
  }

  return (
    <Upload
      onLogout={handleLogout}
      onSessionExpired={handleLogout}
      onOpenDoc={setSelectedDoc}
    />
  )
}
