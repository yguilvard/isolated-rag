import { useState } from 'react'
import { ingest, type IngestResult } from '../api/client'
import styles from './Upload.module.css'

interface Props {
  onLogout: () => void
  onSessionExpired: () => void
}

export default function Upload({ onLogout, onSessionExpired }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [visibility, setVisibility] = useState<'private' | 'public'>('private')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault()
    if (!file) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const r = await ingest(file, visibility)
      setResult(r)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed'
      if (msg === 'Session expired') {
        onSessionExpired()
      } else {
        setError(msg)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <span className={styles.title}>isolated-rag</span>
        <button className={styles.logoutBtn} type="button" onClick={onLogout}>
          Log out
        </button>
      </header>

      <main className={styles.main}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.fileArea}>
            <input
              type="file"
              accept=".txt,.pdf,.md"
              onChange={e => {
                setFile(e.target.files?.[0] ?? null)
                setResult(null)
                setError('')
              }}
              required
            />
            <p className={styles.hint}>Accepted: .txt, .pdf, .md</p>
          </div>

          <div className={styles.toggle} role="group" aria-label="Visibility">
            <button
              type="button"
              className={visibility === 'private' ? styles.activeToggle : styles.inactiveToggle}
              onClick={() => setVisibility('private')}
            >
              Personal
            </button>
            <button
              type="button"
              className={visibility === 'public' ? styles.activeToggle : styles.inactiveToggle}
              onClick={() => setVisibility('public')}
            >
              Shared
            </button>
          </div>

          <button
            className={styles.submitBtn}
            type="submit"
            disabled={!file || loading}
          >
            {loading ? 'Uploading…' : 'Upload'}
          </button>
        </form>

        {result && (
          <p className={styles.success}>
            ✓ {result.chunks_ingested} chunks ingested from {result.document}
          </p>
        )}
        {error && <p className={styles.error}>{error}</p>}
      </main>
    </div>
  )
}
