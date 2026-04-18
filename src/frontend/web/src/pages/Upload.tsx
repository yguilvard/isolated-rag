import { useState, useEffect } from 'react'
import DocumentList from './DocumentList'
import SearchSection from './SearchSection'
import styles from './Upload.module.css'
import type { DocumentSummary } from '../api/client'

interface Props {
  onLogout: () => void
  onSessionExpired: () => void
  onOpenDoc: (doc: DocumentSummary) => void
}

interface AppInfo {
  version: string
  embedding_model: string
}

export default function Upload({ onLogout, onSessionExpired, onOpenDoc }: Props) {
  const [info, setInfo] = useState<AppInfo | null>(null)

  useEffect(() => {
    fetch('/info')
      .then(r => r.json())
      .then((data: AppInfo) => setInfo(data))
      .catch(() => { /* non-critical */ })
  }, [])

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.title}>isolated-rag</span>
          {info && <span className={styles.version}>v{info.version}</span>}
        </div>
        <div className={styles.headerRight}>
          {info && (
            <span className={styles.modelBadge} title="Embedding model">
              {info.embedding_model}
            </span>
          )}
          <button className={styles.logoutBtn} type="button" onClick={onLogout}>
            Log out
          </button>
        </div>
      </header>

      <div className={styles.body}>
        {/* Left pane — document list */}
        <DocumentList refreshKey={0} onSessionExpired={onSessionExpired} onOpenDoc={onOpenDoc} />

        {/* Right pane — search + upload */}
        <main className={styles.main}>
          <div className={styles.searchCard}>
            <h2 className={styles.sectionTitle}>Semantic search</h2>
            <SearchSection onSessionExpired={onSessionExpired} />
          </div>
        </main>
      </div>
    </div>
  )
}
