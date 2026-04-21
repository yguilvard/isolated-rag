import { useState, useEffect } from 'react'
import DocumentList from './DocumentList'
import SearchSection from './SearchSection'
import ChatSection from './ChatSection'
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

type RightTab = 'search' | 'chat'

export default function Upload({ onLogout, onSessionExpired, onOpenDoc }: Props) {
  const [info, setInfo] = useState<AppInfo | null>(null)
  const [tab, setTab] = useState<RightTab>('search')

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

        {/* Right pane — tabbed: Search | Chat */}
        <main className={styles.main}>
          {/* Tab switcher */}
          <div className={styles.tabs}>
            <button
              type="button"
              className={`${styles.tab} ${tab === 'search' ? styles.tabActive : ''}`}
              onClick={() => setTab('search')}
            >
              <span className="material-symbols-outlined">search</span>
              Search
            </button>
            <button
              type="button"
              className={`${styles.tab} ${tab === 'chat' ? styles.tabActive : ''}`}
              onClick={() => setTab('chat')}
            >
              <span className="material-symbols-outlined">chat</span>
              Assistant
            </button>
          </div>

          {tab === 'search' && (
            <div className={styles.searchCard}>
              <h2 className={styles.sectionTitle}>Semantic search</h2>
              <SearchSection onSessionExpired={onSessionExpired} />
            </div>
          )}

          {tab === 'chat' && (
            <div className={styles.chatCard}>
              <ChatSection onSessionExpired={onSessionExpired} />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
