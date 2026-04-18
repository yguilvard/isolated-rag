import { useState, useEffect, useMemo } from 'react'
import { fetchDocumentChunks, type DocumentSummary, type DocumentChunk } from '../api/client'
import styles from './DocumentDetail.module.css'

interface Props {
  doc: DocumentSummary
  onBack: () => void
  onSessionExpired: () => void
}

type SortKey = 'index' | 'chars' | 'words'
type SortDir = 'asc' | 'desc'

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
  })
}

function wordCount(text: string) {
  return text.trim().split(/\s+/).filter(Boolean).length
}

// ── ChunkRow ──────────────────────────────────────────────────────────────────

function ChunkRow({ chunk }: { chunk: DocumentChunk }) {
  const [expanded, setExpanded] = useState(false)
  const chars = chunk.content.length
  const words = wordCount(chunk.content)
  const isLong = chunk.content.length > 300

  return (
    <tr className={styles.row}>
      <td className={styles.cellIdx}>#{chunk.index}</td>
      <td className={styles.cellNum}>{chars.toLocaleString()}</td>
      <td className={styles.cellNum}>{words.toLocaleString()}</td>
      <td className={styles.cellContent}>
        <span className={styles.contentText}>
          {isLong && !expanded
            ? chunk.content.slice(0, 300) + '…'
            : chunk.content}
        </span>
        {isLong && (
          <button
            className={styles.expandBtn}
            onClick={() => setExpanded(e => !e)}
          >
            <span className="material-symbols-outlined">
              {expanded ? 'unfold_less' : 'unfold_more'}
            </span>
            {expanded ? 'Collapse' : 'Expand'}
          </button>
        )}
      </td>
    </tr>
  )
}

// ── DocumentDetail ────────────────────────────────────────────────────────────

export default function DocumentDetail({ doc, onBack, onSessionExpired }: Props) {
  const [chunks, setChunks] = useState<DocumentChunk[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('index')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  useEffect(() => {
    setLoading(true)
    setError('')
    fetchDocumentChunks(doc.name)
      .then(setChunks)
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to load chunks'
        if (msg === 'Session expired') onSessionExpired()
        else setError(msg)
      })
      .finally(() => setLoading(false))
  }, [doc.name, onSessionExpired])

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const enriched = useMemo(() =>
    chunks.map(c => ({
      ...c,
      chars: c.content.length,
      words: wordCount(c.content),
    })),
    [chunks],
  )

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    return q ? enriched.filter(c => c.content.toLowerCase().includes(q)) : enriched
  }, [enriched, filter])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const cmp = a[sortKey] < b[sortKey] ? -1 : a[sortKey] > b[sortKey] ? 1 : 0
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [filtered, sortKey, sortDir])

  // Stats computed over the full chunk set (not the filtered view)
  const stats = useMemo(() => {
    if (!enriched.length) return null
    const charCounts = enriched.map(c => c.chars)
    const wordCounts = enriched.map(c => c.words)
    return {
      totalChars: charCounts.reduce((a, b) => a + b, 0),
      minChars: Math.min(...charCounts),
      maxChars: Math.max(...charCounts),
      avgChars: Math.round(charCounts.reduce((a, b) => a + b, 0) / charCounts.length),
      avgWords: Math.round(wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length),
    }
  }, [enriched])

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <span className={`material-symbols-outlined ${styles.sortIcon}`}>unfold_more</span>
    return (
      <span className={`material-symbols-outlined ${styles.sortIcon} ${styles.sortActive}`}>
        {sortDir === 'asc' ? 'arrow_upward' : 'arrow_downward'}
      </span>
    )
  }

  return (
    <div className={styles.page}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <button className={styles.backBtn} onClick={onBack} aria-label="Back">
          <span className="material-symbols-outlined">arrow_back</span>
        </button>

        <div className={styles.headerMeta}>
          <h1 className={styles.docName} title={doc.name}>{doc.name}</h1>
          <div className={styles.badges}>
            <span className={doc.visibility === 'public' ? styles.badgePublic : styles.badgePrivate}>
              <span className="material-symbols-outlined">{doc.visibility === 'public' ? 'public' : 'lock'}</span>
              {doc.visibility === 'public' ? 'Shared' : 'Personal'}
            </span>
            <span className={styles.badge}>
              <span className="material-symbols-outlined">layers</span>
              {doc.chunks} chunk{doc.chunks !== 1 ? 's' : ''}
            </span>
            <span className={styles.badge}>
              <span className="material-symbols-outlined">calendar_today</span>
              {formatDate(doc.uploaded_at)}
            </span>
          </div>
        </div>
      </header>

      {/* ── Stats bar ── */}
      {stats && (
        <div className={styles.statsBar}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Total chars</span>
            <span className={styles.statValue}>{stats.totalChars.toLocaleString()}</span>
          </div>
          <div className={styles.statDivider} />
          <div className={styles.stat}>
            <span className={styles.statLabel}>Avg chars / chunk</span>
            <span className={styles.statValue}>{stats.avgChars.toLocaleString()}</span>
          </div>
          <div className={styles.statDivider} />
          <div className={styles.stat}>
            <span className={styles.statLabel}>Min / Max chars</span>
            <span className={styles.statValue}>{stats.minChars} / {stats.maxChars}</span>
          </div>
          <div className={styles.statDivider} />
          <div className={styles.stat}>
            <span className={styles.statLabel}>Avg words / chunk</span>
            <span className={styles.statValue}>{stats.avgWords}</span>
          </div>
        </div>
      )}

      {/* ── Toolbar ── */}
      <div className={styles.toolbar}>
        <div className={styles.filterWrap}>
          <span className={`material-symbols-outlined ${styles.filterIcon}`} aria-hidden>search</span>
          <input
            className={styles.filterInput}
            type="search"
            placeholder="Search within chunks…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            aria-label="Filter chunks"
          />
        </div>
        {filter && (
          <span className={styles.filterCount}>
            {sorted.length} / {enriched.length} chunk{enriched.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* ── Table ── */}
      <div className={styles.tableWrap}>
        {loading && (
          <div className={styles.stateMsg}>
            <span className={`material-symbols-outlined ${styles.spin}`}>progress_activity</span>
            Loading chunks…
          </div>
        )}
        {error && <div className={styles.errorMsg}>{error}</div>}

        {!loading && !error && sorted.length === 0 && (
          <div className={styles.stateMsg}>
            {filter ? 'No chunks match the filter.' : 'No chunks found.'}
          </div>
        )}

        {!loading && !error && sorted.length > 0 && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th className={`${styles.th} ${styles.thIdx}`}
                  onClick={() => toggleSort('index')}>
                  # <SortIcon col="index" />
                </th>
                <th className={`${styles.th} ${styles.thNum}`}
                  onClick={() => toggleSort('chars')}>
                  Chars <SortIcon col="chars" />
                </th>
                <th className={`${styles.th} ${styles.thNum}`}
                  onClick={() => toggleSort('words')}>
                  Words <SortIcon col="words" />
                </th>
                <th className={`${styles.th} ${styles.thContent}`}>Content</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(c => (
                <ChunkRow key={c.index} chunk={c} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
