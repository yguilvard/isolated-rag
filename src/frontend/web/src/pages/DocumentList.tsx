import { useState, useEffect, useCallback } from 'react'
import {
  fetchDocuments,
  deleteDocument,
  setDocumentVisibility,
  type DocumentSummary,
} from '../api/client'
import IngestModal from './IngestModal'
import styles from './DocumentList.module.css'

interface Props {
  refreshKey: number
  onSessionExpired: () => void
  onOpenDoc: (doc: DocumentSummary) => void
}

const EXT_ICON: Record<string, string> = {
  pdf: 'picture_as_pdf',
  txt: 'description',
  md: 'article',
}
function fileIconName(name: string) {
  return EXT_ICON[name.split('.').pop()?.toLowerCase() ?? ''] ?? 'folder'
}
function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

// ── DocRow ─────────────────────────────────────────────────────────────────────

interface DocRowProps {
  doc: DocumentSummary
  onDelete: (name: string, e: React.MouseEvent) => void
  onShare: (name: string, visibility: 'private' | 'public', e: React.MouseEvent) => void
  onOpen: (doc: DocumentSummary) => void
  pendingDelete: string | null
  deleting: string | null
  sharing: string | null
}

function DocRow({ doc, onDelete, onShare, onOpen, pendingDelete, deleting, sharing }: DocRowProps) {
  const [expanded, setExpanded] = useState(false)
  const isPending = pendingDelete === doc.name
  const isDeleting = deleting === doc.name
  const isSharing = sharing === doc.name
  const nextVisibility = doc.visibility === 'private' ? 'public' : 'private'

  return (
    <li className={styles.docItem}>
      <div
        className={styles.docRow}
        onClick={() => setExpanded(e => !e)}
        role="button"
        tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && setExpanded(v => !v)}
        aria-expanded={expanded}
      >
        <span className={`material-symbols-outlined ${styles.fileIcon}`} aria-hidden>
          {fileIconName(doc.name)}
        </span>

        <span className={styles.docName} title={doc.name}>{doc.name}</span>

        <span className={`material-symbols-outlined ${styles.chevron} ${expanded ? styles.chevronOpen : ''}`} aria-hidden>
          chevron_right
        </span>

        <button
          className={`${styles.actionBtn} ${styles.actionOpen}`}
          title="Inspect chunks"
          aria-label="Inspect chunks"
          onClick={e => { e.stopPropagation(); onOpen(doc) }}
        >
          <span className="material-symbols-outlined">table_rows</span>
        </button>

        <button
          className={`${styles.actionBtn} ${styles.actionShare}`}
          title={doc.visibility === 'private' ? 'Share with everyone' : 'Make private'}
          aria-label={doc.visibility === 'private' ? 'Share' : 'Make private'}
          disabled={isSharing}
          onClick={e => onShare(doc.name, nextVisibility, e)}
        >
          <span className="material-symbols-outlined">
            {isSharing ? 'more_horiz' : doc.visibility === 'private' ? 'share' : 'lock'}
          </span>
        </button>

        {isPending ? (
          <button
            className={`${styles.actionBtn} ${styles.actionConfirm}`}
            title="Confirm delete"
            aria-label="Confirm delete"
            onClick={e => onDelete(doc.name, e)}
          >
            <span className="material-symbols-outlined">check</span>
          </button>
        ) : (
          <button
            className={`${styles.actionBtn} ${styles.actionDanger}`}
            title="Delete document"
            aria-label="Delete document"
            disabled={isDeleting}
            onClick={e => onDelete(doc.name, e)}
          >
            <span className="material-symbols-outlined">
              {isDeleting ? 'more_horiz' : 'delete'}
            </span>
          </button>
        )}
      </div>

      {isPending && (
        <p className={styles.confirmHint}>Click check to confirm delete.</p>
      )}

      {expanded && (
        <dl className={styles.summary}>
          <div className={styles.summaryRow}>
            <span className={`material-symbols-outlined ${styles.summaryIcon}`}>layers</span>
            <dt>Chunks</dt>
            <dd>{doc.chunks}</dd>
          </div>
          <div className={styles.summaryRow}>
            <span className={`material-symbols-outlined ${styles.summaryIcon}`}>straighten</span>
            <dt>Avg size</dt>
            <dd>~{doc.avg_chars} chars</dd>
          </div>
          <div className={styles.summaryRow}>
            <span className={`material-symbols-outlined ${styles.summaryIcon}`}>calendar_today</span>
            <dt>Uploaded</dt>
            <dd>{formatDate(doc.uploaded_at)}</dd>
          </div>
        </dl>
      )}
    </li>
  )
}

// ── TreeSection ────────────────────────────────────────────────────────────────

interface TreeSectionProps {
  title: string
  icon: string
  docs: DocumentSummary[]
  onDelete: (name: string, e: React.MouseEvent) => void
  onShare: (name: string, visibility: 'private' | 'public', e: React.MouseEvent) => void
  onOpen: (doc: DocumentSummary) => void
  onAdd: (e: React.MouseEvent) => void
  pendingDelete: string | null
  deleting: string | null
  sharing: string | null
}

function TreeSection({ title, icon, docs, onDelete, onShare, onOpen, onAdd, pendingDelete, deleting, sharing }: TreeSectionProps) {
  const [open, setOpen] = useState(true)

  return (
    <div className={styles.tree}>
      <div className={styles.treeHeader}>
        <button
          className={styles.treeRoot}
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
        >
          <span className={`material-symbols-outlined ${styles.treeIcon}`} aria-hidden>{icon}</span>
          <span className={styles.treeLabel}>{title}</span>
          <span className={styles.treeCount}>{docs.length}</span>
          <span className={`material-symbols-outlined ${styles.treeChevron}`} aria-hidden>
            {open ? 'expand_less' : 'expand_more'}
          </span>
        </button>
        <button
          className={styles.addBtn}
          title={`Add to ${title}`}
          aria-label={`Add document to ${title}`}
          onClick={onAdd}
        >
          <span className="material-symbols-outlined">add</span>
        </button>
      </div>

      {open && (
        <ul className={styles.treeList}>
          {docs.length === 0 ? (
            <li className={styles.emptyBranch}>No documents</li>
          ) : (
            docs.map(doc => (
              <DocRow
                key={doc.name}
                doc={doc}
                onDelete={onDelete}
                onShare={onShare}
                onOpen={onOpen}
                pendingDelete={pendingDelete}
                deleting={deleting}
                sharing={sharing}
              />
            ))
          )}
        </ul>
      )}
    </div>
  )
}



// ── DocumentList ───────────────────────────────────────────────────────────────

export default function DocumentList({ refreshKey, onSessionExpired, onOpenDoc }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [sharing, setSharing] = useState<string | null>(null)
  const [modalVisibility, setModalVisibility] = useState<'private' | 'public' | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setDocs(await fetchDocuments())
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load'
      msg === 'Session expired' ? onSessionExpired() : setError(msg)
    } finally {
      setLoading(false)
    }
  }, [onSessionExpired])

  useEffect(() => { load() }, [load, refreshKey])

  // Close pending-delete confirmation when clicking elsewhere
  useEffect(() => {
    if (!pendingDelete) return
    const handler = () => setPendingDelete(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [pendingDelete])

  async function handleShare(name: string, visibility: 'private' | 'public', e: React.MouseEvent) {
    e.stopPropagation()
    setSharing(name)
    try {
      await setDocumentVisibility(name, visibility)
      setDocs(d => d.map(doc => doc.name === name ? { ...doc, visibility } : doc))
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Update failed'
      if (msg === 'Session expired') onSessionExpired()
      else setError(msg)
    } finally {
      setSharing(null)
    }
  }

  async function handleDelete(name: string, e: React.MouseEvent) {
    e.stopPropagation()
    if (pendingDelete !== name) { setPendingDelete(name); return }
    setPendingDelete(null)
    setDeleting(name)
    try {
      await deleteDocument(name)
      setDocs(d => d.filter(doc => doc.name !== name))
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Delete failed'
      if (msg === 'Session expired') onSessionExpired()
      else setError(msg)
    } finally {
      setDeleting(null)
    }
  }

  const filtered = filter.trim()
    ? docs.filter(d => d.name.toLowerCase().includes(filter.toLowerCase()))
    : docs

  const personal = filtered.filter(d => d.visibility === 'private')
  const shared = filtered.filter(d => d.visibility === 'public')

  return (
    <aside className={styles.aside}>
      {/* Heading */}
      <div className={styles.heading}>
        <span className={styles.headingTitle}>Documents</span>
        <button className={styles.refreshBtn} onClick={load} title="Refresh" aria-label="Refresh">
          <span className="material-symbols-outlined">refresh</span>
        </button>
      </div>

      {/* Filter */}
      <div className={styles.filterWrap}>
        <span className={`material-symbols-outlined ${styles.filterIcon}`} aria-hidden>search</span>
        <input
          className={styles.filterInput}
          type="search"
          placeholder="Filter…"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          aria-label="Filter documents"
        />
      </div>

      {/* States */}
      {loading && <p className={styles.placeholder}>Loading…</p>}
      {error && <p className={styles.errorMsg}>{error}</p>}

      {/* Trees */}
      {!loading && !error && (
        <div className={styles.trees}>
          <TreeSection
            title="Personal"
            icon="lock"
            docs={personal}
            onDelete={handleDelete}
            onShare={handleShare}
            onOpen={onOpenDoc}
            onAdd={e => { e.stopPropagation(); setModalVisibility('private') }}
            pendingDelete={pendingDelete}
            deleting={deleting}
            sharing={sharing}
          />
          <TreeSection
            title="Shared"
            icon="public"
            docs={shared}
            onDelete={handleDelete}
            onShare={handleShare}
            onOpen={onOpenDoc}
            onAdd={e => { e.stopPropagation(); setModalVisibility('public') }}
            pendingDelete={pendingDelete}
            deleting={deleting}
            sharing={sharing}
          />
        </div>
      )}

      {/* Ingest modal */}
      {modalVisibility !== null && (
        <IngestModal
          defaultVisibility={modalVisibility}
          onClose={() => setModalVisibility(null)}
          onSuccess={load}
          onSessionExpired={onSessionExpired}
        />
      )}
    </aside>
  )
}
