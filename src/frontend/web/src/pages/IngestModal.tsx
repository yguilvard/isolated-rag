import { useState, useEffect, useRef } from 'react'
import { ingest } from '../api/client'
import styles from './IngestModal.module.css'

interface IngestionSettings {
  chunkSentences: number
  overlapSentences: number
}

interface Props {
  defaultVisibility: 'private' | 'public'
  onClose: () => void
  onSuccess: () => void
  onSessionExpired: () => void
}

export default function IngestModal({ defaultVisibility, onClose, onSuccess, onSessionExpired }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [documentTitle, setDocumentTitle] = useState('')
  const [visibility, setVisibility] = useState<'private' | 'public'>(defaultVisibility)
  const [ingestion, setIngestion] = useState<IngestionSettings>({ chunkSentences: 5, overlapSentences: 1 })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const dialogRef = useRef<HTMLDivElement>(null)

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  function setIng<K extends keyof IngestionSettings>(key: K, value: IngestionSettings[K]) {
    setIngestion(s => ({ ...s, [key]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setError('')
    setLoading(true)
    try {
      await ingest(file, {
        visibility,
        chunkSentences: ingestion.chunkSentences,
        overlapSentences: ingestion.overlapSentences,
        documentTitle: documentTitle.trim() || file.name,
      })
      onSuccess()
      onClose()
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
    <div
      className={styles.backdrop}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
      role="dialog"
      aria-modal="true"
      aria-label="Ingest document"
    >
      <div className={styles.dialog} ref={dialogRef}>
        {/* Header */}
        <div className={styles.header}>
          <span className={`material-symbols-outlined ${styles.headerIcon}`}>upload_file</span>
          <h2 className={styles.title}>Ingest document</h2>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {/* File picker */}
          <div className={styles.fileArea}>
            <span className={`material-symbols-outlined ${styles.fileAreaIcon}`}>
              {file ? 'description' : 'cloud_upload'}
            </span>
            <label className={styles.fileLabel}>
              <span className={styles.fileLabelText}>
                {file ? file.name : 'Choose a file…'}
              </span>
              <input
                type="file"
                accept=".txt,.pdf,.md"
                className={styles.fileInput}
                onChange={e => {
                  const picked = e.target.files?.[0] ?? null
                  setFile(picked)
                  setError('')
                  if (picked) {
                    setDocumentTitle(picked.name.replace(/\.[^.]+$/, ''))
                  }
                }}
              />
            </label>
            <p className={styles.hint}>.txt · .pdf · .md</p>
          </div>

          {/* Document title */}
          <label className={styles.fieldLabel}>
            Title
            <input
              type="text"
              className={styles.textInput}
              placeholder="Document title…"
              value={documentTitle}
              onChange={e => setDocumentTitle(e.target.value)}
              maxLength={255}
            />
          </label>

          {/* Visibility */}
          <div className={styles.fieldLabel}>
            Visibility
            <div className={styles.toggle} role="group" aria-label="Visibility">
              <button
                type="button"
                className={visibility === 'private' ? styles.toggleActive : styles.toggleInactive}
                onClick={() => setVisibility('private')}
              >
                <span className="material-symbols-outlined">lock</span>
                Personal
              </button>
              <button
                type="button"
                className={visibility === 'public' ? styles.toggleActive : styles.toggleInactive}
                onClick={() => setVisibility('public')}
              >
                <span className="material-symbols-outlined">public</span>
                Shared
              </button>
            </div>
          </div>

          {/* Chunking settings */}
          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Chunking</legend>
            <div className={styles.grid}>
              <label className={styles.rangeLabel}>
                <span>Chunk size</span>
                <span className={styles.rangeValue}>{ingestion.chunkSentences} sent.</span>
                <input
                  type="range" min={1} max={20} step={1}
                  value={ingestion.chunkSentences}
                  onChange={e => setIng('chunkSentences', parseInt(e.target.value, 10))}
                />
              </label>
              <label className={styles.rangeLabel}>
                <span>Overlap</span>
                <span className={styles.rangeValue}>{ingestion.overlapSentences} sent.</span>
                <input
                  type="range" min={0} max={10} step={1}
                  value={ingestion.overlapSentences}
                  onChange={e => setIng('overlapSentences', parseInt(e.target.value, 10))}
                />
              </label>
            </div>
          </fieldset>

          {error && <p className={styles.error}>{error}</p>}

          <button className={styles.submitBtn} type="submit" disabled={!file || loading}>
            {loading
              ? <><span className={`material-symbols-outlined ${styles.spin}`}>progress_activity</span> Uploading…</>
              : <><span className="material-symbols-outlined">upload</span> Upload</>
            }
          </button>
        </form>
      </div>
    </div>
  )
}
