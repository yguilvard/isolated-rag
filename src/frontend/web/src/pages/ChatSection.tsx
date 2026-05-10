import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchModels, streamChat } from '../api/client'
import type { ChatProvider, ChatSource } from '../api/client'
import styles from './ChatSection.module.css'

/* ── Source modal helpers ─────────────────────────────────────────────────── */

function isPstSource(doc: string) {
  return doc.includes('::')
}

interface EmailParts {
  from: string; to: string; date: string; subject: string; body: string
}

function parseEmail(content: string): EmailParts {
  const [headerBlock, ...bodyParts] = content.split('\n\n')
  const body = bodyParts.join('\n\n').trim()
  const get = (key: string) => {
    const m = headerBlock.match(new RegExp(`^${key}:\\s*(.*)`, 'mi'))
    return m ? m[1].trim() : ''
  }
  return { from: get('From'), to: get('To'), date: get('Date'), subject: get('Subject'), body }
}

function SourceModal({ source, onClose }: { source: ChatSource; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const isPst = isPstSource(source.document)
  const isAttachment = source.document.includes('::att:')
  const email = isPst && !isAttachment ? parseEmail(source.content) : null

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()} role="dialog" aria-modal>
        <div className={styles.modalHeader}>
          <div className={styles.modalMeta}>
            <span className="material-symbols-outlined">
              {isPst ? (isAttachment ? 'attach_file' : 'mail') : 'description'}
            </span>
            <span className={styles.modalDoc}>{source.document}</span>
            <span className={styles.modalChip}>chunk #{source.chunk_index}</span>
            <span className={styles.modalChip}>{(source.score * 100).toFixed(1)}% match</span>
          </div>
          <button className={styles.modalClose} onClick={onClose} aria-label="Close">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className={styles.modalBody}>
          {email ? (
            <>
              <table className={styles.emailHeaders}>
                <tbody>
                  {email.from    && <tr><td className={styles.emailLabel}>From</td><td>{email.from}</td></tr>}
                  {email.to      && <tr><td className={styles.emailLabel}>To</td><td>{email.to}</td></tr>}
                  {email.date    && <tr><td className={styles.emailLabel}>Date</td><td>{email.date}</td></tr>}
                  {email.subject && <tr><td className={styles.emailLabel}>Subject</td><td><strong>{email.subject}</strong></td></tr>}
                </tbody>
              </table>
              <div className={styles.emailDivider} />
              <pre className={styles.modalContent}>{email.body || '(no body)'}</pre>
            </>
          ) : (
            <pre className={styles.modalContent}>{source.content}</pre>
          )}
        </div>
      </div>
    </div>
  )
}

interface Props {
  onSessionExpired: () => void
}

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  noContext?: boolean
  error?: boolean
}

let _msgId = 0
function nextId() { return ++_msgId }

export default function ChatSection({ onSessionExpired }: Props) {
  const [models, setModels] = useState<ChatProvider[]>([])
  const [modelId, setModelId] = useState<string>('')
  const [useRag, setUseRag] = useState(true)
  const [scope, setScope] = useState<'private' | 'public' | 'all'>('all')
  const [messages, setMessages] = useState<Message[]>([])
  const [query, setQuery] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [selectedSource, setSelectedSource] = useState<ChatSource | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<boolean>(false)

  // Fetch available models on mount
  useEffect(() => {
    fetchModels()
      .then(list => {
        setModels(list)
        if (list.length > 0) setModelId(list[0].id)
      })
      .catch(err => {
        if (err.message === 'Session expired') onSessionExpired()
      })
  }, [onSessionExpired])

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async () => {
    const q = query.trim()
    if (!q || !modelId || streaming) return

    const userMsg: Message = { id: nextId(), role: 'user', content: q }
    const assistantId = nextId()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '' }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setQuery('')
    setStreaming(true)
    abortRef.current = false

    let sources: ChatSource[] = []

    try {
      for await (const event of streamChat({ query: q, modelId, useRag, scope, topK: 5 })) {
        if (abortRef.current) break
        if (event.type === 'sources') {
          sources = event.items
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, sources, noContext: event.no_context } : m)
          )
        } else if (event.type === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId ? { ...m, content: m.content + event.content } : m
            )
          )
        } else if (event.type === 'error') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId ? { ...m, content: event.message, error: true } : m
            )
          )
          break
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      if (msg === 'Session expired') {
        onSessionExpired()
        return
      }
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId ? { ...m, content: msg, error: true } : m
        )
      )
    } finally {
      setStreaming(false)
    }
  }, [query, modelId, useRag, scope, streaming, onSessionExpired])

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  return (
    <div className={styles.wrap}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        {/* Model selector */}
        <div className={styles.toolGroup}>
          <span className={styles.toolLabel}>Model</span>
          <select
            className={styles.select}
            value={modelId}
            onChange={e => setModelId(e.target.value)}
            disabled={streaming || models.length === 0}
          >
            {models.length === 0 && <option value="">No models</option>}
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>

        {/* RAG toggle */}
        <div className={styles.toolGroup}>
          <span className={styles.toolLabel}>Context</span>
          <button
            type="button"
            className={`${styles.pill} ${useRag ? styles.pillActive : ''}`}
            onClick={() => setUseRag(v => !v)}
            disabled={streaming}
            title={useRag ? 'RAG on — answers grounded in your documents' : 'RAG off — pure model knowledge'}
          >
            <span className="material-symbols-outlined">{useRag ? 'library_books' : 'neurology'}</span>
            {useRag ? 'Documents' : 'Training data'}
          </button>
        </div>

        {/* Scope — only when RAG is on */}
        {useRag && (
          <div className={styles.toolGroup}>
            <span className={styles.toolLabel}>Scope</span>
            <div className={styles.toggle}>
              {(['all', 'private', 'public'] as const).map(s => (
                <button
                  key={s}
                  type="button"
                  className={scope === s ? styles.toggleActive : styles.toggleInactive}
                  onClick={() => setScope(s)}
                  disabled={streaming}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Clear */}
        {messages.length > 0 && (
          <button
            type="button"
            className={styles.clearBtn}
            onClick={() => setMessages([])}
            disabled={streaming}
            title="Clear conversation"
          >
            <span className="material-symbols-outlined">delete_sweep</span>
          </button>
        )}
      </div>

      {/* Message history */}
      <div className={styles.history}>
        {messages.length === 0 && (
          <div className={styles.empty}>
            <span className="material-symbols-outlined">chat</span>
            <p>Ask a question{useRag ? ' \u2014 I\u2019ll search your documents for context' : ''}</p>
          </div>
        )}
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`${styles.bubble} ${msg.role === 'user' ? styles.user : styles.assistant} ${msg.error ? styles.errorBubble : ''}`}
          >
            {msg.role === 'user' && (
              <button
                type="button"
                className={styles.retryBtn}
                onClick={() => setQuery(msg.content)}
                title="Edit and resend"
                aria-label="Retry this message"
              >
                <span className="material-symbols-outlined">edit</span>
              </button>
            )}
            {msg.role === 'assistant' && msg.noContext && (
              <div className={styles.noContext}>
                <span className="material-symbols-outlined">search_off</span>
                No relevant content found in your documents — answering from model knowledge
              </div>
            )}
            {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
              <details className={styles.sources}>
                <summary className={styles.sourcesSummary}>
                  <span className="material-symbols-outlined">source</span>
                  {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                </summary>
                <ul className={styles.sourceList}>
                  {msg.sources.map((s, i) => (
                    <li key={i} className={styles.sourceItem} onClick={() => setSelectedSource(s)} title="Click to view full source">
                      <span className={styles.sourceDoc}>
                        <span className="material-symbols-outlined">{s.document.includes('::') ? (s.document.includes('::att:') ? 'attach_file' : 'mail') : 'description'}</span>
                        {s.document}
                      </span>
                      <span className={styles.sourceMeta}>chunk #{s.chunk_index} · {(s.score * 100).toFixed(1)}% match</span>
                      <p className={styles.sourceContent}>{s.content.slice(0, 200)}{s.content.length > 200 ? '…' : ''}</p>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <div className={styles.text}>
              {msg.content || (msg.role === 'assistant' && !msg.error && (
                <span className={styles.cursor} aria-label="typing" />
              ))}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {selectedSource && (
        <SourceModal source={selectedSource} onClose={() => setSelectedSource(null)} />
      )}

      {/* Input area */}
      <div className={styles.inputRow}>
        <textarea
          className={styles.textarea}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          rows={2}
          disabled={streaming || !modelId}
        />
        <button
          type="button"
          className={`${styles.sendBtn} ${streaming ? styles.stopBtn : ''}`}
          onClick={() => streaming ? (abortRef.current = true) : void send()}
          disabled={!streaming && (!query.trim() || !modelId)}
          aria-label={streaming ? 'Stop generation' : 'Send'}
        >
          {streaming
            ? <span className="material-symbols-outlined">stop_circle</span>
            : <span className="material-symbols-outlined">send</span>}
        </button>
      </div>
    </div>
  )
}
