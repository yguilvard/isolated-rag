import { useState } from 'react'
import { search, type SearchResultItem } from '../api/client'
import styles from './SearchSection.module.css'

interface Props {
  onSessionExpired: () => void
}

interface ModelParams {
  temperature: number
  topP: number
  maxTokens: number
  topK: number
}

type Scope = 'private' | 'public' | 'all'

const SCOPE_LABELS: Record<Scope, string> = {
  private: 'Personal',
  public: 'Shared',
  all: 'All',
}

export default function SearchSection({ onSessionExpired }: Props) {
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState<Scope>('all')
  const [paramsOpen, setParamsOpen] = useState(false)
  const [params, setParams] = useState<ModelParams>({
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 512,
    topK: 5,
  })
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SearchResultItem[] | null>(null)
  const [error, setError] = useState('')

  function setParam<K extends keyof ModelParams>(key: K, value: ModelParams[K]) {
    setParams(p => ({ ...p, [key]: value }))
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setError('')
    setResults(null)
    setLoading(true)
    try {
      const res = await search({ query: query.trim(), scope, topK: params.topK })
      setResults(res.results)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Search failed'
      if (msg === 'Session expired') {
        onSessionExpired()
      } else {
        setError(msg)
      }
    } finally {
      setLoading(false)
    }
  }

  const scopeDescription =
    scope === 'private'
      ? 'Searching your personal documents'
      : scope === 'public'
        ? 'Searching shared documents'
        : 'Searching all accessible documents'

  return (
    <section className={styles.section}>
      <form onSubmit={handleSearch} className={styles.form}>
        {/* Search input row */}
        <div className={styles.inputRow}>
          <div className={styles.inputWrap}>
            <span className={`material-symbols-outlined ${styles.searchIcon}`} aria-hidden>search</span>
            <input
              className={styles.input}
              type="search"
              placeholder="Search documents…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              aria-label="Search query"
            />
          </div>
          <button className={styles.searchBtn} type="submit" disabled={!query.trim() || loading}>
            {loading ? '…' : 'Search'}
          </button>
        </div>

        {/* Scope selector + model params row */}
        <div className={styles.controlRow}>
          <div className={styles.scopeGroup} role="group" aria-label="Document scope">
            {(['private', 'public', 'all'] as Scope[]).map(s => (
              <button
                key={s}
                type="button"
                className={scope === s ? styles.scopeActive : styles.scopeBtn}
                onClick={() => setScope(s)}
              >
                {SCOPE_LABELS[s]}
              </button>
            ))}
          </div>
          <span className={styles.scopeHint}>{scopeDescription}</span>

          <button
            type="button"
            className={styles.paramsToggle}
            onClick={() => setParamsOpen(o => !o)}
            aria-expanded={paramsOpen}
          >
            Model parameters
            <span className="material-symbols-outlined">{paramsOpen ? 'expand_less' : 'expand_more'}</span>
          </button>
        </div>

        {/* Collapsible model parameters */}
        {paramsOpen && (
          <div className={styles.paramsPanel}>
            <div className={styles.paramsGrid}>
              <label className={styles.paramLabel}>
                <span className={styles.paramName}>
                  Temperature
                  <span className={`material-symbols-outlined ${styles.infoIcon}`} data-tooltip="Controls randomness. Low values (e.g. 0.2) make output focused and deterministic; high values (e.g. 1.5) make it more creative.">info</span>
                </span>
                <span className={styles.paramValue}>{params.temperature.toFixed(1)}</span>
                <input
                  type="range" min={0} max={2} step={0.1}
                  value={params.temperature}
                  onChange={e => setParam('temperature', parseFloat(e.target.value))}
                />
              </label>

              <label className={styles.paramLabel}>
                <span className={styles.paramName}>
                  Top-P
                  <span className={`material-symbols-outlined ${styles.infoIcon}`} data-tooltip="Nucleus sampling: only tokens whose cumulative probability exceeds this threshold are considered. Lower = more conservative output.">info</span>
                </span>
                <span className={styles.paramValue}>{params.topP.toFixed(2)}</span>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={params.topP}
                  onChange={e => setParam('topP', parseFloat(e.target.value))}
                />
              </label>

              <label className={styles.paramLabel}>
                <span className={styles.paramName}>
                  Max tokens
                  <span className={`material-symbols-outlined ${styles.infoIcon}`} data-tooltip="Maximum tokens the model may generate. Higher values allow longer answers but increase latency.">info</span>
                </span>
                <input
                  type="number" min={64} max={4096} step={64}
                  value={params.maxTokens}
                  onChange={e => setParam('maxTokens', parseInt(e.target.value, 10))}
                  className={styles.numInput}
                />
              </label>

              <label className={styles.paramLabel}>
                <span className={styles.paramName}>
                  Top-K results
                  <span className={`material-symbols-outlined ${styles.infoIcon}`} data-tooltip="Number of document chunks retrieved from the vector store and fed as context to the chat model.">info</span>
                </span>
                <span className={styles.paramValue}>{params.topK}</span>
                <input
                  type="range" min={1} max={20} step={1}
                  value={params.topK}
                  onChange={e => setParam('topK', parseInt(e.target.value, 10))}
                />
              </label>
            </div>
          </div>
        )}
      </form>

      {/* Results */}
      {error && <p className={styles.error}>{error}</p>}

      {results !== null && (
        <div className={styles.results}>
          <p className={styles.resultsMeta}>
            {results.length === 0
              ? 'No results found.'
              : `${results.length} result${results.length === 1 ? '' : 's'} from ${scope === 'private' ? 'personal' : scope === 'public' ? 'shared' : 'all'} documents`}
          </p>
          {results.map((r, i) => (
            <div key={i} className={styles.resultCard}>
              <div className={styles.resultHeader}>
                <span className={styles.resultDoc} title={r.document}>
                  {r.document.split('/').pop() ?? r.document}
                </span>
                <span className={styles.resultMeta}>
                  chunk {r.chunk_index} · {(r.score * 100).toFixed(1)}% match
                </span>
              </div>
              <p className={styles.resultContent}>{r.content}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
