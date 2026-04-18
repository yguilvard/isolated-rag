export interface IngestResult {
  status: string
  chunks_ingested: number
  document: string
}

export interface SearchResultItem {
  content: string
  document: string
  chunk_index: number
  score: number
}

export interface SearchResponse {
  query: string
  scope: string
  results: SearchResultItem[]
}

export async function login(username: string, password: string): Promise<string> {
  // POST credentials as form-encoded (OAuth2PasswordRequestForm)
  const body = new URLSearchParams({ username, password })
  const res = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  if (!res.ok) throw new Error('Invalid credentials')
  const data = await res.json()
  return data.access_token
}

export interface IngestOptions {
  visibility: 'private' | 'public'
  chunkSentences: number
  overlapSentences: number
  documentTitle: string
}

export async function ingest(file: File, options: IngestOptions): Promise<IngestResult> {
  // Build multipart form with file, visibility, chunking settings, and title
  const form = new FormData()
  form.append('file', file)
  form.append('visibility', options.visibility)
  form.append('chunk_sentences', String(options.chunkSentences))
  form.append('overlap_sentences', String(options.overlapSentences))
  form.append('document_title', options.documentTitle)

  const token = localStorage.getItem('token')
  if (!token) {
    throw new Error('Not authenticated')
  }
  const res = await fetch('/ingest', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })

  // Expired or missing token — caller should redirect to login
  if (res.status === 401) {
    localStorage.removeItem('token')
    throw new Error('Session expired')
  }

  if (!res.ok) {
    let detail = 'Upload failed'
    try {
      const err = await res.json()
      detail = err.detail ?? detail
    } catch {
      // non-JSON error body — keep default message
    }
    throw new Error(detail)
  }

  return res.json()
}

export interface DocumentSummary {
  name: string
  chunks: number
  visibility: 'private' | 'public'
  uploaded_at: string
  avg_chars: number
}

export interface DocumentChunk {
  index: number
  content: string
}

export async function fetchDocuments(): Promise<DocumentSummary[]> {
  const token = localStorage.getItem('token')
  if (!token) throw new Error('Not authenticated')

  const res = await fetch('/documents', {
    headers: { Authorization: `Bearer ${token}` },
  })

  if (res.status === 401) {
    localStorage.removeItem('token')
    throw new Error('Session expired')
  }
  if (!res.ok) throw new Error('Failed to load documents')
  return res.json()
}

export async function fetchDocumentChunks(name: string): Promise<DocumentChunk[]> {
  const token = localStorage.getItem('token')
  if (!token) throw new Error('Not authenticated')
  const res = await fetch(`/documents/chunks?name=${encodeURIComponent(name)}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (res.status === 401) { localStorage.removeItem('token'); throw new Error('Session expired') }
  if (!res.ok) throw new Error('Failed to load chunks')
  return res.json()
}

export async function setDocumentVisibility(
  name: string,
  visibility: 'private' | 'public',
): Promise<void> {
  const token = localStorage.getItem('token')
  if (!token) throw new Error('Not authenticated')
  const res = await fetch(
    `/documents/visibility?name=${encodeURIComponent(name)}&visibility=${visibility}`,
    { method: 'PATCH', headers: { Authorization: `Bearer ${token}` } },
  )
  if (res.status === 401) { localStorage.removeItem('token'); throw new Error('Session expired') }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? 'Update failed')
  }
}

export async function deleteDocument(name: string): Promise<void> {
  const token = localStorage.getItem('token')
  if (!token) throw new Error('Not authenticated')
  const res = await fetch(`/documents?name=${encodeURIComponent(name)}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  })
  if (res.status === 401) { localStorage.removeItem('token'); throw new Error('Session expired') }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? 'Delete failed')
  }
}

export interface SearchOptions {
  query: string
  scope: 'private' | 'public' | 'all'
  topK: number
}

export async function search(options: SearchOptions): Promise<SearchResponse> {
  const token = localStorage.getItem('token')
  if (!token) throw new Error('Not authenticated')

  const form = new FormData()
  form.append('query', options.query)
  form.append('scope', options.scope)
  form.append('top_k', String(options.topK))

  const res = await fetch('/search', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })

  if (res.status === 401) {
    localStorage.removeItem('token')
    throw new Error('Session expired')
  }

  if (!res.ok) {
    let detail = 'Search failed'
    try {
      const err = await res.json()
      detail = err.detail ?? detail
    } catch {
      // non-JSON error body — keep default message
    }
    throw new Error(detail)
  }

  return res.json()
}
