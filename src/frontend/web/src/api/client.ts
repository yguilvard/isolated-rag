export interface IngestResult {
  status: string
  chunks_ingested: number
  document: string
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

export async function ingest(
  file: File,
  visibility: 'private' | 'public',
): Promise<IngestResult> {
  // Build multipart form with file and visibility
  const form = new FormData()
  form.append('file', file)
  form.append('visibility', visibility)

  const token = localStorage.getItem('token')
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
    const err = await res.json()
    throw new Error(err.detail ?? 'Upload failed')
  }

  return res.json()
}
