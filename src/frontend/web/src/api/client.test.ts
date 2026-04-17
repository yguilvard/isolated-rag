import { describe, it, expect, vi, beforeEach } from 'vitest'
import { login, ingest } from './client'

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('login', () => {
  it('returns access token on success', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ access_token: 'tok123', token_type: 'bearer' }),
    }))

    const token = await login('alice', 'secret')

    expect(token).toBe('tok123')
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[0]).toBe('/auth/login')
    expect(call[1].method).toBe('POST')
    expect(call[1].headers['Content-Type']).toBe('application/x-www-form-urlencoded')
  })

  it('throws on failed login', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }))
    await expect(login('alice', 'wrong')).rejects.toThrow('Invalid credentials')
  })
})

describe('ingest', () => {
  it('returns IngestResult on success', async () => {
    localStorage.setItem('token', 'my-token')
    const mockResult = { status: 'ok', chunks_ingested: 5, document: 'doc.txt' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockResult),
    }))

    const file = new File(['hello'], 'doc.txt', { type: 'text/plain' })
    const result = await ingest(file, 'private')

    expect(result.chunks_ingested).toBe(5)
    expect(result.document).toBe('doc.txt')
    const call = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].headers['Authorization']).toBe('Bearer my-token')
  })

  it('throws "Session expired" and clears token on 401', async () => {
    localStorage.setItem('token', 'expired')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ status: 401, ok: false }))

    const file = new File(['x'], 'f.txt')
    await expect(ingest(file, 'private')).rejects.toThrow('Session expired')
    expect(localStorage.getItem('token')).toBeNull()
  })

  it('throws API error detail on 422', async () => {
    localStorage.setItem('token', 'tok')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({ detail: 'Unsupported file type: ".xyz"' }),
    }))

    const file = new File(['x'], 'f.xyz')
    await expect(ingest(file, 'private')).rejects.toThrow('Unsupported file type')
  })
})
