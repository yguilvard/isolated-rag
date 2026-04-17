// Polyfill localStorage for Node 25+ environments where the native
// localStorage stub lacks the full Web Storage API (clear, key, etc.)
const store: Record<string, string> = {}

const localStorageMock = {
  get length() {
    return Object.keys(store).length
  },
  clear() {
    for (const key of Object.keys(store)) {
      delete store[key]
    }
  },
  getItem(key: string) {
    return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null
  },
  setItem(key: string, value: string) {
    store[key] = String(value)
  },
  removeItem(key: string) {
    delete store[key]
  },
  key(index: number) {
    return Object.keys(store)[index] ?? null
  },
} as Storage

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
})
