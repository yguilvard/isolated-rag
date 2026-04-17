// Polyfill localStorage for Node 25+ environments where the native
// localStorage stub lacks the full Web Storage API (clear, key, etc.)
const store: Record<string, string> = {}

const localStorageMock: Storage = {
  length: 0,
  clear() {
    for (const key of Object.keys(store)) {
      delete store[key]
    }
    this.length = 0
  },
  getItem(key: string) {
    return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null
  },
  setItem(key: string, value: string) {
    store[key] = String(value)
    this.length = Object.keys(store).length
  },
  removeItem(key: string) {
    delete store[key]
    this.length = Object.keys(store).length
  },
  key(index: number) {
    return Object.keys(store)[index] ?? null
  },
}

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
})
