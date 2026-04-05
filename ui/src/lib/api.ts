const API_BASE = '/_ministack/api'

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`)
  return res.json()
}

export async function fetchRequests(limit = 50, offset = 0) {
  const res = await fetch(`${API_BASE}/requests?limit=${limit}&offset=${offset}`)
  return res.json()
}

export async function fetchLogs(limit = 100) {
  const res = await fetch(`${API_BASE}/logs?limit=${limit}`)
  return res.json()
}

export async function fetchResources(service: string, type?: string) {
  const params = type ? `?type=${type}` : ''
  const res = await fetch(`${API_BASE}/resources/${service}${params}`)
  return res.json()
}

export async function fetchResourceDetail(service: string, type: string, id: string) {
  const res = await fetch(`${API_BASE}/resources/${service}/${type}/${encodeURIComponent(id)}`)
  return res.json()
}

export const SSE_REQUESTS_URL = `${API_BASE}/requests/stream`
export const SSE_LOGS_URL = `${API_BASE}/logs/stream`
