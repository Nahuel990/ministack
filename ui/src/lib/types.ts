export interface ServiceStats {
  status: string
  resources: Record<string, number>
}

export interface StatsResponse {
  services: Record<string, ServiceStats>
  total_resources: number
  uptime_seconds: number
}

export interface RequestEntry {
  id: string
  timestamp: number
  method: string
  path: string
  service: string
  action: string
  status: number
  duration_ms: number
  request_size: number
  response_size: number
}

export interface LogEntry {
  timestamp: string
  level: string
  logger: string
  message: string
}
