import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

export function useSSE<T>(url: string, maxItems = 500): { events: T[]; connected: boolean; clear: () => void } {
  const [events, setEvents] = useState<T[]>([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)
  const wasConnected = useRef(false)

  useEffect(() => {
    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => {
      setConnected(true)
      if (wasConnected.current) {
        toast.success('Reconnected to stream')
      }
      wasConnected.current = true
    }

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as T
        setEvents((prev) => {
          const next = [data, ...prev]
          return next.length > maxItems ? next.slice(0, maxItems) : next
        })
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      setConnected(false)
      if (wasConnected.current) {
        toast.warning('Stream disconnected, reconnecting...')
      }
    }

    return () => {
      es.close()
      esRef.current = null
    }
  }, [url, maxItems])

  const clear = () => setEvents([])

  return { events, connected, clear }
}
