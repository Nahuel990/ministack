import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'

export function useFetch<T>(fetcher: () => Promise<T>, intervalMs?: number) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const result = await fetcher()
      setData(result)
      setError(null)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Fetch failed'
      setError(msg)
      toast.error('Failed to fetch data', { description: msg })
    } finally {
      setLoading(false)
    }
  }, [fetcher])

  useEffect(() => {
    refresh()
    if (intervalMs && intervalMs > 0) {
      const id = setInterval(refresh, intervalMs)
      return () => clearInterval(id)
    }
  }, [refresh, intervalMs])

  return { data, loading, error, refresh }
}
