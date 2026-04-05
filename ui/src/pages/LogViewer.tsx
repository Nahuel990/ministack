import { useState, useRef, useEffect } from 'react'
import { useSSE } from '../hooks/useSSE'
import { SSE_LOGS_URL } from '../lib/api'
import type { LogEntry } from '../lib/types'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { EmptyState } from '@/components/EmptyState'
import { Search, Pause, Play, Trash2, ScrollText } from 'lucide-react'

const LEVEL_VARIANTS: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
  DEBUG: 'secondary',
  INFO: 'default',
  WARNING: 'outline',
  ERROR: 'destructive',
  CRITICAL: 'destructive',
}

export default function LogViewer() {
  const { events, connected, clear } = useSSE<LogEntry>(SSE_LOGS_URL)
  const [filter, setFilter] = useState('')
  const [levelFilter, setLevelFilter] = useState<string>('all')
  const [paused, setPaused] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [displayEvents, setDisplayEvents] = useState<LogEntry[]>([])

  useEffect(() => {
    if (!paused) {
      setDisplayEvents(events)
    }
  }, [events, paused])

  useEffect(() => {
    if (!paused && scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [displayEvents, paused])

  const filtered = displayEvents.filter((log) => {
    if (levelFilter && levelFilter !== 'all' && log.level !== levelFilter) return false
    if (filter && !log.message.toLowerCase().includes(filter.toLowerCase()) && !log.logger.toLowerCase().includes(filter.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="px-4 py-3 border-b bg-card/50 flex items-center gap-3 flex-shrink-0">
        <h2 className="text-sm font-medium mr-2">Logs</h2>

        <Tooltip>
          <TooltipTrigger asChild>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500'}`} />
          </TooltipTrigger>
          <TooltipContent>{connected ? 'Connected' : 'Reconnecting...'}</TooltipContent>
        </Tooltip>

        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Filter logs..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="h-8 w-48 pl-8 text-xs"
            aria-label="Filter logs"
          />
        </div>

        <Select value={levelFilter} onValueChange={setLevelFilter}>
          <SelectTrigger className="h-8 w-32 text-xs" aria-label="Filter by level">
            <SelectValue placeholder="All Levels" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Levels</SelectItem>
            {['DEBUG', 'INFO', 'WARNING', 'ERROR'].map((lvl) => (
              <SelectItem key={lvl} value={lvl}>{lvl}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant={paused ? 'default' : 'outline'}
          size="sm"
          onClick={() => setPaused(!paused)}
          className="h-8 text-xs"
          aria-label={paused ? 'Resume log stream' : 'Pause log stream'}
        >
          {paused ? <Play className="h-3.5 w-3.5 mr-1" /> : <Pause className="h-3.5 w-3.5 mr-1" />}
          {paused ? 'Resume' : 'Pause'}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={clear}
          className="h-8 text-xs"
          aria-label="Clear logs"
        >
          <Trash2 className="h-3.5 w-3.5 mr-1" />
          Clear
        </Button>

        <span className="text-xs text-muted-foreground ml-auto">{filtered.length} entries</span>
      </div>

      {/* Log stream */}
      <ScrollArea className="flex-1" ref={scrollRef}>
        <div aria-live="polite">
          {filtered.length === 0 && (
            <EmptyState
              icon={ScrollText}
              title={connected ? 'Waiting for log entries...' : 'Connecting...'}
              description="Log entries will appear here in real-time."
            />
          )}
          {filtered.map((log, i) => (
            <div
              key={i}
              className="px-4 py-1.5 border-b border-border/50 hover:bg-accent/30 flex items-start gap-3 font-mono text-xs"
            >
              <span className="text-muted-foreground w-36 flex-shrink-0">{log.timestamp}</span>
              <Badge variant={LEVEL_VARIANTS[log.level] ?? 'secondary'} className="text-[10px] w-16 justify-center flex-shrink-0 py-0">
                {log.level}
              </Badge>
              <span className="text-primary/70 w-20 flex-shrink-0 truncate">{log.logger}</span>
              <span className="text-foreground/80 break-all">{log.message}</span>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
