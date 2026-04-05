import { useState, useMemo } from 'react'
import { useSSE } from '../hooks/useSSE'
import { SSE_REQUESTS_URL } from '../lib/api'
import type { RequestEntry } from '../lib/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { EmptyState } from '@/components/EmptyState'
import { JsonViewer } from '@/components/JsonViewer'
import { Activity, Trash2 } from 'lucide-react'

function StatusBadge({ status }: { status: number }) {
  const variant = status < 300 ? 'default' : status < 500 ? 'secondary' : 'destructive'
  return (
    <Badge variant={variant} className="font-mono text-[10px] px-1.5 py-0">
      {status}
    </Badge>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

export default function RequestInspector() {
  const { events, connected, clear } = useSSE<RequestEntry>(SSE_REQUESTS_URL)
  const [serviceFilter, setServiceFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const uniqueServices = useMemo(() => {
    const set = new Set(events.map((e) => e.service))
    return Array.from(set).sort()
  }, [events])

  const filtered = events.filter((req) => {
    if (serviceFilter && serviceFilter !== 'all' && req.service !== serviceFilter) return false
    if (statusFilter === '2xx' && (req.status < 200 || req.status >= 300)) return false
    if (statusFilter === '4xx' && (req.status < 400 || req.status >= 500)) return false
    if (statusFilter === '5xx' && req.status < 500) return false
    return true
  })

  const totalReqs = events.length
  const avgDuration = totalReqs > 0
    ? (events.reduce((sum, r) => sum + r.duration_ms, 0) / totalReqs).toFixed(1)
    : '0'
  const errorRate = totalReqs > 0
    ? ((events.filter((r) => r.status >= 400).length / totalReqs) * 100).toFixed(1)
    : '0'

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="px-4 py-3 border-b bg-card/50 flex items-center gap-3 flex-shrink-0">
        <h2 className="text-sm font-medium mr-2">Requests</h2>

        <Tooltip>
          <TooltipTrigger asChild>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500'}`} />
          </TooltipTrigger>
          <TooltipContent>{connected ? 'Connected' : 'Reconnecting...'}</TooltipContent>
        </Tooltip>

        <Select value={serviceFilter} onValueChange={setServiceFilter}>
          <SelectTrigger className="h-8 w-36 text-xs" aria-label="Filter by service">
            <SelectValue placeholder="All Services" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Services</SelectItem>
            {uniqueServices.map((svc) => (
              <SelectItem key={svc} value={svc}>{svc}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="h-8 w-36 text-xs" aria-label="Filter by status">
            <SelectValue placeholder="All Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="2xx">2xx Success</SelectItem>
            <SelectItem value="4xx">4xx Client Error</SelectItem>
            <SelectItem value="5xx">5xx Server Error</SelectItem>
          </SelectContent>
        </Select>

        <Button
          variant="outline"
          size="sm"
          onClick={clear}
          className="h-8 text-xs"
          aria-label="Clear requests"
        >
          <Trash2 className="h-3.5 w-3.5 mr-1" />
          Clear
        </Button>

        {/* Stats */}
        <div className="ml-auto flex items-center gap-4 text-xs text-muted-foreground">
          <span>{totalReqs} requests</span>
          <span>avg {avgDuration}ms</span>
          <span className={parseFloat(errorRate) > 10 ? 'text-destructive' : ''}>{errorRate}% errors</span>
        </div>
      </div>

      {/* Request table */}
      <div className="flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <EmptyState
            icon={Activity}
            title={connected ? 'Waiting for requests...' : 'Connecting...'}
            description="API requests will appear here in real-time."
          />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[140px]">Time</TableHead>
                <TableHead className="w-[70px]">Status</TableHead>
                <TableHead className="w-[70px]">Method</TableHead>
                <TableHead className="w-[120px]">Service</TableHead>
                <TableHead>Action</TableHead>
                <TableHead className="text-right w-[90px]">Duration</TableHead>
                <TableHead className="text-right w-[80px]">Size</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((req) => (
                <>
                  <TableRow
                    key={req.id}
                    className="cursor-pointer"
                    onClick={() => setExpandedId(expandedId === req.id ? null : req.id)}
                  >
                    <TableCell className="text-muted-foreground font-mono text-xs">
                      {new Date(req.timestamp * 1000).toLocaleTimeString()}
                    </TableCell>
                    <TableCell><StatusBadge status={req.status} /></TableCell>
                    <TableCell className="text-xs text-muted-foreground uppercase">{req.method}</TableCell>
                    <TableCell className="text-xs text-primary">{req.service}</TableCell>
                    <TableCell className="text-xs font-mono truncate max-w-[300px]">{req.action || req.path}</TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">{req.duration_ms.toFixed(1)}ms</TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">{formatBytes(req.response_size)}</TableCell>
                  </TableRow>
                  {expandedId === req.id && (
                    <TableRow key={`${req.id}-detail`}>
                      <TableCell colSpan={7} className="bg-muted/30 p-0">
                        <JsonViewer data={req} />
                      </TableCell>
                    </TableRow>
                  )}
                </>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  )
}
