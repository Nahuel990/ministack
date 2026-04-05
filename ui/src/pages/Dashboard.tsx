import { useCallback } from 'react'
import { useFetch } from '../hooks/useFetch'
import { fetchStats, fetchRequests } from '../lib/api'
import type { StatsResponse, RequestEntry } from '../lib/types'
import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { getServiceIcon } from '@/lib/service-icons'
import { Loader2 } from 'lucide-react'

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function StatusBadge({ status }: { status: string }) {
  return (
    <Badge variant={status === 'available' ? 'default' : 'destructive'} className="text-[10px] px-1.5 py-0">
      {status}
    </Badge>
  )
}

function RequestStatusBadge({ status }: { status: number }) {
  const variant = status < 300 ? 'default' : status < 500 ? 'secondary' : 'destructive'
  return (
    <Badge variant={variant} className="font-mono text-[10px] px-1.5 py-0">
      {status}
    </Badge>
  )
}

export default function Dashboard() {
  const statsFetcher = useCallback(() => fetchStats(), [])
  const requestsFetcher = useCallback(() => fetchRequests(10), [])

  const { data: stats } = useFetch<StatsResponse>(statsFetcher, 5000)
  const { data: reqData } = useFetch<{ requests: RequestEntry[] }>(requestsFetcher, 3000)

  if (!stats) {
    return (
      <div className="p-6 space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton key={i} className="h-32 rounded-lg" />
          ))}
        </div>
      </div>
    )
  }

  const services = Object.entries(stats.services)
  const recentRequests = reqData?.requests ?? []

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <p className="text-sm text-muted-foreground mt-1">
          {services.length} services | {stats.total_resources} resources | uptime {formatUptime(stats.uptime_seconds)}
        </p>
      </div>

      {/* Service Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {services.map(([name, svc]) => {
          const totalRes = Object.values(svc.resources).reduce((a, b) => a + b, 0)
          const Icon = getServiceIcon(name)
          return (
            <Link key={name} to={`/resources/${name}`}>
              <Card className="hover:bg-accent/50 transition-colors h-full">
                <CardHeader className="p-4 pb-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4 text-muted-foreground" />
                      <CardTitle className="text-sm font-medium">{name}</CardTitle>
                    </div>
                    <StatusBadge status={svc.status} />
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <div className="space-y-1">
                    {Object.entries(svc.resources).map(([label, count]) => (
                      <div key={label} className="flex justify-between text-xs">
                        <span className="text-muted-foreground">{label}</span>
                        <span className={count > 0 ? 'text-primary font-medium' : 'text-muted-foreground/50'}>
                          {count}
                        </span>
                      </div>
                    ))}
                    {Object.keys(svc.resources).length === 0 && (
                      <div className="text-xs text-muted-foreground/50">No tracked resources</div>
                    )}
                  </div>
                  {totalRes > 0 && (
                    <div className="mt-2 pt-2 border-t text-xs text-muted-foreground">
                      {totalRes} total
                    </div>
                  )}
                </CardContent>
              </Card>
            </Link>
          )
        })}
      </div>

      {/* Recent Requests */}
      <Card>
        <CardHeader className="p-4 pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">Recent Requests</CardTitle>
            <Link to="/requests" className="text-xs text-primary hover:underline">
              View all
            </Link>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {recentRequests.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              No requests yet. Make an AWS API call to see it here.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[60px]">Status</TableHead>
                  <TableHead className="w-[60px]">Method</TableHead>
                  <TableHead className="w-[120px]">Service</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead className="text-right w-[80px]">Duration</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentRequests.map((req) => (
                  <TableRow key={req.id}>
                    <TableCell><RequestStatusBadge status={req.status} /></TableCell>
                    <TableCell className="text-xs text-muted-foreground uppercase">{req.method}</TableCell>
                    <TableCell className="text-xs text-primary">{req.service}</TableCell>
                    <TableCell className="text-xs truncate max-w-[300px]">{req.action || req.path}</TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">{req.duration_ms.toFixed(1)}ms</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
