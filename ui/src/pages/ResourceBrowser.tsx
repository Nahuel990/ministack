import { useCallback, useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useFetch } from '../hooks/useFetch'
import { fetchStats, fetchResources, fetchResourceDetail } from '../lib/api'
import type { StatsResponse } from '../lib/types'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from '@/components/ui/sheet'
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table'
import { EmptyState } from '@/components/EmptyState'
import { JsonViewer } from '@/components/JsonViewer'
import { getServiceIcon } from '@/lib/service-icons'
import { FolderOpen } from 'lucide-react'

export default function ResourceBrowser() {
  const { service } = useParams<{ service?: string }>()
  const statsFetcher = useCallback(() => fetchStats(), [])
  const { data: stats } = useFetch<StatsResponse>(statsFetcher, 10000)
  const [resources, setResources] = useState<Record<string, unknown[]> | null>(null)
  const [detail, setDetail] = useState<{ service: string; type: string; id: string; detail: unknown } | null>(null)
  const [loadingResources, setLoadingResources] = useState(false)

  useEffect(() => {
    if (!service) {
      setResources(null)
      return
    }
    setLoadingResources(true)
    fetchResources(service)
      .then((data) => setResources(data.resources ?? {}))
      .catch(() => setResources(null))
      .finally(() => setLoadingResources(false))
  }, [service])

  const openDetail = async (svc: string, type: string, id: string) => {
    try {
      const data = await fetchResourceDetail(svc, type, id)
      setDetail(data)
    } catch {
      setDetail(null)
    }
  }

  const services = stats ? Object.entries(stats.services) : []

  return (
    <div className="flex h-full">
      {/* Service sidebar */}
      <ScrollArea className="w-52 border-r bg-card/50">
        <div className="px-3 py-3 border-b">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Services</h3>
        </div>
        <ul className="py-1">
          {services.map(([name, svc]) => {
            const total = Object.values(svc.resources).reduce((a, b) => a + b, 0)
            const Icon = getServiceIcon(name)
            return (
              <li key={name}>
                <Link
                  to={`/resources/${name}`}
                  className={`flex items-center justify-between px-3 py-2 text-sm transition-colors ${
                    service === name
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
                  }`}
                >
                  <span className="flex items-center gap-2 truncate">
                    <Icon className="h-3.5 w-3.5" />
                    {name}
                  </span>
                  {total > 0 && (
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0 ml-2">
                      {total}
                    </Badge>
                  )}
                </Link>
              </li>
            )
          })}
        </ul>
      </ScrollArea>

      {/* Resource content */}
      <div className="flex-1 overflow-auto p-6">
        {!service && (
          <EmptyState
            icon={FolderOpen}
            title="Select a service"
            description="Choose a service from the sidebar to browse its resources."
          />
        )}

        {service && loadingResources && (
          <div className="space-y-4">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        )}

        {service && resources && (
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              {(() => { const Icon = getServiceIcon(service); return <Icon className="h-5 w-5 text-muted-foreground" /> })()}
              <h2 className="text-xl font-bold">{service}</h2>
            </div>

            {Object.entries(resources).map(([type, items]) => (
              <Card key={type}>
                <CardHeader className="p-4 pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">{type}</CardTitle>
                    <Badge variant="secondary" className="text-[10px]">
                      {Array.isArray(items) ? items.length : 0} items
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  {Array.isArray(items) && items.length === 0 && (
                    <div className="px-4 py-6 text-center text-sm text-muted-foreground">Empty</div>
                  )}
                  {Array.isArray(items) && items.length > 0 && (
                    <Table>
                      <TableBody>
                        {(items as Record<string, unknown>[]).map((item, i) => (
                          <TableRow
                            key={i}
                            className="cursor-pointer"
                            onClick={() => openDetail(service, type, String(item.id ?? i))}
                          >
                            <TableCell className="text-primary font-mono font-medium text-xs">
                              {String(item.id ?? i)}
                            </TableCell>
                            <TableCell className="text-muted-foreground text-xs truncate max-w-md">
                              {Object.entries(item)
                                .filter(([k]) => k !== 'id')
                                .map(([k, v]) => `${k}: ${v}`)
                                .join(' | ')}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Detail Sheet */}
        <Sheet open={!!detail} onOpenChange={(open) => !open && setDetail(null)}>
          <SheetContent className="sm:max-w-lg overflow-auto">
            {detail && (
              <>
                <SheetHeader>
                  <SheetTitle>{detail.type} / {detail.id}</SheetTitle>
                  <SheetDescription>{detail.service}</SheetDescription>
                </SheetHeader>
                <JsonViewer data={detail.detail} />
              </>
            )}
          </SheetContent>
        </Sheet>
      </div>
    </div>
  )
}
