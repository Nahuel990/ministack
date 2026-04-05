"""
UI Dashboard API — REST endpoints and SSE streams for the MiniStack dashboard.

Endpoints:
  GET /_ministack/api/stats              — aggregated resource counts
  GET /_ministack/api/requests           — recent request history
  GET /_ministack/api/requests/stream    — SSE real-time request stream
  GET /_ministack/api/logs/stream        — SSE real-time log stream
  GET /_ministack/api/logs               — recent log entries
  GET /_ministack/api/resources/{svc}    — list resources for a service
  GET /_ministack/api/resources/{svc}/{type}/{id} — single resource detail
  GET /_ministack/api/s3/...             — S3-specific browsing (see api/s3.py)

To add a service-specific API:
  1. Create ministack/ui/api/myservice.py with async def handle(rel_path, query_params, send)
  2. Register a route prefix in SERVICE_ROUTES below
"""

import logging

from ministack.ui import interceptor
from ministack.ui.api import _resources, s3
from ministack.ui.api._common import get_query_param, handle_sse, json_response
from ministack.ui.log_handler import ui_log_handler

logger = logging.getLogger("ministack.ui")

API_PREFIX = "/_ministack/api"

# Service-specific route prefixes → handler modules.
# Each module must expose: async def handle(rel_path: str, query_params: dict, send)
SERVICE_ROUTES: dict[str, object] = {
    "/s3/": s3,
}


async def handle(method: str, path: str, query_params: dict, receive, send):
    """Route /_ministack/api/* requests to the appropriate handler."""
    rel = path[len(API_PREFIX):]

    # Core endpoints
    if rel == "/stats" and method == "GET":
        await _resources.handle_stats(send)
        return

    if rel == "/requests/stream" and method == "GET":
        await handle_sse(receive, send, interceptor.subscribe)
        return

    if rel == "/requests" and method == "GET":
        limit = int(get_query_param(query_params, "limit", "50"))
        offset = int(get_query_param(query_params, "offset", "0"))
        await json_response(send, interceptor.get_requests(limit, offset))
        return

    if rel == "/logs/stream" and method == "GET":
        await handle_sse(receive, send, ui_log_handler.subscribe)
        return

    if rel == "/logs" and method == "GET":
        limit = int(get_query_param(query_params, "limit", "100"))
        await json_response(send, {"logs": ui_log_handler.get_recent(limit)})
        return

    # Service-specific routes
    for prefix, module in SERVICE_ROUTES.items():
        if rel.startswith(prefix) and method == "GET":
            await module.handle(rel[len(prefix):], query_params, send)
            return

    # Generic resource browsing (fallback)
    if rel.startswith("/resources/") and method == "GET":
        await _resources.handle_resources(rel[len("/resources/"):], query_params, send)
        return

    await json_response(send, {"error": f"Unknown UI API endpoint: {path}"}, status=404)
