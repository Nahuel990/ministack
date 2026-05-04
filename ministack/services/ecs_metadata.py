"""ECS Task Metadata V4 emulator.

Real ECS injects ECS_CONTAINER_METADATA_URI_V4=http://169.254.170.2/v4/<token>
per container; ministack instead serves the same routes off the gateway port,
keyed by tokens registered from services/ecs.py. State is volatile by design
(stripped on persistence; reset() called by /_ministack/reset).
"""

import re

from ministack.core.responses import json_response

_TASKS: dict[str, dict] = {}
_CONTAINERS: dict[str, dict] = {}

_PATH_RE = re.compile(r"^/v4/(?P<token>[A-Za-z0-9_-]{8,})(?P<rest>/.*)?$")


def register_task(token: str, task_payload: dict, container_payload: dict) -> None:
    _TASKS[token] = task_payload
    _CONTAINERS[token] = container_payload


def unregister_task(token: str) -> None:
    _TASKS.pop(token, None)
    _CONTAINERS.pop(token, None)


def set_docker_id(token: str, docker_id: str) -> None:
    if container := _CONTAINERS.get(token):
        container["DockerId"] = docker_id


def reset() -> None:
    _TASKS.clear()
    _CONTAINERS.clear()


async def handle_request(method, path, headers, body, query_params):
    m = _PATH_RE.match(path)
    if not m:
        return json_response({"message": "not found"}, status=404)
    token = m.group("token")
    if token not in _TASKS:
        return json_response({"message": "unknown token"}, status=404)

    rest = (m.group("rest") or "").rstrip("/")
    if rest == "":
        return json_response(_CONTAINERS[token])
    if rest == "/task":
        return json_response(_TASKS[token])
    if rest in ("/stats", "/task/stats"):
        return json_response({})
    return json_response({"message": "not found"}, status=404)
