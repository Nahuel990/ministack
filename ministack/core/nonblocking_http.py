import asyncio
import os
import urllib.request


def timeout_from_env(env_name: str, default_seconds: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    return parsed if parsed > 0 else float(default_seconds)


def _urlopen_sync(request_or_url, timeout_seconds: float):
    with urllib.request.urlopen(request_or_url, timeout=timeout_seconds) as resp:
        return resp.status, dict(resp.headers.items()), resp.read()


async def urlopen(request_or_url, timeout_seconds: float):
    return await asyncio.to_thread(_urlopen_sync, request_or_url, timeout_seconds)
