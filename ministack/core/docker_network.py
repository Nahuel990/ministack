"""Resolve the Docker network MiniStack itself uses for sibling containers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("ministack.docker_network")

_cached: str | None = None
_resolved = False


def get_ministack_network(client) -> str | None:
    """Return the Docker network MiniStack itself is attached to.

    Precedence:
      1. ``DOCKER_NETWORK`` env (explicit override).
      2. ``LAMBDA_DOCKER_NETWORK`` env (Lambda-only legacy fallback).
      3. Inspect MiniStack's own container by ``HOSTNAME`` and return the first
         attached network.
      4. ``None`` (MiniStack running directly on host, no Docker network).

    Result is cached for the process lifetime so siblings spawned at different
    times land on the same network.
    """
    global _cached, _resolved
    if _resolved:
        return _cached
    explicit = os.environ.get("DOCKER_NETWORK", "") or os.environ.get(
        "LAMBDA_DOCKER_NETWORK", ""
    )
    if explicit:
        _cached = explicit
    else:
        try:
            hostname = os.environ.get("HOSTNAME", "")
            if hostname:
                self_container = client.containers.get(hostname)
                nets = list(
                    self_container.attrs["NetworkSettings"]["Networks"].keys()
                )
                if nets:
                    _cached = nets[0]
        except Exception:
            logger.debug(
                "docker_network: could not inspect MiniStack container",
                exc_info=True,
            )
    _resolved = True
    if _cached:
        logger.info("docker_network: resolved MiniStack network = %s", _cached)
    return _cached


def get_ministack_container_ipv4_on_network(
    client, network_name: str | None = None
) -> str | None:
    """IPv4 address of this MiniStack container on ``network_name``.

    Used so sibling containers (e.g. Lambda) can reach MiniStack services on the
    same Docker network (DNS on UDP/53, HTTP gateway on GATEWAY_PORT).

    ``network_name`` defaults to :func:`get_ministack_network` when omitted.
    """
    hostname = os.environ.get("HOSTNAME", "").strip()
    if not hostname:
        return None
    net = network_name if network_name is not None else get_ministack_network(client)
    if not net:
        return None
    try:
        ctr = client.containers.get(hostname)
        nets = ctr.attrs.get("NetworkSettings", {}).get("Networks", {})
        info = nets.get(net, {})
        ip = (info.get("IPAddress") or "").strip()
        return ip or None
    except Exception:
        logger.debug(
            "docker_network: could not read MiniStack IPv4 on %s",
            net,
            exc_info=True,
        )
        return None
