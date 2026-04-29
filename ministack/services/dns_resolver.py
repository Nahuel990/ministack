"""Wildcard DNS resolver for AppSync Events vhosts on Docker networks.

Resolves ``*.<suffix>`` for AppSync Events HTTP/realtime hostnames to MiniStack's
IPv4 on the shared Docker network. Suffixes come from optional
``APPSYNC_EVENTS_HTTP_HOST_TEMPLATE`` / ``APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE``,
or — when unset — default to ``appsync-api.{region}.{MINISTACK_HOST}`` and
``appsync-realtime-api.{region}.{MINISTACK_HOST}`` (matching AppSync Events
``CreateApi`` ``dns`` defaults). All other queries
are forwarded to upstream resolvers from ``/etc/resolv.conf`` (excluding loopback)
or ``1.1.1.1``.

Disable with ``MINISTACK_DNS_RESOLVER=0``. Port defaults to 53 (``MINISTACK_DNS_PORT``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct

from ministack.core.responses import get_region

logger = logging.getLogger("ministack.dns_resolver")

_dns_transport: asyncio.DatagramTransport | None = None
_dns_protocol: "_DnsUdpProtocol | None" = None
_started: bool = False


def resolver_enabled() -> bool:
    v = os.environ.get("MINISTACK_DNS_RESOLVER", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def is_active() -> bool:
    """True when the UDP resolver bound successfully."""
    return _started


def _default_zone_suffixes() -> list[str]:
    """Zones matching default AppSync Events ``dns`` when templates are unset."""
    region = get_region()
    host = os.environ.get("MINISTACK_HOST", "localhost")
    return [
        f"appsync-api.{region}.{host}",
        f"appsync-realtime-api.{region}.{host}",
    ]


def zone_suffixes() -> list[str]:
    """Host suffixes (e.g. ``appsync-api.eu-west-2.localhost``) for wildcard answers."""
    out: list[str] = []
    for key in (
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
    ):
        tpl = os.environ.get(key, "").strip()
        suf = _suffix_from_appsync_template(tpl)
        if suf and suf not in out:
            out.append(suf)
    if out:
        return out
    return _default_zone_suffixes()


def zone_suffixes_from_templates() -> list[str]:
    """Backward-compatible alias for :func:`zone_suffixes`."""
    return zone_suffixes()


def _suffix_from_appsync_template(tpl: str) -> str | None:
    """Extract ``appsync-api.localhost`` from ``{api_id}.appsync-api.localhost:4566``."""
    if not tpl or "{api_id}" not in tpl:
        return None
    idx = tpl.find("{api_id}")
    rest = tpl[idx + len("{api_id}") :]
    if rest.startswith("."):
        rest = rest[1:]
    host = rest.split(":")[0].split("/")[0].strip()
    return host or None


def read_upstream_nameservers() -> list[str]:
    servers: list[str] = []
    path = "/etc/resolv.conf"
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver "):
                    ip = line.split(None, 1)[1].strip()
                    if ip.startswith("127."):
                        continue
                    servers.append(ip)
    except OSError:
        pass
    if not servers:
        servers = ["1.1.1.1", "8.8.8.8"]
    return servers


def _name_matches_zone(qname: str, zone_suffix: str) -> bool:
    q = qname.rstrip(".").lower()
    z = zone_suffix.rstrip(".").lower()
    return q == z or q.endswith("." + z)


def _parse_name(data: bytes, offset: int) -> tuple[str, int]:
    """Return (dot-separated name, offset after the name — after pointer if used)."""
    labels: list[str] = []
    pos = offset
    end_after_name = offset
    jumped = False

    while pos < len(data):
        ln = data[pos]
        if ln == 0:
            pos += 1
            if not jumped:
                end_after_name = pos
            break
        if (ln & 0xC0) == 0xC0:
            if pos + 1 >= len(data):
                raise ValueError("truncated pointer")
            ptr = ((ln & 0x3F) << 8) | data[pos + 1]
            if not jumped:
                end_after_name = pos + 2
                jumped = True
            pos = ptr
            continue
        if ln > 63:
            raise ValueError("invalid label length")
        pos += 1
        labels.append(data[pos : pos + ln].decode("ascii", errors="replace"))
        pos += ln

    return ".".join(labels), (end_after_name if jumped else pos)


def parse_query_packet(data: bytes) -> tuple[int, str, int, int, bytes] | None:
    """Parse first question: tx_id, qname, qtype, qclass, raw question bytes for echo."""
    if len(data) < 12:
        return None
    tx_id = struct.unpack_from("!H", data, 0)[0]
    qdcount = struct.unpack_from("!H", data, 4)[0]
    if qdcount != 1:
        return None
    try:
        qname, qend = _parse_name(data, 12)
    except ValueError:
        return None
    if qend + 4 > len(data):
        return None
    qtype = struct.unpack_from("!H", data, qend)[0]
    qclass = struct.unpack_from("!H", data, qend + 2)[0]
    qname_bytes = data[12 : qend + 4]
    return tx_id, qname, qtype, qclass, qname_bytes


def _build_response(
    tx_id: int,
    flags: int,
    qname_bytes: bytes,
    answer_a: bytes | None,
    rcode: int = 0,
) -> bytes:
    """Build DNS response. ``flags`` is the query header flags word."""
    ancount = 1 if answer_a else 0
    qr_flags = (flags | 0x8180) & 0xFFF0  # QR, RA; clear RCODE nibble
    qr_flags |= rcode & 0xF
    header = struct.pack(
        "!HHHHHH",
        tx_id,
        qr_flags,
        1,
        ancount,
        0,
        0,
    )
    if answer_a is None:
        return header + qname_bytes

    # Answer: name (compression pointer to offset 12 where qname starts)
    answer = bytearray()
    answer.extend(b"\xC0\x0C")  # pointer to qname at offset 12
    answer.extend(struct.pack("!HHI", 1, 1, 300))  # A, IN, TTL
    answer.extend(struct.pack("!H", len(answer_a)))
    answer.extend(answer_a)
    return header + qname_bytes + bytes(answer)


def forward_udp_sync(query: bytes, upstreams: list[str], timeout: float = 3.0) -> bytes | None:
    """Forward raw DNS datagram to first upstream that responds."""
    for ns in upstreams:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(timeout)
                s.sendto(query, (ns, 53))
                data, _ = s.recvfrom(65535)
                if data:
                    return data
        except OSError as e:
            logger.debug("dns forward to %s failed: %s", ns, e)
            continue
    return None


class _DnsUdpProtocol(asyncio.DatagramProtocol):
    def __init__(
        self,
        answer_ipv4: bytes | None,
        zone_suffixes: list[str],
        upstreams: list[str],
    ):
        self.answer_ipv4 = answer_ipv4
        self.zone_suffixes = zone_suffixes
        self.upstreams = upstreams
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        asyncio.create_task(self._handle(data, addr))

    def error_received(self, exc: Exception) -> None:
        logger.warning("DNS resolver socket error: %s", exc)

    async def _handle(self, data: bytes, addr: tuple) -> None:
        if self.transport is None:
            return
        parsed = parse_query_packet(data)
        if parsed is None:
            resp = await asyncio.to_thread(forward_udp_sync, data, self.upstreams)
            if resp:
                self.transport.sendto(resp, addr)
            return
        tx_id, qname, qtype, qclass, qname_bytes = parsed
        if qclass != 1:  # IN
            resp = await asyncio.to_thread(forward_udp_sync, data, self.upstreams)
            if resp:
                self.transport.sendto(resp, addr)
            return

        flags = struct.unpack_from("!H", data, 2)[0]

        for zone in self.zone_suffixes:
            if _name_matches_zone(qname, zone):
                if qtype == 1:  # A
                    if self.answer_ipv4:
                        resp = _build_response(tx_id, flags, qname_bytes, self.answer_ipv4)
                        self.transport.sendto(resp, addr)
                        return
                    resp = _build_response(tx_id, flags, qname_bytes, None, rcode=2)
                    self.transport.sendto(resp, addr)
                    return
                if qtype == 28:  # AAAA — no IPv6; NOERROR, empty answer
                    qr_flags = (flags | 0x8180) & 0xFFF0
                    hdr = struct.pack("!HHHHHH", tx_id, qr_flags, 1, 0, 0, 0)
                    self.transport.sendto(hdr + qname_bytes, addr)
                    return

        resp = await asyncio.to_thread(forward_udp_sync, data, self.upstreams)
        if resp:
            self.transport.sendto(resp, addr)
        else:
            resp = _build_response(tx_id, flags, qname_bytes, None, rcode=2)
            self.transport.sendto(resp, addr)


async def start_dns_resolver(
    ipv4_answer: bytes | None,
    zones_override: list[str] | None = None,
) -> bool:
    """Bind UDP DNS and start forwarding. Returns True if the socket is up."""
    global _dns_transport, _dns_protocol, _started

    if not resolver_enabled():
        logger.info("DNS resolver disabled (MINISTACK_DNS_RESOLVER=0).")
        return False

    # Tests / rare crashes can leave "_started" set after the transport died.
    if _started and _dns_transport is None:
        _started = False

    if _started:
        return True

    zones = zones_override if zones_override is not None else zone_suffixes()
    upstreams = read_upstream_nameservers()
    port = int(os.environ.get("MINISTACK_DNS_PORT", "53"))

    loop = asyncio.get_running_loop()
    try:
        tr, pr = await loop.create_datagram_endpoint(
            lambda: _DnsUdpProtocol(ipv4_answer, zones, upstreams),
            local_addr=("0.0.0.0", port),
        )
    except OSError as e:
        logger.warning(
            "DNS resolver could not bind 0.0.0.0:%s (%s). "
            "Lambda containers will not use MiniStack DNS.",
            port,
            e,
        )
        return False

    _dns_transport = tr  # type: ignore[assignment]
    _dns_protocol = pr  # type: ignore[assignment]
    _started = True
    logger.info(
        "DNS resolver listening on UDP/%s (zones=%s, upstreams=%s)",
        port,
        zones or "(forward-only)",
        upstreams[:2],
    )
    return True


async def stop_dns_resolver() -> None:
    global _dns_transport, _dns_protocol, _started

    if _dns_transport:
        try:
            _dns_transport.close()
            await asyncio.sleep(0)
        except Exception:
            logger.debug("dns_resolver: transport close failed", exc_info=True)
        _dns_transport = None
    _dns_protocol = None
    _started = False


def inet_aton_ipv4(ip: str) -> bytes | None:
    try:
        return socket.inet_aton(ip)
    except OSError:
        return None


def resolve_ministack_dns_answer_ip() -> bytes | None:
    """IPv4 bytes for A-record answers, or None if unknown."""
    try:
        import docker as docker_lib  # type: ignore[import-untyped]

        dc = docker_lib.from_env()
    except Exception:
        return None

    from ministack.core.docker_network import get_ministack_container_ipv4_on_network

    ip = get_ministack_container_ipv4_on_network(dc)
    if ip:
        return inet_aton_ipv4(ip)
    return None
