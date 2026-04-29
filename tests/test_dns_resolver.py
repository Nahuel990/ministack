"""Tests for ministack.services.dns_resolver."""

from __future__ import annotations

import asyncio
import socket
import struct
from unittest.mock import MagicMock

from ministack.core import docker_network as docker_network_mod
from ministack.services import dns_resolver


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_query(qname: str, qtype: int = 1, tx_id: int = 0xBEEF) -> bytes:
    hdr = struct.pack("!HHHHHH", tx_id, 0x0100, 1, 0, 0, 0)
    buf = bytearray(hdr)
    for part in qname.split("."):
        b = part.encode("ascii")
        buf.append(len(b))
        buf.extend(b)
    buf.append(0)
    buf.extend(struct.pack("!HH", qtype, 1))
    return bytes(buf)


def _udp_query_sync(port: int, payload: bytes, timeout: float = 5.0) -> bytes:
    """Blocking UDP query — run via asyncio.to_thread so the server loop stays free."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout)
        s.sendto(payload, ("127.0.0.1", port))
        data, _ = s.recvfrom(65535)
        return data
    finally:
        s.close()


def test_zone_suffixes_from_templates(monkeypatch):
    monkeypatch.setenv(
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "{api_id}.appsync-api.localhost:4566",
    )
    monkeypatch.setenv(
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
        "{api_id}.appsync-realtime-api.localhost:4566",
    )
    z = dns_resolver.zone_suffixes_from_templates()
    assert "appsync-api.localhost" in z
    assert "appsync-realtime-api.localhost" in z


def test_wildcard_a_record(monkeypatch):
    port = _free_udp_port()
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(port))
    monkeypatch.setenv(
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "{api_id}.appsync-api.localhost:4566",
    )
    monkeypatch.setenv(
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
        "{api_id}.appsync-realtime-api.localhost:4566",
    )
    ip = socket.inet_aton("192.0.2.55")

    async def runner():
        try:
            ok = await dns_resolver.start_dns_resolver(ip)
            assert ok
            assert dns_resolver.is_active()
            q = _make_query("abc123.appsync-api.localhost", 1)
            resp = await asyncio.to_thread(_udp_query_sync, port, q)
            assert resp[-4:] == ip
        finally:
            await dns_resolver.stop_dns_resolver()

    asyncio.run(runner())


def test_wildcard_realtime_suffix(monkeypatch):
    port = _free_udp_port()
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(port))
    monkeypatch.setenv(
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "{api_id}.appsync-api.localhost:4566",
    )
    monkeypatch.setenv(
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
        "{api_id}.appsync-realtime-api.localhost:4566",
    )
    ip = socket.inet_aton("192.0.2.56")

    async def runner():
        try:
            await dns_resolver.start_dns_resolver(ip)
            q = _make_query("xyz.appsync-realtime-api.localhost", 1)
            resp = await asyncio.to_thread(_udp_query_sync, port, q)
            assert resp[-4:] == ip
        finally:
            await dns_resolver.stop_dns_resolver()

    asyncio.run(runner())


def test_forward_when_no_zone_match(monkeypatch):
    port = _free_udp_port()
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(port))
    monkeypatch.setenv(
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "{api_id}.appsync-api.localhost:4566",
    )
    monkeypatch.setenv(
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
        "{api_id}.appsync-realtime-api.localhost:4566",
    )
    ip = socket.inet_aton("192.0.2.1")

    fake = bytes.fromhex(
        "000085000001000100000000076578616d706c6503636f6d0000010001"
        "c00c000100010000003400045bfc0e2c"
    )
    monkeypatch.setattr(dns_resolver, "forward_udp_sync", lambda q, u, timeout=3.0: fake)

    async def runner():
        try:
            await dns_resolver.start_dns_resolver(ip)
            q = _make_query("example.com", 1)
            resp = await asyncio.to_thread(_udp_query_sync, port, q)
            assert resp == fake
        finally:
            await dns_resolver.stop_dns_resolver()

    asyncio.run(runner())


def test_templates_unset_forwards_non_appsync_zones(monkeypatch):
    """Without templates, default zones are region.localhost; unrelated names forward."""
    port = _free_udp_port()
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(port))
    monkeypatch.delenv("APPSYNC_EVENTS_HTTP_HOST_TEMPLATE", raising=False)
    monkeypatch.delenv("APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE", raising=False)

    fake = bytes.fromhex(
        "000085000001000100000000076578616d706c6503636f6d0000010001"
        "c00c000100010000003400045bfc0e2c"
    )
    monkeypatch.setattr(dns_resolver, "forward_udp_sync", lambda q, u, timeout=3.0: fake)

    async def runner():
        try:
            await dns_resolver.start_dns_resolver(socket.inet_aton("192.0.2.2"))
            # Does not match appsync-api.us-east-1.localhost (legacy flat *.appsync-api.localhost)
            q = _make_query("anything.appsync-api.localhost", 1)
            resp = await asyncio.to_thread(_udp_query_sync, port, q)
            assert resp == fake
        finally:
            await dns_resolver.stop_dns_resolver()

    asyncio.run(runner())


def test_resolver_disabled_no_bind(monkeypatch):
    monkeypatch.setenv("MINISTACK_DNS_RESOLVER", "0")
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(_free_udp_port()))

    async def runner():
        ok = await dns_resolver.start_dns_resolver(socket.inet_aton("192.0.2.3"))
        assert not ok
        assert not dns_resolver.is_active()

    asyncio.run(runner())


def test_servfail_when_zone_matches_but_no_ip(monkeypatch):
    port = _free_udp_port()
    monkeypatch.setenv("MINISTACK_DNS_PORT", str(port))
    monkeypatch.setenv(
        "APPSYNC_EVENTS_HTTP_HOST_TEMPLATE",
        "{api_id}.appsync-api.localhost:4566",
    )
    monkeypatch.setenv(
        "APPSYNC_EVENTS_REALTIME_HOST_TEMPLATE",
        "{api_id}.appsync-realtime-api.localhost:4566",
    )

    async def runner():
        try:
            await dns_resolver.start_dns_resolver(None)
            q = _make_query("id.appsync-api.localhost", 1)
            resp = await asyncio.to_thread(_udp_query_sync, port, q)
            flags = struct.unpack_from("!H", resp, 2)[0]
            assert (flags & 0xF) == 2
        finally:
            await dns_resolver.stop_dns_resolver()

    asyncio.run(runner())


def test_parse_query_packet_multi_question_returns_none():
    hdr = struct.pack("!HHHHHH", 1, 0x0100, 2, 0, 0, 0)
    assert dns_resolver.parse_query_packet(hdr + b"\x00") is None


def test_get_ministack_container_ipv4_on_network(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "ministack-test")

    fake_ctr = MagicMock()
    fake_ctr.attrs = {
        "NetworkSettings": {
            "Networks": {
                "bridge": {"IPAddress": "172.18.0.5"},
            }
        }
    }
    fake_dc = MagicMock()
    fake_dc.containers.get.return_value = fake_ctr

    ip = docker_network_mod.get_ministack_container_ipv4_on_network(fake_dc, "bridge")
    assert ip == "172.18.0.5"

