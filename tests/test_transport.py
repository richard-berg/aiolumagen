"""LumagenTransport against a real ``socket://`` peer.

Only the connection lifecycle is tested here; bytes are the protocol's
business and are covered with recorded streams elsewhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from aiolumagen.exceptions import LumagenConnectionError
from aiolumagen.transport import LumagenTransport


class _Peer:
    """A TCP server standing in for a serial bridge."""

    def __init__(self) -> None:
        self.writers: list[asyncio.StreamWriter] = []
        self.accepted = asyncio.Event()
        self.server: asyncio.Server | None = None

    async def _on_client(self, _reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.append(writer)
        self.accepted.set()

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._on_client, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"socket://127.0.0.1:{port}"

    async def close_latest(self) -> None:
        writer = self.writers[-1]
        writer.close()
        await writer.wait_closed()

    async def stop(self) -> None:
        for writer in self.writers:
            writer.close()
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


@pytest.fixture
async def peer() -> AsyncIterator[_Peer]:
    p = _Peer()
    yield p
    await p.stop()


async def _settle() -> None:
    """Long enough for a closed socket's connection_lost to be delivered."""
    await asyncio.sleep(0.2)


async def test_peer_closing_the_connection_is_reported_at_once(peer: _Peer) -> None:
    """Regression: a closed connection was only noticed at the client's next poll tick."""
    transport = LumagenTransport(await peer.start())
    lost = asyncio.Event()
    transport.set_connection_lost_callback(lost.set)
    await transport.connect()
    await asyncio.wait_for(peer.accepted.wait(), 1.0)

    await peer.close_latest()
    await asyncio.wait_for(lost.wait(), 1.0)
    assert not transport.connected
    with pytest.raises(LumagenConnectionError):
        await transport.write(b"ZQS02")


async def test_disconnect_is_not_reported_as_a_loss(peer: _Peer) -> None:
    transport = LumagenTransport(await peer.start())
    losses: list[None] = []
    transport.set_connection_lost_callback(lambda: losses.append(None))
    await transport.connect()
    await transport.disconnect()
    await _settle()
    assert losses == []


async def test_closing_an_old_connection_does_not_mark_its_replacement_down(
    peer: _Peer,
) -> None:
    """A reconnect's first connection may report its close after the second opened."""
    transport = LumagenTransport(await peer.start())
    losses: list[None] = []
    transport.set_connection_lost_callback(lambda: losses.append(None))
    await transport.connect()
    await transport.disconnect()
    await transport.connect()
    await _settle()
    assert transport.connected
    assert losses == []
    await transport.disconnect()
