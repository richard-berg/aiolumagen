"""Reusable local-only serial bridge for cross-layer lifecycle tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import ClassVar


class LoopbackDevice:
    """A fake Lumagen on loopback TCP; no hardware or external network access."""

    replies: ClassVar[dict[bytes, bytes]] = {
        b"ZQS01": b"!S01,RadiancePro,030326,1018,000000\r\n",
        b"ZQS02": b"!S02,1\r\n",
        b"ZQI00": b"!I00,01,A,0\r\n",
        b"ZQI25": (
            b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,"
            b"059,2160,237,2,0,p,P,01,01,178,178,A,1,0,2\r\n"
        ),
        b"ZQI30": b"!I30,Y4N\r\n",
        b"ZQI50": b"!I50,Y\r\n",
        b"ZQI52": b"!I52,0,.0000,0,0\r\n",
        b"ZQI53": b"!I53,0\r\n",
        b"ZQI54": b"!I54,1\r\n",
        b"ZQO01": b"!O01,5994,4096,2160,0,0\r\n",
    }

    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.writers: list[asyncio.StreamWriter] = []
        self.handlers: set[asyncio.Task[None]] = set()
        self.received: list[bytes] = []
        self.fragment_size = 3
        self.silent = False
        self.url = ""

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"socket://127.0.0.1:{port}"
        return self.url

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.append(writer)
        task = asyncio.create_task(self._serve(reader, writer))
        self.handlers.add(task)
        task.add_done_callback(self.handlers.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        pending = bytearray()
        try:
            while data := await reader.read(1024):
                pending.extend(data)
                while pending:
                    if pending[:1] == b"Z" and len(pending) < 3:
                        break
                    if pending.startswith(b"ZE2"):
                        size = 3
                    elif pending.startswith(b"ZQS1"):
                        size = 6
                    elif pending.startswith(b"ZQ"):
                        size = 5
                    elif pending.startswith((b"ZY", b"ZT")):
                        end = pending.find(b"\r")
                        if end < 0:
                            break
                        size = end + 1
                    else:
                        size = 1
                    if len(pending) < size:
                        break
                    command = bytes(pending[:size])
                    del pending[:size]
                    self.received.append(command)
                    reply = self.replies.get(command)
                    if command.startswith(b"ZQS1"):
                        reply = b"!S1" + command[4:5] + b",input " + command[5:] + b"\r\n"
                    if reply is not None and not self.silent:
                        for offset in range(0, len(reply), self.fragment_size):
                            writer.write(reply[offset : offset + self.fragment_size])
                            await writer.drain()
                            await asyncio.sleep(0)
        except ConnectionError:
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    async def drop_latest(self) -> None:
        writer = self.writers[-1]
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for writer in self.writers:
            writer.close()
        if self.handlers:
            await asyncio.gather(*self.handlers)
        self.handlers.clear()
