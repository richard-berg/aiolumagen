"""Adversarial regressions, deterministic stream fuzzing and lifecycle stress."""

from __future__ import annotations

import asyncio
import math
import os
import pty
import random
from collections.abc import AsyncIterator, Callable

import pytest

from aiolumagen.client import LumagenClient
from aiolumagen.commands import (
    fan_speed_command,
    hdr_intensity_mapping_command,
    input_command,
    input_label_command,
    input_restart_command,
    osd_message_command,
    sharpness_command,
    subtitle_shift_command,
)
from aiolumagen.exceptions import LumagenCommandError, LumagenConnectionError
from aiolumagen.formatting import (
    decode_aspect_ratio,
    decode_vertical_rate,
    derive_horizontal_resolution,
)
from aiolumagen.protocol import MAX_RX_BUFFER, LumagenProtocol
from aiolumagen.state import HdrGammaMode, SharpnessSensitivity
from aiolumagen.transport import LumagenTransport, _BridgeProtocol
from tests.conftest import FakeTransport
from tests.loopback_device import LoopbackDevice


@pytest.fixture
async def fast_client(
    monkeypatch: pytest.MonkeyPatch, fake_transport: FakeTransport
) -> AsyncIterator[LumagenClient]:
    async def handshake(client: LumagenClient) -> bool:
        fake_transport.feed(b"!S00\r\n")
        return True

    monkeypatch.setattr(LumagenClient, "_send_startup_sequence", handshake)
    client = LumagenClient(fake_transport, power_poll_interval=None, status_poll_interval=None)
    await client.start()
    yield client
    await client.stop()


def test_overflow_recovery_does_not_parse_a_damaged_line_suffix() -> None:
    wire = b"x" * (MAX_RX_BUFFER + 1) + b"!S02,0\r\n!S00\r\n"
    states = []
    for chunks in ([wire], [wire[: MAX_RX_BUFFER + 1], wire[MAX_RX_BUFFER + 1 :]]):
        protocol = LumagenProtocol(lambda *_: None)
        protocol.feed_bytes(b"!S02,1\r\n")
        for chunk in chunks:
            protocol.feed_bytes(chunk)
        states.append(protocol.state)
        assert protocol.state.power_on is True
        assert protocol.state.alive
    assert states[0] == states[1]


@pytest.mark.parametrize(
    "code,on",
    [("S02", "1"), ("I50", "Y"), ("I53", "1"), ("I54", "1"), ("S02", "0")],
)
@pytest.mark.parametrize("payload", [b"", b"?", b"11", b"1\x00", b"\xff"])
def test_damaged_boolean_reply_cannot_turn_a_known_value_off(
    code: str, on: str, payload: bytes
) -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(f"!{code},{on}\r\n".encode())
    before = protocol.state
    protocol.feed_bytes(b"!" + code.encode() + b"," + payload + b"\r\n")
    assert protocol.state == before


@pytest.mark.parametrize("value", ["nan", "inf", "1e309", "-1"])
def test_hdr_metadata_never_stores_nonfinite_or_negative_values(value: str) -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(b"!I52,1,.0050,1000,4000\r\n")
    protocol.feed_bytes(f"!I52,1,{value},-2,-3\r\n".encode())
    state = protocol.state
    assert state.hdr_source_min_luminance == 0.005
    assert state.hdr_source_max_luminance == 1000
    assert state.hdr_source_max_cll == 4000


def test_legacy_status_numeric_bomb_cannot_raise_or_block_later_lines() -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(b"!I21,1," + b"9" * 400 + b",1080,0,0,178,178\r\n!S00\r\n")
    assert protocol.state.alive
    assert protocol.state.source_refresh_hz is None


@pytest.mark.parametrize("payload", ["", "5994,broken,2160,0,0", "5994,3840,-1,0,0"])
def test_damaged_output_query_does_not_replace_authoritative_geometry(payload: str) -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(b"!O01,5994,4096,2160,0,0\r\n")
    protocol.feed_bytes(f"!O01,{payload}\r\n".encode())
    assert protocol.state.output_width_reported == 4096
    assert protocol.state.output_height_reported == 2160
    assert protocol.state.output_width == 4096


def test_damaged_hdr_flag_does_not_fabricate_an_sdr_transition() -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(b"!I52,1,.0050,1000,4000\r\n")
    before = protocol.state
    protocol.feed_bytes(b"!I52,?,0,0,0\r\n")
    assert protocol.state == before


@pytest.mark.parametrize("payload", ["Y", "Y7?", "Y7Hgarbage", "N9N"])
def test_damaged_sharpness_reply_cannot_partially_replace_a_compound_value(payload: str) -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.feed_bytes(b"!I30,N2N\r\n")
    protocol.feed_bytes(f"!I30,{payload}\r\n".encode())
    assert not protocol.state.sharpness_enabled
    assert protocol.state.sharpness_level == 2
    assert protocol.state.sharpness_sensitivity == "N"


def test_wrong_label_bank_cannot_consume_the_expected_input() -> None:
    protocol = LumagenProtocol(lambda *_: None)
    protocol.expect_input_label(1, memory="A")
    protocol.feed_bytes(b"!S1B,wrong bank\r\n")
    assert protocol.pending_label_input == 1
    assert protocol.state.input_labels == {}
    protocol.feed_bytes(b"!S1A,correct\r\n")
    assert protocol.state.input_labels == {1: "correct"}


def test_state_callback_failure_does_not_drop_responses_or_following_lines() -> None:
    def broken_callback(*_: object) -> None:
        raise RuntimeError("consumer failure")

    protocol = LumagenProtocol(broken_callback)
    seen: list[str] = []
    protocol.add_response_observer(lambda code, _: seen.append(code))
    protocol.feed_bytes(b"!S02,1\r\n!S00\r\n")
    assert seen == ["S02", "S00"]
    assert protocol.state.alive


@pytest.mark.parametrize("seed", range(8))
def test_seeded_damaged_stream_is_chunk_invariant_and_bounded(seed: int) -> None:
    rng = random.Random(seed)
    records: list[bytes] = []
    codes = [b"S02", b"I50", b"I30", b"I52", b"I21", b"I25", b"I99", b"S01", b"O01"]
    for _ in range(2500):
        prefix = rng.choice([b"", b"echo", b"ZQS01"])
        payload = rng.randbytes(rng.randrange(100))
        records.append(prefix + b"!" + rng.choice(codes) + b"," + payload + b"\r\n")
    records.insert(50, b"x" * (MAX_RX_BUFFER * 4) + b"!S02,0\r\n")
    records.append(b"!S02,1\r\n!S00\r\n")
    wire = b"".join(records)
    expected: list[tuple[str, str]] = []
    whole = LumagenProtocol(lambda *_: None)
    whole.add_response_observer(lambda code, payload: expected.append((code, payload)))
    whole.feed_bytes(wire)

    actual: list[tuple[str, str]] = []
    chunked = LumagenProtocol(lambda *_: None)
    chunked.add_response_observer(lambda code, payload: actual.append((code, payload)))
    offset = 0
    while offset < len(wire):
        size = rng.randrange(1, 8192)
        chunked.feed_bytes(wire[offset : offset + size])
        offset += size
        assert len(chunked._rx) <= MAX_RX_BUFFER
    assert actual == expected
    assert chunked.state == whole.state
    assert chunked.state.power_on
    assert chunked.state.alive


@pytest.mark.parametrize("seed", range(4))
def test_seeded_mutations_of_real_response_shapes_preserve_numeric_invariants(seed: int) -> None:
    rng = random.Random(seed)
    protocol = LumagenProtocol(lambda *_: None)
    captures = [
        *LoopbackDevice.replies.values(),
        b"!I21,1,060,1080,0,0,178,178,-,0,000e,0,0,060,1080\r\n",
    ]
    for _ in range(2500):
        frame = bytearray(rng.choice(captures))
        start = rng.randrange(len(frame))
        frame[start : start + rng.randrange(5)] = rng.randbytes(rng.randrange(5))
        frame.extend(b"\n")
        split = rng.randrange(len(frame))
        protocol.feed_bytes(frame[:split])
        protocol.feed_bytes(frame[split:])
        state = protocol.state
        for value in (
            state.source_refresh_hz,
            state.output_refresh_hz,
            state.hdr_source_min_luminance,
        ):
            assert value is None or (math.isfinite(value) and value >= 0)
        assert state.sharpness_level is None or 0 <= state.sharpness_level <= 7
        assert state.output_enabled_mask is None or 0 <= state.output_enabled_mask <= 0xFFFF
        assert len(protocol._rx) <= MAX_RX_BUFFER


@pytest.mark.parametrize(
    "decode",
    [decode_vertical_rate, decode_aspect_ratio, lambda s: derive_horizontal_resolution(s, s)],
)
def test_formatting_overflow_is_an_unknown_value(decode: Callable[[str], object]) -> None:
    assert decode("9" * 400) is None


class _Wire(asyncio.Transport):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def write(self, data: bytes | bytearray | memoryview) -> None:
        if self.closed:
            raise OSError("closed while writing")


async def test_transport_concurrent_connects_open_only_one_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wires: list[_Wire] = []

    async def open_wire(*args: object, **kwargs: object) -> tuple[_Wire, object]:
        await asyncio.sleep(0)
        wire = _Wire()
        wires.append(wire)
        return wire, object()

    monkeypatch.setattr("aiolumagen.transport.serialx.create_serial_connection", open_wire)
    transport = LumagenTransport("socket://unused")
    try:
        await asyncio.gather(*(transport.connect() for _ in range(25)))
        assert len(wires) == 1
    finally:
        await transport.disconnect()
        assert all(w.closed for w in wires)


async def test_cancelled_transport_open_closes_already_created_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _Wire()
    made = asyncio.Event()

    async def open_wire(
        loop: asyncio.AbstractEventLoop, factory: Callable[[], _BridgeProtocol], **kwargs: object
    ) -> tuple[_Wire, _BridgeProtocol]:
        protocol = factory()
        protocol.connection_made(wire)
        made.set()
        await asyncio.Event().wait()
        return wire, protocol

    monkeypatch.setattr("aiolumagen.transport.serialx.create_serial_connection", open_wire)
    transport = LumagenTransport("socket://unused")
    task = asyncio.create_task(transport.connect())
    await made.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire.closed
    assert not transport.connected


def test_old_protocol_cannot_inject_bytes_into_replacement() -> None:
    transport = LumagenTransport("socket://unused")
    old = _BridgeProtocol(transport)
    current = _BridgeProtocol(transport)
    transport._protocol = current
    seen: list[bytes] = []
    transport.set_data_callback(seen.append)
    old.data_received(b"!S02,0\r\n")
    current.data_received(b"!S02,1\r\n")
    assert seen == [b"!S02,1\r\n"]


async def test_transport_write_failure_is_a_connection_error() -> None:
    transport = LumagenTransport("socket://unused")
    wire = _Wire()
    wire.close()
    transport._transport = wire
    transport._connected = True
    with pytest.raises(LumagenConnectionError):
        await transport.write(b"ZQS02")
    assert not transport.connected


async def test_direct_serial_open_is_exclusive_even_through_a_path_alias() -> None:
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    first = LumagenTransport(path)
    alias = LumagenTransport(f"{os.path.dirname(path)}/./{os.path.basename(path)}")
    try:
        await first.connect()
        with pytest.raises(LumagenConnectionError):
            await alias.connect()
        assert first.connected
    finally:
        await first.disconnect()
        await alias.disconnect()
        os.close(slave)
        os.close(master)


async def test_cancelled_start_is_restartable_and_does_not_leak_connection(
    monkeypatch: pytest.MonkeyPatch, fake_transport: FakeTransport
) -> None:
    entered = asyncio.Event()

    async def handshake(client: LumagenClient) -> bool:
        entered.set()
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(LumagenClient, "_send_startup_sequence", handshake)
    client = LumagenClient(fake_transport, power_poll_interval=None, status_poll_interval=None)
    task = asyncio.create_task(client.start())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        assert not client.connected
        assert not client._started
        assert not client._response_waiters
    finally:
        await client.stop()


async def test_cancelling_stop_still_finishes_transport_cleanup(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    original = fake_transport.disconnect

    async def delayed_disconnect() -> None:
        entered.set()
        await release.wait()
        await original()

    fake_transport.disconnect = delayed_disconnect  # type: ignore[method-assign]
    task = asyncio.create_task(fast_client.stop())
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not fast_client.connected
    assert not fast_client._response_waiters


async def test_query_deadline_includes_blocked_transport_write(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    async def blocked_write(data: bytes) -> None:
        await asyncio.Event().wait()

    fake_transport.write = blocked_write  # type: ignore[method-assign]
    async with asyncio.timeout(0.2):
        with pytest.raises(TimeoutError):
            await fast_client.query_and_wait("ZQS02", timeout=0.01)
    assert not fast_client._response_waiters


async def test_sharpness_setter_remains_nonblocking_and_does_not_fabricate_readback(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    fake_transport.feed(b"!I30,Y4N\r\n")
    generation = fast_client.state.sharpness_report_generation
    fake_transport.sent.clear()
    async with asyncio.timeout(0.1):
        await fast_client.set_sharpness(
            enabled=True, level=4, sensitivity=SharpnessSensitivity.HIGH
        )
    assert fake_transport.sent == [b"ZY521Y4H\r", b"ZQI30"]
    assert fast_client.state.sharpness_report_generation == generation
    assert fast_client.state.sharpness_sensitivity == "N"
    fake_transport.feed(b"!I30,\r\n")
    assert fast_client.state.sharpness_report_generation == generation
    fake_transport.feed(b"!I30,Y4N\r\n")
    assert fast_client.state.sharpness_report_generation == generation + 1


async def test_client_can_observe_equal_sharpness_readback_without_a_subscriber_callback(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    fake_transport.feed(b"!I30,Y4N\r\n")
    events: list[tuple[str, ...]] = []
    fast_client.subscribe(lambda _, codes: events.append(codes))
    stale_state = fast_client.state
    query = asyncio.create_task(fast_client.query_and_wait("ZQI30", timeout=0.1))
    await asyncio.sleep(0)
    fake_transport.feed(b"!I30,Y4N\r\n")
    assert await query == "Y4N"
    assert events == []
    assert fast_client.state.sharpness_report_generation > stale_state.sharpness_report_generation


async def test_continuous_junk_cannot_refresh_an_expired_client(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    now = asyncio.get_running_loop().time()
    fast_client._stale_timeout = 0.1
    fast_client._last_response_time = now - 1
    fast_client._connected_at = now - 1
    old_time = fast_client._last_response_time
    junk = [
        b"noise without a terminator",
        b"x" * (MAX_RX_BUFFER + 1),
        b"!S02,1\r\n",  # Still the suffix of the damaged overlong line.
        b"ZE2\r\n",
        b"!S02,?\r\n",
        b"!S02\r\n",
        b"!I25,1,059,broken\r\n",
        b"!I30,Y4\r\n",
        b"!I52,1,nan,-1,-1\r\n",
        b"!S01,RadiancePro\r\n",
        b"!I99,garbage\r\n",
        b"\xff!S00\r\n",
    ]
    for _ in range(20):
        for data in junk:
            fake_transport.feed(data)
            assert fast_client._last_response_time == old_time
            assert not fast_client.available
            assert fast_client._link_down_reason(now) == "no response in 0s"


@pytest.mark.parametrize(
    "reply",
    [
        b"!S00\r\n",
        b"!S02,0\r\n",  # Ordinary standby.
        b"!I25,\r\n",  # Unsupported Full v5 on older firmware.
        b"!I99,\r\n",  # Unsupported undocumented query.
        b"!I25,0,000,0000,0,0,000,000\r\n",  # Legacy/minimal no-signal report.
        b"!I21,0,000,0000,0,0,000,000\r\n",
        b"!S01,RadiancePro,030326,1018,000000\r\n",
        b"!I52,0,.0000,0,0\r\n",
        b"!O01,0,0,0,0,0\r\n",
        b"!I30,Y4N\r\n",
    ],
)
async def test_valid_equal_reports_refresh_liveness_without_state_change_callbacks(
    fast_client: LumagenClient, fake_transport: FakeTransport, reply: bytes
) -> None:
    fake_transport.feed(reply)
    events: list[tuple[str, ...]] = []
    fast_client.subscribe(lambda _, codes: events.append(codes))
    for _ in range(10):
        now = asyncio.get_running_loop().time()
        fast_client._stale_timeout = 0.1
        fast_client._last_response_time = now - 1
        assert not fast_client.available
        fake_transport.feed(reply)
        assert fast_client._last_response_time >= now
        assert fast_client.available
    assert events == []


async def test_initial_noise_does_not_announce_recovery_or_reset_reconnect_backoff(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    fast_client._available = False
    fast_client._reconnect_failures = 3
    fast_client._reconnect_delay = 40
    fast_client._next_reconnect_at = 100
    events: list[tuple[str, ...]] = []
    fast_client.subscribe(lambda _, codes: events.append(codes))
    fake_transport.feed(b"!I30,garbage\r\nnoise\r\n")
    assert not fast_client.available
    assert fast_client._reconnect_failures == 3
    assert fast_client._reconnect_delay == 40
    assert fast_client._next_reconnect_at == 100
    assert ("_available",) not in events


async def test_cancelled_label_read_clears_correlation_context(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    task = asyncio.create_task(fast_client.query_input_labels())
    await asyncio.sleep(0)
    assert fast_client._protocol.pending_label_input == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake_transport.feed(b"!S1A,late\r\n")
    assert fast_client._protocol.pending_label_input is None
    assert fast_client.state.input_labels == {}


async def test_concurrent_label_reads_do_not_share_or_overwrite_a_primer(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    original = fake_transport.write

    async def answer(data: bytes) -> None:
        await original(data)
        number = data[-1:].decode()
        asyncio.get_running_loop().call_later(
            0.001, fake_transport.feed, f"!S1A,input {number}\r\n"
        )

    fake_transport.write = answer  # type: ignore[method-assign]
    assert await asyncio.gather(
        fast_client._read_input_label(1, "A", 0.1),
        fast_client._read_input_label(2, "A", 0.1),
    ) == [True, True]
    assert fast_client.state.input_labels == {1: "input 0", 2: "input 1"}


async def test_abandoned_label_bank_stays_quarantined_until_a_fresh_connection(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    await fast_client.query_input_labels(timeout=0.001)
    await fast_client.query_input_labels(timeout=0.001)
    assert fake_transport.sent == [b"ZQS1A0"]
    await fast_client.stop()
    await fast_client.start()
    await fast_client.query_input_labels(timeout=0.001)
    assert fake_transport.sent == [b"ZQS1A0", b"ZQS1A0"]


async def test_repeated_concurrent_starts_and_stops_leave_no_background_workers(
    monkeypatch: pytest.MonkeyPatch, fake_transport: FakeTransport
) -> None:
    async def handshake(client: LumagenClient) -> bool:
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(LumagenClient, "_send_startup_sequence", handshake)
    fake_transport.connect_hook = lambda: asyncio.sleep(0)
    client = LumagenClient(
        fake_transport, power_poll_interval=1, status_poll_interval=None, stale_timeout=2
    )
    for _ in range(20):
        await asyncio.gather(*(client.start() for _ in range(10)))
        assert client.connected
        await asyncio.gather(*(client.stop() for _ in range(10)))
        assert not client.connected
        assert not client._response_waiters
    assert fake_transport.connect_calls == 20
    assert not any(t.get_name() == "aiolumagen-poll" for t in asyncio.all_tasks())


async def test_real_tcp_fragmentation_drop_reconnect_and_stop_leave_no_tasks() -> None:
    device = LoopbackDevice()
    transport = LumagenTransport(await device.start())
    client = LumagenClient(
        transport, power_poll_interval=0.02, status_poll_interval=0.02, stale_timeout=0.2
    )
    client.RECONNECT_SETTLE = 0
    client.RECONNECT_BACKOFF_INITIAL = 0.02
    client.RECONNECT_BACKOFF_MAX = 0.05
    events: list[tuple[str, ...]] = []
    client.subscribe(lambda _, codes: events.append(codes))
    try:
        async with asyncio.timeout(10):
            await client.start()
            await client.query_input_labels(timeout=0.2)
            assert client.state.input_labels == {i: f"input {i - 1}" for i in range(1, 9)}
            for attempt in range(3):
                await device.drop_latest()
                while len(device.writers) < attempt + 2 or client._handling_link_down:
                    await asyncio.sleep(0.01)
                assert client.available
                assert client.state.model == "RadiancePro"
                assert client.state.output_width == 4096
            assert events.count(("_unavailable",)) == 3
            assert events.count(("_available",)) == 4
    finally:
        await client.stop()
        await device.stop()
    assert not transport.connected
    assert not client._response_waiters
    assert not device.handlers
    assert not any(t.get_name().startswith("aiolumagen-") for t in asyncio.all_tasks())


async def test_waiter_cancellation_timeout_and_disconnect_stress(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    for _ in range(10):
        waiters = [
            asyncio.create_task(fast_client.wait_for_response("I99", timeout=1)) for _ in range(100)
        ]
        await asyncio.sleep(0)
        for waiter in waiters[::3]:
            waiter.cancel()
        fake_transport.feed(b"!I99,value\r\n")
        results = await asyncio.gather(*waiters, return_exceptions=True)
        assert sum(result == "value" for result in results) == 66
        assert not fast_client._response_waiters
    waiters = [
        asyncio.create_task(fast_client.wait_for_response("I99", timeout=0.001)) for _ in range(100)
    ]
    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert all(isinstance(result, TimeoutError) for result in results)
    assert not fast_client._response_waiters


async def test_label_timeout_does_not_misfile_a_late_reply_under_next_input(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    original = fake_transport.write

    async def answer_late(data: bytes) -> None:
        await original(data)
        if data == b"ZQS1A0":
            asyncio.get_running_loop().call_later(0.015, fake_transport.feed, b"!S1A,input one\r\n")

    fake_transport.write = answer_late  # type: ignore[method-assign]
    await fast_client.query_input_labels(timeout=0.01)
    await asyncio.sleep(0.03)
    assert fast_client.state.input_labels == {}
    assert fake_transport.sent == [b"ZQS1A0"]


async def test_disconnect_fails_waiters_even_without_polling(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    waiter = asyncio.create_task(fast_client.wait_for_response("I99", timeout=1))
    await asyncio.sleep(0)
    fake_transport.drop()
    try:
        await asyncio.sleep(0)
        assert waiter.done()
        with pytest.raises(LumagenConnectionError):
            await waiter
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def test_stopped_client_ignores_late_bytes(
    fast_client: LumagenClient, fake_transport: FakeTransport
) -> None:
    await fast_client.stop()
    before = fast_client.state
    fake_transport.feed(b"!S02,1\r\n")
    assert fast_client.state == before
    assert not fast_client._available


@pytest.mark.parametrize("interval", [0.0, -1.0, math.nan, math.inf])
def test_poll_configuration_cannot_create_busy_or_dead_supervisor(
    fake_transport: FakeTransport, interval: float
) -> None:
    with pytest.raises(ValueError):
        LumagenClient(fake_transport, power_poll_interval=interval)


@pytest.mark.parametrize(
    "build",
    [
        input_command,
        fan_speed_command,
        subtitle_shift_command,
        input_restart_command,
        lambda value: sharpness_command(enabled=True, level=value),
        lambda value: input_label_command(value, "label"),
        lambda value: osd_message_command(text="message", duration=value),
        lambda value: hdr_intensity_mapping_command(
            display_max_nits=value, gamma_mode=HdrGammaMode.AUTO
        ),
    ],
)
@pytest.mark.parametrize("value", [1.0, True, None, "1"])
def test_integer_commands_reject_floats_instead_of_emitting_extra_wire_characters(
    build: Callable[[int], str], value: object
) -> None:
    with pytest.raises(LumagenCommandError):
        build(value)  # type: ignore[arg-type]
