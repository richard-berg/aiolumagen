"""Adversarial firmware regressions; synthetic files and simulated UARTs only."""

from __future__ import annotations

import asyncio
import io
import random
import struct
import zipfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace

import pytest

from aiolumagen.exceptions import (
    LumagenConnectionError,
    LumagenFirmwareAbortError,
    LumagenFirmwareError,
    LumagenFirmwareImageError,
)
from aiolumagen.firmware.container import find_containers, parse_container
from aiolumagen.firmware.extract import (
    FirmwareBundle,
    FirmwareImage,
    extract_images,
    extract_resources,
    pe_sections,
    rva_to_offset,
)
from aiolumagen.firmware.plan import plan_update
from aiolumagen.firmware.protocol import (
    BLOCK_SIZE,
    BOOT_SECTOR_LEN,
    SCRATCH_ADDR,
    SECTION1_SLOTS,
    SECTOR_SIZE,
    parse_checksum,
    parse_identity,
)
from aiolumagen.firmware.releases import extract_updater_zip
from aiolumagen.firmware.session import MAX_RX_BYTES, FirmwareSession
from tests.conftest import FakeFirmwareTransport, build_container, build_updater_exe

_REAL_SLEEP = asyncio.sleep


def small_bundle() -> FirmwareBundle:
    return extract_images(
        build_updater_exe(
            swdata=b"\xaa" * BOOT_SECTOR_LEN + b"\x5a" * (2 * BLOCK_SIZE),
            resources={131: build_container(b"\x33" * (2 * BLOCK_SIZE))},
        )
    )


class Wire:
    """Small scripted transport; avoids allocating a flash array per fuzz case."""

    def __init__(self) -> None:
        self.connected = False
        self.callback: Callable[[bytes], None] = lambda _data: None
        self.writes: list[bytes] = []
        self.on_write: Callable[[bytes], None] = lambda _data: None
        self.disconnects = 0

    def set_data_callback(self, callback: Callable[[bytes], None]) -> None:
        self.callback = callback

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = False

    async def write(self, data: bytes) -> None:
        self.writes.append(data)
        self.on_write(data)


class IdentifiedFirmwareTransport(FakeFirmwareTransport):
    def __init__(self, *, model_number: str = "1018", serial: str = "001234") -> None:
        super().__init__()
        self.normal_identity = f"!S01,RadiancePro,030326,{model_number},{serial}\r\n"

    def _handle(self, command: str) -> None:
        if command == "ZQS01":
            self._reply(self.normal_identity)
        else:
            super()._handle(command)


class TestExpectedDevice:
    pytestmark = pytest.mark.usefixtures("no_delays")

    @pytest.mark.parametrize(
        ("model", "serial", "message"),
        [
            ("4242", "1234", "model"),
            ("4242", None, "model"),
            ("4242", "000000", "model"),
            ("1018", "5678", "serial"),
        ],
    )
    async def test_mismatch_refuses_before_updater_entry_or_flash(
        self, model: str, serial: str | None, message: str
    ) -> None:
        transport = IdentifiedFirmwareTransport()
        async with FirmwareSession(transport=transport) as session:
            with pytest.raises(LumagenFirmwareAbortError, match=message):
                await session.run_update(
                    small_bundle(), expected_model_number=model, expected_serial=serial
                )
        assert "M0931" not in transport.commands
        assert not transport.erases
        assert not transport.payload_writes
        assert not transport.copies

    @pytest.mark.parametrize("fresh_serial", ["000000", "", "unknown"])
    async def test_known_serial_requires_a_fresh_nonzero_matching_serial(
        self, fresh_serial: str
    ) -> None:
        transport = IdentifiedFirmwareTransport(serial=fresh_serial)
        async with FirmwareSession(transport=transport) as session:
            with pytest.raises(LumagenFirmwareAbortError, match="serial"):
                await session.run_update(
                    small_bundle(), expected_model_number="1018", expected_serial="001234"
                )
        assert not transport.erases
        assert "M0931" not in transport.commands

    @pytest.mark.parametrize("expected_serial", [None, "", "0", "000000"])
    async def test_unknown_expected_serial_pins_only_the_model(
        self, expected_serial: str | None
    ) -> None:
        transport = IdentifiedFirmwareTransport(serial="999999")
        async with FirmwareSession(transport=transport) as session:
            result = await session.run_update(
                small_bundle(),
                expected_model_number="1018",
                expected_serial=expected_serial,
                dry_run=True,
            )
        assert result.dry_run
        assert "ZQS01" in transport.commands
        assert not transport.erases

    async def test_matching_normal_identity_does_not_confuse_model_number_with_updater_id(
        self,
    ) -> None:
        transport = IdentifiedFirmwareTransport(serial="00001234")
        async with FirmwareSession(transport=transport) as session:
            result = await session.run_update(
                small_bundle(),
                expected_model_number="1018",
                expected_serial="1234",
                only=["section0"],
                promote=False,
                baudrate=9600,
            )
        assert result.written == ("section0",)
        assert transport.commands.index("ZQS01") < transport.commands.index("M0931")

    async def test_truncated_fresh_identity_fails_closed(self) -> None:
        transport = IdentifiedFirmwareTransport()
        transport.normal_identity = "!S01,RadiancePro,030326,1018\r\n"
        async with FirmwareSession(transport=transport) as session:
            with pytest.raises(LumagenFirmwareAbortError, match="fresh device identity"):
                await session.run_update(
                    small_bundle(), expected_model_number="1018", expected_serial="1234"
                )
        assert not transport.erases
        assert "M0931" not in transport.commands

    @pytest.mark.parametrize("serial", ["unknown", "-1", "١٢٣٤"])
    async def test_malformed_expected_serial_is_not_silently_unpinned(self, serial: str) -> None:
        transport = IdentifiedFirmwareTransport()
        async with FirmwareSession(transport=transport) as session:
            with pytest.raises(LumagenFirmwareError, match="expected_serial"):
                await session.run_update(
                    small_bundle(), expected_model_number="1018", expected_serial=serial
                )
        assert not transport.commands

    async def test_convenience_wrapper_forwards_identity_pins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware as firmware

        transport = IdentifiedFirmwareTransport()
        session = FirmwareSession(transport=transport)

        async def load(_source: str) -> FirmwareBundle:
            return small_bundle()

        monkeypatch.setattr(firmware, "load_updater", load)
        monkeypatch.setattr(firmware, "FirmwareSession", lambda *_a, **_kw: session)
        with pytest.raises(LumagenFirmwareAbortError, match="serial"):
            await firmware.update_firmware(
                "synthetic://",
                "synthetic.exe",
                expected_model_number="1018",
                expected_serial="5678",
            )
        assert "M0931" not in transport.commands
        assert not transport.erases


class TestCleanupVerdict:
    pytestmark = pytest.mark.usefixtures("no_delays")

    async def test_failed_z97_cannot_return_a_successful_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        write = transport.write

        async def fail_finish(data: bytes) -> None:
            if data == b"97":
                raise LumagenConnectionError("Z97 delivery failed")
            await write(data)

        monkeypatch.setattr(transport, "write", fail_finish)
        with pytest.raises(LumagenFirmwareError, match="hand-back"):
            async with FirmwareSession(transport=transport) as session:
                await session.run_update(small_bundle(), only=["section1"], baudrate=9600)
        assert transport.left != "Z97"

    async def test_disconnected_before_handoff_is_not_a_success(self) -> None:
        transport = FakeFirmwareTransport()
        with pytest.raises(LumagenFirmwareError, match="hand-back"):
            async with FirmwareSession(transport=transport) as session:
                await session.run_update(small_bundle(), only=["section1"], baudrate=9600)
                await transport.disconnect()
        assert "Z97" not in transport.commands

    async def test_cleanup_error_preserves_the_original_body_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        session = FirmwareSession(transport=transport)

        async def fail(*, failed: bool = False) -> None:
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(session, "hand_back", fail)
        with pytest.raises(ValueError, match="original failure"):
            async with session:
                raise ValueError("original failure")

    async def test_failed_finish_still_closes_the_owned_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware.session as module

        transport = FakeFirmwareTransport()
        write = transport.write

        async def fail_finish(data: bytes) -> None:
            if data == b"97":
                raise LumagenConnectionError("Z97 delivery failed")
            await write(data)

        monkeypatch.setattr(transport, "write", fail_finish)
        monkeypatch.setattr(module, "LumagenTransport", lambda *_a, **_kw: transport)
        with pytest.raises(LumagenFirmwareError, match="hand-back"):
            async with FirmwareSession("synthetic://") as session:
                await session.run_update(small_bundle(), only=["section1"], baudrate=9600)
        assert not transport.connected
        assert getattr(transport._on_data, "__self__", None) is not session


class TestHostileImages:
    def test_every_truncated_pe_header_is_a_domain_error(self) -> None:
        exe = build_updater_exe()
        for size in range(512):
            try:
                extract_images(exe[:size])
            except LumagenFirmwareImageError:
                continue
            pytest.fail(f"accepted a truncated PE header at {size} bytes")

    def test_rva_does_not_map_virtual_only_section_tail(self) -> None:
        exe = bytearray(build_updater_exe())
        section = pe_sections(exe)[-1]
        pe = struct.unpack_from("<I", exe, 0x3C)[0]
        table = pe + 24 + struct.unpack_from("<H", exe, pe + 20)[0]
        struct.pack_into("<I", exe, table + 2 * 40 + 8, section.raw_size + 1024)
        assert rva_to_offset(exe, section.virtual_address + section.raw_size) is None

    def test_resource_graph_cannot_revisit_directories(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware.extract as extract

        exe = bytearray(build_updater_exe())
        root = next(s.raw_address for s in pe_sections(exe) if s.name == ".rsrc")
        struct.pack_into("<HH", exe, root + 12, 0, 40)
        for index in range(40):
            struct.pack_into("<II", exe, root + 16 + 8 * index, 10, 0x80000000)

        calls = 0
        real_u32 = extract._u32

        def guarded_u32(data: bytes, offset: int) -> int:
            nonlocal calls
            calls += 1
            assert calls < 200, "resource graph causes repeated/exponential traversal"
            return real_u32(data, offset)

        monkeypatch.setattr(extract, "_u32", guarded_u32)
        with pytest.raises(LumagenFirmwareImageError):
            extract_resources(bytes(exe))

    @pytest.mark.parametrize("field", [12, 20])
    def test_overlapping_pe_sections_are_rejected(self, field: int) -> None:
        exe = bytearray(build_updater_exe())
        sections = pe_sections(exe)
        pe = struct.unpack_from("<I", exe, 0x3C)[0]
        table = pe + 24 + struct.unpack_from("<H", exe, pe + 20)[0]
        address = sections[0].virtual_address if field == 12 else sections[0].raw_address
        struct.pack_into("<I", exe, table + 40 + field, address)
        with pytest.raises(LumagenFirmwareImageError, match="overlapping"):
            extract_images(bytes(exe))

    def test_container_scan_bounds_candidate_work(self) -> None:
        with pytest.raises(LumagenFirmwareImageError, match="too many container"):
            find_containers(build_container(b"") * 4097)

    async def test_disk_loading_is_bounded_before_allocating_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        import aiolumagen.firmware as firmware

        reads: list[int] = []

        class File(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                reads.append(size)
                return super().read(size)

        monkeypatch.setattr(firmware, "MAX_UPDATER_EXE_BYTES", 32)
        monkeypatch.setattr(Path, "open", lambda *_a, **_kw: File(b"MZ" + b"\x00" * 100))
        with pytest.raises(LumagenFirmwareImageError, match="extraction limit"):
            await firmware.load_updater("synthetic.exe")
        assert reads == [33]

    @pytest.mark.parametrize("size", [0, 128, BOOT_SECTOR_LEN])
    def test_section0_must_contain_bytes_after_the_bootloader(self, size: int) -> None:
        bundle = FirmwareBundle(
            images={"section0": FirmwareImage("section0", b"\xaa" * size, "synthetic")}
        )
        with pytest.raises(LumagenFirmwareImageError):
            plan_update(bundle)

    def test_section0_must_not_overlap_section1_when_promoted(self) -> None:
        bundle = FirmwareBundle(
            images={"section0": FirmwareImage("section0", b"\xaa" * 0x100001, "synthetic")}
        )
        with pytest.raises(LumagenFirmwareImageError):
            plan_update(bundle)

    def test_section1_must_fit_the_smaller_slot(self) -> None:
        raw = build_container(b"\x5a" * 0x400000)
        container = parse_container(raw)
        image = FirmwareImage("section1", container.payload, "synthetic", container=container)
        with pytest.raises(LumagenFirmwareImageError):
            plan_update(FirmwareBundle(images={"section1": image}), only=["section1"])

    def test_forged_container_object_does_not_bypass_checksum_validation(self) -> None:
        container = parse_container(build_container(b"\x5a" * 4096))
        forged = replace(container, raw=container.raw[:-1] + b"\x00")
        image = FirmwareImage("section1", forged.payload, "synthetic", container=forged)
        with pytest.raises(LumagenFirmwareImageError):
            plan_update(FirmwareBundle(images={"section1": image}), only=["section1"])

    def test_shortened_container_size_cannot_silently_discard_zero_tail(self) -> None:
        raw = bytearray(build_container(b"\x5a" * 100 + b"\x00" * 128))
        struct.pack_into("<I", raw, 12, 100)
        with pytest.raises(LumagenFirmwareImageError):
            parse_container(bytes(raw))


class TestReplyValidation:
    @pytest.mark.parametrize(
        "reply",
        ["CS=0", "CS=000", "CS=+0000001", "CS=-0000001", "CS=000_0001", "CS=0x000001"],
    )
    def test_incomplete_or_non_wire_checksums_are_not_verification(self, reply: str) -> None:
        assert parse_checksum(reply) is None

    @pytest.mark.parametrize(
        "reply",
        [
            "1.16.1",
            "030326.+16.0001",
            "030326.0x16.0001",
            "030326.16.-001",
            "030326.16.0001.extra",
            "030326.16.00001",
        ],
    )
    def test_identity_requires_exact_wire_fields(self, reply: str) -> None:
        assert parse_identity(reply) is None


class TestSessionSafety:
    pytestmark = pytest.mark.usefixtures("no_delays")

    @pytest.mark.parametrize("chunk", [0, -1])
    def test_invalid_chunk_cannot_silently_discard_firmware(self, chunk: int) -> None:
        with pytest.raises(LumagenFirmwareError):
            FirmwareSession(transport=Wire(), chunk=chunk)

    async def test_prebuilt_plan_cannot_bypass_device_model_check(self) -> None:
        from tests.conftest import FakeFirmwareTransport

        transport = FakeFirmwareTransport(identity="030326.13.0001")
        async with FirmwareSession(transport=transport) as session:
            with pytest.raises(LumagenFirmwareError):
                await session.run_update(
                    small_bundle(), plan=plan_update(small_bundle()), baudrate=9600
                )
        assert not transport.erases

    async def test_failed_enter_does_not_leak_owned_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware.session as module

        wire = Wire()

        async def fail() -> None:
            wire.connected = True
            raise RuntimeError("connect failed after allocation")

        monkeypatch.setattr(wire, "connect", fail)
        monkeypatch.setattr(module, "LumagenTransport", lambda *_args, **_kw: wire)
        with pytest.raises(RuntimeError, match="connect failed"):
            async with FirmwareSession("synthetic://"):
                pytest.fail("unreachable")
        assert not wire.connected
        assert wire.disconnects == 1

    async def test_close_releases_borrowed_transport_callback(self) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        await session.connect()
        await session.close()
        wire.callback(b"late bytes")
        assert session._buffer == b""
        assert wire.connected

    async def test_read_at_never_returns_undecoded_reply_bytes(self) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        await session.connect()
        wire.on_write = lambda data: wire.callback(b"zzzz\r\n") if data == b"R" else None
        assert await session.read_at(SCRATCH_ADDR, 4) == b""

    async def test_repair_rejects_unaligned_base_before_erasing(self) -> None:
        wire = Wire()
        wire.on_write = lambda _data: pytest.fail("unaligned repair reached the UART")
        session = FirmwareSession(transport=wire)
        await session.connect()
        with pytest.raises(LumagenFirmwareError):
            await session.repair(b"\xaa" * BLOCK_SIZE, SCRATCH_ADDR + 1)
        assert not wire.writes

    @pytest.mark.parametrize(
        ("dest", "source", "length"),
        [(0, SCRATCH_ADDR, 100), (0x20000, 0, 100), (0x20000, SCRATCH_ADDR, 0)],
    )
    async def test_promotion_cannot_overwrite_bootloader_or_copy_unstaged_bytes(
        self, dest: int, source: int, length: int
    ) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        await session.connect()
        with pytest.raises(LumagenFirmwareError):
            await session.promote(dest, source, length)
        assert not wire.writes

    async def test_small_container_checkpoint_never_sends_negative_length(self) -> None:
        from tests.conftest import FakeFirmwareTransport

        transport = FakeFirmwareTransport()
        bundle = extract_images(build_updater_exe(resources={131: build_container(b"\xaa" * 50)}))
        async with FirmwareSession(transport=transport) as session:
            await session.run_update(bundle, only=["section1"], baudrate=9600)
        assert all(not c.startswith("L-") for c in transport.commands)
        assert transport.left == "Z97"

    async def test_postcommit_missing_checksum_is_not_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.conftest import FakeFirmwareTransport

        transport = FakeFirmwareTransport()
        bundle = small_bundle()
        async with FirmwareSession(transport=transport) as session:
            real = session.device_checksum

            async def missing(addr: int, length: int, timeout: float = 15.0) -> int | None:
                if addr == SECTION1_SLOTS["00"].address:
                    return None
                return await real(addr, length, timeout)

            monkeypatch.setattr(session, "device_checksum", missing)
            with pytest.raises(LumagenFirmwareError):
                await session.run_update(bundle, only=["section1"], baudrate=9600)
        assert transport.left != "Z97"

    async def test_postpromotion_missing_ack_and_checksum_is_not_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.conftest import FakeFirmwareTransport

        transport = FakeFirmwareTransport()
        bundle = small_bundle()
        async with FirmwareSession(transport=transport) as session:
            real = session.device_checksum

            async def missing(addr: int, length: int, timeout: float = 15.0) -> int | None:
                return None if session.promoted else await real(addr, length, timeout)

            async def nack(timeout: float = 30.0) -> bool:
                return False

            monkeypatch.setattr(session, "device_checksum", missing)
            monkeypatch.setattr(session, "await_promote_ack", nack)
            with pytest.raises(LumagenFirmwareError):
                await session.run_update(bundle, only=["section0"], baudrate=9600)
        assert transport.left != "Z97"

    async def test_cancelled_context_cleanup_survives_repeated_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware.session as module

        wire = Wire()
        monkeypatch.setattr(module, "LumagenTransport", lambda *_args, **_kw: wire)
        cleaning = asyncio.Event()
        release = asyncio.Event()
        session = FirmwareSession("synthetic://")

        async def hand_back(*, failed: bool = False) -> None:
            assert failed
            cleaning.set()
            await release.wait()

        monkeypatch.setattr(session, "hand_back", hand_back)

        async def update() -> None:
            async with session:
                raise asyncio.CancelledError

        task = asyncio.create_task(update())
        await cleaning.wait()
        task.cancel()
        await _REAL_SLEEP(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not wire.connected
        assert wire.disconnects == 1

    async def test_receive_flood_is_bounded_and_fails_closed(self) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        await session.connect()
        for _ in range(500):
            wire.callback(b"noise" * 4096)
        assert len(session._buffer) <= MAX_RX_BYTES
        with pytest.raises(LumagenFirmwareError, match="overflow"):
            await session.read_bytes(11)

    async def test_flush_timeout_bounds_a_hung_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        session = FirmwareSession(transport=transport, flush_timeout=0.01)
        await session.connect()
        ended = asyncio.Event()

        async def hung(**_kwargs: object) -> str:
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
            return "OK"

        monkeypatch.setattr(transport, "flush", hung)
        with pytest.raises(LumagenFirmwareAbortError, match="flushing"):
            await asyncio.wait_for(session.barrier(4096), timeout=0.25)
        assert ended.is_set()
        assert session.use_flush

    @pytest.mark.parametrize("reply", ["xEr", "xNO", "xO!", "xxOK"])
    async def test_erase_nacks_and_bad_framing_abort_immediately(self, reply: str) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        await session.connect()
        wire.on_write = lambda data: wire.callback(reply.encode()) if data == b"7958a7" else None
        with pytest.raises(LumagenFirmwareAbortError):
            await session.erase_run(SCRATCH_ADDR // SECTOR_SIZE, 1)

    async def test_postcommit_failure_is_not_reported_as_safe_abort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        async with FirmwareSession(transport=transport) as session:

            async def fail(*_args: object, **_kwargs: object) -> None:
                raise LumagenFirmwareAbortError("section0 failed")

            monkeypatch.setattr(session, "_write_section0", fail)
            with pytest.raises(LumagenFirmwareError, match="Do NOT power-cycle") as raised:
                await session.run_update(small_bundle(), baudrate=9600)
            assert not isinstance(raised.value, LumagenFirmwareAbortError)
        assert transport.left == "X"

    async def test_disconnect_after_commit_is_an_unconfirmed_firmware_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        async with FirmwareSession(transport=transport) as session:

            async def fail(*_args: object, **_kwargs: object) -> None:
                raise LumagenConnectionError("link disappeared")

            monkeypatch.setattr(session, "_write_section0", fail)
            with pytest.raises(LumagenFirmwareError, match="Do NOT power-cycle"):
                await session.run_update(small_bundle(), baudrate=9600)
        assert transport.left == "X"

    async def test_cancellation_inside_flush_finishes_the_payload_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = FakeFirmwareTransport()
        flushing = asyncio.Event()
        calls = 0

        async def flush(**_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                flushing.set()
                await asyncio.Event().wait()
            return "OK"

        monkeypatch.setattr(transport, "flush", flush)

        async def stage() -> None:
            async with FirmwareSession(transport=transport, chunk=127) as session:
                session.entered_update_mode = True
                await session.stage_image(b"\xaa" * BLOCK_SIZE, SCRATCH_ADDR)

        task = asyncio.create_task(stage())
        await flushing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport._expect == 0
        assert transport.left == "X"
        assert "Z97" not in transport.commands

    async def test_cancellation_between_erase_completion_bytes_keeps_framing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wire = Wire()
        session = FirmwareSession(transport=wire)
        wire.on_write = lambda data: wire.callback(b"xO") if data == b"7958a7" else None
        real_read = session.read_bytes
        owner: asyncio.Task[None] | None = None

        async def read(count: int, tries: int = 50, interval: float = 0.01) -> bytes:
            token = await real_read(count, tries, interval)
            if token == b"O":
                assert owner is not None
                owner.cancel()
                asyncio.get_running_loop().call_soon(wire.callback, b"K")
                return token
            await _REAL_SLEEP(0)
            return token

        monkeypatch.setattr(session, "read_bytes", read)

        async def erase() -> None:
            async with session:
                session.entered_update_mode = True
                await session.erase_run(SCRATCH_ADDR // SECTOR_SIZE, 1)

        owner = asyncio.create_task(erase())
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert session._erase_state is None
        assert wire.writes[-1] == b"X"

    @pytest.mark.parametrize("chunk", [BLOCK_SIZE, 127])
    @pytest.mark.parametrize("only", [("section0",), ("section1",), ("section0", "section1")])
    async def test_cancellation_at_every_write_boundary_has_no_leaks(
        self, only: tuple[str, ...], chunk: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiolumagen.firmware.session as module

        bundle = small_bundle()
        baseline = FakeFirmwareTransport()
        async with FirmwareSession(transport=baseline, chunk=chunk) as session:
            await session.run_update(bundle, only=only)
            boundaries = len(baseline.writes)

        existing_tasks = asyncio.all_tasks()

        async def cancel_at(boundary: int) -> None:
            transport = FakeFirmwareTransport()
            monkeypatch.setattr(module, "LumagenTransport", lambda *_a, **_k: transport)
            session = FirmwareSession("synthetic://", chunk=chunk)
            real_write = transport.write
            owner: asyncio.Task[None] | None = None
            in_body = True
            writes = 0

            async def interrupt(data: bytes) -> None:
                nonlocal writes
                await real_write(data)
                writes += 1
                if in_body and writes == boundary:
                    assert owner is not None
                    owner.cancel()
                    await _REAL_SLEEP(0)

            monkeypatch.setattr(transport, "write", interrupt)

            async def update() -> None:
                nonlocal in_body
                async with session:
                    try:
                        await session.run_update(bundle, only=only)
                    finally:
                        in_body = False

            owner = asyncio.create_task(update())
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert not transport.connected, (only, boundary)
            assert transport._expect == 0, (only, boundary, "payload counter stranded")
            assert transport.baudrate == 9600, (only, boundary)
            assert "Z97" not in transport.commands, (only, boundary)
            if "M0931" in transport.commands:
                assert transport.left == "X", (only, boundary, transport.commands[-4:])
            assert getattr(transport._on_data, "__self__", None) is not session
            assert asyncio.all_tasks() == existing_tasks, (only, boundary, "task leak")

        for boundary in range(1, boundaries + 1):
            await cancel_at(boundary)


class TestDeterministicFuzz:
    def test_random_non_pe_inputs_only_raise_domain_errors(self) -> None:
        rng = random.Random(90134)
        for _ in range(1000):
            data = rng.randbytes(rng.randrange(1024))
            with pytest.raises(LumagenFirmwareImageError):
                extract_images(data)

    def test_container_single_byte_damage_or_truncation_is_rejected(self) -> None:
        rng = random.Random(12961)
        raw = build_container(rng.randbytes(1024))
        for _ in range(200):
            broken = bytearray(raw)
            offset = rng.randrange(len(raw))
            broken[offset] ^= 1 << rng.randrange(8)
            with pytest.raises(LumagenFirmwareImageError):
                parse_container(bytes(broken))
        for length in range(len(raw)):
            with pytest.raises(LumagenFirmwareImageError):
                parse_container(raw[:length])

    def test_mutated_pe_metadata_never_leaks_struct_errors(self) -> None:
        rng = random.Random(42017)
        original = build_updater_exe()
        for _ in range(2000):
            mutated = bytearray(original)
            for _ in range(rng.randrange(1, 5)):
                offset = rng.randrange(512)
                mutated[offset] ^= 1 << rng.randrange(8)
            with suppress(LumagenFirmwareImageError):
                extract_images(bytes(mutated))

    def test_zip_corrupt_offsets_are_domain_errors(self) -> None:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("radiance_pro030326.exe", b"MZ" + b"A" * 1000)
        raw = out.getvalue()
        bad_offset = bytearray(raw)
        bad_offset[-8] ^= 4
        with pytest.raises(LumagenFirmwareImageError):
            extract_updater_zip(bytes(bad_offset))

        rng = random.Random(4792)
        for _ in range(5000):
            mutated = bytearray(raw)
            for _ in range(rng.randrange(1, 5)):
                mutated[rng.randrange(len(mutated))] ^= 1 << rng.randrange(8)
            with suppress(LumagenFirmwareImageError):
                extract_updater_zip(bytes(mutated))

    def test_zip_forged_entry_count_does_not_bypass_the_allocation_limit(self) -> None:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for index in range(1000):
                archive.writestr(f"{index}.txt", b"")
        forged = bytearray(out.getvalue())
        struct.pack_into("<HH", forged, len(forged) - 14, 1, 1)
        with pytest.raises(LumagenFirmwareImageError, match="too many directory"):
            extract_updater_zip(bytes(forged))
