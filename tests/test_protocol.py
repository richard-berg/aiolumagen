"""LumagenProtocol unit tests.

These exercise the parser against the exact response shapes documented in
``lumagen-research/Tip0011_RS232CommandInterface_111023.pdf``. When changing the
protocol code, update these first.
"""

from __future__ import annotations

import pytest

from aiolumagen.protocol import LumagenProtocol
from aiolumagen.state import (
    AutoAspectStatus,
    Colorspace,
    HdrStatus,
    InputStatus,
    LumagenState,
    SourceMode,
    SubtitleShift,
)


def _collect() -> tuple[list[tuple[LumagenState, tuple[str, ...]]], LumagenProtocol]:
    updates: list[tuple[LumagenState, tuple[str, ...]]] = []

    def on_update(state: LumagenState, codes: tuple[str, ...]) -> None:
        updates.append((state, codes))

    return updates, LumagenProtocol(on_update)


def test_s00_sets_alive_once() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!S00\r\n")
    assert len(updates) == 1
    state, codes = updates[0]
    assert state.alive is True
    assert codes == ("S00",)

    # Second !S00 with no state change should not fire a callback
    proto.feed_bytes(b"!S00\r\n")
    assert len(updates) == 1


def test_s02_power_on_then_off() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!S02,1\r\n")
    proto.feed_bytes(b"!S02,0\r\n")
    assert [u[0].power_on for u in updates] == [True, False]


def test_s01_device_info_splits_model_and_firmware() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!S01,RadiancePro,030225,1018,000000\r\n")
    state, _ = updates[-1]
    assert state.model == "RadiancePro"
    assert state.firmware == "030225"
    assert state.device_info_raw == "RadiancePro,030225,1018,000000"


@pytest.mark.parametrize(
    "damaged",
    [b"!S01,\r\n", b"!S01,RadiancePro,,,\r\n", b"!S01,,030225\r\n", b"!S01, , , , \r\n"],
)
def test_s01_with_empty_fields_keeps_the_known_identity(damaged: bytes) -> None:
    """A reply cut short or damaged must not blank the model or firmware."""
    _, proto = _collect()
    proto.feed_bytes(b"!S01,RadiancePro,030225,1018,003830\r\n")
    proto.feed_bytes(damaged)
    state = proto.state
    assert (state.model, state.firmware, state.model_number, state.serial) == (
        "RadiancePro",
        "030225",
        "1018",
        "003830",
    )


def test_s01_updates_fields_that_are_present() -> None:
    _, proto = _collect()
    proto.feed_bytes(b"!S01,RadiancePro,030225,1018,003830\r\n")
    proto.feed_bytes(b"!S01,RadiancePro,030326,1018,003830\r\n")
    assert proto.state.firmware == "030326"


def test_overlong_line_is_discarded_and_parsing_continues() -> None:
    """Regression: a 10 KB line of 9s arriving with its newline set every I25 field to 9."""
    _, proto = _collect()
    proto.feed_bytes(b"!S02,1\r\n")
    proto.feed_bytes(b"!I25," + b"9," * 5000 + b"\r\n")
    assert proto.state.current_input is None
    proto.feed_bytes(b"!S02,0\r\n")
    assert proto.state.power_on is False


def test_echo_prefix_is_tolerated() -> None:
    """Lumagen can echo the query command before the response on the same line."""
    updates, proto = _collect()
    proto.feed_bytes(b"ZQS01!S01,RadiancePro,030225,1018,000000\r\n")
    assert updates[-1][0].model == "RadiancePro"


def test_i00_input_and_memory() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!I00,03,B,01\r\n")
    state, _ = updates[-1]
    assert state.current_input == "03"
    assert state.input_memory == "B"


def test_i24_full_status_populates_all_fields() -> None:
    updates, proto = _collect()
    # 23 fields per the field-index map in protocol.py
    line = (
        "!I24,"
        "1,"  # [0]  input status = Active
        "060,"  # [1]  source vrate
        "2160,"  # [2]  source resolution
        "0,"  # [3]  D
        "0,"  # [4]  X
        "178,"  # [5]  source aspect = 1.78
        "185,"  # [6]  content aspect = 1.85
        "0,"  # [7]  Y
        "0,"  # [8]  T
        "3840,"  # [9]  WWWW
        "0,"  # [10] C
        "0,"  # [11] B
        "060,"  # [12] output vrate
        "2160,"  # [13] output resolution
        "000,"  # [14] ZZZ
        "3,"  # [15] colorspace = Rec.2100
        "1,"  # [16] HDR flag = HDR
        "p,"  # [17] source mode = progressive
        "0,"  # [18] H
        "05,"  # [19] virtual input = 5
        "00,"  # [20] KK
        "000,"  # [21] JJJ
        "000"  # [22] LLL
        "\r\n"
    )
    proto.feed_bytes(line.encode("ascii"))
    state, codes = updates[-1]
    assert codes == ("I24",)
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_vrate == "060"
    assert state.source_resolution == "2160"
    assert state.source_aspect == "178"
    assert state.content_aspect == "185"
    assert state.output_vrate == "060"
    assert state.output_resolution == "2160"
    assert state.colorspace is Colorspace.REC_2100
    assert state.is_hdr is True
    assert state.hdr_status is HdrStatus.HDR
    assert state.source_mode is SourceMode.PROGRESSIVE
    assert state.current_input == "05"  # I24 field [19] beats I00's field [0]


def test_i24_no_source_leaves_is_hdr_false() -> None:
    updates, proto = _collect()
    # Truncated I24 with only a handful of leading fields — mimics a
    # "no source" report.
    proto.feed_bytes(b"!I24,0,000,0000,0,0,000,000\r\n")
    state, _ = updates[-1]
    assert state.input_status is InputStatus.NO_SOURCE
    assert state.is_hdr is None  # not enough fields to populate HDR


def test_i21_i22_i23_use_shared_leading_fields() -> None:
    updates, proto = _collect()
    # I22 shape — same leading field order as I24
    proto.feed_bytes(b"!I22,1,060,1080,0,0,178,178,0,0,1920,0,0,060,1080\r\n")
    state, _ = updates[-1]
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_resolution == "1080"
    assert state.output_resolution == "1080"
    # I22 does not carry colorspace / HDR
    assert state.colorspace is None
    assert state.is_hdr is None


def test_buffered_partial_line_then_completion() -> None:
    """Inbound bytes may arrive fragmented. The parser must buffer partial lines."""
    updates, proto = _collect()
    proto.feed_bytes(b"!S02,")
    assert updates == []
    proto.feed_bytes(b"1\r\n")
    assert len(updates) == 1
    assert updates[-1][0].power_on is True


def test_two_responses_on_one_feed() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!S02,1\r\n!S00\r\n")
    assert len(updates) == 2
    assert updates[0][0].power_on is True
    assert updates[1][0].alive is True


def test_unknown_code_is_ignored() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!ZZZ,whatever\r\n")
    assert updates == []


def test_state_equality_enables_always_update_false() -> None:
    """Two fresh states must compare equal — prerequisite for coordinator skip."""
    assert LumagenState() == LumagenState()


def test_echo_only_line_without_bang_is_ignored() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"ZQS02\r\n")
    assert updates == []


def test_reset_discards_partial_line() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!S02,")
    proto.reset()
    proto.feed_bytes(b"1\r\n")
    # After reset, the remainder "1\r\n" has no ! and is ignored.
    assert updates == []


def test_i25_full_status_populates_v4_fields_plus_memory_and_power() -> None:
    """!I25 = v4 layout + 2 trailing fields (memory letter, power state).

    Payload taken from a real Full v5 capture against firmware 030225 in
    May 2026: input 1 active, 4K SDR source, output to 4K progressive.
    """
    updates, proto = _collect()
    line = "!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1\r\n"
    proto.feed_bytes(line.encode("ascii"))
    state, codes = updates[-1]
    assert codes == ("I25",)
    # v4-shared fields
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_vrate == "059"
    assert state.source_resolution == "2160"
    assert state.source_aspect == "178"
    assert state.content_aspect == "178"
    assert state.output_vrate == "059"
    assert state.output_resolution == "2160"
    assert state.colorspace is Colorspace.REC_2020  # E=2
    assert state.is_hdr is False  # F=0
    assert state.source_mode is SourceMode.PROGRESSIVE
    assert state.current_input == "01"  # II
    # v5-only fields
    assert state.input_memory == "A"
    assert state.power_on is True


def test_i25_power_off_reports_power_zero() -> None:
    """A real "after standby" capture — last field is 0."""
    updates, proto = _collect()
    line = "!I25,0,059,0000,0,0,178,178,-,0,000f,0,0,059,2160,237,2,0,n,P,01,01,178,178,A,0\r\n"
    proto.feed_bytes(line.encode("ascii"))
    state, _ = updates[-1]
    assert state.power_on is False
    # Memory letter still flows through even when powered off.
    assert state.input_memory == "A"


def test_i25_n_source_mode_recognized() -> None:
    """v5 firmware emits ``n`` for "no input"; older firmware used ``-``.

    Both must populate ``state.source_mode`` (no ValueError, no silent
    drop). The enum exposes them as distinct members so a consumer that
    cares about the wire byte can still see it, but most consumers will
    compare against either one.
    """
    updates, proto = _collect()
    line = "!I25,0,059,0000,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,n,P,01,01,178,178,A,1\r\n"
    proto.feed_bytes(line.encode("ascii"))
    state, _ = updates[-1]
    assert state.source_mode is SourceMode.NO_INPUT_V5
    assert state.source_mode == "n"


def test_i25_input_switch_updates_virtual_input_and_keeps_v5_fields() -> None:
    """Successive !I25s with different II values: v5 fields stay at fixed indices."""
    updates, proto = _collect()
    proto.feed_bytes(
        b"!I25,0,059,0000,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,n,P,03,03,178,178,A,1\r\n"
    )
    state, _ = updates[-1]
    assert state.current_input == "03"
    assert state.input_memory == "A"
    assert state.power_on is True

    proto.feed_bytes(
        b"!I25,0,059,0000,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,n,P,01,01,178,178,B,1\r\n"
    )
    state, _ = updates[-1]
    assert state.current_input == "01"
    assert state.input_memory == "B"  # tracked across the move
    assert state.power_on is True


def test_i25_truncated_payload_is_safe() -> None:
    """Partial !I25 (e.g. only the v4-shared prefix) doesn't crash the parser."""
    updates, proto = _collect()
    proto.feed_bytes(b"!I25,1,060,1080,0,0,178,178,0,0,1920,0,0,060,1080\r\n")
    state, _ = updates[-1]
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_resolution == "1080"
    # Trailing v5 fields not present — should remain at their priors.
    assert state.input_memory is None
    assert state.power_on is None


# ---------- Sharpness / Game mode / Auto aspect (Phase 1 expansions) ----------


def test_i30_sharpness_payload_decoded() -> None:
    """!I30 = ``Y4N`` → enabled=True, level=4, sensitivity=NORMAL."""
    from aiolumagen.state import SharpnessSensitivity

    updates, proto = _collect()
    proto.feed_bytes(b"!I30,Y4N\r\n")
    state, codes = updates[-1]
    assert codes == ("I30",)
    assert state.sharpness_enabled is True
    assert state.sharpness_level == 4
    assert state.sharpness_sensitivity is SharpnessSensitivity.NORMAL
    assert state.sharpness_raw == "Y4N"


def test_i30_sharpness_disabled() -> None:
    """!I30 = ``N0H`` → enabled=False, level=0, sensitivity=HIGH."""
    from aiolumagen.state import SharpnessSensitivity

    updates, proto = _collect()
    proto.feed_bytes(b"!I30,N0H\r\n")
    state, _ = updates[-1]
    assert state.sharpness_enabled is False
    assert state.sharpness_level == 0
    assert state.sharpness_sensitivity is SharpnessSensitivity.HIGH


def test_i30_sharpness_without_comma_delimiter() -> None:
    """``!I30Y4N`` (payload packed onto the code) must still decode.

    Regression: the line splitter only took a payload when byte 4 was a
    comma, so this framing parsed as "no payload" and left every sharpness
    field None — the entities read "unknown" forever even though the
    query was being sent and answered.
    """
    from aiolumagen.state import SharpnessSensitivity

    updates, proto = _collect()
    proto.feed_bytes(b"!I30Y4N\r\n")
    state, codes = updates[-1]
    assert codes == ("I30",)
    assert state.sharpness_enabled is True
    assert state.sharpness_level == 4
    assert state.sharpness_sensitivity is SharpnessSensitivity.NORMAL


def test_i30_sharpness_comma_separated_fields() -> None:
    """``!I30,Y,4,H`` (one field per value) must decode identically."""
    from aiolumagen.state import SharpnessSensitivity

    updates, proto = _collect()
    proto.feed_bytes(b"!I30,Y,4,H\r\n")
    state, _ = updates[-1]
    assert state.sharpness_enabled is True
    assert state.sharpness_level == 4
    assert state.sharpness_sensitivity is SharpnessSensitivity.HIGH


def test_i30_unparseable_payload_leaves_fields_none() -> None:
    """Garbage must not fabricate values, but must still keep the raw bytes."""
    updates, proto = _collect()
    proto.feed_bytes(b"!I30,????\r\n")
    state, _ = updates[-1]
    assert state.sharpness_enabled is None
    assert state.sharpness_level is None
    assert state.sharpness_sensitivity is None
    assert state.sharpness_raw == "????"


def test_i53_game_mode_on_off() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!I53,1\r\n")
    assert updates[-1][0].game_mode is True
    proto.feed_bytes(b"!I53,0\r\n")
    assert updates[-1][0].game_mode is False


def test_i54_auto_aspect_on_off() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!I54,1\r\n")
    assert updates[-1][0].auto_aspect is True
    proto.feed_bytes(b"!I54,0\r\n")
    assert updates[-1][0].auto_aspect is False


# ---------- HDR (Phase 2) ----------


def test_i50_display_rec2020_yes() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!I50,Y\r\n")
    state, codes = updates[-1]
    assert codes == ("I50",)
    assert state.display_supports_rec2020 is True


def test_i50_display_rec2020_no() -> None:
    updates, proto = _collect()
    proto.feed_bytes(b"!I50,N\r\n")
    assert updates[-1][0].display_supports_rec2020 is False


def test_i52_sdr_source_clears_mastering_metadata_and_keeps_none() -> None:
    """SDR source — V=0 with placeholder zeros; mastering fields stay None."""
    updates, proto = _collect()
    # Per the doc, SDR sources are reported as "V=0" with placeholder
    # zeros for the rest of the fields.
    proto.feed_bytes(b"!I52,0,.0000,0,0\r\n")
    state, codes = updates[-1]
    assert codes == ("I52",)
    assert state.is_hdr is False
    assert state.hdr_status is HdrStatus.SDR
    assert state.hdr_source_min_luminance is None
    assert state.hdr_source_max_luminance is None
    assert state.hdr_source_max_cll is None


def test_i52_hdr_source_populates_mastering_metadata() -> None:
    """HDR source — V=1 with real Min/Max/Cll values."""
    updates, proto = _collect()
    # Typical 1000-nit HDR10 title with a 4000-nit master.
    proto.feed_bytes(b"!I52,1,.0050,1000,4000\r\n")
    state, _ = updates[-1]
    assert state.is_hdr is True
    assert state.hdr_status is HdrStatus.HDR
    assert state.hdr_source_min_luminance == pytest.approx(0.0050)
    assert state.hdr_source_max_luminance == 1000
    assert state.hdr_source_max_cll == 4000


def test_i52_transition_hdr_to_sdr_clears_stale_metadata() -> None:
    """When the source flips HDR -> SDR, mastering metadata must clear.

    Without this, a pre-HDR-content sensor reading would persist
    indefinitely after switching to an SDR source, misrepresenting
    the current source.
    """
    updates, proto = _collect()
    proto.feed_bytes(b"!I52,1,.0050,1000,4000\r\n")
    proto.feed_bytes(b"!I52,0,.0000,0,0\r\n")
    state, _ = updates[-1]
    assert state.is_hdr is False
    assert state.hdr_source_min_luminance is None
    assert state.hdr_source_max_luminance is None
    assert state.hdr_source_max_cll is None


def test_i52_malformed_max_field_is_tolerated() -> None:
    """Garbage in a field doesn't poison the rest of the state."""
    updates, proto = _collect()
    proto.feed_bytes(b"!I52,1,.0050,not-a-number,4000\r\n")
    state, _ = updates[-1]
    assert state.is_hdr is True
    assert state.hdr_source_min_luminance == pytest.approx(0.0050)
    assert state.hdr_source_max_luminance is None  # malformed, dropped
    assert state.hdr_source_max_cll == 4000  # following field still parses


def test_i52_truncated_payload_only_populates_what_arrives() -> None:
    """Old firmware with a shorter !I52 — populate what we can read."""
    updates, proto = _collect()
    proto.feed_bytes(b"!I52,1,.0050\r\n")
    state, _ = updates[-1]
    assert state.is_hdr is True
    assert state.hdr_source_min_luminance == pytest.approx(0.0050)
    assert state.hdr_source_max_luminance is None
    assert state.hdr_source_max_cll is None


# ---------- Input labels (ZQS1XY / !S1x) ----------


def test_input_label_correlates_to_pending_input() -> None:
    """A primed input number attributes the next !S1x response to that input."""
    updates, proto = _collect()
    proto.expect_input_label(6)
    proto.feed_bytes(b"!S1B,Apple TV\r\n")
    state, codes = updates[-1]
    assert codes == ("S1B",)
    assert state.input_labels == {6: "Apple TV"}
    assert proto.pending_label_input is None  # consumed


def test_input_label_doc_example() -> None:
    """Tip0011 example: ZQS1B5 -> !S1B,Input 6B (memory B, input 6)."""
    updates, proto = _collect()
    proto.expect_input_label(6)
    proto.feed_bytes(b"!S1B,Input 6B\r\n")
    assert updates[-1][0].input_labels == {6: "Input 6B"}


def test_input_label_without_pending_is_ignored() -> None:
    """An !S1x with no primed input can't be correlated, so it's dropped."""
    updates, proto = _collect()
    proto.feed_bytes(b"!S1A,Orphan\r\n")
    assert updates == []
    assert proto.state.input_labels == {}


def test_input_label_tolerates_echo_prefix() -> None:
    """Echo-on firmware puts the query before the response on one line."""
    updates, proto = _collect()
    proto.expect_input_label(1)
    proto.feed_bytes(b"ZQS1A0!S1A,Roku\r\n")
    assert updates[-1][0].input_labels == {1: "Roku"}


def test_input_label_preserves_commas() -> None:
    """The label is everything after the code — embedded commas are kept."""
    updates, proto = _collect()
    proto.expect_input_label(3)
    proto.feed_bytes(b"!S1A,Foo, Bar\r\n")
    assert updates[-1][0].input_labels == {3: "Foo, Bar"}


def test_input_labels_accumulate_across_queries() -> None:
    """Sequential prime+feed builds the map without dropping earlier entries."""
    updates, proto = _collect()
    proto.expect_input_label(1)
    proto.feed_bytes(b"!S1A,Apple TV\r\n")
    proto.expect_input_label(2)
    proto.feed_bytes(b"!S1A,Roku\r\n")
    assert updates[-1][0].input_labels == {1: "Apple TV", 2: "Roku"}


def test_input_label_update_does_not_mutate_prior_snapshot() -> None:
    """Each label update creates a fresh dict, so earlier snapshots stay stable."""
    updates, proto = _collect()
    proto.expect_input_label(1)
    proto.feed_bytes(b"!S1A,Apple TV\r\n")
    first_snapshot = updates[-1][0].input_labels
    proto.expect_input_label(2)
    proto.feed_bytes(b"!S1A,Roku\r\n")
    # The earlier callback's dict must not have gained input 2.
    assert first_snapshot == {1: "Apple TV"}


def test_reset_clears_pending_label_input() -> None:
    _updates, proto = _collect()
    proto.expect_input_label(4)
    proto.reset()
    assert proto.pending_label_input is None
    proto.feed_bytes(b"!S1A,Late\r\n")
    assert proto.state.input_labels == {}


# ---------- Extended !I24 / !I25 fields (previously parsed and discarded) ----------

# The same real Full v5 capture used above: firmware 030225, input 1 active,
# 4K SDR source, 4K progressive output. 25 payload fields (0-24) — note it
# ends at power, with nothing at 25/26.
I25_REAL_CAPTURE = (
    b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1\r\n"
)


def test_i25_populates_extended_source_fields() -> None:
    """Documented ZQI24 source fields that used to be dropped on the floor."""
    updates, proto = _collect()
    proto.feed_bytes(I25_REAL_CAPTURE)
    state, _ = updates[-1]
    assert state.source_3d_mode == "0"  # [3] D
    assert state.input_config == "0"  # [4] X
    assert state.nls_active is False  # [7] Y = '-' (normal)
    assert state.physical_input == "01"  # [20] KK
    assert state.detected_source_aspect == "178"  # [21] JJJ
    assert state.detected_content_aspect == "178"  # [22] LLL


def test_i25_populates_extended_output_fields() -> None:
    updates, proto = _collect()
    proto.feed_bytes(I25_REAL_CAPTURE)
    state, _ = updates[-1]
    assert state.output_3d_mode == "0"  # [8] T
    assert state.output_enabled_mask == 0x000E  # [9] WWWW
    assert state.active_outputs == (2, 3, 4)  # decoded from the mask
    assert state.output_cms == 0  # [10] C
    assert state.output_style == 0  # [11] B
    assert state.output_aspect == "237"  # [14] ZZZ
    # [18] H arrives uppercase ('P') where source mode is lowercase; both
    # share SourceMode rather than introducing a second enum.
    assert state.output_scan_mode is SourceMode.PROGRESSIVE


def test_i25_nls_engaged_reads_as_true() -> None:
    """Y = 'N' means NLS is engaged — the letter is not a boolean 'no'."""
    updates, proto = _collect()
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,N,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1\r\n"
    )
    assert updates[-1][0].nls_active is True


def test_i25_derives_refresh_rates_and_widths() -> None:
    """Derived values come free off the same line — no extra query.

    The Lumagen never reports horizontal resolution, and reports rate as a
    truncated integer, so both have to be computed. See
    :mod:`aiolumagen.formatting`.
    """
    updates, proto = _collect()
    proto.feed_bytes(I25_REAL_CAPTURE)
    state, _ = updates[-1]
    assert state.source_refresh_hz == pytest.approx(59.94)
    assert state.output_refresh_hz == pytest.approx(59.94)
    # 2160 x 1.78 = 3844.8, snapped to the standard 3840.
    assert state.source_width == 3840
    # 2160 x 2.37 = 5119.2 — too far from any standard width to snap, so the
    # computed value is kept rather than forced to 4096.
    assert state.output_width == 5119


def test_i25_real_capture_has_no_subtitle_or_auto_aspect_fields() -> None:
    """Firmware 030225 stops at power (index 24), so 25/26 stay None.

    Pinning this keeps the empirical mapping honest: the fields are read behind
    a length guard precisely because this capture proves they aren't universal.
    """
    updates, proto = _collect()
    proto.feed_bytes(I25_REAL_CAPTURE)
    state, _ = updates[-1]
    assert state.subtitle_shift is None
    assert state.auto_aspect_status is None
    # ...while the fields that ARE present still decode.
    assert state.power_on is True
    assert state.input_memory == "A"


def test_i25_with_trailing_fields_decodes_subtitle_and_auto_aspect() -> None:
    """Payload 25 = subtitle shift, 26 = auto aspect status (empirical).

    These indices are a hypothesis, not documented behaviour: absent from
    Tip0011 (which predates Full v5) and absent from every capture in this
    repo. The test pins the mapping we implement, so if hardware ever
    contradicts it this is the one place to correct.
    """
    from aiolumagen.state import AutoAspectStatus, SubtitleShift

    updates, proto = _collect()
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,2,2\r\n"
    )
    state, _ = updates[-1]
    assert state.subtitle_shift is SubtitleShift.PERCENT_6  # '2' = 6%
    assert state.auto_aspect_status is AutoAspectStatus.ON  # '2' = on


def test_i25_auto_aspect_status_distinguishes_off_from_disabled() -> None:
    """The tri-state is the reason to carry this field at all.

    ZQI54's boolean can't express "configured but currently inhibited".
    """
    from aiolumagen.state import AutoAspectStatus

    updates, proto = _collect()
    base = (
        "!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,0,{}\r\n"
    )
    proto.feed_bytes(base.format("0").encode("ascii"))
    assert updates[-1][0].auto_aspect_status is AutoAspectStatus.OFF
    proto.feed_bytes(base.format("1").encode("ascii"))
    assert updates[-1][0].auto_aspect_status is AutoAspectStatus.DISABLED
    proto.feed_bytes(base.format("2").encode("ascii"))
    assert updates[-1][0].auto_aspect_status is AutoAspectStatus.ON


def test_i25_index_26_feeds_the_auto_aspect_boolean() -> None:
    """Index 26 drives ``auto_aspect``, so a change lands without a ZQI54 poll.

    Replaces an earlier test that asserted the opposite. Index 26 was kept away
    from the boolean while its mapping was a guess; it has since been confirmed
    on hardware against three independent ways of switching auto aspect off
    (serial ``V``, the OSD menu, and subtitle-shift inhibition), each agreeing
    with ZQI54. The device also pushes !I25 on every auto-aspect change, so the
    push is both sufficient and faster than polling.
    """
    from aiolumagen.state import AutoAspectStatus

    updates, proto = _collect()
    base = (
        "!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,0,{}\r\n"
    )

    proto.feed_bytes(base.format("2").encode("ascii"))
    state = updates[-1][0]
    assert state.auto_aspect_status is AutoAspectStatus.ON
    assert state.auto_aspect is True

    # DISABLED must read as False. Measured: a deliberate off and an inhibited
    # on both report DISABLED, so presenting it as enabled would show auto
    # aspect on immediately after the user switched it off.
    proto.feed_bytes(base.format("1").encode("ascii"))
    state = updates[-1][0]
    assert state.auto_aspect_status is AutoAspectStatus.DISABLED
    assert state.auto_aspect is False

    proto.feed_bytes(base.format("0").encode("ascii"))
    state = updates[-1][0]
    assert state.auto_aspect_status is AutoAspectStatus.OFF
    assert state.auto_aspect is False


def test_i25_without_index_26_leaves_zqi54_boolean_intact() -> None:
    """Firmware that stops the payload early must not have its value wiped.

    The 030225 capture ends at index 24. On such firmware ZQI54 is still polled,
    and a shorter push must leave what it established alone.
    """
    updates, proto = _collect()
    proto.feed_bytes(b"!I54,1\r\n")
    assert updates[-1][0].auto_aspect is True

    # v4-length payload: no index 25 or 26.
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1\r\n"
    )
    state = updates[-1][0]
    assert state.auto_aspect is True
    assert state.auto_aspect_status is None


def test_i25_malformed_extended_fields_do_not_erase_good_state() -> None:
    """A garbled line degrades to "field unknown", never wipes a prior value."""
    updates, proto = _collect()
    proto.feed_bytes(I25_REAL_CAPTURE)
    assert updates[-1][0].output_cms == 0
    assert updates[-1][0].output_enabled_mask == 0x000E

    # Same line with junk in the CMS and mask fields.
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,-,0,zzzz,?,0,059,2160,237,2,0,p,P,01,01,178,178,A,1\r\n"
    )
    state, _ = updates[-1]
    assert state.output_cms == 0  # kept
    assert state.output_enabled_mask == 0x000E  # kept


def test_extended_fields_are_not_applied_to_i21() -> None:
    """!I21 stays on the minimal shared field set, on purpose.

    Tip0011's !I21 signature omits the T and WWWW fields that its own field
    list immediately below it defines, so it's ambiguous whether everything
    from index 8 on is shifted by two. Reading extended fields there would be
    a guess; leaving them None is not.
    """
    updates, proto = _collect()
    proto.feed_bytes(b"!I21,1,060,1080,0,0,178,178,-,0,1920,0,0,060,1080\r\n")
    state, _ = updates[-1]
    # Shared leading fields still parse.
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_resolution == "1080"
    # Extended fields deliberately untouched.
    assert state.nls_active is None
    assert state.output_enabled_mask is None
    assert state.output_cms is None
    assert state.physical_input is None


def test_i24_also_gets_extended_fields() -> None:
    """The extended set applies to the whole I24/I25 path, not just v5."""
    updates, proto = _collect()
    proto.feed_bytes(
        b"!I24,1,059,2160,0,0,178,178,N,0,0003,2,1,059,2160,178,2,0,p,P,01,02,178,178\r\n"
    )
    state, _ = updates[-1]
    assert state.nls_active is True
    assert state.active_outputs == (1, 2)
    assert state.output_cms == 2
    assert state.output_style == 1
    assert state.physical_input == "02"
    assert state.source_width == 3840


# ---------- Response observers (correlation hook) ----------


def test_response_observer_fires_for_each_line() -> None:
    _updates, proto = _collect()
    seen: list[tuple[str, str]] = []
    proto.add_response_observer(lambda code, payload: seen.append((code, payload)))
    proto.feed_bytes(b"!S02,1\r\n!S00\r\n")
    assert seen == [("S02", "1"), ("S00", "")]


def test_response_observer_fires_when_state_is_unchanged() -> None:
    """Broader than the state-update callback, which dedupes no-op responses.

    This is what makes the observer usable for correlation: a repeated poll
    still resolves a waiter even though no listener is woken.
    """
    updates, proto = _collect()
    seen: list[str] = []
    proto.add_response_observer(lambda code, _payload: seen.append(code))
    proto.feed_bytes(b"!S02,1\r\n")
    proto.feed_bytes(b"!S02,1\r\n")  # identical — no state change
    assert len(updates) == 1  # state listener suppressed
    assert seen == ["S02", "S02"]  # observer not suppressed


def test_response_observer_fires_for_unhandled_codes() -> None:
    """A caller can await a code the parser has no handler for."""
    updates, proto = _collect()
    seen: list[tuple[str, str]] = []
    proto.add_response_observer(lambda code, payload: seen.append((code, payload)))
    proto.feed_bytes(b"!I99,whatever\r\n")
    assert updates == []  # no state change
    assert seen == [("I99", "whatever")]


def test_response_observer_sees_committed_state() -> None:
    """Observers run after the new state is swapped in, so reads are fresh."""
    _updates, proto = _collect()
    observed: list[bool | None] = []
    proto.add_response_observer(lambda _code, _payload: observed.append(proto.state.power_on))
    proto.feed_bytes(b"!S02,1\r\n")
    assert observed == [True]


def test_response_observer_unregister() -> None:
    _updates, proto = _collect()
    seen: list[str] = []
    unregister = proto.add_response_observer(lambda code, _payload: seen.append(code))
    proto.feed_bytes(b"!S02,1\r\n")
    unregister()
    proto.feed_bytes(b"!S02,0\r\n")
    assert seen == ["S02"]


def test_raising_response_observer_is_dropped_not_propagated() -> None:
    """A bad observer must not break parsing for everyone else."""
    _updates, proto = _collect()
    seen: list[str] = []

    def _boom(_code: str, _payload: str) -> None:
        raise RuntimeError("observer is broken")

    proto.add_response_observer(_boom)
    proto.add_response_observer(lambda code, _payload: seen.append(code))
    proto.feed_bytes(b"!S02,1\r\n")
    proto.feed_bytes(b"!S02,0\r\n")
    # The healthy observer keeps receiving; the broken one was dropped after
    # its first raise.
    assert seen == ["S02", "S02"]


def test_reset_keeps_response_observers() -> None:
    """The client registers its correlation observer once and expects it to
    survive a reconnect, which calls reset()."""
    _updates, proto = _collect()
    seen: list[str] = []
    proto.add_response_observer(lambda code, _payload: seen.append(code))
    proto.reset()
    proto.feed_bytes(b"!S02,1\r\n")
    assert seen == ["S02"]


# --- !O01 output mode / authoritative output width -------------------------
#
# Payloads below are real captures from a Radiance Pro 4242 on firmware 030326,
# feeding a JVC projector at 4096x2160 through an anamorphic lens. That setup is
# what exposed the bug these tests pin: the output aspect is the *image* aspect
# (2.37), not the raster aspect, so deriving width from it is unsound.


def test_o01_reports_authoritative_output_width() -> None:
    """!O01 field 1 is the real output width and must land in state."""
    updates, proto = _collect()
    proto.feed_bytes(b"!O01,5994,4096,2160,0,0\r\n")
    state = updates[-1][0]
    assert state.output_width_reported == 4096
    assert state.output_height_reported == 2160
    assert state.output_width == 4096
    assert state.output_mode_raw == "5994,4096,2160,0,0"


def test_o01_width_overrides_aspect_derived_width() -> None:
    """The anamorphic regression: 2160 x 2.37 = 5119, but the output is 4096.

    !I25 arrives first and can only infer width from the output aspect, which
    for a scaled output is the image aspect rather than the raster. !O01 then
    supplies the truth and must win — and must keep winning when the next status
    push re-runs the derivation.
    """
    updates, proto = _collect()

    # Full v5 push: output height 2160 (idx 13), output aspect 237 (idx 14).
    i25 = b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,0,2\r\n"
    proto.feed_bytes(i25)
    derived = updates[-1][0].output_width
    assert derived == 5119, "aspect-derived width should be the wrong 5119 here"

    # !O01 corrects it.
    proto.feed_bytes(b"!O01,5994,4096,2160,0,0\r\n")
    assert updates[-1][0].output_width == 4096

    # A later status push must NOT reintroduce the derived value.
    proto.feed_bytes(i25)
    assert updates[-1][0].output_width == 4096
    assert updates[-1][0].output_width_reported == 4096


def test_o01_falls_back_to_derivation_when_unseen() -> None:
    """Without !O01 the aspect-derived width is still better than nothing."""
    updates, proto = _collect()
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,178,2,0,p,P,01,01,178,178,A,1,0,2\r\n"
    )
    state = updates[-1][0]
    assert state.output_width_reported is None
    # 2160 x 1.78 = 3844.8, snapped to the standard 3840.
    assert state.output_width == 3840


def test_o01_zero_geometry_is_treated_as_no_signal() -> None:
    """A 0 width/height is the device's placeholder, not a real raster."""
    updates, proto = _collect()
    proto.feed_bytes(b"!O01,0,0,0,0,0\r\n")
    state = updates[-1][0]
    assert state.output_width_reported is None
    assert state.output_height_reported is None


def test_o01_short_or_malformed_payload_does_not_raise() -> None:
    """Truncated and non-numeric payloads leave the fields unset."""
    for payload in (b"!O01\r\n", b"!O01,5994\r\n", b"!O01,5994,abc,def,0,0\r\n"):
        updates, proto = _collect()
        proto.feed_bytes(payload)
        state = updates[-1][0] if updates else LumagenState()
        assert state.output_width_reported is None


def test_o01_does_not_touch_rate_fields() -> None:
    """!O01 carries a rate too, but ownership stays with the !I25 push.

    Two writers for one value is how precedence bugs start; O01's rate encoding
    (5994) differs from the push's (059) and nothing needs it.
    """
    updates, proto = _collect()
    proto.feed_bytes(
        b"!I25,1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,0,2\r\n"
    )
    before = updates[-1][0]
    proto.feed_bytes(b"!O01,2398,4096,2160,0,0\r\n")
    after = updates[-1][0]
    assert after.output_vrate == before.output_vrate
    assert after.output_refresh_hz == before.output_refresh_hz


# ---------- Status-line validation ----------
#
# Regression: !I24/!I25 fields were accepted one by one with no check that
# they could be what the field holds. The fault proxy's
# "!I25,zz,yy,xx,ww,..." set source_resolution to "xx" and source_vrate to
# "yy", and Home Assistant showed them.

# Firmware 030326: the v4 layout, memory and power, then subtitle shift and
# auto aspect. Every field but power differs from the baseline below, so a
# field that was applied (or kept) is visible. Power is on in both, because the
# old parser read any unknown power value as off.
_V5_LINE = "1,059,2160,0,0,178,178,-,0,000e,0,0,059,2160,237,2,0,p,P,01,01,178,178,A,1,0,2"
_V5_FIELDS = tuple(_V5_LINE.split(","))
_BASELINE = "0,023,1080,2,3,133,240,N,4,0003,5,6,050,1080,178,1,1,i,I,07,08,185,235,C,1,1,0"


def _baseline_protocol() -> tuple[list[tuple[LumagenState, tuple[str, ...]]], LumagenProtocol]:
    updates, proto = _collect()
    proto.feed_bytes(f"!I25,{_BASELINE}\r\n".encode())
    assert updates[-1][0].full_status_raw == _BASELINE
    return updates, proto


def _line(fields: list[str] | tuple[str, ...]) -> bytes:
    return ("!I25," + ",".join(fields) + "\r\n").encode()


def _with(index: int, value: str) -> list[str]:
    fields = list(_V5_FIELDS)
    fields[index] = value
    return fields


def test_the_fault_proxys_garbage_status_line_changes_nothing() -> None:
    updates, proto = _baseline_protocol()
    before = proto.state
    seen: list[tuple[str, str]] = []
    proto.add_response_observer(lambda code, payload: seen.append((code, payload)))
    proto.feed_bytes(b"!I25,zz,yy,xx,ww,vv,uu,tt,ss,rr,qq,pp,oo,nn,mm\r\n")
    assert proto.state == before
    assert proto.state.source_resolution == "1080"
    assert proto.state.source_vrate == "023"
    assert len(updates) == 1
    # Still delivered to a waiter: the line arrived, it just isn't believed.
    assert seen == [("I25", "zz,yy,xx,ww,vv,uu,tt,ss,rr,qq,pp,oo,nn,mm")]


# (index, value of the wrong shape) for every documented field kind.
_MALFORMED = [
    pytest.param(0, "x", id="M-not-a-digit"),
    pytest.param(1, "yy", id="RRR-not-digits"),
    pytest.param(1, "59", id="RRR-too-short"),
    pytest.param(2, "xx", id="VVVV-not-digits"),
    pytest.param(2, "21600", id="VVVV-too-long"),
    pytest.param(3, "a", id="D-not-a-digit"),
    pytest.param(4, "!", id="X-not-a-digit"),
    pytest.param(5, "1.78", id="AAA-not-digits"),
    pytest.param(6, "", id="SSS-empty"),
    pytest.param(7, "--", id="Y-two-characters"),
    pytest.param(8, "T", id="T-not-a-digit"),
    pytest.param(9, "zzzz", id="WWWW-not-hex"),
    pytest.param(9, "00e", id="WWWW-too-short"),
    pytest.param(10, "?", id="C-not-a-digit"),
    pytest.param(11, "10", id="B-two-digits"),
    pytest.param(12, "o59", id="PPP-not-digits"),
    pytest.param(13, "216", id="QQQQ-too-short"),
    pytest.param(14, "2.37", id="ZZZ-not-digits"),
    pytest.param(15, "E", id="E-not-a-digit"),
    pytest.param(16, "yes", id="F-not-a-digit"),
    pytest.param(17, "", id="G-empty"),
    pytest.param(18, "PP", id="H-two-characters"),
    pytest.param(19, "1", id="II-too-short"),
    pytest.param(20, "o1", id="KK-not-digits"),
    pytest.param(21, "17B", id="JJJ-not-digits"),
    pytest.param(22, "1780", id="LLL-too-long"),
    pytest.param(23, "AB", id="memory-two-letters"),
    pytest.param(24, "on", id="power-not-a-digit"),
]


@pytest.mark.parametrize(("index", "value"), _MALFORMED)
def test_a_field_of_the_wrong_shape_rejects_the_whole_line(index: int, value: str) -> None:
    """Nothing from a damaged line is applied, not even its good-looking fields.

    A lost comma shifts every later field into a slot it may well fit, so a
    field that can't be right is reason to distrust the rest of the line.
    """
    updates, proto = _baseline_protocol()
    before = proto.state
    proto.feed_bytes(_line(_with(index, value)))
    assert proto.state == before
    assert len(updates) == 1


def test_a_line_cut_off_mid_field_is_rejected() -> None:
    _updates, proto = _baseline_protocol()
    before = proto.state
    proto.feed_bytes(b"!I25,1,059,21\r\n")
    assert proto.state == before


def test_a_line_cut_off_at_a_comma_applies_what_it_has() -> None:
    _updates, proto = _baseline_protocol()
    proto.feed_bytes(_line(_V5_FIELDS[:3]))
    assert proto.state.input_status is InputStatus.ACTIVE
    assert proto.state.source_resolution == "2160"
    assert proto.state.colorspace is Colorspace.REC_709  # baseline, not reached


# (index, value of the right shape but outside the field's vocabulary,
#  attribute, the baseline value it must keep)
_OUT_OF_RANGE = [
    pytest.param(0, "7", "input_status", InputStatus.NO_SOURCE, id="M"),
    pytest.param(3, "3", "source_3d_mode", "2", id="D"),
    pytest.param(7, "Q", "nls_active", True, id="Y"),
    pytest.param(8, "5", "output_3d_mode", "4", id="T"),
    pytest.param(10, "8", "output_cms", 5, id="C"),
    pytest.param(11, "9", "output_style", 6, id="B"),
    pytest.param(15, "7", "colorspace", Colorspace.REC_709, id="E"),
    pytest.param(16, "2", "hdr_status", HdrStatus.HDR, id="F"),
    pytest.param(17, "x", "source_mode", SourceMode.INTERLACED, id="G"),
    pytest.param(18, "X", "output_scan_mode", SourceMode.INTERLACED, id="H"),
    pytest.param(19, "00", "current_input", "07", id="II-zero"),
    pytest.param(19, "20", "current_input", "07", id="II-above-19"),
    pytest.param(20, "99", "physical_input", "08", id="KK"),
    pytest.param(23, "E", "input_memory", "C", id="memory"),
    pytest.param(24, "7", "power_on", True, id="power"),
    pytest.param(25, "9", "subtitle_shift", SubtitleShift.PERCENT_3, id="subtitle-shift"),
    pytest.param(26, "x", "auto_aspect_status", AutoAspectStatus.OFF, id="auto-aspect"),
]


@pytest.mark.parametrize(("index", "value", "attribute", "kept"), _OUT_OF_RANGE)
def test_an_unknown_value_keeps_that_field_and_applies_the_rest(
    index: int, value: str, attribute: str, kept: object
) -> None:
    """In shape but outside the known vocabulary: most likely newer firmware.

    Rejecting the line for it would freeze every signal sensor on a firmware
    that added one value, so only that field keeps its prior value.
    """
    _updates, proto = _baseline_protocol()
    proto.feed_bytes(_line(_with(index, value)))
    assert getattr(proto.state, attribute) == kept
    # The rest of the line landed.
    assert proto.state.source_resolution == "2160"
    assert proto.state.full_status_raw == ",".join(_with(index, value))


def test_a_valid_line_with_extra_trailing_fields_parses_fully() -> None:
    """Newer firmware appending fields must not cost any that we know."""
    _updates, proto = _baseline_protocol()
    proto.feed_bytes(_line([*_V5_FIELDS, "9", "ABC", "x.y"]))
    state = proto.state
    assert state.input_status is InputStatus.ACTIVE
    assert state.source_vrate == "059"
    assert state.source_resolution == "2160"
    assert state.source_3d_mode == "0"
    assert state.input_config == "0"
    assert state.source_aspect == "178"
    assert state.content_aspect == "178"
    assert state.nls_active is False
    assert state.output_3d_mode == "0"
    assert state.active_outputs == (2, 3, 4)
    assert state.output_cms == 0
    assert state.output_style == 0
    assert state.output_vrate == "059"
    assert state.output_resolution == "2160"
    assert state.output_aspect == "237"
    assert state.colorspace is Colorspace.REC_2020
    assert state.hdr_status is HdrStatus.SDR
    assert state.source_mode is SourceMode.PROGRESSIVE
    assert state.output_scan_mode is SourceMode.PROGRESSIVE
    assert state.current_input == "01"
    assert state.physical_input == "01"
    assert state.detected_source_aspect == "178"
    assert state.detected_content_aspect == "178"
    assert state.input_memory == "A"
    assert state.power_on is True
    assert state.subtitle_shift is SubtitleShift.OFF
    assert state.auto_aspect_status is AutoAspectStatus.ON


def test_i24_lines_are_validated_too() -> None:
    _updates, proto = _collect()
    proto.feed_bytes(b"!I24," + ",".join(_V5_FIELDS[:23]).encode() + b"\r\n")
    before = proto.state
    assert before.source_resolution == "2160"
    proto.feed_bytes(b"!I24,1,059,xxxx," + ",".join(_V5_FIELDS[3:23]).encode() + b"\r\n")
    assert proto.state == before
