"""Decide which firmware sections actually need writing.

Pure sync, no I/O: :func:`plan_update` takes the extracted images plus a snapshot
of what the device reported and returns a plan the caller can show a user
*before* anything is committed. That separation is the point — an update this
consequential should be reviewable, and a plan that can be computed without
touching the device can also be tested without one.

**The vendor's own gate is deliberately not replicated.** The updater decides by
comparing a hardcoded date at a fixed ``.text`` address, which would have to be
re-derived for every release and silently mis-fires when it moves. We have
something strictly better available: the device can checksum its own flash, so we
compare actual bytes and need no version knowledge at all.

Two traps this module exists to avoid, both of which produce plausible-looking
wrong answers rather than errors:

* **A ``MMDDYY`` revision compared as an integer inverts for eight months a
  year.** Handled structurally by
  :class:`~aiolumagen.firmware.protocol.FirmwareRevision`.
* **A byte-perfect section-1 slot never matches the image's raw checksum**,
  because the bootloader stamps four bytes at ``+4`` on commit. Comparing against
  the raw sum therefore reports a mismatch on correct firmware, which makes the
  "is an update needed?" test answer *yes, always* — turning a one-minute update
  into a seven-minute one that rewrites a live slot for nothing. Handled by
  :func:`~aiolumagen.firmware.container.expected_stored_checksum`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from aiolumagen.exceptions import LumagenFirmwareImageError
from aiolumagen.firmware.container import (
    HEADER_LEN,
    ContainerHeader,
    expected_stored_checksum,
    parse_container,
)
from aiolumagen.firmware.extract import (
    CHIP_NAMES,
    SECTION0,
    SECTION1,
    FirmwareBundle,
    FirmwareImage,
)
from aiolumagen.firmware.protocol import (
    BLOCK_DELAY,
    BOOT_SECTOR_LEN,
    CHECKPOINT_SETTLE,
    POST_ERASE_DELAY,
    SECTION0_MAX_WIRE_SIZE,
    SECTION1_MAX_WIRE_SIZE,
    DeviceIdentity,
    FirmwareRevision,
    blocks_for,
    sectors_for,
)

BITS_PER_BYTE: Final = 10
"""8N1: one start bit and one stop bit, so ten bit-times per byte."""

ERASE_SECTOR_SECONDS: Final = 0.34
"""Roughly how long the device takes to erase one sector (it streams one ``x`` each)."""

WRITABLE_SECTIONS: Final = (SECTION0, SECTION1)
"""Sections this library has a qualified write path for.

The chip images are extracted and reported but never written — see the note at
the bottom of :func:`plan_update`. Naming the writable set explicitly means
``only=`` can reject a typo or an out-of-scope request with a useful message
instead of silently planning nothing.
"""


class SectionAction(StrEnum):
    """Whether a section will be written."""

    WRITE = "write"
    SKIP = "skip"


class ProgressStep(StrEnum):
    """One stage of writing a section, in the order the session performs them.

    The unit of `done` passed to :func:`overall_fraction` depends on the step:
    sectors erased for ``ERASE``, bytes handed to the transport for ``WRITE``,
    and 0 or 1 (not yet / complete) for ``SETTLE`` and ``VERIFY``.
    """

    ERASE = "erase"
    SETTLE = "settle"
    """The post-erase settle, :data:`~aiolumagen.firmware.protocol.POST_ERASE_DELAY`."""

    WRITE = "write"
    VERIFY = "verify"
    """Everything after the last block until the section is finished.

    Weighted as one :data:`~aiolumagen.firmware.protocol.CHECKPOINT_SETTLE` — the
    one fixed wait the session always performs before verifying. The checksum
    reads and section 0's internal copy are not modelled (see
    :meth:`UpdatePlan.estimated_seconds`), so this is a floor, not an estimate.
    """


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """What the device reported, as far as planning cares.

    Every field is optional because every field may be unavailable — the caller
    may be planning before it has connected, or a read may have failed. ``None``
    consistently means *unknown*, never *absent* or *zero*, and
    :func:`plan_update` treats unknown conservatively rather than optimistically.
    """

    identity: DeviceIdentity | None = None

    section1_target_code: str | None = None
    """The ``Z35`` reply — the slot the *next* write would target.

    Informational here. It must be re-queried immediately before writing and
    never taken from a plan, because it names the slot the device isn't running
    from and that changes the moment anything is promoted.
    """

    section1_live_header: ContainerHeader | None = None
    """Container header read from the slot the device is currently running."""

    section1_live_checksum: int | None = None
    """The device's own ``C`` over the live slot, across ``HEADER_LEN + size`` bytes.

    Taking the length from the slot's *own* header rather than from the candidate
    image keeps this read image-independent, so one status snapshot can be planned
    against any number of images. A size disagreement is then itself a signal, and
    a cheap one.
    """

    @property
    def revision(self) -> FirmwareRevision | None:
        return self.identity.revision if self.identity else None


@dataclass(frozen=True, slots=True)
class PlannedSection:
    """One section's verdict."""

    name: str
    action: SectionAction
    reason: str
    image: FirmwareImage | None = None

    @property
    def wire_size(self) -> int:
        return len(self.image.wire_bytes) if self.image is not None else 0

    @property
    def blocks(self) -> int:
        return blocks_for(self.wire_size)

    @property
    def sectors(self) -> int:
        return sectors_for(self.wire_size)

    @property
    def will_write(self) -> bool:
        return self.action is SectionAction.WRITE

    def step_seconds(self, step: ProgressStep, baudrate: int, done: int | None = None) -> float:
        """Modelled seconds for `done` units of `step`, or all of it when omitted.

        Uses the same per-unit costs as :meth:`UpdatePlan.estimated_seconds`, so
        the ``ERASE``, ``SETTLE`` and ``WRITE`` steps of every written section sum
        to that estimate at any positive rate. `done` is clamped to the step's
        size, which absorbs the one pad byte the session may append to reach an
        even end address.
        """
        match step:
            case ProgressStep.ERASE:
                sectors = self.sectors if done is None else max(0, min(done, self.sectors))
                return sectors * ERASE_SECTOR_SECONDS
            case ProgressStep.SETTLE:
                return POST_ERASE_DELAY if done is None or done > 0 else 0.0
            case ProgressStep.WRITE:
                size = self.wire_size if done is None else max(0, min(done, self.wire_size))
                wire = size * BITS_PER_BYTE / baudrate if baudrate > 0 else 0.0
                return wire + blocks_for(size) * BLOCK_DELAY
            case ProgressStep.VERIFY:
                return CHECKPOINT_SETTLE if done is None or done > 0 else 0.0


@dataclass(frozen=True, slots=True)
class UpdatePlan:
    """The result of :func:`plan_update`."""

    sections: tuple[PlannedSection, ...] = ()
    warnings: tuple[str, ...] = ()
    device: DeviceStatus = field(default_factory=DeviceStatus)
    release: FirmwareRevision | None = None
    forced: bool = False
    """True when the up-to-date comparison was bypassed. See :func:`plan_update`."""

    only: tuple[str, ...] | None = None
    """Sections the caller restricted this plan to, if any."""

    @property
    def overridden(self) -> bool:
        """True when this plan came from caller overrides rather than comparison.

        Worth surfacing in a UI: an overridden plan is not the library's opinion
        about what the device needs, so "no update available" and "you told me to
        write this anyway" shouldn't look the same to a user.
        """
        return self.forced or self.only is not None

    @property
    def to_write(self) -> tuple[PlannedSection, ...]:
        return tuple(s for s in self.sections if s.will_write)

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to do."""
        return not self.to_write

    @property
    def write_order(self) -> tuple[PlannedSection, ...]:
        """:attr:`to_write` in the order a session writes it: section 1 first.

        Matches the vendor. Section 0 finishes with the promotion, and the
        power-down that loads new firmware follows it, so it has to go last.
        """
        return tuple(sorted(self.to_write, key=lambda s: s.name != SECTION1))

    @property
    def total_bytes(self) -> int:
        return sum(s.wire_size for s in self.to_write)

    @property
    def writes_section1(self) -> bool:
        """Whether this plan touches a live A/B slot.

        The single most useful thing to surface to a user. A section-0-only update
        stages to scratch and is entirely reversible; a plan including section 1
        writes a live slot and takes several times as long.
        """
        return any(s.name == SECTION1 for s in self.to_write)

    def estimated_seconds(self, baudrate: int) -> float:
        """Rough wall-clock estimate: wire time, per-block dead time, and erases.

        Deliberately approximate — it excludes verification reads and the device's
        internal copy. Its job is to distinguish "about a minute" from "about ten"
        so a user can decide whether now is a good time, not to predict a finish
        time. :func:`overall_fraction` reuses the same per-unit costs, which only
        needs them to be right *relative to each other*.
        """
        if baudrate <= 0:
            return 0.0
        total = 0.0
        for section in self.to_write:
            total += section.wire_size * BITS_PER_BYTE / baudrate
            total += section.blocks * BLOCK_DELAY
            total += POST_ERASE_DELAY + section.sectors * ERASE_SECTOR_SECONDS
        return total

    def describe(self) -> str:
        """A short multi-line summary suitable for logging or a confirmation prompt."""
        lines: list[str] = []
        if self.device.identity is not None:
            lines.append(f"device: {self.device.identity}")
        if self.release is not None:
            lines.append(f"update: {self.release}")
        if self.overridden:
            how = ["force"] if self.forced else []
            if self.only is not None:
                how.append(f"only={','.join(self.only)}")
            lines.append(f"OVERRIDDEN ({', '.join(how)}) — not a needs-based plan")
        for section in self.sections:
            verb = "WRITE" if section.will_write else "skip "
            size = f"{section.wire_size:,} bytes" if section.wire_size else "-"
            lines.append(f"  {verb} {section.name:<9} {size:>16}  ({section.reason})")
        lines.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(lines)


def overall_fraction(
    plan: UpdatePlan, baudrate: int, section: str, step: ProgressStep, done: int
) -> float:
    """How much of a whole update is complete, from 0.0 to 1.0.

    Given that every section before `section` in :attr:`UpdatePlan.write_order`
    is finished, every step of `section` before `step` is finished, and `done`
    units of `step` are (see :class:`ProgressStep` for the units).

    Weighted by modelled time rather than bytes, because the steps cost very
    different amounts: a 128 KiB sector erases in about a third of a second,
    while writing the same 128 KiB takes about ten seconds at 230400. A byte-weighted
    bar would stall through every erase and settle, then race through the writes.

    Fixed phases that :meth:`UpdatePlan.estimated_seconds` doesn't model —
    preflight, planning, the rate change — carry no weight; the bar sits at 0.0
    through them. 1.0 means every step of every section is done, so only a
    finished update reaches it.

    :raises ValueError: if `section` is not one the plan writes.
    """
    # One running sum for both numerator and denominator, deliberately not sum():
    # its compensated float summation rounds differently, which left a finished
    # update at 0.9999999999999998.
    elapsed: float | None = None
    total = 0.0
    for planned in plan.write_order:
        for st in ProgressStep:
            if planned.name == section and st is step:
                elapsed = total + planned.step_seconds(st, baudrate, done)
            total += planned.step_seconds(st, baudrate)
    if elapsed is None:
        raise ValueError(f"{section!r} is not written by this plan")
    if total <= 0:
        return 0.0
    return min(1.0, elapsed / total)


def _plan_section0(bundle: FirmwareBundle) -> PlannedSection:
    """Section 0: always written.

    It changes in every release, and it stages to the scratch region rather than
    over live firmware — so a redundant write costs about a minute and risks
    nothing at all. Spending effort to decide whether it's needed would buy less
    than it could get wrong.
    """
    image = bundle.section0
    if image is None:
        raise LumagenFirmwareImageError(
            "no section0 image found in this updater. Section 0 is the main CPU "
            "firmware and lives in the executable's .data segment rather than as "
            "a resource, so either this isn't a Radiance Pro updater or its "
            "layout has changed in a way the extractor doesn't recognise."
        )
    if not image.descriptor_recovered:
        raise LumagenFirmwareImageError(
            "could not locate the swdata descriptor in this updater, so "
            "section0's length fell back to a hardcoded constant. That length "
            "varies between releases, and writing the wrong one would flash a "
            "truncated or over-long image. Refusing to plan an update from it."
        )
    if (
        image.name != SECTION0
        or image.container is not None
        or not (BOOT_SECTOR_LEN < image.size <= BOOT_SECTOR_LEN + SECTION0_MAX_WIRE_SIZE)
    ):
        raise LumagenFirmwareImageError(
            "section0 must contain a boot sector followed by nonempty CPU firmware "
            "that fits before the first section-1 slot"
        )
    return PlannedSection(
        name=SECTION0,
        action=SectionAction.WRITE,
        reason="always written; changes every release and stages to scratch",
        image=image,
    )


def _plan_section1(
    bundle: FirmwareBundle, device: DeviceStatus, *, force: bool = False
) -> PlannedSection | None:
    """Section 1: written only when the live slot demonstrably differs.

    ``None`` when the updater carries no section 1 at all, which is normal.

    The comparison is a payload comparison and needs no version knowledge. Where
    it can't be made, this errs towards writing: an unnecessary section-1 write is
    slow and touches a live slot, but a *skipped* necessary one leaves section 0
    and section 1 from different releases, which is a pairing no vendor session
    ever produces and nothing here can vouch for.

    `force` skips the comparison entirely. This is the only section where force
    changes anything — section 0 is unconditional already.
    """
    image = bundle.section1
    if image is None:
        return None
    if image.name != SECTION1 or image.container is None:
        raise LumagenFirmwareImageError("section1 must be a validated shipped container")
    container = parse_container(image.wire_bytes, name=SECTION1)
    if container.payload != image.payload or container.raw != image.wire_bytes:
        raise LumagenFirmwareImageError("section1 container and image bytes disagree")
    if not container.payload or len(container.raw) > SECTION1_MAX_WIRE_SIZE:
        raise LumagenFirmwareImageError("section1 is empty or exceeds the smaller A/B slot")

    if force:
        return PlannedSection(
            name=SECTION1,
            action=SectionAction.WRITE,
            reason="forced; up-to-date comparison bypassed",
            image=image,
        )

    wire = image.wire_bytes
    header = device.section1_live_header
    stored = device.section1_live_checksum

    if header is None or stored is None:
        return PlannedSection(
            name=SECTION1,
            action=SectionAction.WRITE,
            reason=(
                "could not read the live slot, so it cannot be shown to already hold this image"
            ),
            image=image,
        )

    live_size = HEADER_LEN + header.size
    if live_size != len(wire):
        return PlannedSection(
            name=SECTION1,
            action=SectionAction.WRITE,
            reason=f"live slot holds {live_size:,} bytes, image is {len(wire):,}",
            image=image,
        )

    expected = expected_stored_checksum(wire, header.tag)
    if stored == expected:
        return PlannedSection(
            name=SECTION1,
            action=SectionAction.SKIP,
            reason=f"live slot already matches ({stored:#010x}, tag-corrected)",
            image=image,
        )
    return PlannedSection(
        name=SECTION1,
        action=SectionAction.WRITE,
        reason=f"live slot checksum {stored:#010x}, image expects {expected:#010x}",
        image=image,
    )


def _validate_only(only: Iterable[str]) -> tuple[str, ...]:
    """Normalise and check an `only=` selection.

    Fails loudly rather than quietly planning nothing: a typo'd section name in a
    request to write firmware should not come back as "already up to date".
    """
    requested = tuple(dict.fromkeys(only))  # dedupe, keep caller's order
    if not requested:
        raise LumagenFirmwareImageError(
            "only=() selects no sections. Omit the argument to let the planner "
            f"decide, or name one or more of {', '.join(WRITABLE_SECTIONS)}."
        )
    unknown = [name for name in requested if name not in WRITABLE_SECTIONS]
    if not unknown:
        return requested
    chips = [name for name in unknown if name in CHIP_NAMES]
    if chips:
        raise LumagenFirmwareImageError(
            f"cannot write {', '.join(chips)}: the HDMI chip images are extracted "
            "and reported but deliberately have no write path. No observed vendor "
            "session writes them in Auto mode, and they were unchanged across every "
            "release sampled, so there was nothing to qualify a write path against."
        )
    raise LumagenFirmwareImageError(
        f"unknown section(s) {', '.join(unknown)}. Valid values are {', '.join(WRITABLE_SECTIONS)}."
    )


def plan_update(
    bundle: FirmwareBundle,
    device: DeviceStatus | None = None,
    *,
    force: bool = False,
    only: Iterable[str] | None = None,
) -> UpdatePlan:
    """Decide which sections of `bundle` need writing to the device.

    :param bundle: images from :func:`~aiolumagen.firmware.extract.extract_images`.
    :param device: what the device reported. Omit for a pre-flight estimate; the
        plan will then assume section 1 needs writing, since nothing can prove
        otherwise.
    :param force: bypass the up-to-date comparison and write every selected
        section whether or not the device already holds it. See below for what
        this does *not* override.
    :param only: restrict the plan to these sections (``"section0"`` and/or
        ``"section1"``). Unselected sections are reported as skipped. Chiefly for
        testing and recovery — flashing one section deliberately.
    :raises LumagenFirmwareImageError: if the bundle can't support the requested
        update, `only` names something unwritable, or the device isn't a
        Radiance Pro.

    **What `force` overrides, and what it doesn't.** It overrides exactly one
    thing: the "does the device already have this?" comparison. It deliberately
    does *not* relax any correctness gate, because those aren't about necessity —
    they're about whether the bytes we'd write are the right bytes:

    * an unrecoverable ``swdata`` descriptor (section 0's length would be guessed)
    * a device that isn't a Radiance Pro
    * container checksums, validated during extraction
    * ``Z35`` cross-checking and the header-last commit, both in the session

    So `force` makes an update *unconditional*, never *unchecked*. Anyone reaching
    for it because a correctness gate refused is being told something real.
    """
    device = device or DeviceStatus()
    warnings: list[str] = []
    selected = _validate_only(only) if only is not None else WRITABLE_SECTIONS

    identity = device.identity
    if identity is not None and not identity.is_radiance_pro:
        raise LumagenFirmwareImageError(
            f"device reports {identity.model} (id {identity.device_id:#04x}), not a "
            "Radiance Pro. This updater's images are not valid for it; refusing to "
            "plan an update."
        )

    sections: list[PlannedSection] = []

    if SECTION0 in selected:
        sections.append(_plan_section0(bundle))
    elif bundle.section0 is not None:
        # Reported rather than omitted, so a scoped plan still shows what it left
        # alone. Note the section-0 correctness gates are skipped along with it:
        # `only=["section1"]` must not fail because section 0 has a problem the
        # caller isn't asking us to touch.
        sections.append(
            PlannedSection(
                name=SECTION0,
                action=SectionAction.SKIP,
                reason="not selected",
                image=bundle.section0,
            )
        )

    if SECTION1 in selected:
        section1 = _plan_section1(bundle, device, force=force)
        if section1 is None and only is not None:
            # Absent-and-unasked-for is fine; absent-but-explicitly-requested is
            # not. Silently planning nothing would look like success.
            raise LumagenFirmwareImageError(
                "section1 was requested but this updater contains no section-1 "
                "image (RT_RCDATA 131)."
            )
    else:
        section1 = None

    if section1 is not None:
        sections.append(section1)
        if section1.will_write and not force and device.section1_live_header is None:
            warnings.append(
                "section1 will be written without confirming the device needs it. "
                "It goes into the inactive A/B slot, and the container header is "
                "withheld until the payload verifies, so an abort still leaves the "
                "unit booting its current firmware."
            )
    elif SECTION1 not in selected and bundle.section1 is not None:
        sections.append(
            PlannedSection(
                name=SECTION1,
                action=SectionAction.SKIP,
                reason="not selected",
                image=bundle.section1,
            )
        )

    if force:
        warnings.append(
            "force=True: the up-to-date comparison was bypassed, so selected "
            "sections will be written whether or not the device already holds "
            "them. Correctness gates still apply."
        )
    if only is not None:
        warnings.append(
            f"restricted to {', '.join(selected)} by only=; this plan reflects your "
            "selection, not what the device was found to need."
        )

    # The HDMI chip images are extracted and reported, never written. No observed
    # vendor session writes them in Auto mode, and they were unchanged across every
    # release sampled, so there was nothing to qualify a write path against — a
    # limit of the sample, not a property of the firmware line. "skip" here is a
    # deliberate scope decision, not a missing feature.
    sections.extend(
        PlannedSection(
            name=name,
            action=SectionAction.SKIP,
            reason="chip images are out of scope; no qualified write path",
            image=bundle.images.get(name),
        )
        for name in CHIP_NAMES
        if name in bundle.images
    )

    release, current = bundle.release, device.revision
    if release is not None and current is not None and release < current:
        warnings.append(
            f"this updater ({release}) is OLDER than the firmware on the device "
            f"({current}). Downgrading works, but confirm it's what you intended."
        )

    return UpdatePlan(
        sections=tuple(sections),
        warnings=tuple(warnings),
        device=device,
        release=bundle.release,
        forced=force,
        only=selected if only is not None else None,
    )
