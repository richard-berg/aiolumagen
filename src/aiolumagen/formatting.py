"""Decoders for Lumagen wire codes.

The Lumagen encodes several values in ways that are not directly usable:
refresh rates arrive as 3-digit *truncated* integers (``059`` means 59.94),
aspect ratios as hundredths without a separator (``178`` means 1.78), and the
output-enable set as a 16-bit hex bitmask. Horizontal resolution isn't
reported at all — only vertical resolution and raster aspect, from which the
width has to be derived.

This module is **decoding, not display formatting**. Every function returns a
number (or a tuple of numbers), never a human-readable string: rounding,
units, and localisation are the consumer's business. That keeps the boundary
in ``structure.md`` intact — enum values and display strings stay out of the
library.

Both non-obvious derivations — the NTSC rate family and width-from-aspect —
are documented at their definitions, including why snapping to a standard
width is tolerance-bounded rather than nearest-match (see
:func:`derive_horizontal_resolution`).
"""

from __future__ import annotations

import math

_NTSC_FAMILY: frozenset[int] = frozenset({23, 29, 47, 59, 71, 95, 119})
"""Truncated rate codes belonging to the NTSC ``/1.001`` family.

The Lumagen reports vertical rate as the *truncated* integer part of the real
rate, so the fractional NTSC rates all appear one below their nominal value:
``059`` is 59.94, not 59. Each of these decodes as ``(code + 1) / 1.001``:

===== ============ =========
Code  Nominal      Actual
===== ============ =========
023   24           23.976
029   30           29.970
047   48           47.952
059   60           59.940
071   72           71.928
095   96           95.904
119   120          119.880
===== ============ =========

Using the ``/1.001`` derivation rather than a hand-written lookup table means
the high-rate modes a Radiance Pro supports are covered without anyone having
to remember to extend a dict. Tip0011 only documents ``059``/``060`` by
example, so the rest follow the rule rather than the doc.
"""

COMMON_WIDTHS: tuple[int, ...] = (640, 720, 1280, 1920, 2560, 3840, 4096)
"""Horizontal resolutions a derived width is snapped to when close enough."""

SNAP_TOLERANCE = 0.02
"""Relative distance within which a derived width snaps to a common width.

2% comfortably absorbs the error introduced by the 2-decimal aspect code
(``2160 x 1.78 = 3844.8`` is 0.125% off 3840) while refusing to snap a
genuinely unusual raster onto a nearby standard value.
"""


def decode_vertical_rate(code: str) -> float | None:
    """Decode a Lumagen vertical-rate code to Hz.

    ``"059"`` -> ``59.94``, ``"060"`` -> ``60.0``, ``"023"`` -> ``23.976``.
    Returns ``None`` for ``"000"`` (no signal), an empty field, or anything
    non-numeric.

    Values in the NTSC family are derived as ``(code + 1) / 1.001`` and
    rounded to 3 decimal places — enough to distinguish 59.94 from 60 without
    exposing binary-float noise. Everything else is the integer itself.
    """
    stripped = code.strip()
    if not stripped:
        return None
    try:
        truncated = int(stripped)
    except ValueError:
        return None
    if truncated <= 0:
        # "000" / "0000" is the device's "no signal" placeholder, not 0 Hz.
        return None
    if truncated in _NTSC_FAMILY:
        return round((truncated + 1) / 1.001, 3)
    try:
        return float(truncated)
    except OverflowError:
        return None


def decode_aspect_ratio(code: str) -> float | None:
    """Decode a Lumagen aspect code to a ratio.

    ``"178"`` -> ``1.78``, ``"240"`` -> ``2.4``, ``"133"`` -> ``1.33``.
    Returns ``None`` for ``"000"`` (no signal) or a non-numeric field.

    Applies to every aspect field in the status response: source raster
    (``AAA``), source content (``SSS``), output (``ZZZ``), and the detected
    pair (``JJJ``/``LLL``).
    """
    stripped = code.strip()
    if not stripped:
        return None
    try:
        hundredths = int(stripped)
    except ValueError:
        return None
    if hundredths <= 0:
        return None
    try:
        return hundredths / 100
    except OverflowError:
        return None


def derive_horizontal_resolution(vertical: str, aspect: str) -> int | None:
    """Derive display width from a vertical resolution and raster aspect.

    The Lumagen reports only vertical resolution (``VVVV``) and raster aspect
    (``AAA``/``ZZZ``), so width has to be computed: ``height x aspect``. The
    aspect code carries just two decimal places, so the raw product is off by
    a few pixels (``2160 x 1.78 = 3844.8``); when the result lands within
    :data:`SNAP_TOLERANCE` of a value in :data:`COMMON_WIDTHS` it snaps to
    that value, recovering the exact standard width.

    Outside that tolerance the computed width is returned as-is. Snapping
    unconditionally to the nearest common width is the obvious implementation
    and it's wrong: it turns an unusual but correct raster into a wrong
    standard one, reporting a 5120-wide ultrawide as 4096 — a 20% error
    presented as though it were exact. Refusing the snap keeps the error
    bounded by the aspect code's own precision instead.

    Returns ``None`` when either field is missing, non-numeric, or zero (the
    device's no-signal placeholder).

    Note this is *display geometry*, not necessarily the transmitted pixel
    count: 480i is 720 pixels wide with non-square pixels but a 4:3 raster,
    so this returns 640.
    """
    try:
        height = int(vertical.strip())
    except ValueError:
        return None
    if height <= 0:
        return None
    ratio = decode_aspect_ratio(aspect)
    if ratio is None:
        return None
    try:
        computed = height * ratio
    except OverflowError:
        return None
    if not math.isfinite(computed):
        return None
    nearest = min(COMMON_WIDTHS, key=lambda width: abs(width - computed))
    if abs(nearest - computed) <= computed * SNAP_TOLERANCE:
        return nearest
    return round(computed)


def decode_output_mask(mask: str | int) -> tuple[int, ...] | None:
    """Decode the ``WWWW`` output-enable bitmask to 1-based output numbers.

    Per Tip0011: "Output on. 16 bit hex, b0 to 15 for out 1 to 16. Bit=1 if
    on". So ``"000e"`` (``0b1110``) means outputs 2, 3 and 4 are enabled.

    Accepts the raw hex field or an already-parsed int. Returns ``None`` if
    the field is empty or not valid hex; returns an empty tuple when the mask
    is zero (all outputs off), which is a real state and distinct from
    "couldn't read it".
    """
    if isinstance(mask, str):
        stripped = mask.strip()
        if not stripped:
            return None
        try:
            value = int(stripped, 16)
        except ValueError:
            return None
    else:
        value = mask
    if value < 0:
        return None
    return tuple(bit + 1 for bit in range(16) if value & (1 << bit))
