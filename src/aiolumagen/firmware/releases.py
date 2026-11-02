"""Find Radiance Pro releases on lumagen.com, and unpack the vendor's zip.

Two pure functions sit between "is there new firmware?" and
:func:`~aiolumagen.firmware.extract.extract_images`:

* :func:`parse_release_index` turns the HTML of :data:`RELEASES_URL` into
  :class:`ReleaseListing` records, oldest first.
* :func:`extract_updater_zip` pulls the single updater ``.exe`` out of a
  downloaded release zip.

**No HTTP happens here.** This module is Lumagen-aware (it knows the naming
scheme and the Beta/Production vocabulary) but it takes ``str`` and ``bytes``
and returns data. Fetching the page and the zip is the caller's business — in
Home Assistant, through HA's shared client session — which keeps an HTTP stack
out of this library's dependencies and keeps these functions testable against
synthetic input.

The page format
---------------

The index is a Squarespace page listing every release newest first. Each entry
is a list item whose first link is the download, followed by a heading and free
text::

    <li><p><a href="/s/radiance_pro030326.zip">Download</a></p>
      <ul><li><p><strong>Beta 030326-</strong><em>Posted 042826</em>&nbsp; notes…
        <br><strong><em>Update time ~1 minutes @230k …</em></strong></p></li></ul>
    </li>

Known quirks, all tolerated:

* **A revision can be listed twice.** Listings are deduped on revision, keeping
  the first (newest-posted) occurrence.
* **The heading text is hand-typed and has been wrong** — one entry for the
  ``101524`` zip is headed ``1015524``. So each listing is keyed on the **href
  filename**, and everything in the heading (label, posted date, time estimate)
  is advisory: parsed if it's there, ``None``/``""`` if it isn't.
* **Labels vary.** ``Beta``, ``Production``, and on older entries
  ``Production candidate``. Only a bare ``Production`` is treated as a production
  release; anything else — including a missing or unknown label — is beta, so a
  pre-release is never offered on the production track.

Parsing **fails closed**: if the page has no release links at all, or any
release link has a filename that isn't a plausible ``MMDDYY`` revision, or
resolves somewhere other than an ``https`` URL on the index's own host,
:class:`LumagenReleaseIndexError` is raised rather than a partial answer
returned. A page that has changed shape should result in *no* update being
offered, never a guessed one.
"""

from __future__ import annotations

import datetime
import io
import re
import struct
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final
from urllib.parse import urljoin, urlsplit

from aiolumagen.exceptions import LumagenFirmwareError, LumagenFirmwareImageError
from aiolumagen.firmware.protocol import FirmwareRevision

RELEASES_URL: Final = "https://www.lumagen.com/software-updates/radiance-pro-updates"
"""The vendor's Radiance Pro release index. The caller fetches it."""

MAX_UPDATER_EXE_BYTES: Final = 32 * 1024 * 1024
"""Upper bound on the uncompressed updater EXE. Real ones are about 5 MB."""

MAX_ZIP_ENTRIES: Final = 64
"""Upper bound on archive members. Real release zips hold two."""

_ZIP_HREF_RE: Final = re.compile(r"^radiance_pro(.*)\.zip$", re.IGNORECASE)
_EXE_NAME_RE: Final = re.compile(r"^radiance_pro\w*\.exe$", re.IGNORECASE)
_LABEL_RE: Final = re.compile(r"^(?P<label>beta|production(?:\s+candidate)?)\b", re.IGNORECASE)
"""Only the known labels, so words after them stay in the notes."""
_HEADING_TOKEN_RE: Final = re.compile(r"^\s*(?:\d{6,}\s*-?|\d+\s*-)")
"""The heading's revision-ish token, e.g. ``030326-`` or the mistyped ``1015524-``."""
_POSTED_RE: Final = re.compile(r"Posted\s*(\d{6})", re.IGNORECASE)
_ESTIMATE_RE: Final = re.compile(r"Update time\s*~\s*(\d+)\s*minutes?", re.IGNORECASE)
_UPDATE_SENTENCE_RE: Final = re.compile(r"Update time\b[^.]*\.?", re.IGNORECASE)
_WHITESPACE_RE: Final = re.compile(r"\s+")

_BLOCK_TAGS: Final = frozenset({"br", "p", "li", "ul", "ol", "div", "h1", "h2", "h3", "h4"})
"""Tags that separate words visually, so their boundaries become a space."""

_ENCRYPTED_FLAG: Final = 0x1


class LumagenReleaseIndexError(LumagenFirmwareError, ValueError):
    """The release index could not be parsed with confidence.

    Raised by :func:`parse_release_index` when the page doesn't look like the
    format this module understands. No device was contacted. The right response
    is to **offer nothing** — keep whatever was last known, or report the check
    as failed — never to guess at a release from a page that has changed shape.

    Also subclasses :class:`ValueError`, as the other bad-input firmware error
    does.
    """


class ReleaseChannel(StrEnum):
    """Which releases a consumer wants to be offered."""

    BETA = "beta"
    """Every release. Lumagen ships nearly everything as beta."""

    PRODUCTION = "production"
    """Only releases labelled exactly ``Production``."""


@dataclass(frozen=True, slots=True)
class ReleaseListing:
    """One release on the vendor's index."""

    revision: FirmwareRevision
    """From the href filename — the authoritative source."""

    channel: ReleaseChannel
    label: str
    """The heading's label as written, e.g. ``"Production candidate"``, or ``""``."""

    posted: datetime.date | None
    url: str
    """Absolute ``https`` URL of the zip."""

    notes: str
    """Free-text release notes, with the heading boilerplate removed."""

    est_minutes: int | None
    """The page's "Update time ~N minutes" hint, if present."""

    @property
    def filename(self) -> str:
        """The zip's canonical name, ``radiance_proMMDDYY.zip``."""
        return f"radiance_pro{self.revision.mmddyy}.zip"


@dataclass(slots=True)
class _RawEntry:
    href: str
    group: str
    li_depth: int
    chunks: list[str]


class _IndexParser(HTMLParser):
    """Collect each release anchor's href and the text of its entry.

    An entry's text runs until the ``<li>`` that held its anchor closes, or the
    next release anchor starts. That relies on the page wrapping each release in
    a list item: an anchor outside any ``<li>`` is only ended by the next anchor,
    so text after the last one (a footer) would land in its notes. Notes are
    display-only, so that failure is cosmetic.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[_RawEntry] = []
        self._current: _RawEntry | None = None
        self._li_depth = 0
        self._in_release_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "li":
            self._li_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            match = _match_release_href(href) if href else None
            if href is not None and match is not None:
                self._current = _RawEntry(
                    href=href, group=match.group(1), li_depth=self._li_depth, chunks=[]
                )
                self.entries.append(self._current)
                self._in_release_anchor = True
                return
        if tag in _BLOCK_TAGS:
            self._append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_release_anchor = False
        if tag in _BLOCK_TAGS:
            self._append(" ")
        if tag == "li":
            if self._current is not None and self._li_depth <= self._current.li_depth:
                self._current = None
            self._li_depth = max(0, self._li_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self._in_release_anchor:
            self._append(data)

    def _append(self, text: str) -> None:
        if self._current is not None:
            self._current.chunks.append(text)


def _match_release_href(href: str) -> re.Match[str] | None:
    segment = urlsplit(href).path.rsplit("/", 1)[-1]
    return _ZIP_HREF_RE.match(segment)


def _parse_mmddyy_date(digits: str) -> datetime.date | None:
    try:
        return datetime.date(2000 + int(digits[4:6]), int(digits[0:2]), int(digits[2:4]))
    except ValueError:
        return None


def _resolve_url(href: str, base_url: str) -> str:
    url = urljoin(base_url, href)
    parts = urlsplit(url)
    base_host = (urlsplit(base_url).hostname or "").lower()
    if parts.scheme != "https" or (parts.hostname or "").lower() != base_host:
        raise LumagenReleaseIndexError(
            f"release link {href!r} resolves to {url!r}, which is not https on {base_host!r}"
        )
    return url


def _build_listing(entry: _RawEntry, base_url: str) -> ReleaseListing:
    group = entry.group
    revision = FirmwareRevision.parse(group) if len(group) == 6 and group.isdigit() else None
    if revision is None:
        raise LumagenReleaseIndexError(
            f"release link {entry.href!r} does not name a MMDDYY revision"
        )
    url = _resolve_url(entry.href, base_url)

    block = _WHITESPACE_RE.sub(" ", "".join(entry.chunks).replace("\xa0", " ")).strip()

    label = ""
    rest = block
    label_match = _LABEL_RE.match(block)
    if label_match is not None:
        label = _WHITESPACE_RE.sub(" ", label_match.group("label"))
        rest = block[label_match.end() :]
    rest = _HEADING_TOKEN_RE.sub("", rest, count=1)
    channel = ReleaseChannel.PRODUCTION if label.lower() == "production" else ReleaseChannel.BETA

    posted_match = _POSTED_RE.search(block)
    posted = _parse_mmddyy_date(posted_match.group(1)) if posted_match else None

    estimate_match = _ESTIMATE_RE.search(block)
    est_minutes = int(estimate_match.group(1)) if estimate_match else None

    notes = _POSTED_RE.sub(" ", rest)
    notes = _UPDATE_SENTENCE_RE.sub(" ", notes)
    notes = _WHITESPACE_RE.sub(" ", notes).strip()

    return ReleaseListing(
        revision=revision,
        channel=channel,
        label=label,
        posted=posted,
        url=url,
        notes=notes,
        est_minutes=est_minutes,
    )


def parse_release_index(html: str, base_url: str = RELEASES_URL) -> tuple[ReleaseListing, ...]:
    """Parse the vendor's release index into listings, **oldest first**.

    ``[-1]`` is therefore the newest release of any label; use
    :func:`latest_release` to pick one for a channel.

    :param html: the page body, as fetched by the caller.
    :param base_url: what relative hrefs resolve against, and the only host a
        release link may point at.
    :raises LumagenReleaseIndexError: no release links were found, or any one of
        them has an unparseable filename or points off-site. Fails closed: one
        bad entry rejects the whole page.
    """
    parser = _IndexParser()
    parser.feed(html)
    parser.close()
    if not parser.entries:
        raise LumagenReleaseIndexError("no radiance_pro*.zip release links found on the page")

    listings: dict[FirmwareRevision, ReleaseListing] = {}
    for entry in parser.entries:
        listing = _build_listing(entry, base_url)
        # Newest-first page: the first occurrence of a revision wins.
        listings.setdefault(listing.revision, listing)
    return tuple(sorted(listings.values(), key=lambda listing: listing.revision))


def latest_release(
    listings: Iterable[ReleaseListing], channel: ReleaseChannel
) -> ReleaseListing | None:
    """The newest listing a consumer on `channel` should be offered.

    ``BETA`` means the newest release of any label. ``PRODUCTION`` means the newest
    labelled exactly ``Production``. ``None`` if there is nothing to offer.
    """
    candidates = [
        listing
        for listing in listings
        if channel is ReleaseChannel.BETA or listing.channel is ReleaseChannel.PRODUCTION
    ]
    return max(candidates, key=lambda listing: listing.revision, default=None)


def _is_unsafe_member(name: str) -> bool:
    if not name:
        return True
    posix = name.replace("\\", "/")
    windows = PureWindowsPath(name)
    return (
        posix.startswith("/")
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in PurePosixPath(posix).parts
        or ".." in windows.parts
    )


def extract_updater_zip(data: bytes) -> tuple[str, bytes]:
    """Pull the updater EXE out of a vendor release zip.

    The result feeds ``extract_images(exe, source_name=name)``. Pass the name:
    :func:`~aiolumagen.firmware.load_updater` with bare bytes loses it, and with
    it the bundle's release.

    :returns: ``(basename, exe_bytes)``, e.g. ``("radiance_pro030326.exe", …)``.
    :raises LumagenFirmwareImageError: not a zip; too many members; any member
        name that is absolute or climbs out with ``..``; zero or several
        ``radiance_pro*.exe`` members; an encrypted, oversized or corrupt one.
        No device was contacted.
    """
    if len(data) > MAX_UPDATER_EXE_BYTES + 1024 * 1024:
        raise LumagenFirmwareImageError("release zip exceeds the compressed input limit")
    try:
        # Bound central-directory work before ZipFile allocates a ZipInfo for
        # every member. The EOCD count alone is untrusted metadata.
        end = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
        if end < 0 or end + 22 > len(data):
            raise LumagenFirmwareImageError("downloaded release is not a zip archive")
        _, disk, directory_disk, disk_count, count, size, offset, comment = struct.unpack_from(
            "<4s4H2IH", data, end
        )
        if disk or directory_disk or disk_count != count or end + 22 + comment != len(data):
            raise LumagenFirmwareImageError("unsupported multipart or malformed release zip")
        if count > MAX_ZIP_ENTRIES:
            raise LumagenFirmwareImageError(
                f"release zip has {count} entries; at most {MAX_ZIP_ENTRIES} expected"
            )
        if offset + size != end:
            raise LumagenFirmwareImageError("release zip has an invalid central-directory offset")
        cursor = offset
        seen = 0
        while cursor < end:
            if cursor + 46 > end or data[cursor : cursor + 4] != b"PK\x01\x02":
                raise LumagenFirmwareImageError("release zip has a corrupt central directory")
            seen += 1
            if seen > MAX_ZIP_ENTRIES:
                raise LumagenFirmwareImageError("release zip contains too many directory entries")
            name_len, extra_len, comment_len = struct.unpack_from("<3H", data, cursor + 28)
            cursor += 46 + name_len + extra_len + comment_len
        if cursor != end or seen != count:
            raise LumagenFirmwareImageError("release zip directory length/count does not match")
        stream = io.BytesIO(data)
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise LumagenFirmwareImageError(
                    f"release zip has {len(infos)} entries; at most {MAX_ZIP_ENTRIES} expected"
                )
            for info in infos:
                if _is_unsafe_member(info.filename):
                    raise LumagenFirmwareImageError(
                        f"release zip contains an unsafe path: {info.filename!r}"
                    )

            candidates = [
                info
                for info in infos
                if not info.is_dir()
                and _EXE_NAME_RE.match(PurePosixPath(info.filename.replace("\\", "/")).name)
            ]
            if len(candidates) != 1:
                found = ", ".join(info.filename for info in candidates) or "none"
                raise LumagenFirmwareImageError(
                    f"release zip must hold exactly one radiance_pro*.exe; found {found}"
                )
            info = candidates[0]
            if info.flag_bits & _ENCRYPTED_FLAG:
                raise LumagenFirmwareImageError(f"{info.filename} is encrypted")
            limit = MAX_UPDATER_EXE_BYTES
            if info.file_size > limit:
                raise LumagenFirmwareImageError(
                    f"{info.filename} is {info.file_size} bytes; at most {limit} expected"
                )
            with archive.open(info) as member:
                exe = member.read(limit + 1)
            if len(exe) > limit:
                raise LumagenFirmwareImageError(f"{info.filename} decompresses past {limit} bytes")
    except LumagenFirmwareImageError:
        raise
    except (
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        NotImplementedError,
        OSError,
        ValueError,
    ) as err:
        raise LumagenFirmwareImageError(f"release zip is corrupt: {err}") from err

    return PurePosixPath(info.filename.replace("\\", "/")).name, exe
