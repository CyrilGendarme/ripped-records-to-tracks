from __future__ import annotations

from dataclasses import dataclass
import re


_SUFFIX_RE = re.compile(
    r"^\s*"
    r"(?P<head>.+)\s*[\-\u2013]\s*"
    r"(?P<side>[A-D])\s*"
    r"(?:[\-\u2013]\s*(?P<rpm>(?:33|45)(?:\s*\*?\s*rpm)?))?\s*$",
    re.IGNORECASE,
)

_HEAD_WITH_REF_RE = re.compile(
    r"^\s*"
    r"(?P<artist>.+?)\s+[\-\u2013]\s+"
    r"(?P<track>.+?)\s*"
    r"\[(?P<record_ref>[^\]]+)\]\s*$",
    re.IGNORECASE,
)

_HEAD_SEP_RE = re.compile(r"\s+[\-\u2013]\s+")


_INVALID_FILENAME_CHARS_RE = re.compile(r"[<>:\"/\\|?*]")
_WHITESPACE_RE = re.compile(r"\s+")
_RPM_SUFFIX_RE = re.compile(r"(?:^|[\s\-_])(?P<rpm>33|45)\D*$", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingName:
    artist: str
    track: str
    record_ref: str | None
    side: str
    rpm: str | None = None

    def build_display_name(self) -> str:
        base = f"{self.artist} - {self.track}"
        if self.record_ref:
            base = f"{base} [{self.record_ref}]"
        base = f"{base} - {self.side}"
        if self.rpm:
            return f"{base} - {self.rpm}"
        return base


def _normalize_piece(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip())


def _normalize_rpm(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if digits not in {"33", "45"}:
        return None
    return f"{digits}rpm"


def extract_rpm_label(stem: str) -> str | None:
    parsed = parse_recording_name(stem)
    if parsed and parsed.rpm:
        digits = "".join(ch for ch in parsed.rpm if ch.isdigit())
        if digits in {"33", "45"}:
            return f"({digits} rpm)"

    match = _RPM_SUFFIX_RE.search(stem.strip())
    if not match:
        return None

    digits = match.group("rpm")
    if digits not in {"33", "45"}:
        return None
    return f"({digits} rpm)"


def parse_recording_name(stem: str) -> RecordingName | None:
    suffix_match = _SUFFIX_RE.match(stem)
    if not suffix_match:
        return None

    head = _normalize_piece(suffix_match.group("head"))
    side = _normalize_piece(suffix_match.group("side")).upper()
    rpm = _normalize_rpm(suffix_match.group("rpm"))

    artist: str | None = None
    track: str | None = None
    record_ref: str | None = None

    head_with_ref = _HEAD_WITH_REF_RE.match(head)
    if head_with_ref:
        artist = _normalize_piece(head_with_ref.group("artist"))
        track = _normalize_piece(head_with_ref.group("track"))
        record_ref = _normalize_piece(head_with_ref.group("record_ref"))
    else:
        parts = _HEAD_SEP_RE.split(head, maxsplit=1)
        if len(parts) == 2:
            artist = _normalize_piece(parts[0])
            track = _normalize_piece(parts[1])

    if not artist or not track or not side:
        return None

    return RecordingName(
        artist=artist,
        track=track,
        record_ref=record_ref,
        side=side,
        rpm=rpm,
    )


def build_export_base_name(stem: str) -> str:
    parsed = parse_recording_name(stem)
    source = parsed.build_display_name() if parsed else stem
    cleaned = _INVALID_FILENAME_CHARS_RE.sub("_", source)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()
