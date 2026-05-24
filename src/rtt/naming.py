from __future__ import annotations

from dataclasses import dataclass
import re


_RECORDING_NAME_RE = re.compile(
    r"^\s*"
    r"(?P<artist>.+?)\s*[\-\u2013]\s*"
    r"(?P<track>.+?)\s*"
    r"\[(?P<record_ref>[^\]]+)\]\s*"
    r"[\-\u2013]\s*"
    r"(?P<side>[^\-\u2013\[]+?)\s*"
    r"(?:[\-\u2013]\s*(?P<rpm>(?:33|45)\s*\*?\s*rpm))?\s*$",
    re.IGNORECASE,
)


_INVALID_FILENAME_CHARS_RE = re.compile(r"[<>:\"/\\|?*]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RecordingName:
    artist: str
    track: str
    record_ref: str
    side: str
    rpm: str | None = None

    def build_display_name(self) -> str:
        base = f"{self.artist} - {self.track} [{self.record_ref}] - {self.side}"
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


def parse_recording_name(stem: str) -> RecordingName | None:
    match = _RECORDING_NAME_RE.match(stem)
    if not match:
        return None

    artist = _normalize_piece(match.group("artist"))
    track = _normalize_piece(match.group("track"))
    record_ref = _normalize_piece(match.group("record_ref"))
    side = _normalize_piece(match.group("side"))
    rpm = _normalize_rpm(match.group("rpm"))

    if not artist or not track or not record_ref or not side:
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
