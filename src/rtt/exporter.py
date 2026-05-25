from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List
import zipfile

import numpy as np
import soundfile as sf


_INVALID_FILENAME_CHARS_RE = re.compile(r"[<>:\"/\\|?*]")


@dataclass
class TrackExportMetadata:
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    record_ref: str | None = None


def _sanitize_filename_piece(value: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS_RE.sub("_", value).strip()
    return re.sub(r"\s+", " ", cleaned)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        candidate = path.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _safe_track_name(base_name: str, index: int, _start_s: float, _end_s: float) -> str:
    return f"{base_name}_track_{index:02d}.wav"


def _name_from_metadata(meta: TrackExportMetadata | None, fallback_name: str) -> str:
    if meta and meta.artist and meta.title:
        artist = _sanitize_filename_piece(meta.artist)
        title = _sanitize_filename_piece(meta.title)
        if artist and title:
            return f"{artist} - {title}.wav"
    return fallback_name


def _write_wav_metadata(path: Path, meta: TrackExportMetadata | None) -> None:
    if not meta:
        return
    try:
        from mutagen.id3 import TALB, TIT2, TPE1, TXXX
        from mutagen.wave import WAVE
    except Exception:
        return

    try:
        wav = WAVE(str(path))
        if wav.tags is None:
            wav.add_tags()

        if meta.title:
            wav.tags["TIT2"] = TIT2(encoding=3, text=[meta.title])
        if meta.artist:
            wav.tags["TPE1"] = TPE1(encoding=3, text=[meta.artist])
        if meta.album:
            wav.tags["TALB"] = TALB(encoding=3, text=[meta.album])
        if meta.record_ref:
            wav.tags["TXXX:record_ref"] = TXXX(
                encoding=3,
                desc="record_ref",
                text=[meta.record_ref],
            )
        wav.save()
    except Exception:
        return


def export_tracks_to_wav(
    audio: np.ndarray,
    sr: int,
    boundaries_s: Iterable[float],
    output_dir: Path,
    base_name: str,
    track_metadata: List[TrackExportMetadata] | None = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    b = sorted(set(float(x) for x in boundaries_s))
    written: List[Path] = []

    for idx in range(len(b) - 1):
        start_s = b[idx]
        end_s = b[idx + 1]
        if end_s <= start_s:
            continue
        start_i = max(0, int(round(start_s * sr)))
        end_i = min(len(audio), int(round(end_s * sr)))
        if end_i - start_i <= 0:
            continue

        chunk = audio[start_i:end_i]
        fallback_name = _safe_track_name(base_name, idx + 1, start_s, end_s)
        meta = track_metadata[idx] if track_metadata and idx < len(track_metadata) else None
        out_name = _name_from_metadata(meta, fallback_name)
        out_path = _unique_path(output_dir / out_name)
        sf.write(out_path, chunk, sr)
        _write_wav_metadata(out_path, meta)
        written.append(out_path)

    return written


def build_zip(files: Iterable[Path], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    return zip_path
