from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List

import numpy as np
from pydub import AudioSegment


_INVALID_FILENAME_CHARS_RE = re.compile(r"[<>:\"/\\|?*]")


@dataclass
class TrackExportMetadata:
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    record_ref: str | None = None


@dataclass
class AudioExportSettings:
    format: str = "mp3"
    bitrate: str | None = None


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
    return f"{base_name}_track_{index:02d}.mp3"


def _name_from_metadata(meta: TrackExportMetadata | None, fallback_name: str) -> str:
    if meta and meta.artist and meta.title:
        artist = _sanitize_filename_piece(meta.artist)
        title = _sanitize_filename_piece(meta.title)
        if artist and title:
            return f"{artist} - {title}.mp3"
    return fallback_name


def _write_audio_metadata(path: Path, meta: TrackExportMetadata | None) -> None:
    if not meta:
        return
    try:
        from mutagen.id3 import TALB, TIT2, TPE1, TXXX
        from mutagen.mp3 import MP3
        from mutagen.wave import WAVE
    except Exception:
        return

    try:
        if path.suffix.lower() == ".mp3":
            audio_file = MP3(str(path))
        else:
            audio_file = WAVE(str(path))

        if audio_file.tags is None:
            audio_file.add_tags()

        if meta.title:
            audio_file.tags["TIT2"] = TIT2(encoding=3, text=[meta.title])
        if meta.artist:
            audio_file.tags["TPE1"] = TPE1(encoding=3, text=[meta.artist])
        if meta.album:
            audio_file.tags["TALB"] = TALB(encoding=3, text=[meta.album])
        if meta.record_ref:
            audio_file.tags["TXXX:record_ref"] = TXXX(
                encoding=3,
                desc="record_ref",
                text=[meta.record_ref],
            )
        audio_file.save()
    except Exception:
        return


def infer_export_settings(input_file: Path) -> AudioExportSettings:
    settings = AudioExportSettings(format="mp3", bitrate=None)
    if input_file.suffix.lower() != ".mp3":
        return settings

    try:
        from mutagen.mp3 import MP3
    except Exception:
        return settings

    try:
        info = MP3(str(input_file)).info
        bitrate = getattr(info, "bitrate", None)
        if bitrate and bitrate > 0:
            kbps = max(32, int(round(float(bitrate) / 1000.0)))
            settings.bitrate = f"{kbps}k"
    except Exception:
        return settings

    return settings


def _chunk_to_audio_segment(chunk: np.ndarray, sr: int) -> AudioSegment:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        channels = 1
        pcm = np.clip(arr, -1.0, 1.0)
    else:
        channels = int(arr.shape[1])
        pcm = np.clip(arr, -1.0, 1.0)

    pcm16 = (pcm * 32767.0).astype(np.int16)
    return AudioSegment(
        data=pcm16.tobytes(),
        sample_width=2,
        frame_rate=int(sr),
        channels=channels,
    )


def _trim_chunk_silence(
    chunk: np.ndarray,
    sr: int,
    threshold_db: float,
    frame_ms: float = 10.0,
) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.size == 0:
        return arr

    n_samples = int(arr.shape[0]) if arr.ndim > 1 else int(arr.shape[0])
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))

    if arr.ndim == 1:
        mono = arr
    else:
        mono = np.sqrt(np.mean(np.square(arr), axis=1, dtype=np.float64)).astype(
            np.float32
        )

    usable = (len(mono) // frame_len) * frame_len
    if usable <= 0:
        return arr

    framed = mono[:usable].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(np.square(framed), axis=1, dtype=np.float64))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    active = rms_db >= threshold_db

    # Ignore isolated spikes (clicks/crackles) by requiring sustained activity.
    frame_s = frame_ms / 1000.0
    min_active_run_frames = max(1, int(round(0.12 / frame_s)))

    runs: list[tuple[int, int]] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))

    long_runs = [run for run in runs if (run[1] - run[0] + 1) >= min_active_run_frames]
    if not long_runs:
        return arr

    first, _ = long_runs[0]
    _, last = long_runs[-1]

    # Keep a safety margin around detected activity to avoid shaving attacks/decays.
    pad_frames = max(1, int(round(0.12 / frame_s)))
    first = max(0, first - pad_frames)
    last = min(len(active) - 1, last + pad_frames)

    start_i = max(0, first * frame_len)
    end_i = min(n_samples, (last + 1) * frame_len)

    if end_i - start_i < max(1, int(0.05 * sr)):
        return arr

    if arr.ndim == 1:
        return arr[start_i:end_i]
    return arr[start_i:end_i, :]


def export_tracks_to_wav(
    audio: np.ndarray,
    sr: int,
    boundaries_s: Iterable[float],
    output_dir: Path,
    base_name: str,
    track_metadata: List[TrackExportMetadata] | None = None,
    export_settings: AudioExportSettings | None = None,
    trim_silence_db_threshold: float | None = None,
) -> List[Path]:
    settings = export_settings or AudioExportSettings()
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
        if trim_silence_db_threshold is not None:
            chunk = _trim_chunk_silence(
                chunk=chunk,
                sr=sr,
                threshold_db=float(trim_silence_db_threshold),
            )
        fallback_name = _safe_track_name(base_name, idx + 1, start_s, end_s)
        meta = track_metadata[idx] if track_metadata and idx < len(track_metadata) else None
        out_name = _name_from_metadata(meta, fallback_name)
        out_path = _unique_path(output_dir / out_name)

        segment = _chunk_to_audio_segment(chunk, sr)
        export_kwargs = {"format": settings.format}
        if settings.bitrate:
            export_kwargs["bitrate"] = settings.bitrate
        segment.export(str(out_path), **export_kwargs)

        _write_audio_metadata(out_path, meta)
        written.append(out_path)

    return written
