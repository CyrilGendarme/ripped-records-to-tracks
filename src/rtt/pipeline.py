from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import librosa
import numpy as np

from .exporter import build_zip, export_tracks_to_wav
from .naming import build_export_base_name
from .naming import parse_recording_name
from .segmentation import SegmentationConfig, SegmentationResult, detect_boundaries


@dataclass
class SplitOutput:
    segmentation: SegmentationResult
    files: List[Path]
    zip_path: Path
    discogs_tracks: List[str] = field(default_factory=list)


def _discogs_tracks_from_filename_stem(stem: str) -> List[str]:
    parsed = parse_recording_name(stem)
    if not parsed:
        return []

    try:
        from src.api_services.discogs_service import get_release_tracks_by_record_ref
    except Exception:
        return []

    try:
        tracks = get_release_tracks_by_record_ref(
            record_ref=parsed.record_ref,
            artist=parsed.artist,
        )
        print(
            f"Found {len(tracks)} tracks from Discogs for record_ref={parsed.record_ref}"
        )
        for t in tracks:
            print(f"  - {t}")
    except Exception:
        return []

    return [f"{track.side} - {track.title}" for track in tracks]


def load_audio(file_path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path=str(file_path), sr=None, mono=mono)
    return y, sr


def split_audio_file(
    file_path: Path,
    output_dir: Path,
    cfg: SegmentationConfig,
) -> SplitOutput:
    audio, sr = load_audio(file_path)
    seg = detect_boundaries(audio=audio, sr=sr, cfg=cfg)

    base_name = build_export_base_name(file_path.stem)
    files = export_tracks_to_wav(
        audio=audio,
        sr=sr,
        boundaries_s=seg.boundaries_s,
        output_dir=output_dir,
        base_name=base_name,
    )
    zip_path = build_zip(files, output_dir / f"{base_name}_tracks.zip")
    discogs_tracks = _discogs_tracks_from_filename_stem(file_path.stem)
    return SplitOutput(
        segmentation=seg,
        files=files,
        zip_path=zip_path,
        discogs_tracks=discogs_tracks,
    )
