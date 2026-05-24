from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np

from .exporter import build_zip, export_tracks_to_wav
from .segmentation import SegmentationConfig, SegmentationResult, detect_boundaries


@dataclass
class SplitOutput:
    segmentation: SegmentationResult
    files: List[Path]
    zip_path: Path


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

    base_name = file_path.stem.replace(" ", "_")
    files = export_tracks_to_wav(
        audio=audio,
        sr=sr,
        boundaries_s=seg.boundaries_s,
        output_dir=output_dir,
        base_name=base_name,
    )
    zip_path = build_zip(files, output_dir / f"{base_name}_tracks.zip")
    return SplitOutput(segmentation=seg, files=files, zip_path=zip_path)
