from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import zipfile

import numpy as np
import soundfile as sf


def _safe_track_name(base_name: str, index: int, start_s: float, end_s: float) -> str:
    return f"{base_name}_track_{index:02d}_{start_s:07.2f}s-{end_s:07.2f}s.wav"


def export_tracks_to_wav(
    audio: np.ndarray,
    sr: int,
    boundaries_s: Iterable[float],
    output_dir: Path,
    base_name: str,
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
        out_name = _safe_track_name(base_name, idx + 1, start_s, end_s)
        out_path = output_dir / out_name
        sf.write(out_path, chunk, sr)
        written.append(out_path)

    return written


def build_zip(files: Iterable[Path], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    return zip_path
