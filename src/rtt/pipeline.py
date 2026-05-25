from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import List, Optional, Tuple

import librosa
import numpy as np

from .exporter import TrackExportMetadata, build_zip, export_tracks_to_wav
from .naming import build_export_base_name
from .naming import parse_recording_name
from .segmentation import (
    SegmentationConfig,
    SegmentationResult,
    detect_boundaries,
    detect_boundaries_with_precomputed,
    precompute_segmentation,
)

@dataclass
class SplitOutput:
    segmentation: SegmentationResult
    files: List[Path]
    zip_path: Path
    discogs_tracks: List[str] = field(default_factory=list)


@dataclass
class DiscogsExportInfo:
    display_tracks: List[str] = field(default_factory=list)
    track_titles: List[str] = field(default_factory=list)
    expected_count: Optional[int] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    record_ref: Optional[str] = None


def _discogs_export_info_from_filename_stem(stem: str) -> DiscogsExportInfo:
    parsed = parse_recording_name(stem)
    if not parsed:
        return DiscogsExportInfo()

    try:
        from src.api_services.discogs_service import get_release_info_by_record_ref
    except Exception:
        return DiscogsExportInfo(
            artist=parsed.artist,
            album=parsed.track,
            record_ref=parsed.record_ref,
        )

    try:
        release_info = get_release_info_by_record_ref(
            record_ref=parsed.record_ref,
            artist=parsed.artist,
        )
        if release_info is None:
            return DiscogsExportInfo(
                artist=parsed.artist,
                album=parsed.track,
                record_ref=parsed.record_ref,
            )

        tracks = release_info.tracks
        print(
            f"Found {len(tracks)} tracks from Discogs for record_ref={parsed.record_ref}"
        )
        for t in tracks:
            print(f"  - {t}")
    except Exception:
        return DiscogsExportInfo(
            artist=parsed.artist,
            album=parsed.track,
            record_ref=parsed.record_ref,
        )

    side_hint = (parsed.side or "").strip().upper()[:1]
    selected_tracks = tracks
    if side_hint in {"A", "B", "C", "D"}:
        side_tracks = [track for track in tracks if track.side == side_hint]
        if side_tracks:
            selected_tracks = side_tracks

    return DiscogsExportInfo(
        display_tracks=[f"{track.side} - {track.title}" for track in selected_tracks],
        track_titles=[track.title for track in selected_tracks],
        expected_count=len(selected_tracks) if selected_tracks else None,
        artist=release_info.artist or parsed.artist,
        album=release_info.album or parsed.track,
        record_ref=release_info.record_ref or parsed.record_ref,
    )


def _build_track_metadata(
    seg: SegmentationResult,
    export_info: DiscogsExportInfo,
) -> List[TrackExportMetadata]:
    track_count = max(0, len(seg.boundaries_s) - 1)
    metadata: List[TrackExportMetadata] = []
    for idx in range(track_count):
        title = (
            export_info.track_titles[idx]
            if idx < len(export_info.track_titles)
            else f"Track {idx + 1:02d}"
        )
        metadata.append(
            TrackExportMetadata(
                artist=export_info.artist,
                title=title,
                album=export_info.album,
                record_ref=export_info.record_ref,
            )
        )
    return metadata


def _segment_count(seg: SegmentationResult) -> int:
    return max(0, len(seg.boundaries_s) - 1)


def _force_boundaries_to_target_count(
    boundaries_s: List[float],
    duration_s: float,
    target_count: int,
) -> List[float]:
    """Adjust boundaries to exactly match target_count tracks.

    Starts from detected boundaries, then:
    - merges shortest segments if there are too many,
    - splits longest segments if there are too few.
    """
    if target_count <= 0:
        return sorted(set(round(float(x), 3) for x in boundaries_s))

    b = sorted(set(round(float(x), 3) for x in boundaries_s))
    if not b:
        b = [0.0, round(float(duration_s), 3)]
    if b[0] != 0.0:
        b = [0.0] + b
    dur = round(float(duration_s), 3)
    if b[-1] != dur:
        b.append(dur)

    # Remove boundaries when we have too many segments.
    while len(b) - 1 > target_count and len(b) > 2:
        shortest_i = None
        shortest_len = None
        for i in range(len(b) - 1):
            seg_len = b[i + 1] - b[i]
            if shortest_len is None or seg_len < shortest_len:
                shortest_len = seg_len
                shortest_i = i

        if shortest_i is None:
            break

        # Remove the boundary adjacent to the shortest segment (never first/last).
        if shortest_i == 0:
            del_idx = 1
        elif shortest_i == len(b) - 2:
            del_idx = len(b) - 2
        else:
            left_len = b[shortest_i] - b[shortest_i - 1]
            right_len = b[shortest_i + 2] - b[shortest_i + 1]
            del_idx = shortest_i if left_len <= right_len else shortest_i + 1
        del b[del_idx]

    # Add boundaries when we have too few segments.
    while len(b) - 1 < target_count:
        longest_i = None
        longest_len = 0.0
        for i in range(len(b) - 1):
            seg_len = b[i + 1] - b[i]
            if seg_len > longest_len:
                longest_len = seg_len
                longest_i = i

        if longest_i is None or longest_len <= 0.02:
            # Last-resort uniform split if current segments are degenerate.
            step = duration_s / float(target_count)
            b = [round(i * step, 3) for i in range(target_count)] + [
                round(float(duration_s), 3)
            ]
            break

        mid = round((b[longest_i] + b[longest_i + 1]) / 2.0, 3)
        if mid <= b[longest_i] or mid >= b[longest_i + 1]:
            step = duration_s / float(target_count)
            b = [round(i * step, 3) for i in range(target_count)] + [
                round(float(duration_s), 3)
            ]
            break
        b.insert(longest_i + 1, mid)

    b = sorted(set(b))
    if b[0] != 0.0:
        b = [0.0] + b
    if b[-1] != round(float(duration_s), 3):
        b.append(round(float(duration_s), 3))
    return b


def _detect_boundaries_with_target_count(
    audio: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
    target_count: int,
) -> SegmentationResult:
    if target_count <= 0:
        return detect_boundaries(audio=audio, sr=sr, cfg=cfg)

    duration_s = len(audio) / float(sr)
    est_track_len = duration_s / max(target_count, 1)
    base_min_track = float(
        np.clip(est_track_len * 0.45, 10.0, max(15.0, est_track_len))
    )

    sensitivity_offsets = [0.0, 0.15, -0.15, 0.25, -0.25, 0.35, -0.35]
    distance_factors = [1.0, 0.8, 1.2, 0.65, 1.35]
    min_track_factors = [1.0, 0.8, 1.2]
    total_iterations = (
        len(min_track_factors) * len(distance_factors) * len(sensitivity_offsets)
    )
    iteration = 0
    run_start = time.perf_counter()

    print(
        "Adaptive segmentation started "
        f"(target_count={target_count}, total_iterations={total_iterations})"
    )

    precompute_start = time.perf_counter()
    precomputed = precompute_segmentation(audio=audio, sr=sr, cfg=cfg)
    print(
        "Feature precompute done "
        f"(elapsed={time.perf_counter() - precompute_start:.2f}s)."
    )

    best_seg: Optional[SegmentationResult] = None
    best_score: Optional[Tuple[int, float]] = None

    for min_factor in min_track_factors:
        for distance_factor in distance_factors:
            for sens_offset in sensitivity_offsets:
                iteration += 1
                candidate_cfg = replace(
                    cfg,
                    sensitivity=float(np.clip(cfg.sensitivity + sens_offset, 0.0, 1.0)),
                    novelty_peak_distance_s=max(
                        2.5, cfg.novelty_peak_distance_s * distance_factor
                    ),
                    min_track_len_s=max(8.0, base_min_track * min_factor),
                )
                seg = detect_boundaries_with_precomputed(
                    precomputed=precomputed,
                    sr=sr,
                    cfg=candidate_cfg,
                )
                count = _segment_count(seg)
                score = (
                    abs(count - target_count),
                    abs(seg.duration_s / max(count, 1) - est_track_len),
                )

                elapsed = time.perf_counter() - run_start
                print(
                    f"Iteration {iteration}/{total_iterations}: "
                    f"sensitivity={candidate_cfg.sensitivity:.2f}, "
                    f"peak_distance={candidate_cfg.novelty_peak_distance_s:.2f}, "
                    f"min_track={candidate_cfg.min_track_len_s:.2f}, "
                    f"segments={count}, score={score}, elapsed={elapsed:.2f}s"
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_seg = seg
                    print(
                        "  New best candidate: "
                        f"segments={count}, score={score}, "
                        f"boundaries={seg.boundaries_s}"
                    )

                if count == target_count:
                    print(
                        "Exact target reached. "
                        f"Returning after {iteration}/{total_iterations} iterations."
                    )
                    return seg

    if best_seg is None:
        best_seg = detect_boundaries_with_precomputed(
            precomputed=precomputed,
            sr=sr,
            cfg=cfg,
        )

    if _segment_count(best_seg) != target_count:
        print(
            "No exact target found during adaptive search. "
            f"Forcing boundaries from {_segment_count(best_seg)} to {target_count}."
        )
        forced_boundaries = _force_boundaries_to_target_count(
            boundaries_s=best_seg.boundaries_s,
            duration_s=best_seg.duration_s,
            target_count=target_count,
        )
        best_seg = SegmentationResult(
            boundaries_s=forced_boundaries,
            candidates=best_seg.candidates,
            duration_s=best_seg.duration_s,
            diagnostics=best_seg.diagnostics,
        )

    print(
        "Adaptive segmentation completed "
        f"(final_segments={_segment_count(best_seg)}, total_elapsed={time.perf_counter() - run_start:.2f}s)."
    )

    return best_seg


def load_audio(file_path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path=str(file_path), sr=None, mono=mono)
    return y, sr


def split_audio_file(
    file_path: Path,
    output_dir: Path,
    cfg: SegmentationConfig,
) -> SplitOutput:
    audio, sr = load_audio(file_path)
    export_info = _discogs_export_info_from_filename_stem(file_path.stem)
    expected_count = export_info.expected_count
    if expected_count:
        seg = _detect_boundaries_with_target_count(
            audio=audio,
            sr=sr,
            cfg=cfg,
            target_count=expected_count,
        )
    else:
        seg = detect_boundaries(audio=audio, sr=sr, cfg=cfg)

    base_name = build_export_base_name(file_path.stem)
    track_metadata = _build_track_metadata(seg=seg, export_info=export_info)
    files = export_tracks_to_wav(
        audio=audio,
        sr=sr,
        boundaries_s=seg.boundaries_s,
        output_dir=output_dir,
        base_name=base_name,
        track_metadata=track_metadata,
    )
    zip_path = build_zip(files, output_dir / f"{base_name}_tracks.zip")
    return SplitOutput(
        segmentation=seg,
        files=files,
        zip_path=zip_path,
        discogs_tracks=export_info.display_tracks,
    )
