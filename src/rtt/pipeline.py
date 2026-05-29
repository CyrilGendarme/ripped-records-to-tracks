from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np

from .exporter import (
    TrackExportMetadata,
    export_tracks_to_wav,
    infer_export_settings,
)
from .naming import build_export_base_name
from .naming import extract_rpm_label
from .naming import parse_recording_name
from .segmentation import (
    SegmentationConfig,
    SegmentationResult,
    _bandpass_music_only,
    detect_boundaries_by_silence_target,
)

@dataclass
class SplitOutput:
    segmentation: SegmentationResult
    files: List[Path]
    discogs_tracks: List[str] = field(default_factory=list)
    track_metadata: List[TrackExportMetadata] = field(default_factory=list)


@dataclass
class DiscogsExportInfo:
    display_tracks: List[str] = field(default_factory=list)
    track_titles: List[str] = field(default_factory=list)
    expected_count: Optional[int] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    record_ref: Optional[str] = None
    rpm_label: Optional[str] = None


def _discogs_export_info_from_filename_stem(stem: str) -> DiscogsExportInfo:
    parsed = parse_recording_name(stem)
    rpm_label = extract_rpm_label(stem)
    if not parsed:
        return DiscogsExportInfo(rpm_label=rpm_label)
    if not parsed.record_ref:
        return DiscogsExportInfo(
            artist=parsed.artist,
            album=parsed.track,
            rpm_label=rpm_label,
        )

    try:
        from src.api_services.discogs_service import get_release_info_by_record_ref
    except Exception:
        return DiscogsExportInfo(
            artist=parsed.artist,
            album=parsed.track,
            record_ref=parsed.record_ref,
            rpm_label=rpm_label,
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
                rpm_label=rpm_label,
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
            rpm_label=rpm_label,
        )

    side_hint = (parsed.side or "").strip().upper()[:1]
    selected_tracks = tracks
    if side_hint in {"A", "B", "C", "D"}:
        side_tracks = [track for track in tracks if track.side == side_hint]
        if side_tracks:
            selected_tracks = side_tracks
        else:
            # Some Discogs releases expose positions as 1..N without side letters.
            # For side-rip inputs (A/B), infer halves from sequential numeric positions.
            numeric_positions: List[int] = []
            for track in tracks:
                pos = (track.position or "").strip()
                if not pos.isdigit():
                    numeric_positions = []
                    break
                numeric_positions.append(int(pos))

            if (
                side_hint in {"A", "B"}
                and numeric_positions
                and len(numeric_positions) == len(tracks)
            ):
                ordered = sorted(
                    zip(numeric_positions, tracks), key=lambda item: item[0]
                )
                expected = list(range(1, len(tracks) + 1))
                if [p for p, _ in ordered] == expected and len(ordered) % 2 == 0:
                    midpoint = len(ordered) // 2
                    if side_hint == "A":
                        selected_tracks = [track for _, track in ordered[:midpoint]]
                    else:
                        selected_tracks = [track for _, track in ordered[midpoint:]]

    return DiscogsExportInfo(
        display_tracks=[
            f"{(track.side or track.position or '?')} - {track.title}"
            for track in selected_tracks
        ],
        track_titles=[track.title for track in selected_tracks],
        expected_count=len(selected_tracks) if selected_tracks else None,
        artist=release_info.artist or parsed.artist,
        album=release_info.album or parsed.track,
        record_ref=release_info.record_ref or parsed.record_ref,
        rpm_label=rpm_label,
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
        if export_info.rpm_label and not title.endswith(export_info.rpm_label):
            title = f"{title} {export_info.rpm_label}"
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
        target_count = 1

    print(
        "Silence-only split started "
        f"(target_count={target_count}, threshold_db={cfg.silence_db_threshold}, "
        f"music_band={cfg.music_low_hz:.0f}-{cfg.music_high_hz:.0f}Hz)."
    )
    best_seg = detect_boundaries_by_silence_target(
        audio=audio,
        sr=sr,
        cfg=cfg,
        target_count=target_count,
    )

    if _segment_count(best_seg) != target_count:
        print(
            "No exact target reached with silence-only candidates. "
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
        "Silence-only split completed " f"(final_segments={_segment_count(best_seg)})."
    )

    return best_seg


def _estimate_target_count(duration_s: float, cfg: SegmentationConfig) -> int:
    est = int(round(duration_s / max(10.0, cfg.min_track_len_s)))
    return max(1, est)


def _trim_input_edge_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    min_active_run_s: float,
    frame_ms: float = 10.0,
    music_low_hz: float = 200.0,
    music_high_hz: float = 400.0,
) -> tuple[np.ndarray, float, float]:
    """Trim only global start/end silence from input before any other processing.

    Returns (trimmed_audio, start_trim_s, end_trim_s).
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return arr, 0.0, 0.0

    if arr.ndim == 1:
        mono = arr
        n_samples = int(arr.shape[0])
    else:
        mono = np.sqrt(np.mean(np.square(arr), axis=1, dtype=np.float64)).astype(
            np.float32
        )
        n_samples = int(arr.shape[0])

    # Focus edge detection on the narrow band used elsewhere for vinyl-aware
    # silence analysis so clicks and broad-band surface noise matter less.
    mono_music = _bandpass_music_only(
        y=np.asarray(mono, dtype=np.float32),
        sr=sr,
        low_hz=music_low_hz,
        high_hz=music_high_hz,
    )

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    usable = (len(mono_music) // frame_len) * frame_len
    if usable <= 0:
        return arr, 0.0, 0.0

    framed = mono_music[:usable].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(np.square(framed), axis=1, dtype=np.float64))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))

    # Smooth over a longer window than a click/pop so we only react to sustained
    # music energy. This follows the same spirit as leading-silence detectors
    # that scan in chunks rather than reacting to single frames.
    frame_s = frame_ms / 1000.0
    smooth_frames = max(1, int(round(0.18 / frame_s)))
    smooth_kernel = np.ones(smooth_frames, dtype=np.float64) / float(smooth_frames)
    rms_smooth = np.convolve(rms, smooth_kernel, mode="same")
    rms_db_smooth = 20.0 * np.log10(np.maximum(rms_smooth, 1e-9))

    # Use a stricter effective threshold near the file edges by comparing both
    # to the absolute threshold and to the measured noise floor.
    noise_floor_db = float(np.percentile(rms_db_smooth, 20.0))
    peak_db = float(np.max(rms_db_smooth))
    dynamic_floor_db = noise_floor_db + 8.0
    relative_floor_db = peak_db - 42.0
    effective_threshold_db = max(threshold_db, dynamic_floor_db, relative_floor_db)
    active = rms_db_smooth >= effective_threshold_db

    min_active_run_frames = max(1, int(round(max(0.08, min_active_run_s) / frame_s)))

    # Require dense activity over a window, not just contiguous spikes. This is
    # more resilient to recurring vinyl crackle during long lead-in/out grooves.
    occupancy_frames = max(min_active_run_frames, int(round(0.25 / frame_s)))
    occupancy_kernel = np.ones(occupancy_frames, dtype=np.float64) / float(
        occupancy_frames
    )
    occupancy = np.convolve(active.astype(np.float64), occupancy_kernel, mode="same")
    sustained_active = occupancy >= 0.58

    # Allow very short gaps inside a musical section so soft attacks are not
    # split apart by a few quiet frames.
    bridge_gap_frames = max(1, int(round(0.08 / frame_s)))
    if bridge_gap_frames > 1 and len(sustained_active) > 2:
        bridged = sustained_active.copy()
        i = 0
        while i < len(bridged):
            if not bridged[i]:
                gap_start = i
                while i < len(bridged) and not bridged[i]:
                    i += 1
                gap_end = i - 1
                gap_len = gap_end - gap_start + 1
                left_on = gap_start > 0 and bridged[gap_start - 1]
                right_on = i < len(bridged) and bridged[i]
                if left_on and right_on and gap_len <= bridge_gap_frames:
                    bridged[gap_start : gap_end + 1] = True
            else:
                i += 1
        sustained_active = bridged

    runs: list[tuple[int, int]] = []
    start = None
    for i, is_active in enumerate(sustained_active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(sustained_active) - 1))

    long_runs = [run for run in runs if (run[1] - run[0] + 1) >= min_active_run_frames]
    if not long_runs:
        return arr, 0.0, 0.0

    first, _ = long_runs[0]
    _, last = long_runs[-1]

    # Compensate for smoothing/occupancy detection lag so transient attacks/decays
    # are preserved and not shaved off.
    detection_lag_frames = (smooth_frames // 2) + (occupancy_frames // 2)
    first = max(0, first - detection_lag_frames)
    last = min(len(sustained_active) - 1, last + detection_lag_frames)

    # Keep a small context (pydub-like keep_silence behavior) around music.
    pad_frames = max(1, int(round(0.14 / frame_s)))
    first = max(0, first - pad_frames)
    last = min(len(sustained_active) - 1, last + pad_frames)

    start_i = max(0, first * frame_len)
    end_i = min(n_samples, (last + 1) * frame_len)
    if end_i - start_i <= 0:
        return arr, 0.0, 0.0

    if arr.ndim == 1:
        trimmed = arr[start_i:end_i]
    else:
        trimmed = arr[start_i:end_i, :]

    start_trim_s = float(start_i) / float(sr)
    end_trim_s = float(max(0, n_samples - end_i)) / float(sr)
    return trimmed, start_trim_s, end_trim_s


def load_audio(file_path: Path, mono: bool = False) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path=str(file_path), sr=None, mono=mono)

    # librosa returns multi-channel audio as (channels, samples).
    # Normalize to (samples, channels) so duration and slicing are consistent.
    if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T

    return y, sr


def split_audio_file(
    file_path: Path,
    output_dir: Path,
    cfg: SegmentationConfig,
) -> SplitOutput:
    audio, sr = load_audio(file_path)
    audio, start_trim_s, end_trim_s = _trim_input_edge_silence(
        audio=audio,
        sr=sr,
        threshold_db=cfg.trim_silence_db_threshold,
        min_active_run_s=cfg.input_trim_min_active_s,
        music_low_hz=cfg.music_low_hz,
        music_high_hz=cfg.music_high_hz,
    )
    if start_trim_s > 0.0 or end_trim_s > 0.0:
        print(
            "Input edge trim applied "
            f"(start={start_trim_s:.2f}s, end={end_trim_s:.2f}s, "
            f"threshold={cfg.trim_silence_db_threshold:.1f}dB, "
            f"min_active={cfg.input_trim_min_active_s:.2f}s)."
        )

    duration_s = float(audio.shape[0]) / float(sr)
    export_info = _discogs_export_info_from_filename_stem(file_path.stem)
    expected_count = export_info.expected_count
    target_count = expected_count or _estimate_target_count(
        duration_s=duration_s, cfg=cfg
    )
    if not expected_count:
        print(
            "Discogs target unavailable. "
            f"Falling back to silence-only estimated target_count={target_count}."
        )

    seg = _detect_boundaries_with_target_count(
        audio=audio,
        sr=sr,
        cfg=cfg,
        target_count=target_count,
    )

    base_name = build_export_base_name(file_path.stem)
    track_metadata = _build_track_metadata(seg=seg, export_info=export_info)
    export_settings = infer_export_settings(file_path)
    files = export_tracks_to_wav(
        audio=audio,
        sr=sr,
        boundaries_s=seg.boundaries_s,
        output_dir=output_dir,
        base_name=base_name,
        track_metadata=track_metadata,
        export_settings=export_settings,
        trim_silence_db_threshold=cfg.trim_silence_db_threshold,
    )
    return SplitOutput(
        segmentation=seg,
        files=files,
        discogs_tracks=export_info.display_tracks,
        track_metadata=track_metadata,
    )
