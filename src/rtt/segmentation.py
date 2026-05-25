from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

@dataclass
class SegmentationConfig:
    frame_length: int = 4096
    hop_length: int = 1024
    min_track_len_s: float = 40.0
    max_track_len_s: float = 600.0
    silence_db_threshold: float = -36.0
    silence_min_len_s: float = 1.2
    novelty_peak_distance_s: float = 8.0
    sensitivity: float = 0.55
    weight_silence: float = 0.35
    weight_bpm_change: float = 0.6
    weight_tonality_change: float = 0.2
    weight_spectral_novelty: float = 0.25
    # music_low_hz: float = 60.0
    # music_high_hz: float = 2500.0
    music_low_hz: float = 200.0
    music_high_hz: float = 400.0
    trim_silence_db_threshold: float = -52.0
    input_trim_min_active_s: float = 0.10


@dataclass
class BoundaryCandidate:
    time_s: float
    score: float
    reasons: Dict[str, float]


@dataclass
class SegmentationResult:
    boundaries_s: List[float]
    candidates: List[BoundaryCandidate]
    duration_s: float
    diagnostics: Dict[str, np.ndarray]


@dataclass
class PrecomputedSegmentation:
    duration_s: float
    rms_db: np.ndarray
    times: np.ndarray
    combo_score: np.ndarray
    bpm_score: np.ndarray
    tonality_score: np.ndarray
    novelty_score: np.ndarray
    silence_score: np.ndarray
    silence_idx: np.ndarray


def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or np.isclose(mx, mn):
        return np.zeros_like(x)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 2:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _windowed_feature_deltas(feature: np.ndarray, win_frames: int) -> np.ndarray:
    n_bins, n_frames = feature.shape
    if n_frames < 2:
        return np.zeros(n_frames, dtype=float)
    win_frames = max(1, win_frames)
    out = np.zeros(n_frames, dtype=float)
    for idx in range(win_frames, n_frames - win_frames):
        left = np.mean(feature[:, idx - win_frames : idx], axis=1)
        right = np.mean(feature[:, idx : idx + win_frames], axis=1)
        left_norm = np.linalg.norm(left) + 1e-9
        right_norm = np.linalg.norm(right) + 1e-9
        sim = float(np.dot(left, right) / (left_norm * right_norm))
        out[idx] = 1.0 - np.clip(sim, -1.0, 1.0)
    return out


def _silence_candidates(
    rms_db: np.ndarray,
    times: np.ndarray,
    threshold_db: float,
    min_silence_len_s: float,
) -> Tuple[np.ndarray, List[int]]:
    low = rms_db < threshold_db
    candidates_idx: List[int] = []
    n = len(low)
    i = 0
    while i < n:
        if low[i]:
            start = i
            while i < n and low[i]:
                i += 1
            end = i - 1
            if end >= start:
                dur = times[end] - times[start]
                if dur >= min_silence_len_s:
                    mid = (start + end) // 2
                    candidates_idx.append(mid)
        i += 1

    score = np.zeros_like(rms_db, dtype=float)
    if candidates_idx:
        score[candidates_idx] = 1.0
    return score, candidates_idx


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return np.asarray(audio)


def _bandpass_music_only(
    y: np.ndarray, sr: int, low_hz: float, high_hz: float
) -> np.ndarray:
    nyquist = max(1.0, sr / 2.0)
    low = max(20.0, min(low_hz, nyquist * 0.9))
    high = max(low + 20.0, min(high_hz, nyquist * 0.98))

    if high <= low:
        return y

    try:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        return sosfiltfilt(sos, y)
    except Exception:
        # Filtering should improve robustness, but must never block splitting.
        return y


def _silence_windows_from_rms(
    rms_db: np.ndarray,
    times: np.ndarray,
    threshold_db: float,
    min_silence_len_s: float,
) -> List[Tuple[int, int, int, float, float]]:
    low = rms_db < threshold_db
    windows: List[Tuple[int, int, int, float, float]] = []
    n = len(low)
    i = 0
    while i < n:
        if low[i]:
            start = i
            while i < n and low[i]:
                i += 1
            end = i - 1
            if end >= start:
                dur = float(times[end] - times[start])
                if dur >= min_silence_len_s:
                    local = rms_db[start : end + 1]
                    min_rel = int(np.argmin(local))
                    min_idx = start + min_rel
                    min_db = float(rms_db[min_idx])
                    windows.append((start, end, min_idx, min_db, dur))
        i += 1
    return windows


def _uniform_boundaries(duration_s: float, target_count: int) -> List[float]:
    if target_count <= 0:
        return [0.0, round(float(duration_s), 3)]
    step = duration_s / float(target_count)
    boundaries = [round(i * step, 3) for i in range(target_count)]
    boundaries.append(round(float(duration_s), 3))
    boundaries[0] = 0.0
    return boundaries


def detect_boundaries_by_silence_target(
    audio: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
    target_count: int,
) -> SegmentationResult:
    y = np.asarray(_to_mono(audio), dtype=np.float32)
    duration_s = len(y) / float(sr)
    target_count = max(1, int(target_count))

    y_music = _bandpass_music_only(
        y=y,
        sr=sr,
        low_hz=cfg.music_low_hz,
        high_hz=cfg.music_high_hz,
    )

    rms = librosa.feature.rms(
        y=y_music,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    times = librosa.frames_to_time(
        np.arange(len(rms_db)), sr=sr, hop_length=cfg.hop_length
    )

    windows = _silence_windows_from_rms(
        rms_db=rms_db,
        times=times,
        threshold_db=cfg.silence_db_threshold,
        min_silence_len_s=cfg.silence_min_len_s,
    )

    candidates: List[BoundaryCandidate] = []
    for start, end, min_idx, min_db, dur in windows:
        t = float(times[min_idx])
        depth = max(0.0, cfg.silence_db_threshold - min_db)
        # Keep a scalar score for diagnostics, but selection prioritizes duration.
        score = float(depth * max(dur, 0.1))
        candidates.append(
            BoundaryCandidate(
                time_s=t,
                score=score,
                reasons={
                    "silence": score,
                    "window_duration_s": dur,
                    "min_rms_db": min_db,
                    "silence_depth_db": depth,
                },
            )
        )

    wanted_cuts = max(0, target_count - 1)
    min_track = max(5.0, cfg.min_track_len_s)
    selected: List[BoundaryCandidate] = []

    # Biggest silence windows first, then deepest among equals.
    for cand in sorted(
        candidates,
        key=lambda c: (
            float(c.reasons.get("window_duration_s", 0.0)),
            float(c.reasons.get("silence_depth_db", 0.0)),
        ),
        reverse=True,
    ):
        if len(selected) >= wanted_cuts:
            break
        if cand.time_s < min_track or duration_s - cand.time_s < min_track:
            continue
        if any(abs(cand.time_s - s.time_s) < min_track for s in selected):
            continue
        selected.append(cand)

    boundaries = (
        [0.0] + sorted(round(c.time_s, 3) for c in selected) + [round(duration_s, 3)]
    )
    boundaries = sorted(set(boundaries))

    if len(boundaries) - 1 < target_count:
        boundaries = _uniform_boundaries(
            duration_s=duration_s, target_count=target_count
        )

    diagnostics = {
        "time_s": times,
        "rms_db": rms_db,
        "silence_score": np.array([c.score for c in candidates], dtype=float),
    }
    return SegmentationResult(
        boundaries_s=boundaries,
        candidates=sorted(candidates, key=lambda c: c.time_s),
        duration_s=float(duration_s),
        diagnostics=diagnostics,
    )


def _boundaries_from_candidates(
    candidates: List[BoundaryCandidate],
    duration_s: float,
    min_track_len_s: float,
    max_track_len_s: float,
) -> List[float]:
    boundaries = [0.0]
    last = 0.0
    max_track = max(min_track_len_s + 1.0, max_track_len_s)
    min_track = max(5.0, min_track_len_s)

    for cand in sorted(candidates, key=lambda c: c.time_s):
        if cand.time_s - last < min_track:
            continue
        if cand.time_s - last > max_track:
            forced = min(last + max_track, duration_s)
            if forced - last >= min_track:
                boundaries.append(float(forced))
                last = float(forced)
        if cand.time_s - last >= min_track:
            boundaries.append(cand.time_s)
            last = cand.time_s

    if duration_s - last >= min_track * 0.5:
        boundaries.append(duration_s)
    elif boundaries[-1] < duration_s:
        boundaries[-1] = duration_s

    boundaries = sorted(set(round(v, 3) for v in boundaries))
    if boundaries[0] != 0.0:
        boundaries = [0.0] + boundaries
    if boundaries[-1] != round(duration_s, 3):
        boundaries.append(round(duration_s, 3))
    return boundaries


def precompute_segmentation(
    audio: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
) -> PrecomputedSegmentation:
    if audio.ndim > 1:
        y = np.mean(audio, axis=1)
    else:
        y = audio

    y = np.asarray(y, dtype=np.float32)
    duration_s = len(y) / float(sr)

    rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=cfg.hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
    onset_env = _rolling_mean(onset_env, 5)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=cfg.hop_length)
    tonality_change = _windowed_feature_deltas(chroma, win_frames=max(4, int(sr / cfg.hop_length * 3.0)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=cfg.hop_length)
    mfcc_delta = _windowed_feature_deltas(mfcc, win_frames=max(3, int(sr / cfg.hop_length * 2.0)))
    spec_novelty = _normalize_0_1(0.45 * onset_env + 0.55 * mfcc_delta)

    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=cfg.hop_length,
        aggregate=None,
    )
    if len(tempo_curve) != len(times):
        tempo_curve = np.interp(
            np.arange(len(times)),
            np.linspace(0, max(1, len(times) - 1), num=len(tempo_curve)),
            tempo_curve,
        )
    tempo_grad = np.abs(np.gradient(tempo_curve))

    silence_score, silence_idx = _silence_candidates(
        rms_db=rms_db,
        times=times,
        threshold_db=cfg.silence_db_threshold,
        min_silence_len_s=cfg.silence_min_len_s,
    )

    bpm_score = _normalize_0_1(tempo_grad)
    tonality_score = _normalize_0_1(tonality_change)
    novelty_score = _normalize_0_1(spec_novelty)

    raw_combo = (
        cfg.weight_silence * silence_score
        + cfg.weight_bpm_change * bpm_score
        + cfg.weight_tonality_change * tonality_score
        + cfg.weight_spectral_novelty * novelty_score
    )
    combo_score = _normalize_0_1(_rolling_mean(raw_combo, 3))

    return PrecomputedSegmentation(
        duration_s=float(duration_s),
        rms_db=rms_db,
        times=times,
        combo_score=combo_score,
        bpm_score=bpm_score,
        tonality_score=tonality_score,
        novelty_score=novelty_score,
        silence_score=silence_score,
        silence_idx=np.asarray(silence_idx, dtype=int),
    )


def detect_boundaries_with_precomputed(
    precomputed: PrecomputedSegmentation,
    sr: int,
    cfg: SegmentationConfig,
) -> SegmentationResult:
    min_peak_height = np.clip(0.78 - cfg.sensitivity * 0.55, 0.12, 0.72)
    distance_frames = max(1, int(cfg.novelty_peak_distance_s * sr / cfg.hop_length))
    peak_idx, _ = find_peaks(
        precomputed.combo_score,
        height=min_peak_height,
        distance=distance_frames,
    )

    if precomputed.silence_idx.size:
        peak_idx = np.append(peak_idx, precomputed.silence_idx)
    peak_idx = np.unique(np.sort(peak_idx))

    candidates: List[BoundaryCandidate] = []
    for idx in peak_idx:
        t = float(precomputed.times[idx])
        reasons = {
            "silence": float(precomputed.silence_score[idx]),
            "bpm_change": float(precomputed.bpm_score[idx]),
            "tonality_change": float(precomputed.tonality_score[idx]),
            "spectral_novelty": float(precomputed.novelty_score[idx]),
        }
        candidates.append(
            BoundaryCandidate(
                time_s=t, score=float(precomputed.combo_score[idx]), reasons=reasons
            )
        )

    boundaries = _boundaries_from_candidates(
        candidates=candidates,
        duration_s=precomputed.duration_s,
        min_track_len_s=cfg.min_track_len_s,
        max_track_len_s=cfg.max_track_len_s,
    )

    diagnostics = {
        "time_s": precomputed.times,
        "rms_db": precomputed.rms_db,
        "combo_score": precomputed.combo_score,
        "bpm_score": precomputed.bpm_score,
        "tonality_score": precomputed.tonality_score,
        "novelty_score": precomputed.novelty_score,
        "silence_score": precomputed.silence_score,
    }
    return SegmentationResult(
        boundaries_s=boundaries,
        candidates=candidates,
        duration_s=precomputed.duration_s,
        diagnostics=diagnostics,
    )


def detect_boundaries(
    audio: np.ndarray, sr: int, cfg: SegmentationConfig
) -> SegmentationResult:
    precomputed = precompute_segmentation(audio=audio, sr=sr, cfg=cfg)
    return detect_boundaries_with_precomputed(precomputed=precomputed, sr=sr, cfg=cfg)
