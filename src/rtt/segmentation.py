from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks


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
    weight_bpm_change: float = 0.2
    weight_tonality_change: float = 0.2
    weight_spectral_novelty: float = 0.25


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


def detect_boundaries(audio: np.ndarray, sr: int, cfg: SegmentationConfig) -> SegmentationResult:
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

    min_peak_height = np.clip(0.78 - cfg.sensitivity * 0.55, 0.12, 0.72)
    distance_frames = max(1, int(cfg.novelty_peak_distance_s * sr / cfg.hop_length))
    peak_idx, _ = find_peaks(combo_score, height=min_peak_height, distance=distance_frames)

    for idx in silence_idx:
        if 0 <= idx < len(combo_score):
            peak_idx = np.append(peak_idx, idx)
    peak_idx = np.unique(np.sort(peak_idx))

    candidates: List[BoundaryCandidate] = []
    for idx in peak_idx:
        t = float(times[idx])
        reasons = {
            "silence": float(silence_score[idx]),
            "bpm_change": float(bpm_score[idx]),
            "tonality_change": float(tonality_score[idx]),
            "spectral_novelty": float(novelty_score[idx]),
        }
        candidates.append(BoundaryCandidate(time_s=t, score=float(combo_score[idx]), reasons=reasons))

    boundaries = [0.0]
    last = 0.0
    max_track = max(cfg.min_track_len_s + 1.0, cfg.max_track_len_s)
    min_track = max(5.0, cfg.min_track_len_s)

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

    diagnostics = {
        "time_s": times,
        "rms_db": rms_db,
        "combo_score": combo_score,
        "bpm_score": bpm_score,
        "tonality_score": tonality_score,
        "novelty_score": novelty_score,
        "silence_score": silence_score,
    }
    return SegmentationResult(
        boundaries_s=boundaries,
        candidates=candidates,
        duration_s=float(duration_s),
        diagnostics=diagnostics,
    )
