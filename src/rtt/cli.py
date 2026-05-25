from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import split_audio_file
from .segmentation import SegmentationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a long MP3 recording into tracks.")
    parser.add_argument("input", type=Path, help="Input audio file path (.mp3 supported if backend decoder is available).")
    parser.add_argument("--output", type=Path, default=Path("output_tracks"), help="Output directory.")
    parser.add_argument(
        "--min-track", type=float, default=40.0, help="Minimum track length in seconds."
    )
    parser.add_argument("--silence-db", type=float, default=-36.0, help="Silence threshold in dB.")
    parser.add_argument(
        "--silence-min",
        type=float,
        default=1.2,
        help="Minimum silence window in seconds.",
    )
    parser.add_argument(
        "--music-low-hz",
        type=float,
        default=120.0,
        help="Low frequency bound for music-only silence analysis.",
    )
    parser.add_argument(
        "--music-high-hz",
        type=float,
        default=5000.0,
        help="High frequency bound for music-only silence analysis.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = SegmentationConfig(
        min_track_len_s=args.min_track,
        silence_db_threshold=args.silence_db,
        silence_min_len_s=args.silence_min,
        music_low_hz=args.music_low_hz,
        music_high_hz=args.music_high_hz,
    )

    result = split_audio_file(file_path=args.input, output_dir=args.output, cfg=cfg)

    print(f"Detected boundaries: {result.segmentation.boundaries_s}")
    print(f"Exported {len(result.files)} track files")
    print(f"Zip bundle: {result.zip_path}")
    if result.discogs_tracks:
        print("Discogs tracks (side - title):")
        for entry in result.discogs_tracks:
            print(f"  - {entry}")


if __name__ == "__main__":
    main()
