from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import split_audio_file
from .segmentation import SegmentationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a long MP3 recording into tracks.")
    parser.add_argument("input", type=Path, help="Input audio file path (.mp3 supported if backend decoder is available).")
    parser.add_argument("--output", type=Path, default=Path("output_tracks"), help="Output directory.")
    parser.add_argument("--min-track", type=float, default=40.0, help="Minimum track length in seconds.")
    parser.add_argument("--max-track", type=float, default=600.0, help="Maximum track length in seconds.")
    parser.add_argument("--silence-db", type=float, default=-36.0, help="Silence threshold in dB.")
    parser.add_argument("--sensitivity", type=float, default=0.55, help="Detection sensitivity 0..1.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = SegmentationConfig(
        min_track_len_s=args.min_track,
        max_track_len_s=args.max_track,
        silence_db_threshold=args.silence_db,
        sensitivity=args.sensitivity,
    )

    result = split_audio_file(file_path=args.input, output_dir=args.output, cfg=cfg)

    print(f"Detected boundaries: {result.segmentation.boundaries_s}")
    print(f"Exported {len(result.files)} track files")
    print(f"Zip bundle: {result.zip_path}")


if __name__ == "__main__":
    main()
