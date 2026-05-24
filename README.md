# ripped-records-to-tracks

Split one long recording (for example an entire ripped vinyl side) into probable tracks.

The app combines multiple rules:

- Moment of silence detection
- BPM / tempo curve change detection
- Main tonality (chroma profile) change detection
- Spectral novelty detection (onset + timbral shifts)

## Features

- Native desktop GUI (Tkinter) for local usage
- Tune split behavior with interactive controls
- Preview detected track timeline before exporting
- Export all detected tracks as WAV files in a ZIP archive
- CLI mode for scripting

## Quick Start

1. Create and activate a Python environment (Python 3.10+).
2. Install dependencies:

	pip install -r requirements.txt

3. Run desktop app:

	python app.py

4. Pick your input file and output folder, then click "Split Into Tracks".

## CLI Usage

Example:

python -m src.rtt.cli input_mix.mp3 --output output_tracks --min-track 45 --max-track 480 --silence-db -38 --sensitivity 0.62

## Notes

- Input decoding for MP3 depends on your local audio backend. If MP3 fails to decode, install FFmpeg and retry.
- Output format is WAV for portability and robust writing support.