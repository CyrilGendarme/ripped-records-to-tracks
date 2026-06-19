# ripped-records-to-tracks

Split one long recording (for example an entire ripped vinyl side) into probable tracks.

The app combines multiple rules:

- Moment of silence detection
- BPM / tempo curve change detection
- Main tonality (chroma profile) change detection
- Spectral novelty detection (onset + timbral shifts)


## Prerequisites

Before running the application, make sure you complete the following setup steps:

### Configure the .env file

Create and configure the .env file with the required settings for your environment before launching the application.

### Install FFmpeg

FFmpeg is required for exporting .wav tracks correctly.

Download FFmpeg from the official website: https://ffmpeg.org/download.html
Install it on your system.
Ensure the ffmpeg executable is available in your system PATH.

You can verify the installation by running:

ffmpeg -version

If FFmpeg is correctly installed, version information will be displayed.


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
- Preferred input naming format is: `ArtistName - TrackName [record ref] - side - 33/45rpm`.
- The trailing `- 33/45rpm` segment is optional and only indicates a non-standard recording/playback speed.