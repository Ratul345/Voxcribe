"""
Voxcribe - Output exporters

Write transcription results to TXT, timestamped TXT, SRT, and JSON files.
"""

import json
import os
from pathlib import Path
from typing import List

from voxcribe.transcriber import TranscriptionResult
from voxcribe.utils import format_timestamp, format_srt_timestamp


# ---------------------------------------------------------------------------
# TXT — timestamped readable text (default)
# ---------------------------------------------------------------------------

def export_txt(result: TranscriptionResult, output_path: str) -> str:
    """
    Export transcription as timestamped text.

    Format:
        [00:00:03] Hello everyone.
        [00:00:08] Today we're going to talk about...

    Returns:
        The path that was written.
    """
    _ensure_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in result.segments:
            ts = format_timestamp(seg.start)
            f.write(f"[{ts}] {seg.text}\n")

    return output_path


# ---------------------------------------------------------------------------
# Plain TXT — clean text without timestamps
# ---------------------------------------------------------------------------

def export_plain(result: TranscriptionResult, output_path: str) -> str:
    """
    Export transcription as clean readable text without timestamps.

    Each segment becomes a line, separated by blank lines for readability.
    This format is ideal for reading, studying, and pasting into AI chat.

    Returns:
        The path that was written.
    """
    _ensure_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.segments):
            f.write(seg.text)
            if i < len(result.segments) - 1:
                f.write("\n\n")
        f.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# SRT — standard subtitle format
# ---------------------------------------------------------------------------

def export_srt(result: TranscriptionResult, output_path: str) -> str:
    """
    Export transcription as an SRT subtitle file.

    Format:
        1
        00:00:03,000 --> 00:00:08,000
        Hello everyone.

    Returns:
        The path that was written.
    """
    _ensure_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.segments, start=1):
            start_ts = format_srt_timestamp(seg.start)
            end_ts = format_srt_timestamp(seg.end)
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{seg.text}\n\n")

    return output_path


# ---------------------------------------------------------------------------
# JSON — structured data
# ---------------------------------------------------------------------------

def export_json(result: TranscriptionResult, output_path: str) -> str:
    """
    Export transcription as a structured JSON file.

    Schema:
        {
            "source": "video.mp4",
            "language": "en",
            "language_probability": 0.98,
            "duration": 123.45,
            "segments": [
                {"start": 3.0, "end": 8.0, "text": "Hello everyone."},
                ...
            ]
        }

    Returns:
        The path that was written.
    """
    _ensure_dir(output_path)

    data = {
        "source": os.path.basename(result.source),
        "language": result.language,
        "language_probability": round(result.language_probability, 4),
        "duration": round(result.duration, 2),
        "segments": [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
            }
            for seg in result.segments
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps user-facing format names to (exporter_function, file_extension)
FORMAT_MAP = {
    "txt":       (export_txt, ".txt"),
    "srt":       (export_srt, ".srt"),
    "json":      (export_json, ".json"),
    "plain":     (export_plain, ".plain.txt"),
    "txt-plain": (export_plain, ".plain.txt"),
    "txt-ts":    (export_txt, ".timestamped.txt"),
}

VALID_FORMATS = list(FORMAT_MAP.keys())


def export_all(
    result: TranscriptionResult,
    formats: List[str],
    output_base: str,
) -> List[str]:
    """
    Export the transcription result to multiple formats.

    Args:
        result:      The transcription result.
        formats:     List of format names (e.g. ["txt", "srt", "json"]).
        output_base: Base path without extension (e.g. "output/video_name").

    Returns:
        List of paths that were written.
    """
    written: List[str] = []

    for fmt in formats:
        fmt_lower = fmt.strip().lower()
        if fmt_lower not in FORMAT_MAP:
            print(f"⚠️  Unknown format '{fmt}', skipping. Valid: {', '.join(VALID_FORMATS)}")
            continue

        exporter_fn, ext = FORMAT_MAP[fmt_lower]
        path = output_base + ext
        exporter_fn(result, path)
        written.append(path)

    return written


def _ensure_dir(file_path: str) -> None:
    """Create parent directories if they don't exist."""
    parent = Path(file_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
