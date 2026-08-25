"""
Voxcribe - Utility functions

FFmpeg setup, input file validation, timestamp/duration formatting,
and shared helpers.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Supported media formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".webm",
    ".mpeg", ".mpga", ".avi", ".mov", ".mkv",
}


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def find_ffmpeg() -> Optional[str]:
    """
    Locate an FFmpeg executable.

    Search order:
      1. System PATH (already installed by the user).
      2. imageio-ffmpeg bundled binary (Windows convenience fallback).

    Returns:
        Absolute path to ffmpeg, or None if not found.
    """
    # 1. Check system PATH
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return ffmpeg_on_path

    # 2. Try imageio-ffmpeg (Windows fallback)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception):
        pass

    return None


def setup_ffmpeg(verbose: bool = False) -> Optional[str]:
    """
    Ensure FFmpeg is available. On Windows, adds the imageio-ffmpeg
    binary directory to PATH so that faster-whisper (and its underlying
    libraries) can find it.

    Args:
        verbose: Print extra detail about what was found.

    Returns:
        Path to ffmpeg executable, or None if unavailable.
    """
    ffmpeg_path = find_ffmpeg()

    if ffmpeg_path is None:
        return None

    # Make sure the directory containing ffmpeg is on PATH so that
    # child processes (CTranslate2, av, etc.) can also find it.
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

    if verbose:
        print(f"  FFmpeg: {ffmpeg_path}")

    return ffmpeg_path


# ---------------------------------------------------------------------------
# Input file validation
# ---------------------------------------------------------------------------

def validate_input_file(file_path: str, verbose: bool = False) -> bool:
    """
    Validate that the input file exists, has a supported extension,
    and is a readable media file.

    Args:
        file_path: Path to the media file.
        verbose:   Show additional diagnostics.

    Returns:
        True if the file looks valid, False otherwise (with a printed error).
    """
    path = Path(file_path)

    # --- existence ---
    if not path.exists():
        print(f"❌ Error: File '{file_path}' was not found.")
        return False

    # --- is a file (not a directory) ---
    if not path.is_file():
        print(f"❌ Error: '{file_path}' is not a file.")
        return False

    # --- readable ---
    if not os.access(path, os.R_OK):
        print(f"❌ Error: No permission to read '{file_path}'.")
        return False

    # --- extension check ---
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"❌ Unsupported format '{ext}'.")
        print(f"📋 Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return False

    # --- lightweight media probe (if ffmpeg is available) ---
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        if not _probe_media(file_path, ffmpeg, verbose):
            return False

    return True


def _probe_media(file_path: str, ffmpeg_path: str, verbose: bool = False) -> bool:
    """
    Use ffprobe (or ffmpeg -i) to verify the file contains a valid
    audio or video stream.

    Returns True if the probe succeeds, False with a printed error otherwise.
    """
    # Try ffprobe first (comes with FFmpeg distributions)
    ffprobe = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
    if sys.platform == "win32":
        ffprobe += ".exe"

    prober = ffprobe if shutil.which(ffprobe) or os.path.isfile(ffprobe) else None

    try:
        if prober:
            cmd = [
                prober,
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                file_path,
            ]
        else:
            # Fall back to `ffmpeg -i` which prints stream info to stderr
            cmd = [ffmpeg_path, "-v", "error", "-i", file_path, "-f", "null", "-t", "0.1", "-"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
        )

        if result.returncode != 0:
            stderr_text = result.stderr.decode(errors="replace").strip()
            print(f"❌ Cannot read media file. It may be corrupted or not a valid media file.")
            if verbose and stderr_text:
                print(f"  FFmpeg details: {stderr_text[:300]}")
            return False

    except FileNotFoundError:
        # ffprobe/ffmpeg binary vanished — skip probe silently
        pass
    except subprocess.TimeoutExpired:
        print("⚠️  Media probe timed out. Proceeding anyway.")
    except Exception as e:
        if verbose:
            print(f"⚠️  Media probe error: {e}")

    return True


# ---------------------------------------------------------------------------
# Timestamp / duration formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to [HH:MM:SS] display format.

    Example: 185.3 → "00:03:05"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Example: 185.35 → "00:03:05,350"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_duration(seconds: float) -> str:
    """
    Human-readable duration string.

    Examples:
        45.0   → "0:45"
        125.0  → "2:05"
        3661.0 → "1:01:01"
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Language display mapping
# ---------------------------------------------------------------------------

LANGUAGE_NAMES = {
    "bn": "Bengali (বাংলা)",
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "ar": "Arabic (العربية)",
    "ru": "Russian (Русский)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "ko": "Korean (한국어)",
    "tr": "Turkish (Türkçe)",
    "ur": "Urdu (اردو)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ml": "Malayalam (മലയാളം)",
    "kn": "Kannada (ಕನ್ನಡ)",
}


def get_language_name(code: str) -> str:
    """
    Resolve an ISO 639-1 language code to a user-friendly display name.
    """
    if not code:
        return "Unknown"
    code_lower = code.lower()
    return LANGUAGE_NAMES.get(code_lower, code.upper())

