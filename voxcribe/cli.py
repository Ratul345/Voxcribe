"""
Voxcribe - Command-line interface

Argument parsing, orchestration, and user-facing output.
Preserves backward-compatible CLI from Voxcribe v1.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows (cp1252 cannot display emoji/Unicode)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from voxcribe import __version__
from voxcribe.transcriber import VALID_MODELS, load_and_transcribe
from voxcribe.exporters import VALID_FORMATS, export_all
from voxcribe.utils import (
    setup_ffmpeg,
    validate_input_file,
    format_duration,
    SUPPORTED_EXTENSIONS,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Preserves original short flags: -m, -o, -v
    Adds new options for language, format, device, compute type, VAD.
    """
    parser = argparse.ArgumentParser(
        prog="voxcribe",
        description=(
            "Voxcribe — Local AI Transcription Tool\n"
            "Powered by faster-whisper (CTranslate2)"
        ),
        epilog=(
            "Examples:\n"
            "  voxcribe video.mp4\n"
            "  voxcribe lecture.mp3 --model small --language auto\n"
            "  voxcribe audio.wav --language bn --format txt,srt,json\n"
            "  voxcribe video.mp4 --device cuda --compute-type float16\n"
            "\n"
            "Backward-compatible alias:\n"
            "  python transcribe.py video.mp4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Positional ----
    parser.add_argument(
        "input_file",
        help="Path to the audio or video file",
    )

    # ---- Model ----
    parser.add_argument(
        "-m", "--model",
        choices=VALID_MODELS,
        default="small",
        help="Whisper model size (default: small)",
    )

    # ---- Language ----
    parser.add_argument(
        "-l", "--language",
        default="auto",
        help=(
            "Language code for transcription. "
            "Use 'auto' for automatic detection (default). "
            "Examples: bn, en, hi, es, fr, de, ja, zh"
        ),
    )

    # ---- Output ----
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output path or base name. "
            "Default: output/<input_name>"
        ),
    )

    # ---- Format ----
    parser.add_argument(
        "-f", "--format",
        default="txt",
        help=(
            "Output formats, comma-separated. "
            f"Valid: {', '.join(VALID_FORMATS)} "
            "(default: txt). "
            "Example: --format txt,srt,json"
        ),
    )

    # ---- Device ----
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device (default: cpu)",
    )

    # ---- Compute type ----
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Model precision: int8 (CPU default), float16 (GPU), float32. Default: int8",
    )

    # ---- VAD ----
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument(
        "--vad",
        action="store_true",
        default=True,
        help="Enable Voice Activity Detection (default: enabled)",
    )
    vad_group.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection",
    )

    # ---- Task ----
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: 'transcribe' (default) or 'translate' (translate to English)",
    )

    # ---- Verbose ----
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output",
    )

    # ---- Debug ----
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed diagnostic output with quality metrics",
    )

    # ---- Version ----
    parser.add_argument(
        "--version",
        action="version",
        version=f"Voxcribe v{__version__}",
    )

    return parser


def main() -> None:
    """
    Main entry point for Voxcribe CLI.

    Flow:
      1. Parse arguments
      2. Setup FFmpeg
      3. Validate input file
      4. Load model & transcribe
      5. Export to requested formats
      6. Print summary
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    verbose = args.verbose or args.debug  # Enable verbose if debug is on
    vad_enabled = not args.no_vad

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print()
    print("═" * 56)
    print("  🎤 Voxcribe — Local AI Transcription Tool")
    print("═" * 56)
    print()

    # ------------------------------------------------------------------
    # 1. FFmpeg setup
    # ------------------------------------------------------------------
    ffmpeg_path = setup_ffmpeg(verbose=verbose)
    if ffmpeg_path:
        print(f"✅ FFmpeg available")
    else:
        print("⚠️  FFmpeg not found.")
        print("   Install FFmpeg or run:  pip install imageio-ffmpeg")
        print("   Transcription may fail without FFmpeg.\n")

    # ------------------------------------------------------------------
    # 2. Validate input
    # ------------------------------------------------------------------
    if not validate_input_file(args.input_file, verbose=verbose):
        sys.exit(1)

    input_path = os.path.abspath(args.input_file)
    input_name = Path(input_path).stem

    print(f"🎬 Input: {os.path.basename(input_path)}")
    print()

    # ------------------------------------------------------------------
    # 3. Resolve language
    # ------------------------------------------------------------------
    language = None  # None = auto-detect in faster-whisper
    if args.language.lower() != "auto":
        language = args.language.lower()
        print(f"🌐 Language: {language}")
    else:
        print("🌐 Language: auto-detect")

    print()

    # ------------------------------------------------------------------
    # 4. Transcribe
    # ------------------------------------------------------------------
    result = load_and_transcribe(
        audio_path=input_path,
        model_size=args.model,
        language=language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=vad_enabled,
        task=args.task,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 5. Determine output paths
    # ------------------------------------------------------------------
    # Parse requested formats
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    if not formats:
        formats = ["txt"]

    # Validate formats
    invalid_formats = [f for f in formats if f not in VALID_FORMATS]
    if invalid_formats:
        print(f"⚠️  Unknown format(s): {', '.join(invalid_formats)}")
        print(f"   Valid formats: {', '.join(VALID_FORMATS)}")
        formats = [f for f in formats if f in VALID_FORMATS]
        if not formats:
            print("❌ No valid output formats specified.")
            sys.exit(1)

    # Determine output base path
    if args.output:
        out = Path(args.output)
        # If user gave a path with extension, use it as-is for single format
        if out.suffix and len(formats) == 1:
            output_base = str(out.with_suffix(""))
        else:
            output_base = str(out)
    else:
        # Default: output/<input_name>
        output_base = os.path.join("output", input_name)

    # ------------------------------------------------------------------
    # 6. Export
    # ------------------------------------------------------------------
    print()
    written = export_all(result, formats, output_base)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print()
    print("═" * 56)
    print("  ✅ Transcription Complete")
    print("═" * 56)
    print()
    print(f"  📁 Source:    {os.path.basename(input_path)}")
    print(f"  🤖 Model:     {args.model}")
    print(f"  🌐 Language:  {result.language} ({result.language_probability:.0%})")
    print(f"  ⏱️  Duration:  {format_duration(result.duration)}")
    print(f"  📝 Segments:  {len(result.segments)}")
    print()

    if written:
        print("  💾 Saved:")
        for p in written:
            print(f"     {p}")
    else:
        print("  ⚠️  No output files were written.")

    print()
    print("═" * 56)
    print()


if __name__ == "__main__":
    main()
