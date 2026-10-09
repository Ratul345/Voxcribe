"""
Voxcribe - Transcription engine

Wraps faster-whisper's WhisperModel for model loading and transcription.
Returns structured results with per-segment timestamps.
Supports callbacks for progress tracking and UI streaming.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from voxcribe.utils import format_duration, format_timestamp, get_language_name


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single transcribed segment with timestamps."""
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Complete transcription output."""
    source: str
    language: str
    language_probability: float
    duration: float
    segments: List[Segment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Valid model names
# ---------------------------------------------------------------------------

VALID_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v2", "large-v1", "turbo"]


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def load_and_transcribe(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
    task: str = "transcribe",
    verbose: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    segment_callback: Optional[Callable[[Segment, int, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> TranscriptionResult:
    """
    Load a faster-whisper model and transcribe the given audio/video file.

    Args:
        audio_path:        Path to the media file.
        model_size:        Whisper model name (tiny, base, small, medium, large-v3, turbo).
        language:          Language code (e.g. "bn", "en") or None for auto-detect.
        device:            "cpu" or "cuda".
        compute_type:      "int8", "float16", "float32", etc.
        vad_filter:        Enable Silero VAD to skip silent regions.
        task:              "transcribe" (default) or "translate".
        verbose:           Print extra diagnostics.
        progress_callback: Optional fn(progress_0_to_1, status_text)
        segment_callback:  Optional fn(segment, segment_index, total_duration)
        status_callback:   Optional fn(status_text)
        should_cancel:     Optional fn() -> bool to abort processing

    Returns:
        TranscriptionResult with segments, detected language, and duration.
    """
    def log_status(msg: str):
        print(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Import faster-whisper (fail gracefully)
    # ------------------------------------------------------------------
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        err = "faster-whisper is not installed. Please install it with: pip install faster-whisper"
        log_status(f"❌ {err}")
        raise RuntimeError(err)

    # ------------------------------------------------------------------
    # Validate device choice
    # ------------------------------------------------------------------
    if device == "cuda":
        try:
            import ctranslate2
            if hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("No CUDA-compatible GPU detected on your system. Please switch to CPU.")
        except Exception as e:
            if "No CUDA" in str(e):
                raise

    if should_cancel and should_cancel():
        raise InterruptedError("Transcription cancelled by user.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log_status(f"🔄 Loading Whisper model '{model_size}' on {device.upper()} ({compute_type})...")
    if progress_callback:
        progress_callback(0.05, f"Loading model '{model_size}'...")

    load_start = time.time()
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    except Exception as e:
        err_msg = str(e)
        if "cuda" in err_msg.lower() or "cublas" in err_msg.lower() or "cudnn" in err_msg.lower():
            raise RuntimeError(f"CUDA initialization error: {err_msg}. Try running on CPU with int8.")
        elif "memory" in err_msg.lower() or "oom" in err_msg.lower():
            raise MemoryError(f"Not enough memory for model '{model_size}'. Try 'tiny', 'base', or 'small'.")
        raise RuntimeError(f"Error loading model '{model_size}': {err_msg}")

    load_time = time.time() - load_start
    log_status(f"✅ Model loaded in {load_time:.1f}s.")

    if should_cancel and should_cancel():
        raise InterruptedError("Transcription cancelled by user.")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    log_status("🎤 Transcribing media file...")
    if progress_callback:
        progress_callback(0.12, "Starting transcription...")

    transcribe_start = time.time()

    transcribe_language = None if (language is None or language.lower() in ("auto", "")) else language
    vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=200) if vad_filter else None

    try:
        segments_gen, info = model.transcribe(
            audio_path,
            language=transcribe_language,
            task=task,
            beam_size=5,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            condition_on_previous_text=True,
            repetition_penalty=1.05,
        )
    except Exception as e:
        raise RuntimeError(f"Transcription execution failed: {e}")

    detected_lang = info.language or "unknown"
    detected_prob = info.language_probability or 0.0
    duration = info.duration or 0.0
    lang_display = get_language_name(detected_lang)

    log_status(f"🌐 Detected language: {lang_display} [{detected_lang}] (confidence: {detected_prob:.0%})")
    log_status(f"⏱️ Audio duration: {format_duration(duration)}")

    collected_segments: List[Segment] = []
    segment_count = 0

    for seg in segments_gen:
        if should_cancel and should_cancel():
            log_status("⏹️ Transcription stopped by user.")
            raise InterruptedError("Transcription cancelled by user.")

        text = seg.text.strip()
        if not text:
            continue

        segment_data = Segment(
            start=seg.start,
            end=seg.end,
            text=text,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
        )
        collected_segments.append(segment_data)
        segment_count += 1

        ts = format_timestamp(seg.start)
        preview = text if len(text) <= 70 else text[:67] + "..."
        if verbose:
            print(f"   [{ts}] {preview}")

        # Calculate progress
        if duration > 0:
            # Scale from 15% to 95%
            pct = 0.15 + (min(seg.end / duration, 1.0) * 0.80)
            if progress_callback:
                progress_callback(pct, f"Transcribing [{ts} / {format_duration(duration)}]")

        if segment_callback:
            try:
                segment_callback(segment_data, segment_count, duration)
            except Exception:
                pass

    transcribe_time = time.time() - transcribe_start
    log_status(f"✅ Completed in {transcribe_time:.1f}s ({segment_count} segments)")
    if progress_callback:
        progress_callback(1.0, "Transcription completed successfully.")

    return TranscriptionResult(
        source=audio_path,
        language=detected_lang,
        language_probability=detected_prob,
        duration=duration,
        segments=collected_segments,
    )


def _cuda_unavailable_error(detail: str = ""):
    """Print a clear, user-friendly error when CUDA is requested but cannot be used."""
    print("\n❌ CUDA acceleration is not available.")
    if detail:
        print(f"   Details: {detail}")
    print("\n💡 Recommendation: Run on CPU (default):")
    print("   python transcribe.py <file> --device cpu --compute-type int8\n")
    sys.exit(1)
