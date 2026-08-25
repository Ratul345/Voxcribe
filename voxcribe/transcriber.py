"""
Voxcribe - Transcription engine

Wraps faster-whisper's WhisperModel for model loading and transcription.
Returns structured results with per-segment timestamps.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

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
) -> TranscriptionResult:
    """
    Load a faster-whisper model and transcribe the given audio/video file.

    Args:
        audio_path:    Path to the media file.
        model_size:    Whisper model name (tiny, base, small, medium, large-v3, turbo).
        language:      Language code (e.g. "bn", "en") or None for auto-detect.
        device:        "cpu" or "cuda".
        compute_type:  "int8", "float16", "float32", etc.
        vad_filter:    Enable Silero VAD to skip silent regions.
        task:          "transcribe" (default) or "translate".
        verbose:       Print extra diagnostics.

    Returns:
        TranscriptionResult with segments, detected language, and duration.

    Raises:
        SystemExit on unrecoverable errors (with user-friendly messages).
    """
    # ------------------------------------------------------------------
    # Import faster-whisper (fail gracefully)
    # ------------------------------------------------------------------
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ Error: faster-whisper is not installed.")
        print("   Install it with:  pip install faster-whisper")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate device choice
    # ------------------------------------------------------------------
    if device == "cuda":
        try:
            import ctranslate2
            if hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() == 0:
                _cuda_unavailable_error("No CUDA-compatible GPU detected on your system.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"🔄 Loading model '{model_size}'...")
    load_start = time.time()

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    except ValueError as e:
        err = str(e).lower()
        if "cuda" in err or "gpu" in err:
            _cuda_unavailable_error(str(e))
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    except MemoryError:
        print(f"❌ Not enough memory for model '{model_size}'.")
        print("   Try a smaller model such as 'small' or 'base'.")
        sys.exit(1)
    except Exception as e:
        err_msg = str(e)
        if "cuda" in err_msg.lower() or "cublas" in err_msg.lower() or "cudnn" in err_msg.lower():
            _cuda_unavailable_error(err_msg)
        elif "Could not load" in err_msg or "download" in err_msg.lower():
            print(f"❌ Failed to download model '{model_size}'.")
            print("   Check your internet connection and try again.")
            if verbose:
                print(f"   Details: {err_msg}")
        elif "memory" in err_msg.lower() or "oom" in err_msg.lower():
            print(f"❌ Not enough memory for model '{model_size}'.")
            print("   Try a smaller model such as 'small' or 'base'.")
        else:
            print(f"❌ Error loading model: {err_msg}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    load_time = time.time() - load_start
    print(f"✅ Model loaded. ({load_time:.1f}s)")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    print(f"\n🎤 Transcribing...")
    if verbose:
        print(f"   beam_size=5, vad_filter={vad_filter}, device={device}, compute_type={compute_type}, task={task}")

    transcribe_start = time.time()

    try:
        # language=None triggers auto-detection in faster-whisper
        transcribe_language = None if (language is None or language.lower() == "auto") else language

        vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=200) if vad_filter else None

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
        err_msg = str(e)
        if "cublas" in err_msg.lower() or "cudnn" in err_msg.lower() or "cuda" in err_msg.lower():
            _cuda_unavailable_error(err_msg)
        elif "ffmpeg" in err_msg.lower():
            print("❌ FFmpeg error during transcription.")
            print("   Make sure FFmpeg is installed and the media file is valid.")
        else:
            print(f"❌ Transcription failed: {err_msg}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Collect segments (the generator streams the transcription)
    # ------------------------------------------------------------------
    detected_lang = info.language or "unknown"
    detected_prob = info.language_probability or 0.0
    duration = info.duration or 0.0
    lang_display = get_language_name(detected_lang)

    print(f"🌐 Detected language: {lang_display} [{detected_lang}] (confidence: {detected_prob:.0%})")
    print(f"⏱️  Duration: {format_duration(duration)}")
    print()

    collected_segments: List[Segment] = []
    segment_count = 0

    try:
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            # Collect segment with quality metrics
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

            # Real-time progress output
            ts = format_timestamp(seg.start)
            preview = text if len(text) <= 70 else text[:67] + "..."
            print(f"   [{ts}] {preview}")
            
            # Debug output with quality metrics
            if verbose:
                print(f"      logprob={seg.avg_logprob:.3f}, "
                      f"compression={seg.compression_ratio:.2f}, "
                      f"no_speech={seg.no_speech_prob:.3f}")

    except MemoryError:
        print(f"\n❌ Ran out of memory during transcription.")
        print("   Try a smaller model or a shorter file.")
        sys.exit(1)
    except Exception as e:
        err_msg = str(e)
        if "cublas" in err_msg.lower() or "cudnn" in err_msg.lower() or "cuda" in err_msg.lower():
            _cuda_unavailable_error(err_msg)
        else:
            print(f"\n❌ Error during transcription: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    transcribe_time = time.time() - transcribe_start
    print()
    print(f"✅ Transcription completed. ({transcribe_time:.1f}s, {segment_count} segments)")

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
        if "cublas" in detail.lower() or "cudnn" in detail.lower() or "not found" in detail.lower():
            print("   Required NVIDIA CUDA/cuDNN DLL libraries were not found on this system.")
            print("   To use GPU acceleration, install the CUDA 12 toolkit and cuDNN, or run:")
            print("     pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        else:
            print(f"   Details: {detail}")
    print("\n💡 Recommendation: Run on CPU (default):")
    print("   python transcribe.py <file> --device cpu --compute-type int8\n")
    sys.exit(1)

