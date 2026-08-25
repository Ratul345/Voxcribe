# 🎤 Voxcribe — Local Video/Audio Transcription Tool

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Engine](https://img.shields.io/badge/engine-faster--whisper-orange.svg)
![Privacy](https://img.shields.io/badge/privacy-local%20only-brightgreen.svg)

A privacy-first, local transcription tool for studying educational videos and audio content.

Voxcribe converts your local video/audio files into high-quality transcripts — with timestamps, subtitle files, and structured JSON — all processed **entirely on your machine**.

## 🌟 What's New in v2

Voxcribe started as a Bengali-only speech-to-text tool. Version 2 evolves it into a **multilingual transcription tool** while keeping its Bengali heritage and simple design.

| v1 | v2 |
|---|---|
| OpenAI Whisper | **faster-whisper** (CTranslate2, faster & more efficient) |
| Bengali only (`language="bn"`) | **Multilingual** with auto-detection |
| Plain text output | **TXT + Timestamped TXT + SRT + JSON** |
| No timestamps | **Per-segment timestamps** |
| `base` model | **`small` model** (better quality default) |
| CPU only, no choice | **CPU (int8) + optional CUDA (float16)** |
| No VAD | **Voice Activity Detection** enabled by default |


## ✨ Features

- 🎯 **Multilingual**: Bengali, English, Hindi, and 90+ languages supported by Whisper
- 🔒 **Privacy-first**: All processing happens locally — no uploads, no cloud, no tracking
- ⚡ **Fast**: Uses faster-whisper (CTranslate2) for efficient CPU/GPU inference
- ⏱️ **Timestamps**: Every segment has start/end times
- 📄 **Multiple outputs**: TXT, Timestamped TXT, SRT subtitles, JSON
- 🎤 **VAD**: Voice Activity Detection reduces hallucination during silence
- 🌐 **Auto-detect**: Automatic language detection with confidence score
- 🗣️ **Mixed language**: Handles Bangla+English, Banglish, technical terms naturally
- 📺 **Video support**: MP4, MKV, AVI, MOV, WebM — audio extracted automatically
- 🎵 **Audio support**: MP3, WAV, M4A, MPEG, MPGA
- 💻 **CPU-friendly**: Works without a GPU using int8 quantization
- 🎮 **Optional GPU**: CUDA support for faster processing
- 📴 **Offline**: After model download, works without internet

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **FFmpeg** (see [FFmpeg Setup](#-ffmpeg-setup) below)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ratul345/Voxcribe.git
cd Voxcribe

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate     # Windows (.venv\Scripts\Activate.ps1 in PowerShell)
# source .venv/bin/activate  # macOS/Linux

# Install project in editable mode (creates the `voxcribe` command)
pip install -e .
```

> **Note**: The first time you use a model, it will be downloaded automatically (~460 MB for `small`). After that, transcription works offline. Models are cached in `~/.cache/huggingface/hub/`.

### Basic Usage

```bash
# Transcribe with auto language detection (default)
voxcribe video.mp4

# Specify language
voxcribe lecture.mp3 --language bn
voxcribe podcast.wav --language en

# Choose model quality
voxcribe audio.m4a --model medium

# Multiple output formats (timestamped TXT, clean plain TXT, SRT, JSON)
voxcribe video.mp4 --format txt,plain,srt,json

# Custom output path
voxcribe video.mp4 --output my_transcript

# Backward-compatible / development alias (works identically)
python transcribe.py video.mp4
```

## 📋 Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `tiny` | ~75 MB | ⚡ Fastest | 🟡 Basic | Quick tests, short clips |
| `base` | ~145 MB | ⚡ Fast | 🟢 Good | Lightweight, faster processing |
| **`small`** | **~460 MB** | **🟡 Balanced** | **🟢 Recommended** | **Default — best quality/speed balance** |
| `medium` | ~1.5 GB | 🔴 Slow | 🔵 High | Important content, better accuracy |
| `large-v3` | ~3 GB | 🔴 Slowest | 🔵 Highest | Maximum quality, needs more RAM |

**💡 Recommendation**: Start with `small`. It provides noticeably better accuracy than `base`, especially for Bengali and mixed-language content, while remaining practical on a normal Windows PC with CPU.

> **⚠️ RAM usage**: `medium` needs ~5 GB RAM, `large-v3` needs ~10 GB+. If your PC has 8 GB RAM, stick with `small` or `base`.

## 🌐 Language Support

Voxcribe supports all languages available in the Whisper model, including:

| Code | Language |
|------|----------|
| `auto` | Automatic detection (default) |
| `bn` | Bengali / বাংলা |
| `en` | English |
| `hi` | Hindi / हिन्दी |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `ja` | Japanese |
| `zh` | Chinese |
| `ar` | Arabic |
| ... | [90+ languages](https://github.com/openai/whisper#available-models-and-languages) |

### Mixed Language

Voxcribe does **not** translate speech. It transcribes what was actually spoken.

If someone says:

> "আজ আমরা React-এর component architecture নিয়ে কথা বলব"

The transcript will be:

> আজ আমরা React-এর component architecture নিয়ে কথা বলব

English technical terms inside Bengali speech are preserved naturally.

## 📄 Output Formats

### TXT — Timestamped text (default)

```
[00:00:03] Hello everyone.
[00:00:08] Today we're going to talk about the first concept.
[00:00:15] The first important concept is...
```

Best for: studying and referencing specific parts of a video.

### Plain TXT (`--format plain`)

```
Hello everyone.

Today we're going to talk about the first concept.

The first important concept is...
```

Best for: reading, studying, and pasting into AI chat for discussion without timestamps.

### SRT — Standard subtitle format (`--format srt`)

```
1
00:00:03,000 --> 00:00:08,000
Hello everyone.

2
00:00:08,000 --> 00:00:15,000
Today we're going to talk about the first concept.
```

Best for: loading as subtitles in video players (VLC, YouTube, etc.).

### JSON — Structured data (`--format json`)

```json
{
  "source": "video.mp4",
  "language": "en",
  "language_probability": 0.98,
  "duration": 123.45,
  "segments": [
    {"start": 3.0, "end": 8.0, "text": "Hello everyone."}
  ]
}
```

Best for: programmatic analysis, developer integrations, and AI pipelines.

## 🔧 CLI Reference

```
usage: voxcribe [-h] [-m MODEL] [-l LANGUAGE] [-o OUTPUT] [-f FORMAT]
                [--device {cpu,cuda}] [--compute-type TYPE]
                [--vad | --no-vad] [--task {transcribe,translate}]
                [-v] [--version]
                input_file
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `input_file` | | (required) | Path to audio/video file |
| `--model` | `-m` | `small` | Model: tiny, base, small, medium, large-v3, turbo |
| `--language` | `-l` | `auto` | Language code or "auto" |
| `--output` | `-o` | `output/<name>` | Output path/base name |
| `--format` | `-f` | `txt` | Output formats (comma-separated): txt, plain, srt, json |
| `--device` | | `cpu` | Compute device: cpu, cuda |
| `--compute-type` | | `int8` | Precision: int8 (CPU default), float16 (GPU), float32 |
| `--vad` | | (on) | Enable VAD (default) |
| `--no-vad` | | | Disable VAD |
| `--task` | | `transcribe` | Task: transcribe (default) or translate (to English) |
| `--verbose` | `-v` | | Show debug output |
| `--version` | | | Show version |

## 💻 CPU Usage

Voxcribe works on CPU by default — no GPU required.

```bash
# Default: CPU with int8 quantization
voxcribe video.mp4

# Explicitly:
voxcribe video.mp4 --device cpu --compute-type int8
```

This is the recommended setup for most Windows PCs.

## 🎮 GPU / CUDA Usage (Optional)

If you have an NVIDIA GPU with CUDA configured:

```bash
voxcribe video.mp4 --device cuda --compute-type float16
```

**Requirements for GPU:**
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- cuDNN installed
- `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (or equivalent for your CUDA version)

> **Note**: If you request `--device cuda` and CUDA is not available, Voxcribe will show a clear error instead of silently falling back to CPU.

## 🔊 FFmpeg Setup

Voxcribe needs FFmpeg to decode audio/video files.

### Windows

**Option A — Automatic** (recommended for beginners):
```bash
pip install imageio-ffmpeg
```
This installs a bundled FFmpeg binary. Voxcribe finds it automatically.

**Option B — Manual**:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g. `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH

### macOS
```bash
brew install ffmpeg
```

### Linux
```bash
sudo apt install ffmpeg
```

## 📁 Project Structure

```
Voxcribe/
├── pyproject.toml         # Build & console script definition (voxcribe = voxcribe.cli:main)
├── transcribe.py          # Backward-compatible CLI entry point
├── voxcribe/
│   ├── __init__.py        # Package metadata
│   ├── cli.py             # Argument parsing & orchestration
│   ├── transcriber.py     # faster-whisper model & transcription
│   ├── exporters.py       # TXT, Plain, SRT, JSON output writers
│   └── utils.py           # FFmpeg, validation, formatting
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── CONTRIBUTING.md        # Contribution guide
├── LICENSE                # MIT License
└── .gitignore
```

## 🎯 Examples

### Transcribe a Bengali lecture
```bash
voxcribe bangla_lecture.mp4 --language bn --format txt,srt
```

### Transcribe an English podcast with all outputs
```bash
voxcribe podcast.mp3 --language en --format txt,plain,srt,json
```

### Auto-detect language from a Reel or educational clip
```bash
voxcribe reel_video.mp4 --format txt
```

### High-quality transcription
```bash
voxcribe important_lecture.mp4 --model medium --format txt,srt,json
```

### Quick transcription of a short clip
```bash
voxcribe clip.mp3 --model tiny
```

## 🔍 Troubleshooting

### "FFmpeg not found"
- Install imageio-ffmpeg: `pip install imageio-ffmpeg`
- Or install FFmpeg system-wide and add it to PATH

### "CUDA is not available"
- You requested `--device cuda` but CUDA is not configured
- Run with `--device cpu` (default) or install CUDA toolkit

### "Not enough memory"
- Use a smaller model: `--model base` or `--model tiny`
- Close other applications to free RAM
- `large-v3` requires 10+ GB RAM

### "Failed to download model"
- Check your internet connection
- The model downloads only once and is cached locally
- Cache location: `~/.cache/huggingface/hub/`

### Poor transcription quality
- Use a larger model: `--model medium` or `--model large-v3`
- Specify the language explicitly: `--language bn`
- Ensure the audio is clear with minimal background noise
- Try `--no-vad` if VAD is cutting off speech

### Slow transcription
- Use a smaller model: `--model base` or `--model tiny`
- If you have a CUDA GPU: `--device cuda --compute-type float16`

## 🔒 Privacy

Voxcribe is designed for privacy:

- ✅ **All processing is local** — audio/video never leaves your machine
- ✅ **No analytics or telemetry** — zero tracking
- ✅ **No cloud APIs** — no OpenAI API key needed
- ✅ **No automatic uploads** — your content stays yours
- ✅ **Offline capable** — after model download, no internet needed

The only network activity is the one-time model download from Hugging Face when you first use a model.

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — CTranslate2-based Whisper reimplementation by SYSTRAN
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Original Whisper speech recognition research by OpenAI
- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** — Fast inference engine

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Ratul345/Voxcribe/issues)
- 💡 **Feature Requests**: Open a discussion
- ❓ **Questions**: Check the troubleshooting section above

---

**⭐ If Voxcribe helps your learning, give it a star!**
