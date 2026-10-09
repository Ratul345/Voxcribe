# 🎙️ Voxcribe — AI Desktop Transcriber

A modern, high-performance local AI desktop speech-to-text application and CLI powered by `faster-whisper` (CTranslate2). Specially tailored for Bengali (বাংলা) transcription while offering complete multilingual speech recognition, automatic language detection, real-time streaming, and translation.

---

## ✨ Features

- **🎨 Modern Desktop GUI**: Built with CustomTkinter featuring clean dark and light modes, card-based layouts, and responsive controls.
- **⚡ High Performance & Acceleration**:
  - GPU (NVIDIA CUDA) support with automatic detection.
  - Multi-threaded CPU execution with `int8` quantization for fast CPU transcription.
- **🇧🇩 Bangla First + Multilingual**:
  - Bengali (বাংলা) prioritized along with English, Hindi, Spanish, French, German, Japanese, Chinese, Arabic, and auto-detection.
- **🕒 Live Real-Time Transcription**: Watch timestamped speech segments appear dynamically as your media is processed.
- **🔇 Voice Activity Detection (Silero VAD)**: Automatically skips background silence to boost speed and precision.
- **🌐 Speech Translation**: One-click translation of foreign/Bengali speech into English.
- **💾 Multi-Format Export**:
  - 📝 **Timestamped Text (.txt)**
  - 📄 **Clean Text (.txt)**
  - 🎬 **Subtitles (.srt)**
  - 📊 **Structured JSON (.json)**
  - 📋 **Direct Copy to Clipboard**
- **💻 Dual Interface**: Launch as a desktop GUI or use standard command-line tools.

---

## 🚀 Quick Start

### 1. Installation

Ensure Python 3.10+ is installed, then install requirements:

```bash
pip install -r requirements.txt
```

### 2. Launching Desktop Application

You can launch the GUI using any of the following:

```bash
python gui_app.py
```
or
```bash
python transcribe.py
```

### 3. Command-Line Usage (CLI)

You can also transcribe directly from the command line:

```bash
# Basic transcription (auto-detect or default model)
python transcribe.py video.mp4

# Bengali transcription with high accuracy model
python transcribe.py bangla.mp4 --language bn --model small --format txt,srt

# GPU acceleration
python transcribe.py bangla.mp4 --device cuda --compute-type float16
```

---

## 🛠️ Project Structure

```text
BanglaSTT/
├── gui_app.py           # Desktop Application launcher
├── transcribe.py        # Dual launcher (launches GUI or CLI depending on arguments)
├── requirements.txt     # Python dependencies
├── voxcribe/
│   ├── gui.py           # CustomTkinter Desktop Application UI
│   ├── transcriber.py   # faster-whisper core engine & progress streaming
│   ├── exporters.py     # TXT, SRT, JSON exporters
│   ├── utils.py         # FFmpeg and audio probes
│   └── cli.py           # CLI argument parsing and runner
```

---

## 📜 License

MIT License. Free for personal, academic, and commercial use.
