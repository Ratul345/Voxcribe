import gradio as gr
from faster_whisper import WhisperModel
import spaces

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}

def get_model(model_size: str = "large-v3"):
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = WhisperModel(
            model_size, device="cuda", compute_type="float16"
        )
    return _MODEL_CACHE[model_size]


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Transcription logic
# ---------------------------------------------------------------------------
@spaces.GPU
def transcribe_audio(audio_path, model_size, language, task):
    if not audio_path:
        return "⚠️ Please upload or record an audio file.", "", ""

    model = get_model(model_size)
    lang_code = language if language != "auto" else None

    segments, info = model.transcribe(
        audio_path,
        language=lang_code,
        task=task,
        vad_filter=True,
    )

    full_text = []
    srt_lines = []

    for i, seg in enumerate(segments, 1):
        txt = seg.text.strip()
        full_text.append(txt)
        srt_lines.append(
            f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{txt}\n"
        )

    transcription = " ".join(full_text)
    srt_content = "\n".join(srt_lines)
    meta = (
        f"🌐 **Language:** {info.language}  •  "
        f"📊 **Confidence:** {info.language_probability:.1%}  •  "
        f"⏱️ **Duration:** {info.duration:.1f}s"
    )
    return meta, transcription, srt_content


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
custom_css = """
/* ── Global ──────────────────────────────────────────────── */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header banner ───────────────────────────────────────── */
#header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    text-align: center;
    color: #fff;
}
#header-banner h1 {
    font-size: 2rem;
    margin: 0 0 4px 0;
    color: #fff !important;
}
#header-banner p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
    color: #e8e8ff !important;
}

/* ── Cards ───────────────────────────────────────────────── */
.card-section {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    background: #fafbfc;
}

/* ── Primary button ──────────────────────────────────────── */
#run-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: #fff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.45) !important;
}

/* ── Meta info strip ─────────────────────────────────────── */
#meta-strip {
    background: linear-gradient(90deg, #eef2ff 0%, #faf5ff 100%);
    border-radius: 10px;
    padding: 12px 18px;
    border-left: 4px solid #764ba2;
}

/* ── Outputs ─────────────────────────────────────────────── */
#transcript-box textarea, #srt-box textarea {
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(css=custom_css, title="Voxcribe – Speech to Text") as demo:

    # Header
    gr.HTML(
        """
        <div id="header-banner">
            <h1>🎙️ Voxcribe</h1>
            <p>Bangla &amp; Multilingual Speech-to-Text · powered by faster-whisper &amp; ZeroGPU</p>
        </div>
        """
    )

    # ── Input section ────────────────────────────────────────
    with gr.Group(elem_classes="card-section"):
        gr.Markdown("### 🎧 Audio Input")
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Upload a file or record from your microphone",
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=["large-v3", "turbo", "medium", "small", "base"],
                value="large-v3",
                label="🧠 Model",
                info="Larger models are more accurate but slower.",
            )
            lang_dd = gr.Dropdown(
                choices=[
                    ("Auto Detect", "auto"),
                    ("Bangla  বাংলা", "bn"),
                    ("English", "en"),
                    ("Hindi  हिन्दी", "hi"),
                    ("Arabic  العربية", "ar"),
                    ("Spanish", "es"),
                ],
                value="bn",
                label="🌍 Language",
            )
            task_radio = gr.Radio(
                choices=[
                    ("Transcribe", "transcribe"),
                    ("Translate → English", "translate"),
                ],
                value="transcribe",
                label="📝 Task",
            )

    transcribe_btn = gr.Button(
        "🚀  Start Transcription",
        variant="primary",
        elem_id="run-btn",
    )

    # ── Output section ───────────────────────────────────────
    with gr.Group(elem_classes="card-section"):
        gr.Markdown("### 📄 Results")
        meta_display = gr.Markdown(elem_id="meta-strip")

        with gr.Tabs():
            with gr.Tab("Transcript"):
                text_output = gr.Textbox(
                    label="Full Text",
                    lines=10,
                    show_copy_button=True,
                    elem_id="transcript-box",
                )
            with gr.Tab("Subtitles (SRT)"):
                srt_output = gr.Textbox(
                    label="SRT Format",
                    lines=10,
                    show_copy_button=True,
                    elem_id="srt-box",
                )

    # ── Wiring ───────────────────────────────────────────────
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dd, lang_dd, task_radio],
        outputs=[meta_display, text_output, srt_output],
    )

    # Footer
    gr.Markdown(
        "<center style='color:#888;font-size:0.82rem;margin-top:12px'>"
        "Built with ❤️ by <b>Ratul</b> · MIT License · "
        "<a href='https://github.com/Ratul345/BanglaSTT' target='_blank'>GitHub</a>"
        "</center>"
    )

if __name__ == "__main__":
    demo.launch()
