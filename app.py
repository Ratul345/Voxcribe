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
        f"🌐 **Language:** `{info.language}` | "
        f"📊 **Confidence:** `{info.language_probability:.1%}` | "
        f"⏱️ **Duration:** `{info.duration:.1f}s`"
    )
    return meta, transcription, srt_content

# ---------------------------------------------------------------------------
# Clean, modern CSS with strict icon size control
# ---------------------------------------------------------------------------
custom_css = """
/* Fix huge SVG icons on Hugging Face Spaces */
svg, .upload-container svg, button svg, .icon svg {
    max-width: 28px !important;
    max-height: 28px !important;
    width: auto !important;
    height: auto !important;
    display: inline-block !important;
}

/* Audio input box styling */
.gradio-container {
    max-width: 900px !important;
    margin: 20px auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.header-box {
    text-align: center;
    padding: 24px 16px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border-radius: 12px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
}

.header-box h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: white !important;
}

.header-box p {
    font-size: 14px;
    opacity: 0.85;
    margin: 0;
    color: #e0e7ff !important;
}

#transcribe-btn {
    background: #4f46e5 !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    border: none !important;
    margin: 12px 0 !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
}

#transcribe-btn:hover {
    background: #4338ca !important;
}
"""

with gr.Blocks(title="Voxcribe - Speech to Text", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.HTML(
        """
        <div class="header-box">
            <h1>🎙️ Voxcribe</h1>
            <p>Bangla & Multilingual AI Speech-to-Text • Powered by ZeroGPU</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio / Voice Input"
            )

            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=["large-v3", "turbo", "medium", "small", "base"],
                    value="large-v3",
                    label="Model Size"
                )
                lang_dd = gr.Dropdown(
                    choices=[
                        ("Auto Detect", "auto"),
                        ("Bangla (বাংলা)", "bn"),
                        ("English", "en"),
                        ("Hindi (हिन्दी)", "hi"),
                        ("Arabic (العربية)", "ar"),
                        ("Spanish", "es")
                    ],
                    value="bn",
                    label="Language"
                )

            task_radio = gr.Radio(
                choices=[("Transcribe", "transcribe"), ("Translate to English", "translate")],
                value="transcribe",
                label="Task"
            )

            transcribe_btn = gr.Button("🚀 Start Transcription", elem_id="transcribe-btn")

        with gr.Column(scale=1):
            meta_display = gr.Markdown()
            text_output = gr.Textbox(
                label="Transcription Text",
                lines=9,
                placeholder="Transcribed text will appear here..."
            )
            srt_output = gr.Textbox(
                label="Subtitles (SRT)",
                lines=9,
                placeholder="SRT timestamps will appear here..."
            )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dd, lang_dd, task_radio],
        outputs=[meta_display, text_output, srt_output]
    )

if __name__ == "__main__":
    demo.launch()
