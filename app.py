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
# Clean, modern UI (compatible with Gradio 5)
# ---------------------------------------------------------------------------
custom_css = """
/* Constrain SVG icons to normal sizes */
svg {
    max-width: 24px !important;
    max-height: 24px !important;
}
.header-box {
    text-align: center;
    padding: 24px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
}
.header-box h1 {
    font-size: 28px;
    font-weight: bold;
    margin: 0 0 6px 0;
    color: white !important;
}
.header-box p {
    font-size: 15px;
    margin: 0;
    color: #ede9fe !important;
}
#start-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: none !important;
}
"""

with gr.Blocks(title="Voxcribe - Speech to Text", css=custom_css, theme=gr.themes.Soft()) as demo:
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
                label="Audio Input"
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

            transcribe_btn = gr.Button("🚀 Start Transcription", elem_id="start-btn")

        with gr.Column(scale=1):
            meta_display = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Transcript"):
                    text_output = gr.Textbox(
                        label="Full Text",
                        lines=10,
                        placeholder="Transcribed text will appear here...",
                        show_copy_button=True
                    )
                with gr.Tab("Subtitles (SRT)"):
                    srt_output = gr.Textbox(
                        label="Subtitles (SRT)",
                        lines=10,
                        placeholder="Subtitles will appear here...",
                        show_copy_button=True
                    )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dd, lang_dd, task_radio],
        outputs=[meta_display, text_output, srt_output]
    )

if __name__ == "__main__":
    demo.launch()
