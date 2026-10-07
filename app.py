import os
# pyrefly: ignore [missing-import]
import spaces
# pyrefly: ignore [missing-import]
import gradio as gr
# pyrefly: ignore [missing-import]
from faster_whisper import WhisperModel

# Global model cache to avoid re-allocating when reusing container
_MODEL_CACHE = {}

def get_model(model_size: str = "large-v3"):
    if model_size not in _MODEL_CACHE:
        # ZeroGPU allocates CUDA automatically when running inside @spaces.GPU
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cuda", compute_type="float16")
    return _MODEL_CACHE[model_size]

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

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
        vad_filter=True
    )

    full_text_list = []
    srt_list = []

    for i, seg in enumerate(segments, 1):
        txt = seg.text.strip()
        full_text_list.append(txt)
        start_ts = format_timestamp(seg.start)
        end_ts = format_timestamp(seg.end)
        srt_list.append(f"{i}\n{start_ts} --> {end_ts}\n{txt}\n")

    transcription = " ".join(full_text_list)
    srt_content = "\n".join(srt_list)
    
    meta_info = (
        f"**Detected Language:** `{info.language}` | "
        f"**Confidence:** `{info.language_probability:.1%}` | "
        f"**Duration:** `{info.duration:.1f}s`"
    )

    return meta_info, transcription, srt_content

# Custom UI
with gr.Blocks(title="Voxcribe - Speech to Text", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Voxcribe — Bangla & Multilingual Speech to Text
        Fast, accurate AI transcription powered by **faster-whisper** on **Hugging Face ZeroGPU**.
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
                model_dropdown = gr.Dropdown(
                    choices=["large-v3", "turbo", "medium", "small", "base"],
                    value="large-v3",
                    label="Model Size"
                )
                lang_dropdown = gr.Dropdown(
                    choices=[
                        ("Auto Detect", "auto"),
                        ("Bangla (বাংলা)", "bn"),
                        ("English", "en"),
                        ("Hindi (हिन्दी)", "hi"),
                        ("Arabic (العربية)", "ar"),
                        ("Spanish", "es"),
                    ],
                    value="bn",
                    label="Language"
                )
            
            task_choice = gr.Radio(
                choices=[("Transcribe", "transcribe"), ("Translate to English", "translate")],
                value="transcribe",
                label="Task"
            )
            
            transcribe_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")

        with gr.Column(scale=1):
            meta_display = gr.Markdown(label="Info")
            text_output = gr.Textbox(label="Transcription Text", lines=8)
            srt_output = gr.Textbox(label="Subtitles (SRT format)", lines=8)

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dropdown, lang_dropdown, task_choice],
        outputs=[meta_display, text_output, srt_output]
    )

if __name__ == "__main__":
    demo.launch()
