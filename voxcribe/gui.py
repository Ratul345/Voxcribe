"""
Voxcribe Desktop GUI Application
================================
Modern, high-performance desktop interface for BanglaSTT / Voxcribe.
Built with CustomTkinter for sleek aesthetics, dark/light mode support,
real-time transcription streaming, and multiple export formats.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List

import customtkinter as ctk
from tkinter import filedialog, messagebox

from voxcribe import __version__
from voxcribe.transcriber import VALID_MODELS, Segment, TranscriptionResult, load_and_transcribe
from voxcribe.exporters import export_txt, export_plain, export_srt, export_json
from voxcribe.utils import (
    find_ffmpeg,
    setup_ffmpeg,
    format_duration,
    format_timestamp,
    get_language_name,
    SUPPORTED_EXTENSIONS,
)

# Supported language list with Bangla as top priority
LANGUAGE_OPTIONS = [
    ("Bengali (বাংলা) - bn", "bn"),
    ("Auto Detect - auto", "auto"),
    ("English - en", "en"),
    ("Hindi (हिन्दी) - hi", "hi"),
    ("Arabic (العربية) - ar", "ar"),
    ("Spanish - es", "es"),
    ("French - fr", "fr"),
    ("German - de", "de"),
    ("Japanese - ja", "ja"),
    ("Chinese - zh", "zh"),
    ("Russian - ru", "ru"),
    ("Urdu (اردو) - ur", "ur"),
]


class BanglaSTTApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Voxcribe")
        self.geometry("1100x780")
        self.minsize(960, 680)

        # Set appearance
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # State variables
        self.selected_file_path: Optional[str] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.is_transcribing = False
        self.cancel_requested = False
        self.current_result: Optional[TranscriptionResult] = None
        self.segments_data: List[Segment] = []

        # Ensure FFmpeg is registered
        setup_ffmpeg()

        # Build UI layout
        self._create_layout()

        # Check GPU status on boot
        self._check_hardware_support()

    def _create_layout(self):
        # Configure grid weight
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)


        # -----------------------------------------------------------
        # Left Sidebar (Controls & Settings)
        # -----------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.rowconfigure(12, weight=1)

        # App Brand Title
        self.brand_title = ctk.CTkLabel(
            self.sidebar_frame,
            text="🎙️ BanglaSTT Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.brand_title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.brand_sub = ctk.CTkLabel(
            self.sidebar_frame,
            text=f"Powered by faster-whisper • v{__version__}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.brand_sub.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Separator
        sep1 = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color=("#3B4252", "#2E3440"))
        sep1.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))

        # Model Selector
        self.lbl_model = ctk.CTkLabel(self.sidebar_frame, text="Whisper Model:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_model.grid(row=3, column=0, padx=20, pady=(0, 4), sticky="w")

        self.combo_model = ctk.CTkComboBox(
            self.sidebar_frame,
            values=["small (Best Balance)", "base (Fast)", "tiny (Ultra Fast)", "medium (High Accuracy)", "large-v3 (Studio Quality)", "turbo (Fast v3)"],
            width=280,
            command=self._on_model_change,
        )
        self.combo_model.set("small (Best Balance)")
        self.combo_model.grid(row=4, column=0, padx=20, pady=(0, 15), sticky="w")

        # Language Selector
        self.lbl_lang = ctk.CTkLabel(self.sidebar_frame, text="Language:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_lang.grid(row=5, column=0, padx=20, pady=(0, 4), sticky="w")

        lang_names = [item[0] for item in LANGUAGE_OPTIONS]
        self.combo_lang = ctk.CTkComboBox(
            self.sidebar_frame,
            values=lang_names,
            width=280,
        )
        self.combo_lang.set("Bengali (বাংলা) - bn")
        self.combo_lang.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="w")

        # Compute Device (CPU vs GPU)
        self.lbl_device = ctk.CTkLabel(self.sidebar_frame, text="Acceleration Device:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_device.grid(row=7, column=0, padx=20, pady=(0, 4), sticky="w")

        self.seg_device = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["CPU", "GPU (CUDA)"],
            width=280,
            command=self._on_device_toggle,
        )
        self.seg_device.set("CPU")
        self.seg_device.grid(row=8, column=0, padx=20, pady=(0, 15), sticky="w")

        # Compute Precision / Type
        self.lbl_compute = ctk.CTkLabel(self.sidebar_frame, text="Compute Type:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_compute.grid(row=9, column=0, padx=20, pady=(0, 4), sticky="w")

        self.combo_compute = ctk.CTkComboBox(
            self.sidebar_frame,
            values=["int8", "float16", "float32"],
            width=280,
        )
        self.combo_compute.set("int8")
        self.combo_compute.grid(row=10, column=0, padx=20, pady=(0, 15), sticky="w")

        # VAD Filter & Task Options
        self.chk_vad = ctk.CTkCheckBox(self.sidebar_frame, text="Silero VAD (Remove silence)")
        self.chk_vad.select()
        self.chk_vad.grid(row=11, column=0, padx=20, pady=(0, 10), sticky="w")

        self.chk_translate = ctk.CTkCheckBox(self.sidebar_frame, text="Translate speech to English")
        self.chk_translate.grid(row=12, column=0, padx=20, pady=(0, 15), sticky="nw")

        # Appearance Mode Selector at bottom of sidebar
        self.lbl_theme = ctk.CTkLabel(self.sidebar_frame, text="Theme:", font=ctk.CTkFont(size=11))
        self.lbl_theme.grid(row=13, column=0, padx=20, pady=(0, 2), sticky="w")

        self.combo_theme = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme,
            width=280,
            height=28,
        )
        self.combo_theme.set("Dark")
        self.combo_theme.grid(row=14, column=0, padx=20, pady=(0, 20), sticky="w")

        # -----------------------------------------------------------
        # Right Main Panel
        # -----------------------------------------------------------
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.rowconfigure(3, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # File Drop/Select Banner Card
        self.file_card = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color=("#E5E9F0", "#2E3440"))
        self.file_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        self.file_card.columnconfigure(0, weight=1)

        file_card_inner = ctk.CTkFrame(self.file_card, fg_color="transparent")
        file_card_inner.pack(fill="both", expand=True, padx=20, pady=18)

        self.lbl_selected_file = ctk.CTkLabel(
            file_card_inner,
            text="📁 No Audio/Video file selected",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        self.lbl_selected_file.pack(side="left", fill="x", expand=True)

        self.btn_browse = ctk.CTkButton(
            file_card_inner,
            text="Browse Media File",
            command=self._browse_file,
            width=160,
            height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.btn_browse.pack(side="right", padx=(10, 0))

        # Action and Progress Bar Row
        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.action_frame.columnconfigure(0, weight=1)

        self.btn_transcribe = ctk.CTkButton(
            self.action_frame,
            text="🚀 Start Transcription",
            command=self._start_transcription,
            height=44,
            fg_color=("#007ACC", "#1F6AA5"),
            hover_color=("#005999", "#144F7A"),
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.btn_transcribe.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(
            self.action_frame,
            text="⏹️ Cancel",
            command=self._cancel_transcription,
            height=44,
            width=110,
            fg_color="#D08770",
            hover_color="#BF616A",
            state="disabled",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_cancel.grid(row=0, column=1)

        # Progress bar and status indicator
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=12)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.lbl_status = ctk.CTkLabel(
            self.progress_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self.lbl_status.grid(row=1, column=0, sticky="w")

        # Results Tabview (Live Segments / Full Text / Logs)
        self.tabview = ctk.CTkTabview(self.main_frame, corner_radius=10)
        self.tabview.grid(row=3, column=0, sticky="nsew", pady=(0, 15))

        self.tab_segments = self.tabview.add("🕒 Timestamped Segments")
        self.tab_fulltext = self.tabview.add("📄 Full Clean Text")
        self.tab_logs = self.tabview.add("📋 Execution Logs")

        # 1. Timestamped segments textbox
        self.tab_segments.rowconfigure(0, weight=1)
        self.tab_segments.columnconfigure(0, weight=1)
        self.txt_segments = ctk.CTkTextbox(
            self.tab_segments,
            wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.txt_segments.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # 2. Full clean text textbox
        self.tab_fulltext.rowconfigure(0, weight=1)
        self.tab_fulltext.columnconfigure(0, weight=1)
        self.txt_fulltext = ctk.CTkTextbox(
            self.tab_fulltext,
            wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=14),
        )
        self.txt_fulltext.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # 3. System execution logs
        self.tab_logs.rowconfigure(0, weight=1)
        self.tab_logs.columnconfigure(0, weight=1)
        self.txt_logs = ctk.CTkTextbox(
            self.tab_logs,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.txt_logs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Export Controls Bar (Footer)
        self.export_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color=("#E5E9F0", "#2E3440"))
        self.export_frame.grid(row=4, column=0, sticky="ew")

        export_inner = ctk.CTkFrame(self.export_frame, fg_color="transparent")
        export_inner.pack(fill="both", expand=True, padx=15, pady=12)

        lbl_export = ctk.CTkLabel(
            export_inner,
            text="Save / Export As:",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        lbl_export.pack(side="left", padx=(0, 10))

        self.btn_export_txt = ctk.CTkButton(
            export_inner,
            text="💾 Text (.txt)",
            width=100,
            command=lambda: self._export_result("txt"),
        )
        self.btn_export_txt.pack(side="left", padx=5)

        self.btn_export_srt = ctk.CTkButton(
            export_inner,
            text="🎬 Subtitles (.srt)",
            width=120,
            command=lambda: self._export_result("srt"),
        )
        self.btn_export_srt.pack(side="left", padx=5)

        self.btn_export_json = ctk.CTkButton(
            export_inner,
            text="{ } JSON (.json)",
            width=110,
            command=lambda: self._export_result("json"),
        )
        self.btn_export_json.pack(side="left", padx=5)

        self.btn_copy = ctk.CTkButton(
            export_inner,
            text="📋 Copy to Clipboard",
            width=140,
            fg_color=("#4C566A", "#3B4252"),
            hover_color=("#434C5E", "#2E3440"),
            command=self._copy_to_clipboard,
        )
        self.btn_copy.pack(side="right", padx=5)

        self._update_export_buttons_state(False)

    def _check_hardware_support(self):
        """Check if CUDA GPU is available on system and set default."""
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count() if hasattr(ctranslate2, "get_cuda_device_count") else 0
            if cuda_count > 0:
                self.seg_device.set("GPU (CUDA)")
                self.combo_compute.set("float16")
                self._append_log(f"⚡ NVIDIA GPU detected ({cuda_count} device). GPU acceleration ready.")
            else:
                self.seg_device.set("CPU")
                self.combo_compute.set("int8")
                self._append_log("ℹ️ Running in CPU mode (Optimized with int8 quantization).")
        except Exception as e:
            self._append_log(f"Hardware check note: {e}")

    def _change_theme(self, mode: str):
        ctk.set_appearance_mode(mode)

    def _on_model_change(self, val: str):
        self._append_log(f"Model selected: {val}")

    def _on_device_toggle(self, val: str):
        if val == "GPU (CUDA)":
            self.combo_compute.set("float16")
        else:
            self.combo_compute.set("int8")

    def _browse_file(self):
        extensions_desc = "Media Files (" + " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS)) + ")"
        file_path = filedialog.askopenfilename(
            title="Select Audio or Video File for Transcription",
            filetypes=[
                ("All Supported Media", ";".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))),
                ("Audio Files", "*.mp3;*.wav;*.m4a"),
                ("Video Files", "*.mp4;*.mkv;*.mov;*.webm;*.avi"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_file_path = os.path.abspath(file_path)
            file_name = os.path.basename(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.lbl_selected_file.configure(
                text=f"📁 {file_name} ({file_size_mb:.1f} MB)",
                text_color=("#2E3440", "#ECEFF4"),
            )
            self.lbl_status.configure(text=f"Ready to transcribe: {file_name}")
            self._append_log(f"Selected input file: {self.selected_file_path} ({file_size_mb:.1f} MB)")

    def _start_transcription(self):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            messagebox.showwarning("Select Media File", "Please choose an audio or video file first!")
            return

        if self.is_transcribing:
            return

        self.is_transcribing = True
        self.cancel_requested = False
        self.current_result = None
        self.segments_data.clear()

        # Update UI state
        self.btn_transcribe.configure(state="disabled", text="⏳ Transcribing in progress...")
        self.btn_cancel.configure(state="normal")
        self.btn_browse.configure(state="disabled")
        self.progress_bar.set(0.0)
        self._update_export_buttons_state(False)

        # Clear text displays
        self.txt_segments.delete("1.0", "end")
        self.txt_fulltext.delete("1.0", "end")

        # Parse options
        model_str = self.combo_model.get().split()[0]
        selected_lang_label = self.combo_lang.get()
        lang_code = "auto"
        for label, code in LANGUAGE_OPTIONS:
            if label == selected_lang_label:
                lang_code = code
                break

        device_mode = "cuda" if "GPU" in self.seg_device.get() else "cpu"
        compute_type = self.combo_compute.get()
        use_vad = bool(self.chk_vad.get())
        is_translate = bool(self.chk_translate.get())
        task_name = "translate" if is_translate else "transcribe"

        self._append_log(
            f"\n--- Starting Transcription ---\n"
            f"File: {self.selected_file_path}\n"
            f"Model: {model_str} | Language: {lang_code} | Device: {device_mode} ({compute_type})\n"
            f"Task: {task_name} | VAD: {use_vad}"
        )

        # Launch worker thread
        self.transcription_thread = threading.Thread(
            target=self._run_transcription_worker,
            args=(self.selected_file_path, model_str, lang_code, device_mode, compute_type, use_vad, task_name),
            daemon=True,
        )
        self.transcription_thread.start()

    def _run_transcription_worker(self, path, model_size, lang, device, compute_type, vad, task):
        start_time = time.time()
        try:
            def on_progress(pct, text):
                self.after(0, lambda: self._update_progress(pct, text))

            def on_segment(seg: Segment, count: int, total_dur: float):
                self.after(0, lambda: self._add_segment_to_ui(seg, count, total_dur))

            def on_status(text: str):
                self.after(0, lambda: self._append_log(text))

            def check_cancel():
                return self.cancel_requested

            result = load_and_transcribe(
                audio_path=path,
                model_size=model_size,
                language=lang,
                device=device,
                compute_type=compute_type,
                vad_filter=vad,
                task=task,
                progress_callback=on_progress,
                segment_callback=on_segment,
                status_callback=on_status,
                should_cancel=check_cancel,
            )

            elapsed = time.time() - start_time
            self.after(0, lambda: self._on_transcription_completed(result, elapsed))

        except InterruptedError:
            self.after(0, self._on_transcription_cancelled)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_transcription_failed(err_msg))

    def _update_progress(self, pct: float, text: str):
        self.progress_bar.set(pct)
        self.lbl_status.configure(text=f"Status: {text} ({int(pct * 100)}%)")

    def _add_segment_to_ui(self, seg: Segment, count: int, total_dur: float):
        self.segments_data.append(seg)

        # Add to timestamp view
        ts_start = format_timestamp(seg.start)
        ts_end = format_timestamp(seg.end)
        seg_entry = f"[{ts_start} → {ts_end}]  {seg.text}\n"
        self.txt_segments.insert("end", seg_entry)
        self.txt_segments.see("end")

        # Add to clean full text view
        self.txt_fulltext.insert("end", seg.text + "\n\n")
        self.txt_fulltext.see("end")

    def _on_transcription_completed(self, result: TranscriptionResult, elapsed: float):
        self.is_transcribing = False
        self.current_result = result
        self.btn_transcribe.configure(state="normal", text="🚀 Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.progress_bar.set(1.0)

        lang_name = get_language_name(result.language)
        msg = f"Completed in {elapsed:.1f}s • {len(result.segments)} segments • Language: {lang_name} ({result.language_probability:.0%})"
        self.lbl_status.configure(text=f"Status: {msg}")
        self._append_log(f"🎉 {msg}")
        self._update_export_buttons_state(True)

        # Automatically switch to segments tab
        self.tabview.set("🕒 Timestamped Segments")

    def _cancel_transcription(self):
        if self.is_transcribing:
            self.cancel_requested = True
            self.lbl_status.configure(text="Status: Cancelling transcription...")
            self.btn_cancel.configure(state="disabled")

    def _on_transcription_cancelled(self):
        self.is_transcribing = False
        self.btn_transcribe.configure(state="normal", text="🚀 Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.lbl_status.configure(text="Status: Transcription cancelled by user.")
        self._append_log("⏹️ Transcription stopped by user.")
        if len(self.segments_data) > 0:
            self._update_export_buttons_state(True)

    def _on_transcription_failed(self, error_message: str):
        self.is_transcribing = False
        self.btn_transcribe.configure(state="normal", text="🚀 Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.lbl_status.configure(text="Status: Transcription failed.")
        self._append_log(f"❌ Error: {error_message}")
        messagebox.showerror("Transcription Error", f"Failed to transcribe media file:\n\n{error_message}")

    def _update_export_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_export_txt.configure(state=state)
        self.btn_export_srt.configure(state=state)
        self.btn_export_json.configure(state=state)
        self.btn_copy.configure(state=state)

    def _copy_to_clipboard(self):
        active_tab = self.tabview.get()
        if "Segments" in active_tab:
            content = self.txt_segments.get("1.0", "end-1c")
        elif "Text" in active_tab:
            content = self.txt_fulltext.get("1.0", "end-1c")
        else:
            content = self.txt_logs.get("1.0", "end-1c")

        if content.strip():
            self.clipboard_clear()
            self.clipboard_append(content)
            messagebox.showinfo("Copied", "Transcription text copied to system clipboard!")

    def _export_result(self, format_type: str):
        if not self.segments_data:
            messagebox.showwarning("No Data", "There are no transcribed segments to export.")
            return

        # Prepare dummy TranscriptionResult if needed
        res = self.current_result or TranscriptionResult(
            source=self.selected_file_path or "untitled",
            language="auto",
            language_probability=1.0,
            duration=self.segments_data[-1].end if self.segments_data else 0.0,
            segments=self.segments_data,
        )

        base_name = Path(self.selected_file_path).stem if self.selected_file_path else "transcription"
        filetypes_dict = {
            "txt": [("Text Document", "*.txt")],
            "srt": [("SRT Subtitles", "*.srt")],
            "json": [("JSON File", "*.json")],
        }

        save_path = filedialog.asksaveasfilename(
            title=f"Export as {format_type.upper()}",
            initialfile=f"{base_name}.{format_type}",
            defaultextension=f".{format_type}",
            filetypes=filetypes_dict.get(format_type, [("All Files", "*.*")]),
        )

        if not save_path:
            return

        try:
            if format_type == "txt":
                export_txt(res, save_path)
            elif format_type == "srt":
                export_srt(res, save_path)
            elif format_type == "json":
                export_json(res, save_path)

            self._append_log(f"💾 File successfully saved to: {save_path}")
            messagebox.showinfo("Saved", f"Exported successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save file: {e}")

    def _append_log(self, text: str):
        self.txt_logs.insert("end", text + "\n")
        self.txt_logs.see("end")


def launch_gui():
    """Main entrypoint for desktop GUI app."""
    app = BanglaSTTApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
