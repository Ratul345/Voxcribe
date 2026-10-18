"""
Voxcribe Desktop GUI Application
================================
Modern, high-performance desktop interface for Voxcribe / Voxcribe.
Built with CustomTkinter for sleek aesthetics, dark/light mode support,
real-time transcription streaming, and multiple export formats.

Redesigned v3.1 — High-contrast typography, custom window & taskbar icon,
Pixora-inspired layout, and completely anonymized settings.
"""

import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Optional, List

# Windows taskbar grouping & icon attribution
if sys.platform == "win32":
    try:
        import ctypes
        app_user_model_id = "Voxcribe.voxcribe.studio.v2"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_user_model_id)
    except Exception:
        pass

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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

# ─── Asset paths ──────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_JPG = ASSETS_DIR / "logo.jpg"
LOGO_ICO = ASSETS_DIR / "logo.ico"
LOGO_PNG = ASSETS_DIR / "logo.png"

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

# ─── Color Palette (High-Contrast, Modern SaaS Polish) ────────────
# Tuples are (light_mode, dark_mode)
C = {
    # Primary accent — modern vibrant indigo/purple
    "accent":           ("#6366F1", "#818CF8"),
    "accent_hover":     ("#4F46E5", "#6366F1"),
    "accent_deep":      ("#4338CA", "#A5B4FC"),
    "accent_soft":      ("#EEF2FF", "#1E1B4B"),
    "accent_badge_bg":  ("#EDE9FE", "#2E1065"),
    "accent_badge_text":("#5B21B6", "#DDD6FE"),

    # Backgrounds
    "bg_root":          ("#F8FAFC", "#0F172A"),
    "bg_sidebar":       ("#FFFFFF", "#1E293B"),
    "bg_card":          ("#FFFFFF", "#1E293B"),
    "bg_input":         ("#F1F5F9", "#334155"),
    "bg_hover":         ("#E2E8F0", "#334155"),

    # Text hierarchy — clear, crisp contrast
    "text_primary":     ("#0F172A", "#F8FAFC"),  # High-contrast headings
    "text_secondary":   ("#334155", "#CBD5E1"),  # Readability-focused body & labels
    "text_muted":       ("#475569", "#94A3B8"),  # Supporting secondary descriptions
    "text_on_accent":   ("#FFFFFF", "#FFFFFF"),

    # Borders & separators
    "border":           ("#E2E8F0", "#334155"),
    "border_sidebar":   ("#E2E8F0", "#334155"),

    # Semantic states
    "success":          ("#059669", "#34D399"),
    "success_bg":       ("#ECFDF5", "#064E3B"),
    "danger":           ("#DC2626", "#F87171"),
    "warning":          ("#D97706", "#FBBF24"),
    "warning_bg":       ("#FFFBEB", "#451A03"),

    # Navigation sidebar
    "nav_active_bg":    ("#EEF2FF", "#1E1B4B"),
    "nav_active_text":  ("#4F46E5", "#A5B4FC"),
    "nav_inactive":     ("#334155", "#94A3B8"),

    # Export pill buttons
    "pill_txt":         ("#6366F1", "#818CF8"),
    "pill_srt":         ("#0891B2", "#22D3EE"),
    "pill_json":        ("#D97706", "#F59E0B"),
}


# ═══════════════════════════════════════════════════════════════════
#  NAV BUTTON — Custom sidebar navigation item
# ═══════════════════════════════════════════════════════════════════
class NavButton(ctk.CTkButton):
    """A sidebar navigation button that looks like a modern nav link."""

    def __init__(self, master, text, icon="", is_active=False, **kwargs):
        super().__init__(
            master,
            text=f"  {icon}   {text}" if icon else f"   {text}",
            anchor="w",
            height=40,
            corner_radius=10,
            border_spacing=8,
            font=ctk.CTkFont(size=13, weight="bold" if is_active else "normal"),
            fg_color=C["nav_active_bg"] if is_active else "transparent",
            hover_color=C["bg_hover"],
            text_color=C["nav_active_text"] if is_active else C["nav_inactive"],
            **kwargs,
        )
        self._is_active = is_active

    def set_active(self, active: bool):
        self._is_active = active
        self.configure(
            fg_color=C["nav_active_bg"] if active else "transparent",
            text_color=C["nav_active_text"] if active else C["nav_inactive"],
            font=ctk.CTkFont(size=13, weight="bold" if active else "normal"),
        )


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════
class VoxcribeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Voxcribe")
        self.geometry("1240x840")
        self.minsize(1060, 720)

        # Set titlebar and taskbar icon
        self._set_app_icon()

        # Appearance mode — defaults to Light
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=C["bg_root"])

        # State variables
        self.selected_file_path: Optional[str] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.is_transcribing = False
        self.cancel_requested = False
        self.current_result: Optional[TranscriptionResult] = None
        self.segments_data: List[Segment] = []
        self._current_page = "transcribe"

        # Load logo image for sidebar header
        self._logo_image = None
        if HAS_PIL and LOGO_PNG.exists():
            try:
                self._logo_image = ctk.CTkImage(
                    light_image=Image.open(LOGO_PNG),
                    dark_image=Image.open(LOGO_PNG),
                    size=(38, 38),
                )
            except Exception:
                pass
        elif HAS_PIL and LOGO_JPG.exists():
            try:
                self._logo_image = ctk.CTkImage(
                    light_image=Image.open(LOGO_JPG),
                    dark_image=Image.open(LOGO_JPG),
                    size=(38, 38),
                )
            except Exception:
                pass

        # Ensure FFmpeg is registered
        setup_ffmpeg()

        # Build layout & views
        self._create_layout()

        # Check GPU status on boot
        self._check_hardware_support()

    def _set_app_icon(self):
        """Configure both window titlebar icon and Windows taskbar icon."""
        if LOGO_ICO.exists():
            try:
                self.iconbitmap(str(LOGO_ICO))
            except Exception:
                pass

        if LOGO_PNG.exists():
            try:
                if HAS_PIL:
                    self._tk_icon = ImageTk.PhotoImage(file=str(LOGO_PNG))
                    self.iconphoto(False, self._tk_icon)
                else:
                    self._tk_icon = tk.PhotoImage(file=str(LOGO_PNG))
                    self.iconphoto(False, self._tk_icon)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════
    #  LAYOUT BUILDER
    # ══════════════════════════════════════════════════════════════
    def _create_layout(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # ─────────────────────────────────────────────────────────
        #  LEFT SIDEBAR — Navigation & Quick Controls
        # ─────────────────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(
            self, width=260, corner_radius=0,
            fg_color=C["bg_sidebar"],
            border_width=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # ── Brand Header ──────────────────────────────────
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=18, pady=(24, 6))

        if self._logo_image:
            logo_label = ctk.CTkLabel(brand_frame, image=self._logo_image, text="")
            logo_label.pack(side="left", padx=(0, 10))
        else:
            logo_label = ctk.CTkLabel(
                brand_frame, text="🎙️",
                font=ctk.CTkFont(size=28),
            )
            logo_label.pack(side="left", padx=(0, 10))

        brand_text = ctk.CTkFrame(brand_frame, fg_color="transparent")
        brand_text.pack(side="left")

        ctk.CTkLabel(
            brand_text, text="Voxcribe",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_text, text="Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C["accent"],
        ).pack(anchor="w")

        # Version badge — crisp high-contrast pill
        ver_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        ver_frame.pack(fill="x", padx=20, pady=(2, 14))

        ver_badge = ctk.CTkLabel(
            ver_frame,
            text=f" v{__version__} • AI Studio ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["accent_badge_text"],
            fg_color=C["accent_badge_bg"],
            corner_radius=6,
        )
        ver_badge.pack(side="left")

        # Separator
        ctk.CTkFrame(
            self.sidebar, height=1, fg_color=C["border_sidebar"]
        ).pack(fill="x", padx=16, pady=(0, 14))

        # ── Navigation Section ────────────────────────────
        nav_header = ctk.CTkLabel(
            self.sidebar, text="  MAIN MENU",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text_secondary"], anchor="w",
        )
        nav_header.pack(fill="x", padx=18, pady=(0, 6))

        nav_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_container.pack(fill="x", padx=12, pady=(0, 8))

        self.nav_buttons = {}

        self.nav_buttons["transcribe"] = NavButton(
            nav_container, text="Transcribe", icon="✦", is_active=True,
            command=lambda: self._switch_page("transcribe"),
        )
        self.nav_buttons["transcribe"].pack(fill="x", pady=2)

        self.nav_buttons["settings"] = NavButton(
            nav_container, text="Settings", icon="⚙",
            command=lambda: self._switch_page("settings"),
        )
        self.nav_buttons["settings"].pack(fill="x", pady=2)

        self.nav_buttons["logs"] = NavButton(
            nav_container, text="Logs", icon="📋",
            command=lambda: self._switch_page("logs"),
        )
        self.nav_buttons["logs"].pack(fill="x", pady=2)

        # Separator
        ctk.CTkFrame(
            self.sidebar, height=1, fg_color=C["border_sidebar"]
        ).pack(fill="x", padx=16, pady=(8, 14))

        # ── Quick Settings (always visible) ──────────────
        quick_header = ctk.CTkLabel(
            self.sidebar, text="  QUICK SETTINGS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text_secondary"], anchor="w",
        )
        quick_header.pack(fill="x", padx=18, pady=(0, 8))

        settings_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        settings_container.pack(fill="x", padx=16)

        # Model
        self._sidebar_label(settings_container, "Model")
        self.combo_model = self._sidebar_combo(
            settings_container,
            values=["small (Best Balance)", "base (Fast)", "tiny (Ultra Fast)",
                    "medium (High Accuracy)", "large-v3 (Studio)", "turbo (Fast v3)"],
            default="small (Best Balance)",
            command=self._on_model_change,
        )

        # Language
        self._sidebar_label(settings_container, "Language")
        lang_names = [item[0] for item in LANGUAGE_OPTIONS]
        self.combo_lang = self._sidebar_combo(
            settings_container, values=lang_names,
            default="Bengali (বাংলা) - bn",
        )

        # Device toggle
        self._sidebar_label(settings_container, "Acceleration Device")
        self.seg_device = ctk.CTkSegmentedButton(
            settings_container,
            values=["CPU", "GPU"],
            height=32, corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text_primary"],
            selected_color=C["accent"],
            selected_hover_color=C["accent_hover"],
            unselected_color=C["bg_input"],
            unselected_hover_color=C["bg_hover"],
            command=self._on_device_toggle,
        )
        self.seg_device.set("CPU")
        self.seg_device.pack(fill="x", pady=(0, 10))

        # ── Spacer ──
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # ── Bottom: Theme Toggle Card ─────────────────────
        theme_card = ctk.CTkFrame(
            self.sidebar, corner_radius=12,
            fg_color=C["accent_soft"],
            border_width=0,
        )
        theme_card.pack(fill="x", padx=14, pady=(0, 16))

        theme_inner = ctk.CTkFrame(theme_card, fg_color="transparent")
        theme_inner.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            theme_inner, text="🎨  Appearance Theme",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["text_primary"],
        ).pack(anchor="w", pady=(0, 6))

        self.seg_theme = ctk.CTkSegmentedButton(
            theme_inner,
            values=["Light", "Dark", "System"],
            height=30, corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text_primary"],
            selected_color=C["accent"],
            selected_hover_color=C["accent_hover"],
            unselected_color=C["bg_input"],
            unselected_hover_color=C["bg_hover"],
            command=self._change_theme,
        )
        self.seg_theme.set("Light")
        self.seg_theme.pack(fill="x")

        # ─────────────────────────────────────────────────────────
        #  RIGHT MAIN PANEL — Page container
        # ─────────────────────────────────────────────────────────
        self.main_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self.main_panel.rowconfigure(0, weight=1)
        self.main_panel.columnconfigure(0, weight=1)

        # Create page frames
        self._pages = {}
        self._build_transcribe_page()
        self._build_settings_page()
        self._build_logs_page()

        # Show default page
        self._switch_page("transcribe")

    # ─────────────────────────────────────────────────────────────
    #  PAGE: TRANSCRIBE (main workspace)
    # ─────────────────────────────────────────────────────────────
    def _build_transcribe_page(self):
        page = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self._pages["transcribe"] = page

        page.rowconfigure(4, weight=1)
        page.columnconfigure(0, weight=1)

        # ── Header ──
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 4))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="My Workspace",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["text_muted"],
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header, text="Speech-to-Text Studio",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=C["text_primary"],
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Status pill on the right
        self.lbl_status_pill = ctk.CTkLabel(
            header, text="  ● Ready  ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["success"],
            fg_color=C["success_bg"],
            corner_radius=12,
        )
        self.lbl_status_pill.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 4))

        # ── File Selection Card ──
        self.file_card = ctk.CTkFrame(
            page, corner_radius=16,
            fg_color=C["bg_card"],
            border_width=1, border_color=C["border"],
        )
        self.file_card.grid(row=1, column=0, sticky="ew", padx=20, pady=(14, 0))

        file_inner = ctk.CTkFrame(self.file_card, fg_color="transparent")
        file_inner.pack(fill="both", padx=24, pady=20)

        # Left: icon + text
        file_left = ctk.CTkFrame(file_inner, fg_color="transparent")
        file_left.pack(side="left", fill="x", expand=True)

        # Circular icon holder
        self.file_icon_holder = ctk.CTkFrame(
            file_left, width=48, height=48,
            corner_radius=24,
            fg_color=C["accent_soft"],
        )
        self.file_icon_holder.pack(side="left", padx=(0, 16))
        self.file_icon_holder.pack_propagate(False)

        self.lbl_file_icon = ctk.CTkLabel(
            self.file_icon_holder, text="📂",
            font=ctk.CTkFont(size=20),
        )
        self.lbl_file_icon.place(relx=0.5, rely=0.5, anchor="center")

        file_text = ctk.CTkFrame(file_left, fg_color="transparent")
        file_text.pack(side="left", fill="x", expand=True)

        self.lbl_file_title = ctk.CTkLabel(
            file_text, text="No media file selected",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=C["text_primary"], anchor="w",
        )
        self.lbl_file_title.pack(anchor="w")

        # Backward compatibility alias
        self.lbl_selected_file = self.lbl_file_title

        self.lbl_file_sub = ctk.CTkLabel(
            file_text, text="Click browse to select an audio or video file",
            font=ctk.CTkFont(size=12),
            text_color=C["text_secondary"], anchor="w",
        )
        self.lbl_file_sub.pack(anchor="w", pady=(3, 0))

        self.btn_browse = ctk.CTkButton(
            file_inner, text="Browse Media File",
            command=self._browse_file,
            width=170, height=40, corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["text_on_accent"],
        )
        self.btn_browse.pack(side="right")

        # ── Action Row ──
        action_row = ctk.CTkFrame(page, fg_color="transparent")
        action_row.grid(row=2, column=0, sticky="ew", padx=20, pady=(14, 0))
        action_row.columnconfigure(0, weight=1)

        self.btn_transcribe = ctk.CTkButton(
            action_row, text="✦  Start Transcription",
            command=self._start_transcription,
            height=50, corner_radius=14,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["text_on_accent"],
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.btn_transcribe.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(
            action_row, text="⏹  Cancel",
            command=self._cancel_transcription,
            height=50, width=130, corner_radius=14,
            fg_color=C["danger"], hover_color=("#B91C1C", "#EF4444"),
            text_color=C["text_on_accent"],
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
        )
        self.btn_cancel.grid(row=0, column=1)

        # ── Progress ──
        progress_frame = ctk.CTkFrame(page, fg_color="transparent")
        progress_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            progress_frame, height=7, corner_radius=4,
            progress_color=C["accent"],
            fg_color=C["bg_input"],
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 4))

        self.lbl_status = ctk.CTkLabel(
            progress_frame, text="Status: Ready",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["text_secondary"], anchor="w",
        )
        self.lbl_status.pack(anchor="w")

        # ── Results Card ──
        results_card = ctk.CTkFrame(
            page, corner_radius=16,
            fg_color=C["bg_card"],
            border_width=1, border_color=C["border"],
        )
        results_card.grid(row=4, column=0, sticky="nsew", padx=20, pady=(12, 0))
        results_card.rowconfigure(1, weight=1)
        results_card.columnconfigure(0, weight=1)

        # Tab header inside card
        tab_header = ctk.CTkFrame(results_card, fg_color="transparent")
        tab_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))

        self.tab_buttons = {}
        self._active_tab = "segments"

        for key, label in [("segments", "🕒 Segments"), ("fulltext", "📄 Full Text")]:
            is_active = key == "segments"
            btn = ctk.CTkButton(
                tab_header, text=label,
                height=34, corner_radius=8,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=C["accent_soft"] if is_active else "transparent",
                hover_color=C["bg_hover"],
                text_color=C["accent_deep"] if is_active else C["text_secondary"],
                command=lambda k=key: self._switch_tab(k),
            )
            btn.pack(side="left", padx=(0, 6))
            self.tab_buttons[key] = btn

        # Text areas (stacked, we toggle visibility)
        self.txt_segments = ctk.CTkTextbox(
            results_card, wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=C["text_primary"],
            corner_radius=12,
            fg_color=C["bg_input"],
            border_width=0,
        )
        self.txt_segments.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 14))

        self.txt_fulltext = ctk.CTkTextbox(
            results_card, wrap="word",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=C["text_primary"],
            corner_radius=12,
            fg_color=C["bg_input"],
            border_width=0,
        )
        # Hidden initially
        self.txt_fulltext.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 14))
        self.txt_fulltext.grid_remove()

        # ── Export Bar ──
        export_bar = ctk.CTkFrame(
            page, corner_radius=14,
            fg_color=C["bg_card"],
            border_width=1, border_color=C["border"],
        )
        export_bar.grid(row=5, column=0, sticky="ew", padx=20, pady=(10, 20))

        export_inner = ctk.CTkFrame(export_bar, fg_color="transparent")
        export_inner.pack(fill="x", padx=18, pady=12)

        ctk.CTkLabel(
            export_inner, text="Export",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C["text_primary"],
        ).pack(side="left", padx=(0, 14))

        self.btn_export_txt = self._pill_button(export_inner, "💾 Text", C["pill_txt"], lambda: self._export_result("txt"))
        self.btn_export_srt = self._pill_button(export_inner, "🎬 SRT", C["pill_srt"], lambda: self._export_result("srt"))
        self.btn_export_json = self._pill_button(export_inner, "{ } JSON", C["pill_json"], lambda: self._export_result("json"))

        self.btn_copy = ctk.CTkButton(
            export_inner, text="📋 Copy",
            width=95, height=34, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=C["bg_input"],
            hover_color=C["bg_hover"],
            text_color=C["text_primary"],
            text_color_disabled=("#94A3B8", "#64748B"),
            border_width=1, border_color=C["border"],
            command=self._copy_to_clipboard,
        )
        self.btn_copy.pack(side="right", padx=3)

        self._update_export_buttons_state(False)

    # ─────────────────────────────────────────────────────────────
    #  PAGE: SETTINGS (engine configuration)
    # ─────────────────────────────────────────────────────────────
    def _build_settings_page(self):
        page = ctk.CTkScrollableFrame(
            self.main_panel, fg_color="transparent",
            corner_radius=0,
        )
        self._pages["settings"] = page

        # Header
        ctk.CTkLabel(
            page, text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=C["text_primary"],
        ).pack(anchor="w", padx=24, pady=(24, 4))

        ctk.CTkLabel(
            page, text="Configure transcription engine and processing options",
            font=ctk.CTkFont(size=13),
            text_color=C["text_secondary"],
        ).pack(anchor="w", padx=24, pady=(0, 16))

        # ── Card: Compute Settings ──
        compute_card = self._settings_card(page, "⚡  Compute Configuration")

        self._settings_label(compute_card, "Compute Precision / Type")
        self.combo_compute = ctk.CTkComboBox(
            compute_card,
            values=["int8", "float16", "float32"],
            width=280, height=36, corner_radius=8,
            border_width=1, border_color=C["border"],
            fg_color=C["bg_input"],
            text_color=C["text_primary"],
            button_color=C["accent"],
            button_hover_color=C["accent_hover"],
            dropdown_fg_color=C["bg_card"],
            dropdown_text_color=C["text_primary"],
            font=ctk.CTkFont(size=13),
        )
        self.combo_compute.set("int8")
        self.combo_compute.pack(anchor="w", padx=20, pady=(0, 16))

        self._settings_label(compute_card, "Inference Options")

        self.chk_vad = ctk.CTkCheckBox(
            compute_card, text="  Enable Silero VAD (Remove silence)",
            font=ctk.CTkFont(size=13, weight="bold"),
            checkbox_width=22, checkbox_height=22, corner_radius=6,
            border_width=2,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["text_primary"],
        )
        self.chk_vad.select()
        self.chk_vad.pack(anchor="w", padx=20, pady=(0, 8))

        self.chk_translate = ctk.CTkCheckBox(
            compute_card, text="  Translate speech to English",
            font=ctk.CTkFont(size=13, weight="bold"),
            checkbox_width=22, checkbox_height=22, corner_radius=6,
            border_width=2,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["text_primary"],
        )
        self.chk_translate.pack(anchor="w", padx=20, pady=(0, 20))

        # ── Card: About (Completely Anonymized) ──
        about_card = self._settings_card(page, "ℹ️  About Voxcribe")

        about_text = (
            f"Voxcribe v{__version__}\n"
            "Powered by faster-whisper • CTranslate2\n"
            "Privacy-first, local offline transcription engine\n\n"
            "License: MIT"
        )
        ctk.CTkLabel(
            about_card, text=about_text,
            font=ctk.CTkFont(size=13),
            text_color=C["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 20))

    # ─────────────────────────────────────────────────────────────
    #  PAGE: LOGS (execution logs)
    # ─────────────────────────────────────────────────────────────
    def _build_logs_page(self):
        page = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self._pages["logs"] = page
        page.rowconfigure(1, weight=1)
        page.columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Execution Logs",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=C["text_primary"],
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header, text="Real-time engine output and diagnostic messages",
            font=ctk.CTkFont(size=13),
            text_color=C["text_secondary"],
        ).grid(row=1, column=0, sticky="w")

        btn_clear = ctk.CTkButton(
            header, text="🗑 Clear Logs",
            width=110, height=32, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=C["bg_input"],
            hover_color=C["bg_hover"],
            text_color=C["text_primary"],
            border_width=1, border_color=C["border"],
            command=lambda: self.txt_logs.delete("1.0", "end"),
        )
        btn_clear.grid(row=0, column=1, rowspan=2, sticky="e")

        # Logs card
        logs_card = ctk.CTkFrame(
            page, corner_radius=16,
            fg_color=C["bg_card"],
            border_width=1, border_color=C["border"],
        )
        logs_card.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        logs_card.rowconfigure(0, weight=1)
        logs_card.columnconfigure(0, weight=1)

        self.txt_logs = ctk.CTkTextbox(
            logs_card, wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=C["text_primary"],
            corner_radius=12,
            fg_color=C["bg_input"],
            border_width=0,
        )
        self.txt_logs.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

    # ══════════════════════════════════════════════════════════════
    #  HELPER WIDGET FACTORIES
    # ══════════════════════════════════════════════════════════════
    def _sidebar_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["text_primary"],
        ).pack(anchor="w", padx=4, pady=(0, 4))

    def _sidebar_combo(self, parent, values, default, command=None):
        combo = ctk.CTkComboBox(
            parent, values=values,
            height=34, corner_radius=8,
            border_width=1, border_color=C["border"],
            fg_color=C["bg_input"],
            text_color=C["text_primary"],
            button_color=C["accent"],
            button_hover_color=C["accent_hover"],
            dropdown_fg_color=C["bg_card"],
            dropdown_text_color=C["text_primary"],
            font=ctk.CTkFont(size=11, weight="bold"),
            command=command,
        )
        combo.set(default)
        combo.pack(fill="x", pady=(0, 10))
        return combo

    def _settings_card(self, parent, title):
        card = ctk.CTkFrame(
            parent, corner_radius=16,
            fg_color=C["bg_card"],
            border_width=1, border_color=C["border"],
        )
        card.pack(fill="x", padx=20, pady=(0, 14))

        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=C["text_primary"],
        ).pack(anchor="w", padx=20, pady=(18, 12))

        return card

    def _settings_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(0, 4))

    def _pill_button(self, parent, text, color, command):
        btn = ctk.CTkButton(
            parent, text=text,
            width=95, height=34, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=color,
            hover_color=color,
            text_color=C["text_on_accent"],
            text_color_disabled=("#E2E8F0", "#475569"),
            command=command,
        )
        btn.pack(side="left", padx=3)
        return btn

    # ══════════════════════════════════════════════════════════════
    #  PAGE / TAB SWITCHING
    # ══════════════════════════════════════════════════════════════
    def _switch_page(self, page_name: str):
        for name, frame in self._pages.items():
            frame.grid_remove()

        self._pages[page_name].grid(row=0, column=0, sticky="nsew")
        self._current_page = page_name

        for name, btn in self.nav_buttons.items():
            btn.set_active(name == page_name)

    def _switch_tab(self, tab_name: str):
        self._active_tab = tab_name

        for key, btn in self.tab_buttons.items():
            is_active = key == tab_name
            btn.configure(
                fg_color=C["accent_soft"] if is_active else "transparent",
                text_color=C["accent_deep"] if is_active else C["text_secondary"],
                font=ctk.CTkFont(size=12, weight="bold"),
            )

        if tab_name == "segments":
            self.txt_fulltext.grid_remove()
            self.txt_segments.grid()
        else:
            self.txt_segments.grid_remove()
            self.txt_fulltext.grid()

    # ══════════════════════════════════════════════════════════════
    #  EVENT HANDLERS & TRANSCRIPTION ENGINE
    # ══════════════════════════════════════════════════════════════
    def _check_hardware_support(self):
        """Check if CUDA GPU is available on system and set default."""
        try:
            import ctranslate2
            cuda_count = ctranslate2.get_cuda_device_count() if hasattr(ctranslate2, "get_cuda_device_count") else 0
            if cuda_count > 0:
                self.seg_device.set("GPU")
                self.combo_compute.set("float16")
                self._append_log(f"⚡ NVIDIA GPU detected ({cuda_count} device). GPU acceleration ready.")
            else:
                self.seg_device.set("CPU")
                self.combo_compute.set("int8")
                self._append_log("ℹ️ Running in CPU mode (Optimized with int8 quantization).")
        except Exception as e:
            self._append_log(f"Hardware check note: {e}")

    def _change_theme(self, mode: str):
        clean = mode.replace("☀ ", "").replace("🌙 ", "")
        ctk.set_appearance_mode(clean)

    def _on_model_change(self, val: str):
        self._append_log(f"Model selected: {val}")

    def _on_device_toggle(self, val: str):
        if val == "GPU":
            self.combo_compute.set("float16")
        else:
            self.combo_compute.set("int8")

    def _browse_file(self):
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

            self.lbl_file_title.configure(
                text=f"{file_name}  ({file_size_mb:.1f} MB)",
                text_color=C["text_primary"],
            )
            self.lbl_file_sub.configure(
                text=f"Ready  •  {os.path.dirname(self.selected_file_path)}",
                text_color=C["text_secondary"],
            )
            self.lbl_file_icon.configure(text="🎵")
            self.file_icon_holder.configure(fg_color=("#E0F2FE", "#1E3A5F"))
            self.lbl_status.configure(text=f"Status: Ready — {file_name}")
            self._append_log(f"Selected file: {self.selected_file_path} ({file_size_mb:.1f} MB)")

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

        # Update UI
        self.btn_transcribe.configure(state="disabled", text="⏳  Transcribing...")
        self.btn_cancel.configure(state="normal")
        self.btn_browse.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.lbl_status_pill.configure(
            text="  ● Processing  ",
            text_color=C["warning"],
            fg_color=C["warning_bg"],
        )
        self._update_export_buttons_state(False)

        # Clear text
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
            f"\n{'─'*40}\n"
            f"Starting Transcription\n"
            f"File: {self.selected_file_path}\n"
            f"Model: {model_str} | Lang: {lang_code} | Device: {device_mode} ({compute_type})\n"
            f"Task: {task_name} | VAD: {use_vad}\n"
            f"{'─'*40}"
        )

        # Ensure we're on the transcribe page
        self._switch_page("transcribe")

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

        ts_start = format_timestamp(seg.start)
        ts_end = format_timestamp(seg.end)
        seg_entry = f"[{ts_start} → {ts_end}]  {seg.text}\n"
        self.txt_segments.insert("end", seg_entry)
        self.txt_segments.see("end")

        self.txt_fulltext.insert("end", seg.text + "\n\n")
        self.txt_fulltext.see("end")

    def _on_transcription_completed(self, result: TranscriptionResult, elapsed: float):
        self.is_transcribing = False
        self.current_result = result
        self.btn_transcribe.configure(state="normal", text="✦  Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.progress_bar.set(1.0)

        lang_name = get_language_name(result.language)
        msg = f"Done in {elapsed:.1f}s • {len(result.segments)} segments • {lang_name} ({result.language_probability:.0%})"
        self.lbl_status.configure(text=f"Status: {msg}")
        self.lbl_status_pill.configure(
            text="  ● Complete  ",
            text_color=C["success"],
            fg_color=C["success_bg"],
        )
        self._append_log(f"🎉 {msg}")
        self._update_export_buttons_state(True)
        self._switch_tab("segments")

    def _cancel_transcription(self):
        if self.is_transcribing:
            self.cancel_requested = True
            self.lbl_status.configure(text="Status: Cancelling...")
            self.btn_cancel.configure(state="disabled")

    def _on_transcription_cancelled(self):
        self.is_transcribing = False
        self.btn_transcribe.configure(state="normal", text="✦  Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.lbl_status.configure(text="Status: Cancelled by user.")
        self.lbl_status_pill.configure(
            text="  ● Ready  ",
            text_color=C["success"],
            fg_color=C["success_bg"],
        )
        self._append_log("⏹️ Transcription stopped by user.")
        if len(self.segments_data) > 0:
            self._update_export_buttons_state(True)

    def _on_transcription_failed(self, error_message: str):
        self.is_transcribing = False
        self.btn_transcribe.configure(state="normal", text="✦  Start Transcription")
        self.btn_cancel.configure(state="disabled")
        self.btn_browse.configure(state="normal")
        self.lbl_status.configure(text="Status: Transcription failed.")
        self.lbl_status_pill.configure(
            text="  ● Error  ",
            text_color=C["danger"],
            fg_color=("#FEF2F2", "#450A0A"),
        )
        self._append_log(f"❌ Error: {error_message}")
        messagebox.showerror("Transcription Error", f"Failed to transcribe:\n\n{error_message}")

    def _update_export_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_export_txt.configure(state=state)
        self.btn_export_srt.configure(state=state)
        self.btn_export_json.configure(state=state)
        self.btn_copy.configure(state=state)

    def _copy_to_clipboard(self):
        if self._active_tab == "segments":
            content = self.txt_segments.get("1.0", "end-1c")
        else:
            content = self.txt_fulltext.get("1.0", "end-1c")

        if content.strip():
            self.clipboard_clear()
            self.clipboard_append(content)
            messagebox.showinfo("Copied", "Text copied to clipboard!")

    def _export_result(self, format_type: str):
        if not self.segments_data:
            messagebox.showwarning("No Data", "No transcribed segments to export.")
            return

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

            self._append_log(f"💾 Saved: {save_path}")
            messagebox.showinfo("Saved", f"Exported to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save: {e}")

    def _append_log(self, text: str):
        self.txt_logs.insert("end", text + "\n")
        self.txt_logs.see("end")


def launch_gui():
    """Main entrypoint for desktop GUI app."""
    app = VoxcribeApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
