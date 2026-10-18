#!/usr/bin/env python3
"""
Voxcribe - Local Video/Audio Transcription Tool

Thin entry point:
    python transcribe.py audio_file.mp3

License: MIT
"""

import sys
from voxcribe.cli import main
from voxcribe.gui import launch_gui

if __name__ == "__main__":
    if len(sys.argv) == 1 or "--gui" in sys.argv:
        launch_gui()
    else:
        main()