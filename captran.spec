# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Spec File for CapTran (JA->EN Live Captioner).

Bundles the application, Silero VAD ONNX model, Faster-Whisper CTranslate2 models,
Argos Translate ja->en packages, and all required C++/runtime DLLs into a standalone
100% offline Windows executable.
"""

from pathlib import Path
import sys
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None
project_root = Path.cwd()

# 1. Collect all package data and dynamic libraries for ML runtimes
datas = []
binaries = []
hiddenimports = [
    "faster_whisper",
    "ctranslate2",
    "argostranslate",
    "argostranslate.package",
    "argostranslate.translate",
    "onnxruntime",
    "pyaudiowpatch",
    "scipy",
    "scipy.signal",
    "scipy.special",
    "numpy",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "queue",
    "threading",
    "dataclasses",
    "src",
    "src.domain",
    "src.domain.entities",
    "src.domain.ports",
    "src.application",
    "src.application.live_caption_use_case",
    "src.infrastructure",
    "src.infrastructure.audio",
    "src.infrastructure.audio.factory",
    "src.infrastructure.audio.silero_vad",
    "src.infrastructure.audio.windows_loopback",
    "src.infrastructure.stt",
    "src.infrastructure.stt.faster_whisper_stt",
    "src.infrastructure.translation",
    "src.infrastructure.translation.argos_translator",
    "src.infrastructure.settings",
    "src.infrastructure.settings.local_settings_repository",
    "src.infrastructure.ui",
    "src.infrastructure.ui.control_window",
    "src.infrastructure.ui.overlay_presenter",
    "src.infrastructure.ui.console_presenter",
    "src.composition_root_gui",
    "src.composition_root_cli",
]

# Collect submodules and data from runtime packages
for pkg in ["ctranslate2", "faster_whisper", "argostranslate", "onnxruntime", "pyaudiowpatch"]:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as e:
        print(f"[NOTE] collect_all({pkg}) notice: {e}")

# 2. Bundle local offline model assets (Silero VAD, Whisper weights, Argos packages)
models_dir = project_root / "models"
if models_dir.exists():
    datas.append((str(models_dir), "models"))

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "torch", "torchvision"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="captran",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True so both CLI and GUI output/logs are available
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="captran",
)
