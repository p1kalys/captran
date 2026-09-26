"""Argos-Translate implementation of the Translator domain port.

Runs offline OpenNMT/CTranslate2-based translations locally with zero runtime network calls.
Language models are installed once and cached locally.
Compatible with PyInstaller frozen bundles and source execution.
"""

from pathlib import Path
import sys
from typing import Optional

from src.domain.ports import Translator

try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    argostranslate = None  # type: ignore


def _resolve_resource_path(relative_or_abs_path: str) -> Path:
    """Resolve path relative to sys._MEIPASS when frozen with PyInstaller, or cwd."""
    p = Path(relative_or_abs_path)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_candidate = Path(sys._MEIPASS) / relative_or_abs_path
        if meipass_candidate.exists():
            return meipass_candidate
    return p.resolve()


class ArgosTranslateTranslator(Translator):
    """Offline Japanese-to-English translator using Argos Translate."""

    def __init__(
        self,
        from_code: str = "ja",
        to_code: str = "en",
        packages_dir: Optional[str] = None,
    ) -> None:
        self.from_code = from_code
        self.to_code = to_code
        self._translation_model: Optional[object] = None

        # Check for bundled packages inside frozen package or models/argos
        if packages_dir is None:
            bundled_candidate = _resolve_resource_path("models/argos_packages")
            if bundled_candidate.exists():
                self.packages_dir = str(bundled_candidate)
            else:
                self.packages_dir = None
        else:
            self.packages_dir = str(_resolve_resource_path(packages_dir))

    def _ensure_translation_loaded(self) -> object:
        """Load and cache the installed language translation pair."""
        if self._translation_model is not None:
            return self._translation_model

        if argostranslate is None:
            raise ImportError(
                "argostranslate is not installed. Please install argostranslate."
            )

        # If bundled package directory is available, install from bundled .argosmodel files
        if self.packages_dir and Path(self.packages_dir).exists():
            for pkg_file in Path(self.packages_dir).glob("*.argosmodel"):
                try:
                    argostranslate.package.install_from_path(str(pkg_file))
                except Exception:
                    pass

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next(
            (lang for lang in installed_languages if lang.code == self.from_code), None
        )
        to_lang = next(
            (lang for lang in installed_languages if lang.code == self.to_code), None
        )

        if from_lang is None or to_lang is None:
            raise RuntimeError(
                f"Argos Translate language package '{self.from_code}->{self.to_code}' is not installed.\n"
                f"Please run 'python scripts/install_translation_model.py' to install it offline."
            )

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            raise RuntimeError(
                f"No direct or pivot translation route found from '{self.from_code}' to '{self.to_code}'."
            )

        self._translation_model = translation
        return self._translation_model

    def translate(self, japanese_text: str) -> str:
        """Translate Japanese text to English text offline."""
        cleaned_text = japanese_text.strip()
        if not cleaned_text:
            return ""

        translation_model = self._ensure_translation_loaded()
        translated: str = translation_model.translate(cleaned_text)  # type: ignore
        return translated.strip()
