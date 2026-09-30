"""Argos-Translate implementation of the Translator domain port.

Runs offline OpenNMT/CTranslate2-based translations locally with zero runtime network calls.
Language models are installed once and cached locally.
Compatible with PyInstaller frozen bundles and source execution.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union

from src.domain.entities import Language
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
    """Offline translator using Argos Translate."""

    def __init__(
        self,
        from_code: Union[str, Language] = Language.JAPANESE,
        to_code: Union[str, Language] = Language.ENGLISH,
        packages_dir: Optional[str] = None,
    ) -> None:
        self.from_code = (Language.from_code(from_code) if isinstance(from_code, str) else from_code).value
        self.to_code = (Language.from_code(to_code) if isinstance(to_code, str) else to_code).value
        self._translation_models: Dict[Tuple[str, str], object] = {}

        # Check for bundled packages inside frozen package or models/argos
        if packages_dir is None:
            bundled_candidate = _resolve_resource_path("models/argos_packages")
            if bundled_candidate.exists():
                self.packages_dir = str(bundled_candidate)
            else:
                self.packages_dir = None
        else:
            self.packages_dir = str(_resolve_resource_path(packages_dir))

    def _ensure_translation_loaded(self, from_code: str, to_code: str) -> object:
        """Load and cache the installed language translation pair."""
        pair = (from_code, to_code)
        if pair in self._translation_models:
            return self._translation_models[pair]

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
            (lang for lang in installed_languages if lang.code == from_code), None
        )
        to_lang = next(
            (lang for lang in installed_languages if lang.code == to_code), None
        )

        if from_lang is None or to_lang is None:
            raise RuntimeError(
                f"Argos Translate language package '{from_code}->{to_code}' is not installed.\n"
                f"Please run 'python scripts/install_translation_model.py' to install it offline."
            )

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            raise RuntimeError(
                f"No direct or pivot translation route found from '{from_code}' to '{to_code}'."
            )

        self._translation_models[pair] = translation
        return translation

    def translate(
        self,
        text: str,
        source_language: Language = Language.JAPANESE,
        target_language: Language = Language.ENGLISH,
    ) -> str:
        """Translate text from source_language to target_language offline."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return ""

        src_lang = Language.from_code(source_language) if isinstance(source_language, str) else source_language
        tgt_lang = Language.from_code(target_language) if isinstance(target_language, str) else target_language

        if not isinstance(src_lang, Language):
            raise ValueError(f"source_language must be a valid Language enum, got {source_language}")
        if not isinstance(tgt_lang, Language):
            raise ValueError(f"target_language must be a valid Language enum, got {target_language}")

        if src_lang == tgt_lang:
            return cleaned_text

        translation_model = self._ensure_translation_loaded(src_lang.value, tgt_lang.value)
        translated: str = translation_model.translate(cleaned_text)  # type: ignore
        return translated.strip()
