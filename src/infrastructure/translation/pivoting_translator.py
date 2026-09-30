"""PivotingTranslator implementation of the Translator domain port.

Uses direct Argos Translate packages when available for a language pair,
otherwise pivots through English (e.g., ja -> en -> de).
Caches loaded models in memory for fast multi-lingual switching during sessions.
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


class PivotingTranslator(Translator):
    """Offline translator supporting direct and English-pivoting language pairs."""

    def __init__(self, packages_dir: Optional[str] = None) -> None:
        self._translation_models: Dict[Tuple[str, str], object] = {}

        # Check for bundled packages inside frozen package or models/argos_packages
        if packages_dir is None:
            bundled_candidate = _resolve_resource_path("models/argos_packages")
            if bundled_candidate.exists():
                self.packages_dir = str(bundled_candidate)
            else:
                self.packages_dir = None
        else:
            self.packages_dir = str(_resolve_resource_path(packages_dir))

        self._installed_packages_scanned = False

    def _scan_bundled_packages(self) -> None:
        """Scan and install bundled .argosmodel packages if present."""
        if self._installed_packages_scanned:
            return
        self._installed_packages_scanned = True

        if argostranslate is not None and self.packages_dir and Path(self.packages_dir).exists():
            for pkg_file in Path(self.packages_dir).glob("*.argosmodel"):
                try:
                    argostranslate.package.install_from_path(str(pkg_file))
                except Exception:
                    pass

    def _get_direct_translation_model(self, from_code: str, to_code: str) -> Optional[object]:
        """Load and cache a direct translation model from_code -> to_code."""
        pair = (from_code, to_code)
        if pair in self._translation_models:
            return self._translation_models[pair]

        if argostranslate is None:
            raise ImportError(
                "argostranslate is not installed. Please install argostranslate."
            )

        self._scan_bundled_packages()

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next(
            (lang for lang in installed_languages if lang.code == from_code), None
        )
        to_lang = next(
            (lang for lang in installed_languages if lang.code == to_code), None
        )

        if from_lang is None or to_lang is None:
            return None

        # Look for direct package translation, excluding composite/identity pivot routes
        translation = next(
            (
                t
                for t in from_lang.translations_from
                if t.to_lang.code == to_code
                and type(t).__name__ not in ("CompositeTranslation", "IdentityTranslation")
            ),
            None,
        )
        if translation is not None:
            self._translation_models[pair] = translation
        return translation

    def _translate_step(self, text: str, from_code: str, to_code: str) -> str:
        """Execute a single translation step using cached or loaded model."""
        if from_code == to_code:
            return text

        model = self._get_direct_translation_model(from_code, to_code)
        if model is None:
            raise RuntimeError(
                f"Argos Translate package '{from_code}->{to_code}' is not installed locally."
            )

        translated: str = model.translate(text)  # type: ignore
        return translated.strip()

    def translate(
        self,
        text: str,
        source_language: Union[str, Language] = Language.JAPANESE,
        target_language: Union[str, Language] = Language.ENGLISH,
    ) -> str:
        """Translate text using direct package or two-step pivot via English."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return ""

        src_lang = (
            Language.from_code(source_language)
            if isinstance(source_language, str)
            else source_language
        )
        tgt_lang = (
            Language.from_code(target_language)
            if isinstance(target_language, str)
            else target_language
        )

        if not isinstance(src_lang, Language):
            raise ValueError(f"source_language must be a valid Language enum, got {source_language}")
        if not isinstance(tgt_lang, Language):
            raise ValueError(f"target_language must be a valid Language enum, got {target_language}")

        if src_lang == tgt_lang:
            return cleaned_text

        src_code = src_lang.value
        tgt_code = tgt_lang.value

        # 1. Try Direct Route (when either source or target is English, or direct package registered)
        if src_code == "en" or tgt_code == "en" or (src_code, tgt_code) in self._translation_models:
            direct_model = self._get_direct_translation_model(src_code, tgt_code)
            if direct_model is not None:
                translated: str = direct_model.translate(cleaned_text)  # type: ignore
                return translated.strip()

        # 2. Try Pivot Route via English
        if src_code != "en" and tgt_code != "en":
            src_to_en = self._get_direct_translation_model(src_code, "en")
            en_to_tgt = self._get_direct_translation_model("en", tgt_code)

            if src_to_en is not None and en_to_tgt is not None:
                intermediate_en = src_to_en.translate(cleaned_text).strip()  # type: ignore
                if not intermediate_en:
                    return ""
                final_out = en_to_tgt.translate(intermediate_en).strip()  # type: ignore
                return final_out

        # 3. Neither Direct nor Pivot route available
        raise RuntimeError(
            f"No direct or pivot translation route available for '{src_code}->{tgt_code}'.\n"
            f"Required packages: direct '{src_code}->{tgt_code}' or pivot '{src_code}->en' + 'en->{tgt_code}'."
        )
