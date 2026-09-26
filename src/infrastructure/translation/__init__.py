"""Translation infrastructure package."""

from src.infrastructure.translation.adapter import OfflineTranslationAdapter
from src.infrastructure.translation.argos_translator import ArgosTranslateTranslator

__all__ = ["OfflineTranslationAdapter", "ArgosTranslateTranslator"]
