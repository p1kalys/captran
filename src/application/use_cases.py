"""Application use case orchestrators.

This layer depends ONLY on the domain layer (entities and ports).
No infrastructure or external dependencies are imported here.
"""

from src.application.live_caption_use_case import LiveCaptionUseCase

# Backward-compatibility alias
LiveCaptioningUseCase = LiveCaptionUseCase

__all__ = ["LiveCaptionUseCase", "LiveCaptioningUseCase"]
