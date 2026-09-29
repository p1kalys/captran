"""Local JSON file adapter implementing CustomVocabularyRepository domain port.

Persists domain-specific custom vocabulary (terms, product names, acronyms, and substitution rules)
to local disk at ~/.ja-en-captioner/custom_vocab.json.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.domain.entities import CustomVocabulary
from src.domain.ports import CustomVocabularyRepository


class JsonCustomVocabularyRepository(CustomVocabularyRepository):
    """Saves and loads CustomVocabulary from a local JSON configuration file."""

    DEFAULT_CONFIG_PATH = Path.home() / ".ja-en-captioner" / "custom_vocab.json"

    def __init__(self, config_file_path: Optional[Path] = None) -> None:
        self.config_path = (
            Path(config_file_path).resolve()
            if config_file_path is not None
            else self.DEFAULT_CONFIG_PATH.resolve()
        )

    def load(self) -> CustomVocabulary:
        """Load vocabulary from local JSON file. Returns empty CustomVocabulary if not found."""
        if not self.config_path.exists():
            return CustomVocabulary()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return CustomVocabulary()

            raw_prompt_terms = data.get("prompt_terms")
            prompt_terms = [str(t) for t in raw_prompt_terms] if isinstance(raw_prompt_terms, list) else []

            raw_source_subs = data.get("source_substitutions")
            source_subs = (
                {str(k): str(v) for k, v in raw_source_subs.items()}
                if isinstance(raw_source_subs, dict)
                else {}
            )

            raw_target_subs = data.get("target_substitutions")
            target_subs = (
                {str(k): str(v) for k, v in raw_target_subs.items()}
                if isinstance(raw_target_subs, dict)
                else {}
            )

            return CustomVocabulary(
                prompt_terms=prompt_terms,
                source_substitutions=source_subs,
                target_substitutions=target_subs,
            )
        except Exception as e:
            print(f"[WARNING] Failed to load custom vocabulary from {self.config_path} ({e}), using defaults.")
            return CustomVocabulary()

    def save(self, vocabulary: CustomVocabulary) -> None:
        """Persist vocabulary to local JSON file on disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "prompt_terms": list(vocabulary.prompt_terms),
                "source_substitutions": dict(vocabulary.source_substitutions),
                "target_substitutions": dict(vocabulary.target_substitutions),
            }
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.config_path)
        except Exception as e:
            print(f"[ERROR] Failed to save custom vocabulary to {self.config_path}: {e}")

    # Convenience helper methods
    def add_prompt_term(self, term: str) -> None:
        """Add a term to the prompt hints list and persist."""
        vocab = self.load()
        terms = list(vocab.prompt_terms)
        if term not in terms:
            terms.append(term)
            updated = CustomVocabulary(
                prompt_terms=terms,
                source_substitutions=vocab.source_substitutions,
                target_substitutions=vocab.target_substitutions,
            )
            self.save(updated)

    def add_source_substitution(self, pattern: str, replacement: str) -> None:
        """Add a pre-translation substitution rule and persist."""
        vocab = self.load()
        subs = dict(vocab.source_substitutions)
        subs[pattern] = replacement
        updated = CustomVocabulary(
            prompt_terms=vocab.prompt_terms,
            source_substitutions=subs,
            target_substitutions=vocab.target_substitutions,
        )
        self.save(updated)

    def add_target_substitution(self, pattern: str, replacement: str) -> None:
        """Add a post-translation substitution rule and persist."""
        vocab = self.load()
        subs = dict(vocab.target_substitutions)
        subs[pattern] = replacement
        updated = CustomVocabulary(
            prompt_terms=vocab.prompt_terms,
            source_substitutions=vocab.source_substitutions,
            target_substitutions=subs,
        )
        self.save(updated)

    def get_initial_prompt(self) -> str:
        """Retrieve initial prompt string from loaded vocabulary."""
        return self.load().get_initial_prompt()

    def apply_source_substitutions(self, text: str) -> str:
        """Apply pre-translation substitutions using current vocabulary."""
        return self.load().apply_source_substitutions(text)

    def apply_target_substitutions(self, text: str) -> str:
        """Apply post-translation substitutions using current vocabulary."""
        return self.load().apply_target_substitutions(text)


# Backward compatibility aliases
LocalCustomVocabularyAdapter = JsonCustomVocabularyRepository
LocalCustomVocabularyRepository = JsonCustomVocabularyRepository
