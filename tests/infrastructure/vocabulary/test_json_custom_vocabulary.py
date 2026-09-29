"""Unit tests for CustomVocabulary entity and JsonCustomVocabularyRepository adapter."""

from pathlib import Path
import tempfile
import pytest

from src.domain.entities import CustomVocabulary
from src.infrastructure.vocabulary.json_custom_vocabulary import (
    JsonCustomVocabularyRepository,
)


def test_custom_vocabulary_entity_prompt_and_substitutions() -> None:
    vocab = CustomVocabulary(
        prompt_terms=["CapTran", "Kubernetes", "PySide6", "Antigravity"],
        source_substitutions={
            "キャプトラン": "CapTran",
            "クバネティス": "Kubernetes",
        },
        target_substitutions={
            "cap tran": "CapTran",
            "k8s": "Kubernetes",
        },
    )

    # Test initial prompt generation
    assert vocab.get_initial_prompt() == "CapTran, Kubernetes, PySide6, Antigravity"

    # Test pre-translation source substitution
    ja_raw = "本日はキャプトランとクバネティスのデモを行います。"
    ja_sub = vocab.apply_source_substitutions(ja_raw)
    assert "CapTran" in ja_sub
    assert "Kubernetes" in ja_sub
    assert "キャプトラン" not in ja_sub
    assert "クバネティス" not in ja_sub

    # Test post-translation target substitution (case-insensitive)
    en_raw = "Welcome to the cap tran presentation running on k8s."
    en_sub = vocab.apply_target_substitutions(en_raw)
    assert "CapTran" in en_sub
    assert "Kubernetes" in en_sub
    assert "cap tran" not in en_sub.lower()


def test_json_custom_vocabulary_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "custom_vocab.json"
        repo = JsonCustomVocabularyRepository(config_file_path=json_path)

        # 1. Non-existent file returns empty vocabulary
        initial_vocab = repo.load()
        assert len(initial_vocab.prompt_terms) == 0
        assert len(initial_vocab.source_substitutions) == 0

        # 2. Add prompt terms and substitutions
        repo.add_prompt_term("CapTran")
        repo.add_prompt_term("FastAPI")
        repo.add_source_substitution("ファストAPI", "FastAPI")
        repo.add_target_substitution("fast api", "FastAPI")

        # 3. Reload and verify persistence
        loaded = repo.load()
        assert loaded.prompt_terms == ("CapTran", "FastAPI")
        assert loaded.source_substitutions == {"ファストAPI": "FastAPI"}
        assert loaded.target_substitutions == {"fast api": "FastAPI"}

        # 4. Verify helper methods on repo
        assert repo.get_initial_prompt() == "CapTran, FastAPI"
        assert repo.apply_source_substitutions("ファストAPIの紹介") == "FastAPIの紹介"
        assert repo.apply_target_substitutions("Introduction to fast api") == "Introduction to FastAPI"


def test_json_custom_vocabulary_invalid_types_fallback() -> None:
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "corrupt_vocab.json"
        # Write corrupt types (e.g. prompt_terms as string, substitutions as list)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt_terms": "not a list",
                    "source_substitutions": ["not", "a", "dict"],
                    "target_substitutions": 12345,
                },
                f,
            )

        repo = JsonCustomVocabularyRepository(config_file_path=json_path)
        loaded = repo.load()
        assert loaded.prompt_terms == ()
        assert loaded.source_substitutions == {}
        assert loaded.target_substitutions == {}
