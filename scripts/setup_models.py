"""Offline model setup script for CapTran.

Downloads and caches:
1. Silero VAD ONNX model
2. Multilingual faster-whisper model (once)
3. Argos Translate language packages (direct pairs + pivot via English)

Features:
- Estimates total download and disk size before downloading
- Supports --languages flag to set up a subset of the 7 supported languages
- Stages packages into models/argos_packages/ for complete offline execution
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import List, Set, Tuple
import urllib.request

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUPPORTED_LANGUAGES = ["hi", "ja", "en", "es", "fr", "de", "ko"]

LANG_NAMES = {
    "hi": "Hindi",
    "ja": "Japanese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ko": "Korean",
}


def get_required_argos_pairs(languages: List[str]) -> List[Tuple[str, str]]:
    """Determine necessary Argos direct packages to enable translation among selected languages.
    
    To connect any selected non-English language with others via English pivot,
    we need (lang -> 'en') and ('en' -> lang).
    """
    pairs: Set[Tuple[str, str]] = set()
    for lang in languages:
        if lang != "en":
            pairs.add((lang, "en"))
            pairs.add(("en", lang))
    return sorted(list(pairs))


def setup_whisper_model(model_size: str = "small", download_root: Path = Path("models/whisper")) -> None:
    """Download and cache faster-whisper multilingual model."""
    try:
        from faster_whisper import download_model
    except ImportError:
        print("[ERROR] faster-whisper is not installed. Run: pip install faster-whisper")
        sys.exit(1)

    download_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/3] Downloading Faster-Whisper [{model_size}] multilingual model weights...")
    model_path = download_model(model_size, output_dir=str(download_root))
    print(f"[SUCCESS] Faster-Whisper [{model_size}] saved to: {model_path}")


def setup_silero_vad(dest_path: Path = Path("models/silero_vad.onnx")) -> None:
    """Download Silero VAD ONNX model if not present."""
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        print(f"\n[2/3] Silero VAD model already present at: {dest_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[2/3] Downloading Silero VAD ONNX model...")
    url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

    fd, temp_file_path = tempfile.mkstemp(prefix="silero_vad_", suffix=".tmp", dir=str(dest_path.parent))
    os.close(fd)
    temp_path = Path(temp_file_path)

    try:
        urllib.request.urlretrieve(url, str(temp_path))
        if not temp_path.exists() or temp_path.stat().st_size <= 1000000:
            raise ValueError("Downloaded VAD model failed validation (missing or file size too small).")
        temp_path.replace(dest_path)
        print(f"[SUCCESS] Silero VAD saved to: {dest_path}")
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise e


def setup_argos_packages(
    required_pairs: List[Tuple[str, str]],
    staging_dir: Path = Path("models/argos_packages"),
) -> None:
    """Download and install required Argos Translate packages."""
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        print("[ERROR] argostranslate is not installed. Run: pip install argostranslate")
        sys.exit(1)

    staging_dir.mkdir(parents=True, exist_ok=True)
    print("\n[3/3] Updating Argos Translate package index...")
    try:
        argostranslate.package.update_package_index()
    except Exception as e:
        print(f"[WARNING] Could not update remote package index ({e}). Using cached index.")

    available_packages = argostranslate.package.get_available_packages()
    installed_langs = argostranslate.translate.get_installed_languages()

    def is_installed(src: str, tgt: str) -> bool:
        s_lang = next((l for l in installed_langs if l.code == src), None)
        t_lang = next((l for l in installed_langs if l.code == tgt), None)
        return s_lang is not None and t_lang is not None and s_lang.get_translation(t_lang) is not None

    failed_pairs: List[Tuple[str, str, str]] = []

    print(f"\nInstalling {len(required_pairs)} Argos Translate language pairs:")
    for src, tgt in required_pairs:
        pair_str = f"{src}->{tgt}"
        src_name = LANG_NAMES.get(src, src)
        tgt_name = LANG_NAMES.get(tgt, tgt)

        if is_installed(src, tgt):
            print(f"  * {pair_str:<10} ({src_name} -> {tgt_name}): Already installed [OK]")
            continue

        pkg = next((p for p in available_packages if p.from_code == src and p.to_code == tgt), None)
        if pkg is None:
            msg = "Not found in index"
            print(f"  * {pair_str:<10} ({src_name} -> {tgt_name}): [WARNING] {msg}")
            failed_pairs.append((src, tgt, msg))
            continue

        print(f"  * Downloading {pair_str:<10} ({src_name} -> {tgt_name})...")
        try:
            downloaded_path = pkg.download()
            dest_file = staging_dir / Path(downloaded_path).name
            shutil.copyfile(downloaded_path, str(dest_file))
            argostranslate.package.install_from_path(str(dest_file))
            print(f"    Installed & staged to {dest_file.name}")
        except Exception as err:
            print(f"    [ERROR] Failed to install {pair_str}: {err}")
            failed_pairs.append((src, tgt, str(err)))

    if failed_pairs:
        summary = ", ".join(f"{s}->{t} ({reason})" for s, t, reason in failed_pairs)
        raise RuntimeError(f"Failed to install {len(failed_pairs)} required translation pairs: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and stage all offline models for CapTran."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=SUPPORTED_LANGUAGES,
        choices=SUPPORTED_LANGUAGES,
        help="Subset of languages to configure (default: all 7: hi ja en es fr de ko)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Faster-Whisper model size to download (default: small)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated disk usage and download size without downloading",
    )
    args = parser.parse_args()

    required_pairs = get_required_argos_pairs(args.languages)

    # Estimate sizes:
    # Whisper small: ~480 MB download, ~500 MB disk
    # Whisper tiny: ~75 MB, base: ~145 MB, medium: ~1.5 GB
    whisper_sizes = {
        "tiny": (75, 80),
        "base": (145, 150),
        "small": (480, 500),
        "medium": (1500, 1600),
        "large-v3": (3100, 3200),
    }
    w_dl, w_disk = whisper_sizes.get(args.whisper_model, (500, 500))

    # Silero VAD: ~2 MB
    vad_dl, vad_disk = 2, 2

    # Argos packages: ~45 MB download & ~90 MB disk per pair
    argos_dl = len(required_pairs) * 45
    argos_disk = len(required_pairs) * 90

    total_dl = w_dl + vad_dl + argos_dl
    total_disk = w_disk + vad_disk + argos_disk

    print("=" * 70)
    print("  CAPTRAN MODEL ASSET SETUP & OFFLINE PREPARATION")
    print("=" * 70)
    print(f" Configured Languages ({len(args.languages)}): {', '.join(args.languages)}")
    print(f" Whisper Model Size:    {args.whisper_model}")
    print(f" Argos Packages:        {len(required_pairs)} direct pairs ({', '.join(f'{s}->{t}' for s, t in required_pairs)})")
    print("-" * 70)
    print(f" Estimated Download Size: ~{total_dl} MB")
    print(f" Estimated Disk Usage:    ~{total_disk} MB")
    print("=" * 70)

    if args.dry_run:
        print("\n[Dry run complete. No models downloaded.]")
        return

    setup_whisper_model(model_size=args.whisper_model)
    setup_silero_vad()
    setup_argos_packages(required_pairs=required_pairs)

    print("\n" + "=" * 70)
    print("  ALL REQUESTED MODELS CONFIGURED & READY FOR OFFLINE EXECUTION!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
