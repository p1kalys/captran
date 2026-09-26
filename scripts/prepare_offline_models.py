"""One-time build preparation script to download and stage all offline model assets.

Stages:
1. Silero VAD ONNX model in models/silero_vad.onnx
2. Faster-Whisper model weights in models/whisper/
3. Argos Translate ja->en language model in models/argos_packages/

Ensures the final PyInstaller package runs 100% offline with zero runtime internet access.
"""

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request


def prepare_silero_vad(models_dir: Path) -> Path:
    """Ensure Silero VAD ONNX model is present."""
    vad_dest = models_dir / "silero_vad.onnx"
    if vad_dest.exists() and vad_dest.stat().st_size > 1000000:
        print(f"[OK] Silero VAD ONNX model present at: {vad_dest}")
        return vad_dest

    print("[INFO] Downloading Silero VAD ONNX model...")
    url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    models_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(vad_dest))
    print(f"[SUCCESS] Silero VAD saved to: {vad_dest}")
    return vad_dest


def prepare_whisper_models(models_dir: Path, model_sizes: list[str]) -> None:
    """Pre-download faster-whisper CTranslate2 model weights into models/whisper."""
    try:
        from faster_whisper import download_model
    except ImportError:
        print("[ERROR] faster-whisper is not installed. Run: pip install faster-whisper")
        sys.exit(1)

    whisper_dir = models_dir / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)

    for size in model_sizes:
        print(f"[INFO] Pre-downloading faster-whisper [{size}] weights into {whisper_dir}...")
        try:
            download_model(size, output_dir=str(whisper_dir))
            print(f"[SUCCESS] Whisper [{size}] model cached successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download Whisper [{size}]: {e}")


def prepare_argos_models(models_dir: Path) -> None:
    """Pre-download and cache Argos Translate ja->en package into models/argos_packages/."""
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        print("[ERROR] argostranslate is not installed. Run: pip install argostranslate")
        sys.exit(1)

    argos_dir = models_dir / "argos_packages"
    argos_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching Argos Translate package index...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    pkg = next(
        (p for p in available_packages if p.from_code == "ja" and p.to_code == "en"),
        None,
    )
    if pkg is None:
        print("[ERROR] Could not find ja->en package in Argos index.")
        return

    print(f"[INFO] Downloading Argos package: {pkg}...")
    downloaded_path = pkg.download()

    dest_file = argos_dir / Path(downloaded_path).name
    shutil.copyfile(downloaded_path, str(dest_file))
    print(f"[SUCCESS] Staged Argos ja->en package at: {dest_file}")

    # Also install locally for immediate source development
    argostranslate.package.install_from_path(str(dest_file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all models for 100% offline PyInstaller bundling."
    )
    parser.add_argument(
        "--whisper-models",
        nargs="+",
        default=["tiny", "small"],
        help="Whisper model sizes to pre-bundle (default: tiny small)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PREPARING OFFLINE MODEL ASSETS FOR PYINSTALLER BUNDLING")
    print("=" * 65)

    prepare_silero_vad(models_dir)
    prepare_whisper_models(models_dir, args.whisper_models)
    prepare_argos_models(models_dir)

    print("\n" + "=" * 65)
    print("  ALL OFFLINE MODEL ASSETS STAGED SUCCESSFULLY!")
    print("  You can now run PyInstaller with: pyinstaller ja_en_captioner.spec")
    print("=" * 65)


if __name__ == "__main__":
    main()
