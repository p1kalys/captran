"""One-time offline setup script for Argos Translate Japanese->English package.

Downloads and installs the ja->en language model package locally so translations
run fully offline with zero runtime API or cloud network calls.
"""

import sys

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    print("[ERROR] argostranslate is not installed.")
    print("Please install it via:")
    print("    pip install argostranslate")
    sys.exit(1)


def is_package_installed(from_code: str = "ja", to_code: str = "en") -> bool:
    """Check if the translation language pair is already installed locally."""
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
    to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

    if from_lang is not None and to_lang is not None:
        translation = from_lang.get_translation(to_lang)
        return translation is not None
    return False


def install_ja_en_package() -> None:
    print("=" * 60)
    print(" Argos Translate: Japanese -> English Model Setup")
    print("=" * 60)

    if is_package_installed("ja", "en"):
        print("\n[SUCCESS] Japanese -> English (ja->en) translation package is already installed!")
        # Test translation
        test_out = argostranslate.translate.translate("こんにちは", "ja", "en")
        print(f"Smoke test: 'こんにちは' -> '{test_out}'")
        return

    print("\nUpdating Argos Translate package index...")
    argostranslate.package.update_package_index()

    available_packages = argostranslate.package.get_available_packages()
    print(f"Found {len(available_packages)} available packages in index.")

    package_to_install = next(
        (pkg for pkg in available_packages if pkg.from_code == "ja" and pkg.to_code == "en"),
        None,
    )

    if package_to_install is None:
        print("[ERROR] Could not find 'ja' -> 'en' package in Argos repository index.")
        sys.exit(1)

    print(f"Downloading package: {package_to_install}...")
    download_path = package_to_install.download()

    print(f"Installing package from: {download_path}...")
    argostranslate.package.install_from_path(download_path)

    if is_package_installed("ja", "en"):
        print("\n[SUCCESS] Installed Japanese -> English package successfully!")
        test_out = argostranslate.translate.translate("こんにちは", "ja", "en")
        print(f"Smoke test: 'こんにちは' -> '{test_out}'")
    else:
        print("\n[ERROR] Installation completed but translation route 'ja' -> 'en' was not found.")
        sys.exit(1)


if __name__ == "__main__":
    install_ja_en_package()
