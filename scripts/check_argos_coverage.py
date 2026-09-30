"""Script to check Argos Translate package availability and pivot coverage.

Queries the Argos Translate package index and prints a matrix of all 7x6 = 42 language
pairs among hi, ja, en, es, fr, de, ko, marking each as:
- 'direct package available'
- 'needs pivot' (via English)
- 'unsupported'
"""

from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    print("[ERROR] argostranslate is not installed. Run 'pip install argostranslate'")
    sys.exit(1)

from src.domain.entities import Language

LANGUAGES = [
    Language.HINDI,
    Language.JAPANESE,
    Language.ENGLISH,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.KOREAN,
]

LANG_NAMES = {
    "hi": "Hindi",
    "ja": "Japanese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ko": "Korean",
}


def get_available_package_pairs(update_index: bool = True) -> Set[Tuple[str, str]]:
    """Retrieve available (from_code, to_code) pairs from Argos package index."""
    if update_index:
        try:
            print("Updating Argos Translate package index...")
            argostranslate.package.update_package_index()
        except Exception as err:
            print(f"[WARNING] Could not update remote package index ({err}). Using cached index.")

    packages = argostranslate.package.get_available_packages()
    pairs = set()
    for pkg in packages:
        pairs.add((pkg.from_code, pkg.to_code))

    # Also include any locally installed translation routes
    try:
        installed = argostranslate.translate.get_installed_languages()
        for from_lang in installed:
            for to_lang in installed:
                if from_lang != to_lang and from_lang.get_translation(to_lang):
                    pairs.add((from_lang.code, to_lang.code))
    except Exception:
        pass

    return pairs


def analyze_pair_coverage(
    direct_pairs: Set[Tuple[str, str]],
    languages: List[Language] = LANGUAGES,
) -> Dict[Tuple[str, str], str]:
    """Classify all pairs as 'direct package available', 'needs pivot', or 'unsupported'."""
    coverage: Dict[Tuple[str, str], str] = {}

    for src in languages:
        for tgt in languages:
            if src == tgt:
                continue
            s_code = src.value
            t_code = tgt.value

            if (s_code, t_code) in direct_pairs:
                status = "direct package available"
            elif (s_code, "en") in direct_pairs and ("en", t_code) in direct_pairs:
                status = "needs pivot"
            else:
                status = "unsupported"

            coverage[(s_code, t_code)] = status

    return coverage


def print_coverage_report(coverage: Dict[Tuple[str, str], str]) -> None:
    """Print formatted matrix and pair breakdown report."""
    print("\n" + "=" * 80)
    print("  ARGOS TRANSLATE COVERAGE MATRIX (7x6 = 42 Language Pairs)")
    print("=" * 80)

    codes = [lang.value for lang in LANGUAGES]

    # Header row
    col_label = "Src \\ Tgt"
    header = f"{col_label:<10}" + "".join(f"{c:>10}" for c in codes)
    print(header)
    print("-" * len(header))

    counts = {"direct package available": 0, "needs pivot": 0, "unsupported": 0}

    for src_code in codes:
        row_str = f"{src_code:<10}"
        for tgt_code in codes:
            if src_code == tgt_code:
                cell = "--"
            else:
                status = coverage.get((src_code, tgt_code), "unsupported")
                counts[status] += 1
                if status == "direct package available":
                    cell = "DIRECT"
                elif status == "needs pivot":
                    cell = "PIVOT(en)"
                else:
                    cell = "UNSUPP"
            row_str += f"{cell:>10}"
        print(row_str)

    print("-" * len(header))
    print(f"\nSummary of {len(coverage)} pairs:")
    print(f"  - Direct packages available: {counts['direct package available']}")
    print(f"  - Needs pivot (via English):  {counts['needs pivot']}")
    print(f"  - Unsupported:               {counts['unsupported']}")
    print("=" * 80)

    # Detailed list of all 42 pairs
    print("\nDetailed Pair Breakdown:")
    print(f"{'Source':<15} {'Target':<15} {'Pair':<10} {'Status'}")
    print("-" * 65)
    for (src, tgt), status in sorted(coverage.items()):
        src_label = f"{LANG_NAMES.get(src, src)} ({src})"
        tgt_label = f"{LANG_NAMES.get(tgt, tgt)} ({tgt})"
        pair_str = f"{src}->{tgt}"
        print(f"{src_label:<15} {tgt_label:<15} {pair_str:<10} {status}")
    print("=" * 80)


def main() -> None:
    pairs = get_available_package_pairs()
    print(f"Found {len(pairs)} direct language translation packages in Argos index.")
    coverage = analyze_pair_coverage(pairs)
    print_coverage_report(coverage)


if __name__ == "__main__":
    main()
