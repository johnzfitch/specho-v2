#!/usr/bin/env python3
"""
SpecHO CLI - AI Text Detection

Usage:
    # Analyze text directly
    ./specho_cli.py "Your text here"

    # From file
    ./specho_cli.py -f document.txt

    # From stdin
    cat document.txt | ./specho_cli.py

    # JSON output for scripting
    ./specho_cli.py --json "Your text"

    # Verbose with feature breakdown
    ./specho_cli.py -v "Your text"

    # Feature analysis mode
    ./specho_cli.py --analyze "Deep feature analysis"
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='SpecHO AI Text Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "First, we examine. Second, we analyze."
  %(prog)s -f essay.txt --verbose
  cat text.txt | %(prog)s --json
  %(prog)s --analyze "Deep feature analysis"
        """
    )

    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('-f', '--file', help='Read text from file')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed breakdown')
    parser.add_argument('--analyze', action='store_true',
                       help='Deep feature analysis mode')
    parser.add_argument('--top', type=int, default=10,
                       help='Show top N features (default: 10)')

    args = parser.parse_args()

    # Get input text
    text = None

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            parser.print_help()
            print("\nError: No input text provided")
            sys.exit(1)

    if not text or len(text.strip()) < 50:
        print("Error: Text too short (need at least 50 characters)")
        sys.exit(1)

    # Import registry
    try:
        from extractors.registry import get_registry
    except ImportError:
        print("Error: Could not import registry. Run from specho-v2 directory.")
        sys.exit(1)

    # Extract features
    start = time.perf_counter()
    registry = get_registry(quiet=True)
    result = registry.extract_all(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Analyze mode - deep feature inspection
    if args.analyze or args.verbose:
        print("=" * 60)
        print("SPECHO FEATURE ANALYSIS")
        print("=" * 60)

        print(f"\nText length: {len(text)} chars")
        print(f"Extractors: {len(registry._extractors)}")
        print(f"Features: {result.n_features}")
        print(f"Extraction time: {elapsed_ms:.1f}ms")

        if result.n_failed > 0:
            print(f"\nWarnings: {result.n_failed} extractors had issues")
            for err in result.errors:
                print(f"  - {err}")

        print(f"\n--- TOP {args.top} FEATURES (by absolute value) ---")
        sorted_features = sorted(
            result.features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:args.top]

        for name, value in sorted_features:
            # Find group
            group = "?"
            for ext_name, ext in registry._extractors.items():
                if name in ext.feature_names:
                    group = ext.group[:8]
                    break
            print(f"  {name:35} {value:8.4f}  [{group}]")

        print(f"\n--- BY GROUP ---")
        for group_name in sorted(registry.groups):
            group_result = result.by_group.get(
                [n for n, e in registry._extractors.items() if e.group == group_name][0]
            )
            if group_result:
                n_zero = sum(1 for v in group_result.features.values() if v == 0)
                n_total = len(group_result.features)
                status = "⚠" if n_zero > n_total * 0.5 else "✓"
                print(f"  {group_name:15} {n_total:3}D  zeros={n_zero:2}  {status}")

        if not args.json:
            return

    # Output
    if args.json:
        output = {
            "text_length": len(text),
            "n_features": result.n_features,
            "extraction_time_ms": round(elapsed_ms, 1),
            "features": {k: round(v, 6) for k, v in result.features.items()},
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nExtracted {result.n_features} features in {elapsed_ms:.1f}ms")
        print(f"Groups: {', '.join(sorted(registry.groups))}")

        # Quick summary
        non_zero = sum(1 for v in result.features.values() if v != 0)
        print(f"Non-zero features: {non_zero}/{result.n_features}")


if __name__ == '__main__':
    main()
