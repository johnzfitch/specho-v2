#!/usr/bin/env python3
"""
Test Harness for SpecHO v2 Extractors

Run interactively to test extractors individually or all at once.

Usage:
    python test_harness.py                    # Test all extractors
    python test_harness.py --group pos        # Test one group
    python test_harness.py --feature sentence_length_cv  # Test one feature
    python test_harness.py --diagnose         # Full diagnostic report
    python test_harness.py --compare          # Compare human vs AI samples
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractors import get_registry, reset_registry


# Sample texts
HUMAN_SAMPLE = """
I've been thinking about this problem for a while now, and honestly?
I'm not entirely sure what the best approach is. Maybe we could try
something different—like starting from scratch and seeing where it leads.

But here's the thing: every time I try a new method, I run into the
same issues. It's frustrating! Still, I think there might be a way
forward if we just... keep experimenting?

What do you think? Have you ever dealt with something like this before?
"""

AI_SAMPLE = """
The implementation of effective strategies requires careful consideration
of multiple factors. First, it is important to establish clear objectives.
Second, the methodology should be systematically evaluated. Third, the
results must be thoroughly analyzed.

Furthermore, the data indicates that optimal outcomes are achieved through
consistent application of established principles. Additionally, research
has demonstrated the efficacy of structured approaches in similar contexts.

In conclusion, the evidence strongly supports the adoption of comprehensive
frameworks for addressing complex challenges. Therefore, organizations
should prioritize the development of robust systems.
"""


def print_header(title: str, width: int = 60):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_features(features: dict, highlight_nonzero: bool = True):
    """Print features in a nice format."""
    for name, value in sorted(features.items()):
        if highlight_nonzero and value != 0.0:
            print(f"  {name:40s} = {value:8.4f} *")
        else:
            print(f"  {name:40s} = {value:8.4f}")


def test_all(text: str = None):
    """Test all extractors."""
    reset_registry()
    registry = get_registry(quiet=False)

    if text is None:
        text = HUMAN_SAMPLE

    print_header("Registry Status")
    print(registry.status())

    print_header("Extraction Results")
    result = registry.extract_all(text)

    print(f"\nTotal features: {result.n_features}")
    print(f"Total time: {result.total_time_ms:.2f}ms")
    print(f"Failed: {result.n_failed}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    print("\nFeatures by group:")
    for group_name, group_result in result.by_group.items():
        print(f"\n  [{group_name}] ({group_result.extraction_time_ms:.2f}ms)")
        for name, value in sorted(group_result.features.items()):
            marker = " *" if value != 0.0 else ""
            print(f"    {name:38s} = {value:8.4f}{marker}")


def test_group(group: str, text: str = None):
    """Test a specific group."""
    reset_registry()
    registry = get_registry(quiet=True)

    if text is None:
        text = HUMAN_SAMPLE

    print_header(f"Group: {group}")

    if group not in registry.groups:
        print(f"Unknown group '{group}'")
        print(f"Available groups: {', '.join(registry.groups)}")
        return

    result = registry.extract_group(group, text)
    print(f"Time: {result.total_time_ms:.2f}ms")
    print(f"Features: {result.n_features}")
    print()
    print_features(result.features)


def test_feature(feature: str, text: str = None):
    """Test a single feature."""
    reset_registry()
    registry = get_registry(quiet=True)

    if text is None:
        text = HUMAN_SAMPLE

    print_header(f"Feature: {feature}")

    if feature not in registry.all_feature_names:
        print(f"Unknown feature '{feature}'")
        print(f"\nAvailable features ({len(registry.all_feature_names)} total):")
        for f in sorted(registry.all_feature_names):
            print(f"  - {f}")
        return

    value = registry.extract_feature(feature, text)
    print(f"Value: {value:.4f}")


def diagnose():
    """Run full diagnostic report."""
    reset_registry()
    registry = get_registry(quiet=False)

    print_header("Diagnostic Report")
    report = registry.test_all()

    print(f"\nSummary:")
    print(f"  Total extractors: {report['summary']['total_extractors']}")
    print(f"  Working: {report['summary']['working']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Total features: {report['summary']['total_features']}")
    print(f"  Groups: {', '.join(report['summary']['groups'])}")

    print("\nExtractor details:")
    for name, info in report["extractors"].items():
        status = "OK" if info["result"]["success"] else "FAILED"
        time = info["result"]["extraction_time_ms"]
        print(f"\n  [{name}] {status} ({time:.2f}ms)")
        print(f"    Group: {info['group']}")
        print(f"    Features: {info['n_features']}")
        print(f"    Dependencies: {info['dependencies'] or 'None'}")
        if not info["result"]["success"]:
            print(f"    Error: {info['result']['error']}")


def compare():
    """Compare human vs AI samples."""
    reset_registry()
    registry = get_registry(quiet=True)

    print_header("Human vs AI Comparison")

    human_result = registry.extract_all(HUMAN_SAMPLE)
    ai_result = registry.extract_all(AI_SAMPLE)

    print(f"\nHuman text: {len(HUMAN_SAMPLE.split())} words, {human_result.total_time_ms:.2f}ms")
    print(f"AI text: {len(AI_SAMPLE.split())} words, {ai_result.total_time_ms:.2f}ms")

    print("\n{:40s} {:>10s} {:>10s} {:>10s}".format(
        "Feature", "Human", "AI", "Diff"
    ))
    print("-" * 72)

    # Sort by absolute difference
    features_with_diff = []
    for name in human_result.features:
        h_val = human_result.features[name]
        a_val = ai_result.features[name]
        diff = h_val - a_val
        features_with_diff.append((name, h_val, a_val, diff))

    # Sort by absolute difference (largest first)
    features_with_diff.sort(key=lambda x: abs(x[3]), reverse=True)

    for name, h_val, a_val, diff in features_with_diff:
        direction = "+" if diff > 0 else ""
        significance = "***" if abs(diff) > 0.5 else ("**" if abs(diff) > 0.2 else ("*" if abs(diff) > 0.1 else ""))
        print(f"{name:40s} {h_val:10.4f} {a_val:10.4f} {direction}{diff:9.4f} {significance}")

    print("\n*** = large difference (>0.5)")
    print("**  = medium difference (>0.2)")
    print("*   = small difference (>0.1)")


def interactive():
    """Interactive mode for testing custom text."""
    reset_registry()
    registry = get_registry(quiet=True)

    print_header("Interactive Mode")
    print("Enter text to analyze (empty line to finish, 'quit' to exit):")

    while True:
        lines = []
        print("\n> ", end="")
        while True:
            line = input()
            if line == "":
                break
            if line.lower() == "quit":
                return
            lines.append(line)

        if not lines:
            continue

        text = "\n".join(lines)
        result = registry.extract_all(text)

        print(f"\n[Analysis: {len(text.split())} words, {result.total_time_ms:.2f}ms]")
        print_features(result.features)


def main():
    parser = argparse.ArgumentParser(description="Test SpecHO v2 extractors")
    parser.add_argument("--group", "-g", help="Test specific group")
    parser.add_argument("--feature", "-f", help="Test specific feature")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run diagnostic")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare human vs AI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--text", "-t", help="Custom text to analyze")

    args = parser.parse_args()

    # Read text from file if provided
    text = None
    if args.text:
        if Path(args.text).exists():
            text = Path(args.text).read_text()
        else:
            text = args.text

    if args.diagnose:
        diagnose()
    elif args.compare:
        compare()
    elif args.interactive:
        interactive()
    elif args.group:
        test_group(args.group, text)
    elif args.feature:
        test_feature(args.feature, text)
    else:
        test_all(text)


if __name__ == "__main__":
    main()
