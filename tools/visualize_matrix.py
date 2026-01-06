#!/usr/bin/env python3
"""
Enhanced Matrix Visualization with ASCII Graphs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import statistics


def sparkline(values: List[float], width: int = 10) -> str:
    """Create ASCII sparkline from values."""
    if not values or all(v == 0 for v in values):
        return "▁" * width

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return "▄" * width

    bars = "▁▂▃▄▅▆▇█"
    result = []

    for v in values[:width]:
        normalized = (v - min_val) / range_val
        bar_idx = min(int(normalized * (len(bars) - 1)), len(bars) - 1)
        result.append(bars[bar_idx])

    return ''.join(result)


def bar_chart(value: float, min_val: float, max_val: float, width: int = 20) -> str:
    """Create horizontal bar chart."""
    if max_val == min_val:
        filled = width // 2
    else:
        normalized = (value - min_val) / (max_val - min_val)
        filled = int(normalized * width)

    return "█" * filled + "░" * (width - filled)


def percent_diff(ai_value: float, human_value: float) -> str:
    """Calculate and format percentage difference from human."""
    if human_value == 0:
        if ai_value == 0:
            return "  0%"
        return "+∞%"

    diff = ((ai_value - human_value) / human_value) * 100

    if diff > 0:
        return f"+{diff:3.0f}%"
    else:
        return f"{diff:4.0f}%"


def color_code(diff_percent: str) -> str:
    """Add color coding based on difference magnitude."""
    # Terminal color codes
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    if "∞" in diff_percent:
        return f"{RED}{diff_percent}{RESET}"

    try:
        value = float(diff_percent.replace("%", "").replace("+", ""))
        if abs(value) > 50:
            return f"{RED}{diff_percent}{RESET}"
        elif abs(value) > 25:
            return f"{YELLOW}{diff_percent}{RESET}"
        else:
            return f"{GREEN}{diff_percent}{RESET}"
    except:
        return diff_percent


def visualize_dimension(dim_name: str, human_val: float, model_vals: Dict[str, float],
                       all_values: List[float]) -> str:
    """Create enhanced visualization for a single dimension."""
    min_val = min(all_values)
    max_val = max(all_values)

    lines = []
    lines.append(f"\n  {dim_name}")
    lines.append(f"    Human: {human_val:8.3f} {bar_chart(human_val, min_val, max_val, 15)}")

    for model, val in model_vals.items():
        diff = percent_diff(val, human_val)
        diff_colored = color_code(diff)
        chart = bar_chart(val, min_val, max_val, 15)
        lines.append(f"    {model:12}: {val:8.3f} {chart} {diff_colored}")

    # Add sparkline of all model values
    model_values_list = list(model_vals.values())
    spark = sparkline(model_values_list, len(model_vals))
    lines.append(f"    Distribution: {spark}")

    return '\n'.join(lines)


def identify_key_differentiators(results: Dict, top_n: int = 5) -> List[tuple]:
    """Identify dimensions with largest variance across models."""
    human = results["human_baseline"]
    model_data = results["by_model"]

    dimension_variances = []

    # Get all dimensions
    if human["samples"] > 0:
        for dim in human["cognitive_features"].keys():
            # Get values for this dimension across all models
            values = []

            # Human
            if dim in human["cognitive_features"]:
                human_vals = human["cognitive_features"][dim]
                if human_vals:
                    values.append(statistics.mean(human_vals))

            # AI models
            for model_key, model_info in model_data.items():
                if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                    model_vals = model_info["cognitive_features"][dim]
                    if model_vals:
                        values.append(statistics.mean(model_vals))

            # Calculate variance
            if len(values) > 1:
                variance = statistics.variance(values)
                dimension_variances.append((dim, variance, values))

    # Sort by variance
    dimension_variances.sort(key=lambda x: x[1], reverse=True)

    return dimension_variances[:top_n]


def enhanced_matrix_viz(results_path: Path):
    """Create enhanced matrix visualization."""
    with results_path.open('r') as f:
        results = json.load(f)

    human = results["human_baseline"]
    model_data = results["by_model"]

    MODEL_NAMES = {
        "gemma-2-9b": "Gemini-2-9B",
        "mistral-7B": "Mistral-7B",
        "qwen-2-72B": "Qwen-2-72B",
        "llama-8B": "Llama-3-8B",
        "accounts/yi-01-ai/models/yi-large": "Yi-Large",
        "GPT_4-o": "GPT-4o",
    }

    print("\n" + "="*100)
    print("🔍 ENHANCED 39D FINGERPRINT ANALYSIS")
    print("="*100)

    # Show key differentiators first
    print("\n📊 TOP 5 DIMENSIONS WITH HIGHEST MODEL VARIANCE")
    print("="*100)

    differentiators = identify_key_differentiators(results, top_n=5)

    for i, (dim, variance, values) in enumerate(differentiators, 1):
        # Get human and model values
        human_mean = statistics.mean(human["cognitive_features"][dim]) if dim in human["cognitive_features"] else 0

        model_vals = {}
        for model_key, model_info in model_data.items():
            if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                vals = model_info["cognitive_features"][dim]
                if vals:
                    model_name = MODEL_NAMES.get(model_key, model_key)[:12]
                    model_vals[model_name] = statistics.mean(vals)

        all_vals = [human_mean] + list(model_vals.values())

        print(f"\n{i}. {dim.upper()} (variance={variance:.4f})")
        viz = visualize_dimension(dim, human_mean, model_vals, all_vals)
        print(viz)

    # Detailed layer analysis
    dimension_groups = {
        "TRAJECTORY (5D) - High Trust": [
            "concept_jump_mean", "concept_jump_variance", "path_tortuosity",
            "turning_angle_mean", "return_rate"
        ],
        "SPECHO EXTENDED (15D) - Medium-High Trust": [
            "phonetic_mean", "semantic_mean", "burstiness",
            "cooccurrence_rate", "geometric_mean"
        ],
        "EPISTEMIC (6D) - Medium Trust": [
            "hedge_density", "hedge_clustering", "confidence_mean",
            "confidence_variance", "confidence_arc"
        ],
        "SYNTACTIC (7D) - Lower Trust": [
            "clause_length_mean", "sentence_complexity", "comma_density",
            "parenthetical_rate"
        ],
    }

    for layer_name, dimensions in dimension_groups.items():
        print(f"\n{'='*100}")
        print(f"📐 {layer_name}")
        print("="*100)

        for dim in dimensions:
            if dim not in human["cognitive_features"]:
                continue

            human_vals = human["cognitive_features"][dim]
            if not human_vals:
                continue

            human_mean = statistics.mean(human_vals)

            model_vals = {}
            all_vals = [human_mean]

            for model_key, model_info in model_data.items():
                if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                    vals = model_info["cognitive_features"][dim]
                    if vals:
                        model_name = MODEL_NAMES.get(model_key, model_key)[:12]
                        model_mean = statistics.mean(vals)
                        model_vals[model_name] = model_mean
                        all_vals.append(model_mean)

            viz = visualize_dimension(dim, human_mean, model_vals, all_vals)
            print(viz)

    print("\n" + "="*100)
    print("✅ ANALYSIS COMPLETE")
    print("="*100)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced matrix visualization")
    parser.add_argument("--results", type=str,
                       default="data/validation_39d_results.json",
                       help="Path to results JSON file")

    args = parser.parse_args()

    results_path = Path(__file__).parent.parent / args.results

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    enhanced_matrix_viz(results_path)


if __name__ == "__main__":
    main()
