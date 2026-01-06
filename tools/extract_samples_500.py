#!/usr/bin/env python3
"""
Extract 500 High-Quality Samples from gsingh1-py/train HuggingFace Dataset

Requirements:
- Human_story length: 100-3000 characters
- At least 4 AI model variants present (out of 6 possible)
- Good category distribution (news, obituary, analysis, media, opinion)
"""

import sys
from pathlib import Path
from typing import Dict, List
import json
from collections import defaultdict

# Add pact-corpus to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset


# Model list from dataset
MODELS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]


def categorize_prompt(prompt: str) -> str:
    """Categorize article by headline pattern."""
    prompt_lower = prompt.lower()

    if "dies at" in prompt_lower or "dead at" in prompt_lower:
        return "obituary"
    elif any(word in prompt_lower for word in ["photo", "picture", "image", "video"]):
        return "media"
    elif "?" in prompt:
        return "analysis"
    elif any(word in prompt_lower for word in ["opinion", "editorial", "letter"]):
        return "opinion"
    else:
        return "news"


def extract_samples(target_count: int = 500, allow_flexible: bool = True) -> Dict:
    """
    Extract samples from HuggingFace dataset with balanced category distribution.

    Args:
        target_count: Target number of samples (default 500)
        allow_flexible: If True, adjust category targets based on available data

    Returns dict with:
    - samples: List of validated samples
    - metadata: Statistics about extraction
    """
    print(f"Loading gsingh1-py/train dataset (7,321 samples)...")
    dataset = load_dataset("gsingh1-py/train", split="train")
    print(f"✅ Dataset loaded: {len(dataset)} samples\n")

    samples = []
    categories = defaultdict(int)
    skipped_reasons = defaultdict(int)

    # Target distribution for balanced sampling (flexible based on data availability)
    category_targets = {
        "news": int(target_count * 0.35),      # 35%
        "analysis": int(target_count * 0.30),  # 30%
        "media": int(target_count * 0.15),     # 15%
        "opinion": int(target_count * 0.12),   # 12%
        "obituary": int(target_count * 0.08),  # 8%
    }
    category_counts = {cat: 0 for cat in category_targets}

    print("Processing dataset...")
    for idx, sample in enumerate(dataset):
        if len(samples) >= target_count:
            break

        # Status update
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx} samples, extracted {len(samples)}...", end='\r')

        # Extract fields
        prompt = sample.get("prompt", "")
        human_text = sample.get("Human_story", "")

        # Validate prompt
        if not prompt or not prompt.strip():
            skipped_reasons["missing_prompt"] += 1
            continue

        # Validate human text length
        if not human_text or len(human_text) < 100:
            skipped_reasons["human_too_short"] += 1
            continue

        if len(human_text) > 3000:
            skipped_reasons["human_too_long"] += 1
            continue

        # Get all AI variants
        ai_variants = {}
        for model_key in MODELS:
            ai_text = sample.get(model_key, "")
            if ai_text and ai_text.strip():
                ai_variants[model_key] = ai_text

        # Require at least 4 models
        if len(ai_variants) < 4:
            skipped_reasons[f"insufficient_models_{len(ai_variants)}"] += 1
            continue

        # Categorize
        category = categorize_prompt(prompt)

        # Check if we still need this category
        if category_counts[category] >= category_targets[category]:
            skipped_reasons[f"quota_full_{category}"] += 1
            continue

        # Sample meets all criteria
        sample_data = {
            "id": f"nyt_{len(samples):04d}",
            "prompt": prompt,
            "category": category,
            "human": human_text,
            "ai_variants": ai_variants,
        }

        samples.append(sample_data)
        categories[category] += 1
        category_counts[category] += 1

    print(f"\n✅ Extracted {len(samples)} samples\n")

    # Print category distribution
    print("Category Distribution:")
    for cat in ["news", "analysis", "opinion", "obituary", "media"]:
        count = categories.get(cat, 0)
        target = category_targets[cat]
        pct = 100 * count / len(samples) if samples else 0
        status = "✅" if count >= target * 0.9 else "⚠️"
        print(f"  {status} {cat:12}: {count:3} / {target:3} ({pct:5.1f}%)")

    # Model coverage
    model_counts = defaultdict(int)
    for sample in samples:
        for model in sample["ai_variants"]:
            model_counts[model] += 1

    print("\nModel Coverage Across Samples:")
    for model in MODELS:
        count = model_counts[model]
        pct = 100 * count / len(samples) if samples else 0
        print(f"  {model:40}: {count:3} ({pct:5.1f}%)")

    # Text length statistics
    lengths = [len(s["human"]) for s in samples]
    print(f"\nHuman Text Length Statistics:")
    print(f"  Average: {sum(lengths) / len(lengths):.0f} chars")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")

    # Skip reasons
    print(f"\nSkipped Samples ({sum(skipped_reasons.values())}):")
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")

    # Build metadata
    metadata = {
        "total_samples": len(samples),
        "category_distribution": {
            cat: {
                "count": count,
                "percentage": round(100 * count / len(samples), 1) if samples else 0,
            }
            for cat, count in categories.items()
        },
        "avg_human_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_human_length": min(lengths) if lengths else 0,
        "max_human_length": max(lengths) if lengths else 0,
        "models_per_sample": {
            model: count
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1])
        },
        "avg_models_per_sample": round(sum(len(s["ai_variants"]) for s in samples) / len(samples), 2) if samples else 0,
    }

    return {
        "samples": samples,
        "metadata": metadata,
    }


def save_samples(data: Dict, output_path: Path):
    """Save samples to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w') as f:
        json.dump(data, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Samples saved to: {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")


def main():
    """Extract and save 500 samples."""
    print("=" * 80)
    print("Extract 500 High-Quality Samples from gsingh1-py/train")
    print("=" * 80 + "\n")

    # Extract samples
    data = extract_samples(target_count=500)

    if not data["samples"]:
        print("❌ ERROR: No samples extracted")
        sys.exit(1)

    # Save samples
    output_path = Path(__file__).parent.parent / "data" / "samples_500.json"
    save_samples(data, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total samples: {len(data['samples'])}")
    print(f"Average models per sample: {data['metadata']['avg_models_per_sample']}")
    print(f"Average human text length: {data['metadata']['avg_human_length']:.0f} chars")


if __name__ == "__main__":
    main()
