#!/usr/bin/env python3
# Suppress pkg_resources deprecation warning from pronouncing library
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

"""
Full SpecHO Validation: All 6 Models from HuggingFace Dataset

Extracts human+AI pairs across 6 different models and validates:
1. Semantic scores vary (not constant 0.5) - semantic fix verification
2. Detection patterns differ by model - model fingerprinting
3. Prompt/headline categories show different patterns

Zero-cost approach using gsingh1-py/train dataset (7,321 NYT articles).
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

# Friendly names for display
MODEL_NAMES = {
    "gemma-2-9b": "Gemini-2-9B",
    "mistral-7B": "Mistral-7B",
    "qwen-2-72B": "Qwen-2-72B",
    "llama-8B": "Llama-3-8B",
    "accounts/yi-01-ai/models/yi-large": "Yi-Large",
    "GPT_4-o": "GPT-4o",
}


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


def extract_pairs(limit: int = 50) -> List[Dict]:
    """
    Extract human+AI pairs from HuggingFace dataset across all 6 models.

    Returns list of samples, each with:
    - id: Sample identifier
    - prompt: Article headline
    - category: Article type (news, obituary, analysis, etc.)
    - human: Original human text
    - ai_variants: Dict of {model_name: ai_text}
    """
    print(f"Loading gsingh1-py/train dataset (7,321 samples)...")
    dataset = load_dataset("gsingh1-py/train", split="train")

    pairs = []
    categories = defaultdict(int)

    for idx, sample in enumerate(dataset):
        if len(pairs) >= limit:
            break

        # Extract fields
        prompt = sample.get("prompt", "")
        human_text = sample.get("Human_story", "")

        # Get all AI variants
        ai_variants = {}
        for model_key in MODELS:
            ai_text = sample.get(model_key, "")
            if ai_text and ai_text.strip():
                ai_variants[model_key] = ai_text

        # Validate
        if not human_text or len(human_text) < 100:
            continue

        # Require at least 4 models (some samples may have empty fields)
        if len(ai_variants) < 4:
            continue

        # Check length (100-3000 chars for validation)
        if len(human_text) > 3000:
            continue

        # Categorize
        category = categorize_prompt(prompt)
        categories[category] += 1

        pairs.append({
            "id": f"nyt_{idx}",
            "prompt": prompt,
            "category": category,
            "human": human_text,
            "ai_variants": ai_variants,
        })

        if len(pairs) % 10 == 0:
            print(f"  Extracted {len(pairs)}/{limit} samples...")

    print(f"\n✅ Extracted {len(pairs)} samples")
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:12} : {count:3} ({100*count/len(pairs):.1f}%)")

    return pairs


def run_specho_analysis(pairs: List[Dict]) -> Dict:
    """Run SpecHO analysis on all AI variants across all models."""
    # Import SpecHO after extraction to avoid dependency issues
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specHO"))
        from specHO.preprocessor.pipeline import LinguisticPreprocessor
        from specHO.clause_identifier.pipeline import ClauseIdentifier
        from specHO.echo_engine.pipeline import EchoAnalysisEngine
        from specHO.scoring.pipeline import ScoringModule
    except ImportError as e:
        print(f"ERROR: Failed to import SpecHO: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Initializing SpecHO with semantic fix...")
    print("=" * 70)

    # Initialize pipeline components
    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()
    echo_engine = EchoAnalysisEngine(semantic_model_path="all-MiniLM-L6-v2")
    scorer = ScoringModule()

    print("✅ Pipeline initialized")
    print(f"✅ Semantic analyzer loaded: {echo_engine.semantic_analyzer.is_loaded}")

    results = {
        "pairs": [],
        "by_model": defaultdict(lambda: {
            "samples": 0,
            "semantic_scores": [],
            "detection_scores": [],
            "phonetic_scores": [],
            "structural_scores": [],
        }),
        "by_category": defaultdict(lambda: defaultdict(list)),
        "human_baseline": {
            "samples": 0,
            "semantic_scores": [],
            "detection_scores": [],
            "phonetic_scores": [],
            "structural_scores": [],
        },
    }

    print(f"\nAnalyzing {len(pairs)} samples across {len(MODELS)} models...")
    print("=" * 70)

    for pair_idx, pair in enumerate(pairs):
        print(f"\n[{pair_idx + 1}/{len(pairs)}] {pair['category'].upper()}: {pair['prompt'][:60]}...")

        pair_results = {
            "id": pair["id"],
            "prompt": pair["prompt"],
            "category": pair["category"],
            "human_length": len(pair["human"]),
            "analyses": {}
        }

        # Analyze human baseline first
        try:
            tokens, doc = preprocessor.process(pair["human"])
            clause_pairs = clause_identifier.identify_pairs(tokens, doc)
            echo_scores = [echo_engine.analyze_pair(cp) for cp in clause_pairs]
            final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

            phonetic = sum(s.phonetic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
            structural = sum(s.structural_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
            semantic = sum(s.semantic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0

            # Store human baseline
            pair_results["analyses"]["HUMAN"] = {
                "final_score": final_score,
                "phonetic_score": phonetic,
                "structural_score": structural,
                "semantic_score": semantic,
                "num_pairs": len(clause_pairs),
            }

            results["human_baseline"]["samples"] += 1
            results["human_baseline"]["semantic_scores"].append(semantic)
            results["human_baseline"]["detection_scores"].append(final_score)
            results["human_baseline"]["phonetic_scores"].append(phonetic)
            results["human_baseline"]["structural_scores"].append(structural)

            print(f"  HUMAN BASELINE: Final={final_score:.3f} | P={phonetic:.3f} S={structural:.3f} Sem={semantic:.3f}")
        except Exception as e:
            print(f"  HUMAN BASELINE: ERROR - {e}")

        # Analyze each AI variant
        for model_key, ai_text in pair["ai_variants"].items():
            model_name = MODEL_NAMES.get(model_key, model_key)

            try:
                # Run pipeline
                tokens, doc = preprocessor.process(ai_text)
                clause_pairs = clause_identifier.identify_pairs(tokens, doc)
                echo_scores = [echo_engine.analyze_pair(pair) for pair in clause_pairs]
                final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

                # Calculate component scores
                phonetic = sum(s.phonetic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
                structural = sum(s.structural_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
                semantic = sum(s.semantic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0

                # Store results
                pair_results["analyses"][model_key] = {
                    "final_score": final_score,
                    "phonetic_score": phonetic,
                    "structural_score": structural,
                    "semantic_score": semantic,
                    "num_pairs": len(clause_pairs),
                }

                # Aggregate by model
                results["by_model"][model_key]["samples"] += 1
                results["by_model"][model_key]["semantic_scores"].append(semantic)
                results["by_model"][model_key]["detection_scores"].append(final_score)
                results["by_model"][model_key]["phonetic_scores"].append(phonetic)
                results["by_model"][model_key]["structural_scores"].append(structural)

                # Aggregate by category
                results["by_category"][pair["category"]][model_key].append(final_score)

                print(f"  {model_name:12} : Final={final_score:.3f} | "
                      f"P={phonetic:.3f} S={structural:.3f} Sem={semantic:.3f}")

            except Exception as e:
                print(f"  {model_name:12} : ERROR - {e}")
                pair_results["analyses"][model_key] = {"error": str(e)}

        results["pairs"].append(pair_results)

    return results


def analyze_results(results: Dict):
    """Generate comparative analysis across models."""
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS BY MODEL")
    print("=" * 70)

    # Show human baseline first
    human = results["human_baseline"]
    if human["samples"] > 0:
        h_sem_mean = sum(human["semantic_scores"]) / len(human["semantic_scores"])
        h_det_mean = sum(human["detection_scores"]) / len(human["detection_scores"])
        h_pho_mean = sum(human["phonetic_scores"]) / len(human["phonetic_scores"])
        h_str_mean = sum(human["structural_scores"]) / len(human["structural_scores"])

        print(f"\nHUMAN BASELINE (n={human['samples']}):")
        print(f"  Detection: {h_det_mean:.3f} | Phonetic: {h_pho_mean:.3f} | "
              f"Structural: {h_str_mean:.3f} | Semantic: {h_sem_mean:.3f}")
        print()

    print(f"{'Model':<20} {'Samples':<8} {'Detection':<12} {'Delta':<10} {'Semantic':<12} {'Delta'}")
    print("-" * 80)

    for model_key in MODELS:
        if model_key not in results["by_model"]:
            continue

        model_data = results["by_model"][model_key]
        model_name = MODEL_NAMES.get(model_key, model_key)

        sem_scores = model_data["semantic_scores"]
        det_scores = model_data["detection_scores"]

        if sem_scores and human["samples"] > 0:
            sem_mean = sum(sem_scores) / len(sem_scores)
            det_mean = sum(det_scores) / len(det_scores)

            # Calculate deltas from human
            sem_delta = sem_mean - h_sem_mean
            det_delta = det_mean - h_det_mean

            print(f"{model_name:<20} {len(sem_scores):<8} "
                  f"{det_mean:.3f}       {det_delta:+.3f}     "
                  f"{sem_mean:.3f}       {sem_delta:+.3f}")

    # Rank models by distance from human
    if human["samples"] > 0:
        print("\n" + "=" * 70)
        print("MODELS RANKED BY SIMILARITY TO HUMAN (closest first)")
        print("=" * 70)

        model_distances = []
        for model_key in MODELS:
            if model_key in results["by_model"]:
                model_data = results["by_model"][model_key]
                det_scores = model_data["detection_scores"]
                if det_scores:
                    det_mean = sum(det_scores) / len(det_scores)
                    det_delta = abs(det_mean - h_det_mean)
                    model_distances.append((model_key, det_delta, det_mean))

        # Sort by absolute delta (closest to human first)
        model_distances.sort(key=lambda x: x[1])

        for rank, (model_key, delta, score) in enumerate(model_distances, 1):
            model_name = MODEL_NAMES.get(model_key, model_key)
            print(f"  {rank}. {model_name:<20} (Δ={delta:.3f}, score={score:.3f})")

    # Category analysis
    print("\n" + "=" * 70)
    print("DETECTION SCORES BY CATEGORY")
    print("=" * 70)

    for category, model_scores in results["by_category"].items():
        print(f"\n{category.upper()}:")
        for model_key, scores in model_scores.items():
            if scores:
                model_name = MODEL_NAMES.get(model_key, model_key)
                mean_score = sum(scores) / len(scores)
                print(f"  {model_name:<20} : {mean_score:.3f} (n={len(scores)})")


def verify_semantic_fix(results: Dict) -> bool:
    """Verify semantic scores vary across all models."""
    print("\n" + "=" * 70)
    print("SEMANTIC FIX VERIFICATION")
    print("=" * 70)

    all_semantic = []
    for model_data in results["by_model"].values():
        all_semantic.extend(model_data["semantic_scores"])

    if not all_semantic:
        print("❌ ERROR: No semantic scores collected")
        return False

    unique_scores = set(round(s, 2) for s in all_semantic)

    print(f"Total semantic scores: {len(all_semantic)}")
    print(f"Unique values: {len(unique_scores)}")
    print(f"Range: {min(all_semantic):.3f} - {max(all_semantic):.3f}")

    if len(unique_scores) == 1 and 0.49 <= list(unique_scores)[0] <= 0.51:
        print("❌ FAIL: All semantic scores are 0.5 (fallback mode)")
        return False

    print("✅ PASS: Semantic scores vary properly")
    return True


def save_results(results: Dict, output_path: Path):
    """Save results with float32 conversion."""
    import numpy as np

    def convert_floats(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_floats(item) for item in obj]
        elif isinstance(obj, defaultdict):
            return {k: convert_floats(v) for k, v in dict(obj).items()}
        return obj

    results_serializable = convert_floats(results)

    with output_path.open('w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Run full validation across all 6 models."""
    print("=" * 70)
    print("SpecHO Full Validation - All 6 Models")
    print("=" * 70)

    # Extract pairs
    pairs = extract_pairs(limit=55)

    if not pairs:
        print("❌ ERROR: No pairs extracted")
        sys.exit(1)

    # Run analysis
    results = run_specho_analysis(pairs)

    # Analyze results
    analyze_results(results)

    # Verify semantic fix
    semantic_ok = verify_semantic_fix(results)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "validation_full_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)

    # Summary
    print("\n" + "=" * 70)
    if semantic_ok:
        print("✅ VALIDATION PASSED: All models analyzed with working semantic")
    else:
        print("❌ VALIDATION FAILED: Semantic analyzer issues detected")
    print("=" * 70)


if __name__ == "__main__":
    main()
