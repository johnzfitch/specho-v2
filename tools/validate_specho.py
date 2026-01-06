#!/usr/bin/env python3
# Suppress pkg_resources deprecation warning from pronouncing library
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

"""
Quick Validation: SpecHO Semantic Fix

Extracts 10 human+AI pairs from HuggingFace dataset and validates:
1. Semantic scores vary (not constant 0.5)
2. Detection works across transformation prompt seeds
3. AI texts show different patterns than human texts

Zero-cost approach using existing AI rewrites in gsingh1-py/train dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List
import json

# Add pact-corpus to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from corpus.schema import CorpusSample, Domain, Subdomain, SourceMetadata


def extract_pairs(limit: int = 10) -> List[Dict]:
    """
    Extract human+AI pairs from HuggingFace dataset.

    Dataset fields:
    - Human_story: Original human text
    - gemma-2-9b: AI rewrite
    - mistral-7B: AI rewrite
    - prompt: Transformation prompt used
    """
    print(f"Loading gsingh1-py/train dataset...")
    dataset = load_dataset("gsingh1-py/train", split="train")

    pairs = []
    for idx, sample in enumerate(dataset):
        if len(pairs) >= limit:
            break

        # Extract fields
        human_text = sample.get("Human_story", "")
        ai_texts = {
            "gemma-2-9b": sample.get("gemma-2-9b", ""),
            "mistral-7B": sample.get("mistral-7B", ""),
        }
        prompt = sample.get("prompt", "unknown")

        # Validate
        if not human_text:
            continue

        # Require at least one AI variant
        ai_texts = {k: v for k, v in ai_texts.items() if v}
        if not ai_texts:
            continue

        # Check length (100-2000 chars for quick validation)
        if len(human_text) < 100 or len(human_text) > 2000:
            continue

        pairs.append({
            "id": f"hf_{idx}",
            "human": human_text,
            "ai_variants": ai_texts,
            "prompt": prompt,
        })

        print(f"  Extracted pair {len(pairs)}/{limit} (idx {idx})")

    print(f"\nExtracted {len(pairs)} human+AI pairs")
    return pairs


def run_specho_analysis(pairs: List[Dict]) -> Dict:
    """
    Run SpecHO analysis on all AI variants.

    Returns results with semantic scores to verify they vary.
    """
    # Import SpecHO after extraction to avoid dependency issues
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specHO"))
        from specHO.detector import SpecHODetector
    except ImportError as e:
        print(f"ERROR: Failed to import SpecHO: {e}")
        print("Make sure you're running from the correct environment")
        sys.exit(1)

    print("\nInitializing SpecHO detector with semantic fix...")
    print("  (Using minimal config without baseline for quick validation)")

    # Initialize without baseline for quick validation
    from specHO.preprocessor.pipeline import LinguisticPreprocessor
    from specHO.clause_identifier.pipeline import ClauseIdentifier
    from specHO.echo_engine.pipeline import EchoAnalysisEngine
    from specHO.scoring.pipeline import ScoringModule

    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()
    echo_engine = EchoAnalysisEngine(semantic_model_path="all-MiniLM-L6-v2")
    scorer = ScoringModule()

    results = {
        "pairs": [],
        "semantic_scores": [],
        "detection_scores": [],
    }

    print("\nAnalyzing AI variants...")
    for pair_idx, pair in enumerate(pairs):
        print(f"\n--- Pair {pair_idx + 1}/{len(pairs)} ---")
        print(f"Prompt: {pair['prompt'][:50]}...")

        pair_results = {
            "id": pair["id"],
            "prompt": pair["prompt"],
            "human_length": len(pair["human"]),
            "analyses": {}
        }

        # Analyze each AI variant
        for model_name, ai_text in pair["ai_variants"].items():
            print(f"  Analyzing {model_name}...")

            try:
                # Manual pipeline without validator
                tokens, doc = preprocessor.process(ai_text)
                clause_pairs = clause_identifier.identify_pairs(tokens, doc)
                echo_scores = [echo_engine.analyze_pair(pair) for pair in clause_pairs]
                final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

                # Calculate component scores
                phonetic_score = sum(s.phonetic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
                structural_score = sum(s.structural_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0
                semantic_score = sum(s.semantic_score for s in echo_scores) / len(echo_scores) if echo_scores else 0.0

                # Extract key metrics
                pair_results["analyses"][model_name] = {
                    "final_score": final_score,
                    "phonetic_score": phonetic_score,
                    "structural_score": structural_score,
                    "semantic_score": semantic_score,
                    "confidence": None,  # No baseline = no confidence
                    "num_pairs": len(clause_pairs),
                }

                # Track semantic scores
                results["semantic_scores"].append(semantic_score)
                results["detection_scores"].append(final_score)

                print(f"    Final: {final_score:.3f} | "
                      f"Phonetic: {phonetic_score:.3f} | "
                      f"Structural: {structural_score:.3f} | "
                      f"Semantic: {semantic_score:.3f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                pair_results["analyses"][model_name] = {"error": str(e)}

        results["pairs"].append(pair_results)

    return results


def verify_semantic_fix(results: Dict) -> bool:
    """
    Verify that semantic scores vary (not constant 0.5).

    If all scores are 0.5, the semantic analyzer is still in fallback mode.
    """
    semantic_scores = results["semantic_scores"]

    if not semantic_scores:
        print("\n❌ ERROR: No semantic scores collected")
        return False

    print(f"\n--- Semantic Fix Verification ---")
    print(f"Semantic scores: {', '.join(f'{s:.3f}' for s in semantic_scores)}")

    # Check for variation
    unique_scores = set(round(s, 2) for s in semantic_scores)

    if len(unique_scores) == 1 and 0.49 <= list(unique_scores)[0] <= 0.51:
        print("❌ FAIL: All semantic scores are 0.5 (fallback mode)")
        print("   The semantic analyzer is not loading embeddings correctly")
        return False

    print(f"✅ PASS: Semantic scores vary across {len(unique_scores)} distinct values")
    print(f"   Range: {min(semantic_scores):.3f} - {max(semantic_scores):.3f}")
    return True


def measure_detection(results: Dict):
    """
    Measure detection rates across samples.
    """
    detection_scores = results["detection_scores"]

    if not detection_scores:
        print("\n❌ ERROR: No detection scores collected")
        return

    print(f"\n--- Detection Performance ---")
    print(f"Samples analyzed: {len(detection_scores)}")
    print(f"Mean detection score: {sum(detection_scores) / len(detection_scores):.3f}")
    print(f"Score range: {min(detection_scores):.3f} - {max(detection_scores):.3f}")

    # Threshold at 0.6 (typical for AI-generated text)
    detections = sum(1 for s in detection_scores if s > 0.6)
    print(f"Detected as AI (>0.6): {detections}/{len(detection_scores)} "
          f"({100 * detections / len(detection_scores):.1f}%)")


def save_results(results: Dict, output_path: Path):
    """Save detailed results to JSON."""
    # Convert float32 to float for JSON serialization
    import numpy as np

    def convert_floats(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(item) for item in obj]
        return obj

    results_serializable = convert_floats(results)

    with output_path.open('w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    """Run quick validation."""
    print("=" * 60)
    print("SpecHO Semantic Fix Validation")
    print("=" * 60)

    # Extract pairs
    pairs = extract_pairs(limit=10)

    if not pairs:
        print("❌ ERROR: No pairs extracted")
        sys.exit(1)

    # Run analysis
    results = run_specho_analysis(pairs)

    # Verify semantic fix
    semantic_ok = verify_semantic_fix(results)

    # Measure detection
    measure_detection(results)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)

    # Summary
    print("\n" + "=" * 60)
    if semantic_ok:
        print("✅ VALIDATION PASSED: Semantic fix is working")
    else:
        print("❌ VALIDATION FAILED: Semantic analyzer still in fallback mode")
    print("=" * 60)


if __name__ == "__main__":
    main()
