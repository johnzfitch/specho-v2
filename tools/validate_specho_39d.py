#!/usr/bin/env python3
"""
Full 39D Cognitive Fingerprint Validation

Extends SpecHO validation with complete cognitive fingerprints:
- Layer A: Trajectory (5 dims)
- Layer B: SpecHO Extended (15 dims)
- Layer C: Epistemic + Transitions (12 dims)
- Layer D: Syntactic (7 dims)

Total: 39 dimensions per text
"""

# Suppress pkg_resources deprecation warning from pronouncing library
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import sys
from pathlib import Path
from typing import Dict, List
import json
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specHO"))

from datasets import load_dataset
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.scoring.pipeline import ScoringModule

# Import fingerprint extractor
from fingerprint.extractor import CognitiveExtractor


# Model list from dataset
MODELS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]

MODEL_NAMES = {
    "gemma-2-9b": "Gemini-2-9B",
    "mistral-7B": "Mistral-7B",
    "qwen-2-72B": "Qwen-2-72B",
    "llama-8B": "Llama-3-8B",
    "accounts/yi-01-ai/models/yi-large": "Yi-Large",
    "GPT_4-o": "GPT-4o",
}


def load_samples(samples_path: Path) -> List[Dict]:
    """Load samples from JSON file."""
    print(f"Loading samples from {samples_path}...")

    with samples_path.open('r') as f:
        data = json.load(f)

    samples = data["samples"]
    metadata = data["metadata"]

    print(f"✅ Loaded {len(samples)} samples")
    print(f"   Category distribution:")
    for cat, info in metadata["category_distribution"].items():
        print(f"     {cat:12}: {info['count']:3} ({info['percentage']:.1f}%)")

    return samples


def process_single_sample(args):
    """Process a single sample with all its AI variants (for multiprocessing)."""
    pair, pair_idx, total_samples = args

    # Initialize SpecHO pipeline in this worker process
    from specHO.preprocessor.pipeline import LinguisticPreprocessor
    from specHO.clause_identifier.pipeline import ClauseIdentifier
    from specHO.echo_engine.pipeline import EchoAnalysisEngine
    from specHO.scoring.pipeline import ScoringModule
    from fingerprint.extractor import CognitiveExtractor

    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()
    echo_engine = EchoAnalysisEngine(semantic_model_path="all-MiniLM-L6-v2")
    scorer = ScoringModule()
    cognitive_extractor = CognitiveExtractor(embedding_model="all-MiniLM-L6-v2")

    sample_results = {
        "id": pair["id"],
        "prompt": pair["prompt"],
        "category": pair["category"],
        "human_length": len(pair["human"]),
        "analyses": {}
    }

    # Analyze human baseline
    try:
        start = time.time()

        tokens, doc = preprocessor.process(pair["human"])
        clause_pairs = clause_identifier.identify_pairs(tokens, doc)
        echo_scores = [echo_engine.analyze_pair(cp) for cp in clause_pairs]
        final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

        fingerprint = cognitive_extractor.extract_from_specho(
            pair["human"],
            echo_scores
        )

        elapsed = (time.time() - start) * 1000

        sample_results["analyses"]["HUMAN"] = {
            "final_score": final_score,
            "num_pairs": len(clause_pairs),
            "fingerprint_39d": fingerprint.to_dict(),
            "extraction_time_ms": elapsed,
        }

    except Exception as e:
        sample_results["analyses"]["HUMAN"] = {"error": str(e)}

    # Analyze each AI variant
    for model_key, ai_text in pair["ai_variants"].items():
        try:
            start = time.time()

            tokens, doc = preprocessor.process(ai_text)
            clause_pairs = clause_identifier.identify_pairs(tokens, doc)
            echo_scores = [echo_engine.analyze_pair(cp) for cp in clause_pairs]
            final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

            fingerprint = cognitive_extractor.extract_from_specho(
                ai_text,
                echo_scores
            )

            elapsed = (time.time() - start) * 1000

            sample_results["analyses"][model_key] = {
                "final_score": final_score,
                "num_pairs": len(clause_pairs),
                "fingerprint_39d": fingerprint.to_dict(),
                "extraction_time_ms": elapsed,
            }

        except Exception as e:
            sample_results["analyses"][model_key] = {"error": str(e)}

    return pair_idx, sample_results


def run_39d_analysis(samples: List[Dict], limit: int = None, workers: int = None, verbose: bool = False) -> Dict:
    """Run full 39D cognitive fingerprint analysis."""
    if limit:
        samples = samples[:limit]
        print(f"\nLimiting to first {limit} samples for testing")

    # Determine worker count
    if workers is None:
        workers = max(1, cpu_count() - 1)  # Leave 1 core free

    print("\n" + "=" * 70)
    print(f"Parallel Processing with {workers} workers")
    print("=" * 70)

    results = {
        "samples": [None] * len(samples),  # Pre-allocate to maintain order
        "by_model": defaultdict(lambda: {
            "samples": 0,
            "specho_scores": defaultdict(list),
            "cognitive_features": defaultdict(list),
        }),
        "human_baseline": {
            "samples": 0,
            "specho_scores": defaultdict(list),
            "cognitive_features": defaultdict(list),
        },
    }

    print(f"\nAnalyzing {len(samples)} samples with 39D fingerprints...")
    print("=" * 70)

    total_start = time.time()

    # Prepare arguments for processing
    processing_args = [(pair, idx, len(samples)) for idx, pair in enumerate(samples)]

    completed = 0

    if workers == 1:
        # Sequential processing (no multiprocessing, avoids CUDA fork issues)
        print("Running in sequential mode (no multiprocessing overhead)")

        for arg in processing_args:
            pair_idx, sample_results = process_single_sample(arg)
            completed += 1

            # Store in correct position
            results["samples"][pair_idx] = sample_results

            # Aggregate human baseline
            if "HUMAN" in sample_results["analyses"] and "error" not in sample_results["analyses"]["HUMAN"]:
                results["human_baseline"]["samples"] += 1
                fingerprint_dict = sample_results["analyses"]["HUMAN"]["fingerprint_39d"]
                for key, value in fingerprint_dict.items():
                    results["human_baseline"]["cognitive_features"][key].append(value)

            # Aggregate by model
            for model_key in MODELS:
                if model_key in sample_results["analyses"] and "error" not in sample_results["analyses"][model_key]:
                    results["by_model"][model_key]["samples"] += 1
                    fingerprint_dict = sample_results["analyses"][model_key]["fingerprint_39d"]
                    for key, value in fingerprint_dict.items():
                        results["by_model"][model_key]["cognitive_features"][key].append(value)

            # Progress update
            elapsed = time.time() - total_start
            rate = completed / elapsed if elapsed > 0 else 0

            if verbose:
                # Show every sample in verbose mode
                print(f"✓ Sample {completed}/{len(samples)} | {rate:.2f} samples/sec | {elapsed:.1f}s elapsed")
            elif completed % 10 == 0 or completed == len(samples):
                # Show every 10 samples in normal mode
                print(f"Progress: {completed}/{len(samples)} samples ({completed/len(samples)*100:.1f}%) | "
                      f"{rate:.1f} samples/sec | {elapsed:.1f}s elapsed")

    else:
        # Parallel processing with multiprocessing
        print(f"Running in parallel mode with {workers} workers")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_sample, arg): arg for arg in processing_args}

            for future in as_completed(futures):
                pair_idx, sample_results = future.result()
                completed += 1

                # Store in correct position
                results["samples"][pair_idx] = sample_results

                # Aggregate human baseline
                if "HUMAN" in sample_results["analyses"] and "error" not in sample_results["analyses"]["HUMAN"]:
                    results["human_baseline"]["samples"] += 1
                    fingerprint_dict = sample_results["analyses"]["HUMAN"]["fingerprint_39d"]
                    for key, value in fingerprint_dict.items():
                        results["human_baseline"]["cognitive_features"][key].append(value)

                # Aggregate by model
                for model_key in MODELS:
                    if model_key in sample_results["analyses"] and "error" not in sample_results["analyses"][model_key]:
                        results["by_model"][model_key]["samples"] += 1
                        fingerprint_dict = sample_results["analyses"][model_key]["fingerprint_39d"]
                        for key, value in fingerprint_dict.items():
                            results["by_model"][model_key]["cognitive_features"][key].append(value)

                # Progress update
                elapsed = time.time() - total_start
                rate = completed / elapsed if elapsed > 0 else 0

                if verbose:
                    # Show every sample in verbose mode
                    print(f"✓ Sample {completed}/{len(samples)} | {rate:.2f} samples/sec | {elapsed:.1f}s elapsed")
                elif completed % 10 == 0 or completed == len(samples):
                    # Show every 10 samples in normal mode
                    print(f"Progress: {completed}/{len(samples)} samples ({completed/len(samples)*100:.1f}%) | "
                          f"{rate:.1f} samples/sec | {elapsed:.1f}s elapsed")

    total_elapsed = time.time() - total_start
    print(f"\n✅ Analysis complete in {total_elapsed:.1f}s")

    return results


def analyze_results(results: Dict):
    """Generate comprehensive 39D matrix analysis."""
    print("\n" + "=" * 100)
    print("39D COGNITIVE FINGERPRINT MATRIX")
    print("=" * 100)

    # All 39 dimensions organized by layer
    dimension_groups = {
        "LAYER A: TRAJECTORY (5D)": [
            "concept_jump_mean", "concept_jump_variance", "path_tortuosity",
            "turning_angle_mean", "return_rate"
        ],
        "LAYER B: SPECHO EXTENDED (15D)": [
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max", "burstiness"
        ],
        "LAYER C: EPISTEMIC (6D)": [
            "hedge_density", "hedge_clustering", "hedge_position_bias",
            "confidence_mean", "confidence_variance", "confidence_arc"
        ],
        "LAYER C: TRANSITIONS (6D)": [
            "additive_rate", "contrastive_rate", "causal_rate",
            "temporal_rate", "exemplifying_rate", "reformulating_rate"
        ],
        "LAYER D: SYNTACTIC (7D)": [
            "clause_length_mean", "clause_length_std", "clause_rhythm_autocorr",
            "sentence_complexity", "comma_density", "semicolon_rate", "parenthetical_rate"
        ]
    }

    # Calculate means for all dimensions
    human = results["human_baseline"]
    human_means = {}
    if human["samples"] > 0:
        for dim in dimension_groups["LAYER A: TRAJECTORY (5D)"] + \
                   dimension_groups["LAYER B: SPECHO EXTENDED (15D)"] + \
                   dimension_groups["LAYER C: EPISTEMIC (6D)"] + \
                   dimension_groups["LAYER C: TRANSITIONS (6D)"] + \
                   dimension_groups["LAYER D: SYNTACTIC (7D)"]:
            if dim in human["cognitive_features"]:
                values = human["cognitive_features"][dim]
                human_means[dim] = sum(values) / len(values) if values else 0.0

    # Build model means
    model_means = {}
    for model_key in MODELS:
        if model_key not in results["by_model"]:
            continue
        model_data = results["by_model"][model_key]
        if model_data["samples"] > 0:
            model_means[model_key] = {}
            for dim in dimension_groups["LAYER A: TRAJECTORY (5D)"] + \
                       dimension_groups["LAYER B: SPECHO EXTENDED (15D)"] + \
                       dimension_groups["LAYER C: EPISTEMIC (6D)"] + \
                       dimension_groups["LAYER C: TRANSITIONS (6D)"] + \
                       dimension_groups["LAYER D: SYNTACTIC (7D)"]:
                if dim in model_data["cognitive_features"]:
                    values = model_data["cognitive_features"][dim]
                    model_means[model_key][dim] = sum(values) / len(values) if values else 0.0

    # Print comprehensive matrix
    for layer_name, dimensions in dimension_groups.items():
        print(f"\n{layer_name}")
        print("-" * 100)

        # Header
        header = f"{'Dimension':<30} {'Human':<12}"
        for model_key in MODELS:
            if model_key in model_means:
                model_name = MODEL_NAMES.get(model_key, model_key)[:10]
                header += f"{model_name:<12}"
        print(header)
        print("-" * 100)

        # Each dimension
        for dim in dimensions:
            row = f"{dim:<30} "

            # Human value
            if dim in human_means:
                row += f"{human_means[dim]:<12.3f}"
            else:
                row += f"{'N/A':<12}"

            # Model values
            for model_key in MODELS:
                if model_key in model_means and dim in model_means[model_key]:
                    row += f"{model_means[model_key][dim]:<12.3f}"
                else:
                    row += f"{'N/A':<12}"

            print(row)


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

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Results saved to: {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")


def main():
    """Run full 39D validation."""
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="39D Cognitive Fingerprint Validation")
    parser.add_argument("--samples", type=str,
                       default="data/samples_500.json",
                       help="Path to samples JSON file")
    parser.add_argument("--limit", type=int,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--workers", type=int,
                       help=f"Number of parallel workers (default: {max(1, cpu_count() - 1)})")
    parser.add_argument("--output", type=str,
                       default="data/validation_39d_results.json",
                       help="Output path for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed progress for each sample")

    args = parser.parse_args()

    print("=" * 70)
    print("SpecHO + 39D Cognitive Fingerprint Validation")
    print("=" * 70)

    # Load samples
    samples_path = Path(__file__).parent.parent / args.samples
    samples = load_samples(samples_path)

    if not samples:
        print("❌ ERROR: No samples loaded")
        sys.exit(1)

    # Set multiprocessing mode based on device
    workers = args.workers if args.workers else max(1, multiprocessing.cpu_count() - 1)
    if workers > 1:
        try:
            # Use spawn for CUDA compatibility (slower startup but works with GPU)
            multiprocessing.set_start_method('spawn', force=True)
            print(f"ℹ️  Using spawn mode for {workers} workers (CUDA compatible, slower startup)")
        except RuntimeError:
            pass  # Already set

    # Run analysis
    results = run_39d_analysis(samples, limit=args.limit, workers=args.workers, verbose=args.verbose)

    # Analyze results
    analyze_results(results)

    # Save results
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)

    # Summary
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
