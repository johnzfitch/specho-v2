#!/usr/bin/env python3
"""
Extract Cognitive Fingerprints (39-dim)

Full cognitive fingerprint extraction:
- Layer A: Trajectory (5 dims) - semantic path geometry
- Layer B: SpecHO (15 dims) - clause echo patterns
- Layer C: Epistemic (6 dims) + Transitions (6 dims)
- Layer D: Syntactic (7 dims) - structural rhythm

Total: 39 dimensions, layered by trust level.

Usage:
    python extract_cognitive.py --corpus data/corpus/merged/corpus.json
    python extract_cognitive.py --corpus data/corpus/ --recursive
    python extract_cognitive.py --text "Some text to analyze"
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from collections import Counter
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "fingerprint"))

from cognitive import CognitiveExtractor, CognitiveFingerprint


def load_corpus(path: str, recursive: bool = False) -> list[dict]:
    """Load corpus from file or directory."""
    corpus = []
    path = Path(path)
    
    if path.is_file():
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        corpus.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    corpus = data
    elif path.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        for file in path.glob(pattern):
            if file.name.startswith("."):
                continue
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            corpus.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        corpus.extend(data)
    
    return corpus


def extract_fingerprints(
    corpus_path: str = None,
    text: str = None,
    output_path: str = None,
    specHO_path: str = None,
    recursive: bool = False,
) -> list[CognitiveFingerprint]:
    """Extract cognitive fingerprints from corpus or text."""
    
    extractor = CognitiveExtractor(specHO_path=specHO_path)
    fingerprints = []
    
    if text:
        # Single text analysis
        fp = extractor.extract(text, sample_id="input", source_model="unknown")
        fingerprints.append(fp)
        
    elif corpus_path:
        # Corpus analysis
        print(f"\nLoading corpus from {corpus_path}...")
        corpus = load_corpus(corpus_path, recursive)
        print(f"Loaded {len(corpus)} samples")
        
        print(f"\nExtracting 39-dim cognitive fingerprints...")
        start_time = time.time()
        
        for i, sample in enumerate(corpus):
            sample_id = sample.get("sample_id", f"sample_{i}")
            source_model = sample.get("model_name") or sample.get("model_id") or "unknown"
            text = sample.get("text", "")
            
            if not text or len(text.split()) < 10:
                continue
            
            print(f"  [{i+1}/{len(corpus)}] {sample_id} ({source_model})...", end=" ", flush=True)
            
            fp = extractor.extract(text, sample_id, source_model)
            fingerprints.append(fp)
            
            print(f"✓ ({fp.extraction_time_ms:.0f}ms)")
        
        elapsed = time.time() - start_time
        print(f"\nExtracted {len(fingerprints)} fingerprints in {elapsed:.1f}s")
    
    # Save if output path provided
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            json.dump([asdict(fp) for fp in fingerprints], f, indent=2)
        
        print(f"\nSaved to {output}")
    
    return fingerprints


def print_fingerprint_summary(fp: CognitiveFingerprint):
    """Print detailed fingerprint summary."""
    
    print("\n" + "="*60)
    print("COGNITIVE FINGERPRINT")
    print("="*60)
    
    print(f"\nSample: {fp.sample_id}")
    print(f"Model: {fp.source_model}")
    print(f"Words: {fp.word_count}, Sentences: {fp.n_sentences}, Clauses: {fp.n_clauses}")
    print(f"Extraction time: {fp.extraction_time_ms:.0f}ms")
    
    print(f"\n[LAYER A] Trajectory (High Trust) - 5 dims")
    print(f"  concept_jump_mean:     {fp.concept_jump_mean:.4f}")
    print(f"  concept_jump_variance: {fp.concept_jump_variance:.4f}")
    print(f"  path_tortuosity:       {fp.path_tortuosity:.4f}")
    print(f"  turning_angle_mean:    {fp.turning_angle_mean:.4f}")
    print(f"  return_rate:           {fp.return_rate:.4f}")
    
    print(f"\n[LAYER B] SpecHO (Medium-High Trust) - 15 dims")
    print(f"  phonetic:    μ={fp.phonetic_mean:.3f} σ={fp.phonetic_std:.3f} max={fp.phonetic_max:.3f}")
    print(f"  structural:  μ={fp.structural_mean:.3f} σ={fp.structural_std:.3f} max={fp.structural_max:.3f}")
    print(f"  semantic:    μ={fp.semantic_mean:.3f} σ={fp.semantic_std:.3f} max={fp.semantic_max:.3f}")
    print(f"  cooccurrence: {fp.cooccurrence_rate:.3f}")
    print(f"  geometric:    {fp.geometric_mean:.3f}")
    print(f"  overall:      μ={fp.overall_mean:.3f} σ={fp.overall_std:.3f}")
    print(f"  burstiness:   {fp.burstiness:.3f}")
    
    print(f"\n[LAYER C] Epistemic (Medium Trust) - 6 dims")
    print(f"  hedge_density:       {fp.hedge_density:.2f}")
    print(f"  hedge_clustering:    {fp.hedge_clustering:.2f}")
    print(f"  hedge_position_bias: {fp.hedge_position_bias:.2f}")
    print(f"  confidence_mean:     {fp.confidence_mean:.2f}")
    print(f"  confidence_variance: {fp.confidence_variance:.3f}")
    print(f"  confidence_arc:      {fp.confidence_arc:+.2f}")
    
    print(f"\n[LAYER C] Transitions (Medium Trust) - 6 dims")
    print(f"  additive:      {fp.additive_rate:.2f}")
    print(f"  contrastive:   {fp.contrastive_rate:.2f}")
    print(f"  causal:        {fp.causal_rate:.2f}")
    print(f"  temporal:      {fp.temporal_rate:.2f}")
    print(f"  exemplifying:  {fp.exemplifying_rate:.2f}")
    print(f"  reformulating: {fp.reformulating_rate:.2f}")
    
    print(f"\n[LAYER D] Syntactic (Lower Trust) - 7 dims")
    print(f"  clause_length:       μ={fp.clause_length_mean:.1f} σ={fp.clause_length_std:.1f}")
    print(f"  rhythm_autocorr:     {fp.clause_rhythm_autocorr:+.2f}")
    print(f"  sentence_complexity: {fp.sentence_complexity:.1f}")
    print(f"  comma_density:       {fp.comma_density:.1f}")
    print(f"  semicolon_rate:      {fp.semicolon_rate:.2f}")
    print(f"  parenthetical_rate:  {fp.parenthetical_rate:.2f}")
    
    print(f"\n[VECTORS]")
    print(f"  Layer A (5):  {fp.to_vector_layer_a()}")
    print(f"  Layer AB (20): [{fp.to_vector_layer_ab()[:5]}...]")
    print(f"  Full (39):     [{fp.to_vector_full()[:5]}...]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 39-dim cognitive fingerprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The 39-dimensional cognitive fingerprint:

  Layer A (High Trust):        5 dims - Trajectory geometry
  Layer B (Medium-High Trust): 15 dims - SpecHO echo patterns
  Layer C (Medium Trust):      12 dims - Epistemic + Transitions
  Layer D (Lower Trust):       7 dims - Syntactic rhythm

Trust layers can be used independently for different verification scenarios.
"""
    )
    parser.add_argument(
        "--corpus", type=str,
        help="Path to corpus file or directory"
    )
    parser.add_argument(
        "--text", type=str,
        help="Single text to analyze"
    )
    parser.add_argument(
        "--output", type=str, default="data/fingerprints/cognitive.json",
        help="Output path (default: data/fingerprints/cognitive.json)"
    )
    parser.add_argument(
        "--specHO", type=str,
        help="Path to SpecHO installation"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search directories"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON for single text"
    )
    
    args = parser.parse_args()
    
    if not args.corpus and not args.text:
        parser.print_help()
        print("\nError: Provide --corpus or --text")
        return
    
    fingerprints = extract_fingerprints(
        corpus_path=args.corpus,
        text=args.text,
        output_path=args.output if args.corpus else None,
        specHO_path=args.specHO,
        recursive=args.recursive,
    )
    
    if args.text:
        if args.json:
            print(json.dumps(asdict(fingerprints[0]), indent=2))
        else:
            print_fingerprint_summary(fingerprints[0])
    elif fingerprints:
        # Print corpus summary
        model_counts = Counter(fp.source_model for fp in fingerprints)
        
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"\nTotal fingerprints: {len(fingerprints)}")
        print(f"Dimensions: 39 (layered)")
        
        print("\nBy model:")
        for model, count in sorted(model_counts.items()):
            fps = [fp for fp in fingerprints if fp.source_model == model]
            
            # Compute average features per model
            trajectory = np.mean([fp.concept_jump_mean for fp in fps])
            hedge = np.mean([fp.hedge_density for fp in fps])
            contrastive = np.mean([fp.contrastive_rate for fp in fps])
            
            print(f"  {model}: {count} samples")
            print(f"    trajectory={trajectory:.3f}, hedge={hedge:.2f}, contrastive={contrastive:.2f}")


if __name__ == "__main__":
    main()
