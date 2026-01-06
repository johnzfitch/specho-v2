#!/usr/bin/env python3
# Suppress pkg_resources deprecation warning from pronouncing library
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

"""
Manual Validation Tool - Inspect Single Document in Detail

Shows exactly what features are being detected for manual verification.
"""

import sys
from pathlib import Path
import json
from typing import Dict
import re

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "specHO"))

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.scoring.pipeline import ScoringModule
from fingerprint.extractor import CognitiveExtractor


def show_text_with_annotations(text: str, title: str):
    """Display text with character counts and line numbers."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Length: {len(text)} characters")
    print(f"Word count: {len(text.split())} words")

    # Show text with line numbers
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip():
            print(f"{i:3}: {line}")
    print()


def manual_counts(text: str) -> Dict:
    """Manually count features we can verify."""
    counts = {
        "commas": text.count(','),
        "semicolons": text.count(';'),
        "parentheses_open": text.count('('),
        "parentheses_close": text.count(')'),
        "sentences_approx": len(re.findall(r'[.!?]+', text)),
        "words": len(text.split()),
    }

    # Count hedge words manually
    from fingerprint.epistemic import HEDGE_MARKERS
    hedge_count = 0
    hedge_found = []
    text_lower = text.lower()

    for category, words in HEDGE_MARKERS.items():
        for word in words:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                hedge_count += len(matches)
                hedge_found.extend([(word, m.start()) for m in matches])

    counts["hedge_words"] = hedge_count
    counts["hedge_details"] = sorted(hedge_found, key=lambda x: x[1])

    return counts


def inspect_document(text: str, label: str = "DOCUMENT"):
    """Deep inspection of a single document."""

    print("\n" + "="*80)
    print(f"DETAILED INSPECTION: {label}")
    print("="*80)

    # Show original text
    show_text_with_annotations(text, "ORIGINAL TEXT")

    # Manual counts (ground truth)
    manual = manual_counts(text)
    print("\n" + "="*80)
    print("MANUAL COUNTS (Ground Truth)")
    print("="*80)
    print(f"Characters:        {len(text)}")
    print(f"Words:             {manual['words']}")
    print(f"Sentences (approx):{manual['sentences_approx']}")
    print(f"Commas:            {manual['commas']}")
    print(f"Semicolons:        {manual['semicolons']}")
    print(f"Parentheses:       {manual['parentheses_open']} ( ) {manual['parentheses_close']}")
    print(f"Hedge words:       {manual['hedge_words']}")

    if manual['hedge_details']:
        print("\nHedge words found:")
        for word, pos in manual['hedge_details'][:10]:  # Show first 10
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + len(word) + 20)
            context = text[context_start:context_end]
            print(f"  '{word}' at pos {pos}: ...{context}...")

    # Initialize analyzers
    print("\n" + "="*80)
    print("RUNNING AUTOMATED ANALYSIS...")
    print("="*80)

    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()
    echo_engine = EchoAnalysisEngine(semantic_model_path="all-MiniLM-L6-v2")
    scorer = ScoringModule()
    cognitive_extractor = CognitiveExtractor(embedding_model="all-MiniLM-L6-v2")

    # SpecHO analysis
    tokens, doc = preprocessor.process(text)
    clause_pairs = clause_identifier.identify_pairs(tokens, doc)
    echo_scores = [echo_engine.analyze_pair(cp) for cp in clause_pairs]
    final_score = scorer.score_document(echo_scores) if echo_scores else 0.0

    # Show clause detection
    print("\n" + "="*80)
    print("CLAUSE DETECTION")
    print("="*80)
    print(f"Total clause pairs: {len(clause_pairs)}")
    print(f"\nFirst 5 clause pairs:")
    for i, cp in enumerate(clause_pairs[:5], 1):
        print(f"\n  Pair {i} ({cp.pair_type}):")
        # Get clause text from tokens
        clause_a_text = ' '.join([t.text for t in cp.clause_a.tokens])
        clause_b_text = ' '.join([t.text for t in cp.clause_b.tokens])
        print(f"    Clause A: '{clause_a_text[:60]}...'")
        print(f"    Clause B: '{clause_b_text[:60]}...'")
        print(f"    Zone A (terminal): {[t.text for t in cp.zone_a_tokens]}")
        print(f"    Zone B (initial):  {[t.text for t in cp.zone_b_tokens]}")

    # Show echo scores with detailed breakdowns
    if echo_scores:
        print("\n" + "="*80)
        print("ECHO SCORES - DETAILED BREAKDOWN")
        print("="*80)
        print(f"Phonetic mean: {sum(s.phonetic_score for s in echo_scores) / len(echo_scores):.3f}")
        print(f"Structural mean: {sum(s.structural_score for s in echo_scores) / len(echo_scores):.3f}")
        print(f"Semantic mean: {sum(s.semantic_score for s in echo_scores) / len(echo_scores):.3f}")
        print(f"Final score: {final_score:.3f}")

        print(f"\nDetailed analysis of first 3 clause pairs:")
        for i, (cp, score) in enumerate(zip(clause_pairs[:3], echo_scores[:3]), 1):
            print(f"\n  === Pair {i} ({cp.pair_type}) ===")

            # Show the zones being compared
            zone_a_words = [t.text for t in cp.zone_a_tokens]
            zone_b_words = [t.text for t in cp.zone_b_tokens]

            print(f"    Zone A (terminal): {zone_a_words}")
            print(f"    Zone B (initial):  {zone_b_words}")

            # Show phonetic analysis
            print(f"\n    PHONETIC (score={score.phonetic_score:.3f}):")
            # Get phonetic transcriptions
            for j, token in enumerate(cp.zone_a_tokens):
                if hasattr(token, 'phonetic') and token.phonetic:
                    print(f"      A{j+1}: '{token.text}' → /{token.phonetic}/")
            for j, token in enumerate(cp.zone_b_tokens):
                if hasattr(token, 'phonetic') and token.phonetic:
                    print(f"      B{j+1}: '{token.text}' → /{token.phonetic}/")

            # Show structural analysis
            print(f"\n    STRUCTURAL (score={score.structural_score:.3f}):")
            for j, token in enumerate(cp.zone_a_tokens):
                print(f"      A{j+1}: '{token.text}' → {token.pos_tag}")
            for j, token in enumerate(cp.zone_b_tokens):
                print(f"      B{j+1}: '{token.text}' → {token.pos_tag}")

            # Show semantic similarity
            print(f"\n    SEMANTIC (score={score.semantic_score:.3f}):")
            print(f"      Comparing: {zone_a_words} ↔ {zone_b_words}")
            print(f"      Similarity: {score.semantic_score:.3f}")

            # Show overall
            combined = (score.phonetic_score + score.structural_score + score.semantic_score) / 3
            print(f"\n    COMBINED ECHO: {combined:.3f}")

    # 39D cognitive fingerprint
    fingerprint = cognitive_extractor.extract_from_specho(text, echo_scores)
    fp = fingerprint.to_dict()

    print("\n" + "="*80)
    print("39D COGNITIVE FINGERPRINT")
    print("="*80)

    print("\nTRAJECTORY (5D):")
    print(f"  concept_jump_mean:     {fp['concept_jump_mean']:.3f}")
    print(f"  concept_jump_variance: {fp['concept_jump_variance']:.3f}")
    print(f"  path_tortuosity:       {fp['path_tortuosity']:.3f}")
    print(f"  turning_angle_mean:    {fp['turning_angle_mean']:.3f}")
    print(f"  return_rate:           {fp['return_rate']:.3f}")

    print("\nSPECHO EXTENDED (15D):")
    print(f"  phonetic_mean:         {fp['phonetic_mean']:.3f}")
    print(f"  structural_mean:       {fp['structural_mean']:.3f}")
    print(f"  semantic_mean:         {fp['semantic_mean']:.3f}")
    print(f"  cooccurrence_rate:     {fp['cooccurrence_rate']:.3f}")
    print(f"  burstiness:            {fp['burstiness']:.3f}")

    print("\nEPISTEMIC (6D):")
    print(f"  hedge_density:         {fp['hedge_density']:.3f} (manual count: {manual['hedge_words']}, words: {manual['words']})")
    print(f"  hedge_clustering:      {fp['hedge_clustering']:.3f}")
    print(f"  hedge_position_bias:   {fp['hedge_position_bias']:.3f}")
    print(f"  confidence_mean:       {fp['confidence_mean']:.3f}")
    print(f"  confidence_variance:   {fp['confidence_variance']:.3f}")
    print(f"  confidence_arc:        {fp['confidence_arc']:.3f}")

    print("\nTRANSITIONS (6D):")
    print(f"  additive_rate:         {fp['additive_rate']:.3f}")
    print(f"  contrastive_rate:      {fp['contrastive_rate']:.3f}")
    print(f"  causal_rate:           {fp['causal_rate']:.3f}")
    print(f"  temporal_rate:         {fp['temporal_rate']:.3f}")
    print(f"  exemplifying_rate:     {fp['exemplifying_rate']:.3f}")
    print(f"  reformulating_rate:    {fp['reformulating_rate']:.3f}")

    print("\nSYNTACTIC (7D):")
    print(f"  clause_length_mean:    {fp['clause_length_mean']:.1f}")
    print(f"  clause_length_std:     {fp['clause_length_std']:.1f}")
    print(f"  clause_rhythm_autocorr:{fp['clause_rhythm_autocorr']:.3f}")
    print(f"  sentence_complexity:   {fp['sentence_complexity']:.2f}")
    print(f"  comma_density:         {fp['comma_density']:.3f} (manual count: {manual['commas']})")
    print(f"  semicolon_rate:        {fp['semicolon_rate']:.3f} (manual count: {manual['semicolons']})")
    print(f"  parenthetical_rate:    {fp['parenthetical_rate']:.3f} (manual count: {manual['parentheses_open']})")

    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    # Check comma density
    computed_commas = fp['comma_density'] * manual['words'] / 100  # Convert from per-100-words
    print(f"Comma density check:")
    print(f"  Manual count:    {manual['commas']}")
    print(f"  Computed approx: {computed_commas:.1f}")
    print(f"  Match: {'✅' if abs(computed_commas - manual['commas']) < 2 else '❌'}")

    # Check hedge density
    computed_hedges = fp['hedge_density'] * manual['words'] / 100
    print(f"\nHedge density check:")
    print(f"  Manual count:    {manual['hedge_words']}")
    print(f"  Computed approx: {computed_hedges:.1f}")
    print(f"  Match: {'✅' if abs(computed_hedges - manual['hedge_words']) < 2 else '❌'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect single document with manual validation")
    parser.add_argument("--text", type=str, help="Direct text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--sample-id", type=str, help="Sample ID from samples_500.json")
    parser.add_argument("--variant", type=str, default="human",
                       help="Variant to analyze (human, gemma-2-9b, etc.)")

    args = parser.parse_args()

    if args.text:
        text = args.text
        label = "DIRECT TEXT"
    elif args.file:
        text = Path(args.file).read_text()
        label = f"FILE: {args.file}"
    elif args.sample_id:
        samples_path = Path(__file__).parent.parent / "data/samples_500.json"
        with samples_path.open('r') as f:
            data = json.load(f)

        sample = next((s for s in data["samples"] if s["id"] == args.sample_id), None)
        if not sample:
            print(f"❌ Sample {args.sample_id} not found")
            sys.exit(1)

        if args.variant == "human":
            text = sample["human"]
            label = f"SAMPLE {args.sample_id} - HUMAN"
        else:
            text = sample["ai_variants"].get(args.variant)
            if not text:
                print(f"❌ Variant {args.variant} not found")
                print(f"Available: {list(sample['ai_variants'].keys())}")
                sys.exit(1)
            label = f"SAMPLE {args.sample_id} - {args.variant}"
    else:
        # Default: Use first sample from dataset
        samples_path = Path(__file__).parent.parent / "data/samples_500.json"
        with samples_path.open('r') as f:
            data = json.load(f)

        sample = data["samples"][0]
        text = sample["human"]
        label = f"SAMPLE {sample['id']} - HUMAN (default)"
        print(f"\nℹ️  No input specified, using first sample from dataset")
        print(f"   Use --sample-id, --text, or --file to specify input\n")

    inspect_document(text, label)


if __name__ == "__main__":
    main()
