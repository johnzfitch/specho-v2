#!/usr/bin/env python3
"""Basic usage example for SpecHO Unified Fingerprint System."""

import sys
sys.path.insert(0, '..')

from specho_unified.specho_unified import SpecHOUnified, UnifiedFingerprint

def main():
    human_text = """
    The morning sun cast long shadows across the kitchen floor. 
    Sarah reached for her coffee, still half-asleep. The cat wound 
    between her legs, demanding breakfast. She smiled despite her 
    exhaustion. Another day, another deadline. But first, coffee.
    """
    
    print("Initializing SpecHO Unified...")
    extractor = SpecHOUnified()
    
    print("Extracting fingerprint...")
    fp = extractor.extract(human_text)
    
    print(f"\nFingerprint extracted in {fp.extraction_time_ms:.1f}ms")
    print(f"Total features: {len(fp.to_flat_vector())}")
    print(f"Final score: {fp.final_score:.3f}")

if __name__ == "__main__":
    main()
