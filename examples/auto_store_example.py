#!/usr/bin/env python3
"""
Example: Auto-accumulating fingerprint database.

Shows how fingerprints are automatically stored per model,
building a reference database that improves over time.
"""

import sys
sys.path.insert(0, "..")

from specho_unified import SpecHO

def main():
    # Initialize with auto-storage enabled
    specho = SpecHO(
        data_dir="demo_fingerprints",
        auto_store=True,
    )
    
    print("SpecHO Fingerprint System")
    print("=" * 50)
    
    # Sample texts (in real use, these come from API responses)
    samples = [
        {
            "text": """
            The morning sun cast long shadows across the kitchen floor. 
            Sarah reached for her coffee, still half-asleep. The cat wound 
            between her legs, demanding breakfast. She smiled despite her 
            exhaustion. Another day, another deadline. But first, coffee.
            """,
            "model_id": "human",
        },
        {
            "text": """
            The morning sun illuminated the kitchen with its warm golden rays, 
            creating an atmosphere of tranquility and peace. The individual 
            reached for their beverage container, experiencing the residual 
            effects of insufficient sleep. The domestic feline companion 
            exhibited characteristic food-seeking behavior.
            """,
            "model_id": "gpt-4o",
        },
        {
            "text": """
            As the sun rose, its light filled the kitchen space with warmth.
            The person, still drowsy, grabbed their morning coffee. Meanwhile,
            their cat circled around, clearly wanting to be fed. Despite 
            feeling tired, they managed a smile. Work awaited, but coffee first.
            """,
            "model_id": "claude-sonnet-4",
        },
    ]
    
    # Analyze and store each sample
    print("\n[1] Analyzing and storing samples...")
    for sample in samples:
        result = specho.analyze(
            text=sample["text"],
            model_id=sample["model_id"],
        )
        print(f"  {sample['model_id']}: stored={result.stored}, "
              f"tortuosity={result.path_tortuosity:.2f}, "
              f"semantic={result.semantic_mean:.3f}")
    
    # Check status
    print("\n[2] System status:")
    print(specho.report())
    
    # Now verify an unknown sample
    print("\n[3] Verifying unknown text...")
    unknown_text = """
    The early morning light streamed through the window, casting a 
    gentle glow across the room. She reached for her coffee cup,
    fighting off the remnants of sleep. Her cat meowed persistently,
    a clear demand for breakfast. She couldn't help but smile.
    """
    
    result = specho.verify(unknown_text)
    print(f"  Detected: {result.detected_model}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Trust Level: {result.trust_level.value}")
    
    # Verify with a claim
    print("\n[4] Verifying with identity claim...")
    result = specho.verify(unknown_text, claimed_model="gpt-4o")
    print(f"  Claimed: gpt-4o")
    print(f"  Detected: {result.detected_model}")
    print(f"  Claim verified: {result.claim_verified}")
    
    # AURORA integration
    print("\n[5] AURORA trust evidence...")
    evidence = specho.get_aurora_evidence(unknown_text, claimed_model="claude-sonnet-4")
    print(f"  Trust score: {evidence['trust_score']:.2f}")
    print(f"  Verified: {evidence['verified']}")
    if evidence['alerts']:
        print(f"  Alert: {evidence['alerts'][0]['message']}")


if __name__ == "__main__":
    main()
