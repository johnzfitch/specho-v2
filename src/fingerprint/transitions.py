#!/usr/bin/env python3
"""
Transition Analyzer

Measures how models connect ideas - the "connective tissue" of thought.

Each model has preferences:
- Claude: Heavy on contrastive ("however", "that said", "on the other hand")
- GPT: Heavy on additive ("moreover", "additionally", "furthermore")
- Gemini: Heavy on exemplifying ("for example", "such as", "specifically")

The 6 Transition Features:
1. additive_rate       - "and also" style connections
2. contrastive_rate    - "but however" style turns
3. causal_rate         - "because therefore" reasoning chains
4. temporal_rate       - "then next" sequencing
5. exemplifying_rate   - "for example" illustration
6. reformulating_rate  - "in other words" clarification

These are medium-trust features (relatively easy to mask consciously).
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


# =============================================================================
# TRANSITION LEXICONS
# =============================================================================

TRANSITION_MARKERS = {
    "additive": [
        # Simple addition
        "also", "too", "as well", "in addition", "additionally",
        "moreover", "furthermore", "besides", "what's more",
        "not only", "along with", "coupled with",
        # Similarity
        "similarly", "likewise", "in the same way", "equally",
        "correspondingly", "by the same token",
    ],
    "contrastive": [
        # Direct contrast
        "however", "but", "yet", "although", "though", "even though",
        "nevertheless", "nonetheless", "still", "despite", "in spite of",
        "whereas", "while", "whilst", "on the other hand",
        "on the contrary", "conversely", "in contrast",
        "that said", "having said that", "then again",
        # Concession
        "admittedly", "granted", "of course", "certainly",
    ],
    "causal": [
        # Cause
        "because", "since", "as", "due to", "owing to",
        "on account of", "given that", "seeing that",
        # Effect
        "therefore", "thus", "hence", "so", "consequently",
        "as a result", "accordingly", "for this reason",
        "that's why", "which is why",
    ],
    "temporal": [
        # Sequence
        "first", "second", "third", "finally", "lastly",
        "then", "next", "afterward", "afterwards", "subsequently",
        "previously", "before", "after", "meanwhile", "simultaneously",
        # Beginning/End
        "initially", "originally", "eventually", "ultimately",
        "in the end", "at last",
    ],
    "exemplifying": [
        # Examples
        "for example", "for instance", "such as", "like",
        "including", "especially", "particularly", "notably",
        "specifically", "in particular", "to illustrate",
        "as an illustration", "e.g.", "i.e.",
        # Case
        "in this case", "in that case", "take the case of",
    ],
    "reformulating": [
        # Clarification
        "in other words", "that is", "that is to say",
        "namely", "specifically", "more precisely",
        "to put it another way", "put differently",
        "what i mean is", "essentially", "basically",
        # Summary
        "in short", "in brief", "in summary", "to summarize",
        "in conclusion", "to conclude", "overall", "all in all",
    ],
}


@dataclass
class TransitionFeatures:
    """Transition signature of a text."""
    
    # Core features (6 dims) - rates normalized to sum to 1
    additive_rate: float = 0.0
    contrastive_rate: float = 0.0
    causal_rate: float = 0.0
    temporal_rate: float = 0.0
    exemplifying_rate: float = 0.0
    reformulating_rate: float = 0.0
    
    # Diagnostic
    total_transitions: int = 0
    transition_density: float = 0.0  # Per 100 words
    transition_counts: dict = None
    
    def to_vector(self) -> np.ndarray:
        """Return the 6 core features as a vector."""
        return np.array([
            self.additive_rate,
            self.contrastive_rate,
            self.causal_rate,
            self.temporal_rate,
            self.exemplifying_rate,
            self.reformulating_rate,
        ])
    
    @property
    def feature_names(self) -> list[str]:
        return [
            "additive_rate",
            "contrastive_rate",
            "causal_rate",
            "temporal_rate",
            "exemplifying_rate",
            "reformulating_rate",
        ]
    
    @property
    def dominant_type(self) -> str:
        """Return the most common transition type."""
        rates = {
            "additive": self.additive_rate,
            "contrastive": self.contrastive_rate,
            "causal": self.causal_rate,
            "temporal": self.temporal_rate,
            "exemplifying": self.exemplifying_rate,
            "reformulating": self.reformulating_rate,
        }
        return max(rates, key=rates.get)


class TransitionAnalyzer:
    """
    Analyzes transition patterns in text.
    
    Measures how a model connects ideas and moves between concepts.
    """
    
    def __init__(self):
        # Compile patterns for efficiency
        self._patterns = {}
        for category, markers in TRANSITION_MARKERS.items():
            # Sort by length (longest first) to match multi-word phrases first
            markers_sorted = sorted(markers, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(m) for m in markers_sorted) + r')\b'
            self._patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def _find_transitions(self, text: str) -> list[tuple[int, str, str]]:
        """
        Find all transitions in text.
        
        Returns:
            List of (position, marker, category)
        """
        transitions = []
        text_lower = text.lower()
        
        # Track positions to avoid double-counting overlaps
        used_positions = set()
        
        for category, pattern in self._patterns.items():
            for match in pattern.finditer(text_lower):
                start, end = match.start(), match.end()
                
                # Check for overlap
                if any(start <= p < end or start < p <= end for p in used_positions):
                    continue
                
                transitions.append((start, match.group(), category))
                used_positions.update(range(start, end))
        
        # Sort by position
        transitions.sort(key=lambda x: x[0])
        return transitions
    
    def analyze(self, text: str) -> TransitionFeatures:
        """
        Analyze transition features of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            TransitionFeatures with 6 core metrics
        """
        words = text.split()
        n_words = len(words)
        
        if n_words < 10:
            return TransitionFeatures(transition_counts={})
        
        # Find all transitions
        transitions = self._find_transitions(text)
        
        # Count by category
        counts = Counter(t[2] for t in transitions)
        total = len(transitions)
        
        # Calculate rates (normalized distribution)
        if total > 0:
            additive_rate = counts.get("additive", 0) / total
            contrastive_rate = counts.get("contrastive", 0) / total
            causal_rate = counts.get("causal", 0) / total
            temporal_rate = counts.get("temporal", 0) / total
            exemplifying_rate = counts.get("exemplifying", 0) / total
            reformulating_rate = counts.get("reformulating", 0) / total
        else:
            additive_rate = contrastive_rate = causal_rate = 0.0
            temporal_rate = exemplifying_rate = reformulating_rate = 0.0
        
        # Transition density (per 100 words)
        transition_density = total / n_words * 100
        
        return TransitionFeatures(
            additive_rate=float(additive_rate),
            contrastive_rate=float(contrastive_rate),
            causal_rate=float(causal_rate),
            temporal_rate=float(temporal_rate),
            exemplifying_rate=float(exemplifying_rate),
            reformulating_rate=float(reformulating_rate),
            total_transitions=total,
            transition_density=float(transition_density),
            transition_counts=dict(counts),
        )


# =============================================================================
# TRANSITION SEQUENCE ANALYSIS
# =============================================================================

class TransitionSequenceAnalyzer:
    """
    Analyzes transition sequences (n-grams of transition types).
    
    This captures patterns like:
    - "additive → contrastive" (Claude's "moreover... however" pattern)
    - "causal → causal" (reasoning chains)
    - "exemplifying → reformulating" (example then summary)
    """
    
    def __init__(self):
        self.base_analyzer = TransitionAnalyzer()
    
    def get_transition_sequence(self, text: str) -> list[str]:
        """Get sequence of transition types in order."""
        transitions = self.base_analyzer._find_transitions(text)
        return [t[2] for t in transitions]
    
    def get_bigrams(self, text: str) -> Counter:
        """Get transition type bigrams."""
        sequence = self.get_transition_sequence(text)
        bigrams = []
        for i in range(len(sequence) - 1):
            bigrams.append(f"{sequence[i]}→{sequence[i+1]}")
        return Counter(bigrams)
    
    def analyze_flow(self, text: str) -> dict:
        """
        Analyze the "flow" of transitions through text.
        
        Returns dict with:
        - sequence: List of transition types in order
        - bigrams: Counter of transition pairs
        - flow_entropy: How unpredictable is the transition sequence?
        """
        sequence = self.get_transition_sequence(text)
        bigrams = self.get_bigrams(text)
        
        # Compute entropy of bigram distribution
        if bigrams:
            total = sum(bigrams.values())
            probs = [c / total for c in bigrams.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            entropy = 0.0
        
        return {
            "sequence": sequence,
            "bigrams": dict(bigrams),
            "flow_entropy": float(entropy),
            "n_transitions": len(sequence),
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the transition analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transition analysis demo")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    
    args = parser.parse_args()
    
    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        print("="*60)
        print("TRANSITION ANALYSIS DEMO")
        print("="*60)
        
        analyzer = TransitionAnalyzer()
        seq_analyzer = TransitionSequenceAnalyzer()
        
        # Different transition styles
        text_additive = """
        Machine learning has transformed many industries. Additionally, it has
        created new job categories. Moreover, the technology continues to improve.
        Furthermore, costs are decreasing. Also, accessibility is increasing.
        Similarly, related fields like data science are growing.
        """
        
        text_contrastive = """
        Machine learning has many benefits. However, it also raises concerns.
        Although the technology is powerful, it requires careful deployment.
        On the other hand, not all applications are problematic. Nevertheless,
        we must remain vigilant. That said, the potential is enormous.
        """
        
        text_causal = """
        Machine learning requires large datasets. Because data is abundant,
        models can learn effectively. Therefore, accuracy has improved.
        Consequently, more applications are viable. Since computing power
        is cheaper, training is faster. Thus, deployment is more practical.
        """
        
        for name, text in [("Additive-heavy", text_additive), 
                          ("Contrastive-heavy", text_contrastive),
                          ("Causal-heavy", text_causal)]:
            print(f"\n--- {name} ---")
            features = analyzer.analyze(text)
            
            print(f"  Transition density: {features.transition_density:.1f} per 100 words")
            print(f"  Distribution:")
            for cat in ["additive", "contrastive", "causal", "temporal", "exemplifying", "reformulating"]:
                rate = getattr(features, f"{cat}_rate")
                bar = "█" * int(rate * 30)
                print(f"    {cat:14s}: {rate:.2f} {bar}")
            print(f"  Dominant type: {features.dominant_type}")
            
            # Sequence analysis
            flow = seq_analyzer.analyze_flow(text)
            print(f"  Sequence: {' → '.join(flow['sequence'][:5])}...")
            print(f"  Flow entropy: {flow['flow_entropy']:.2f}")
        
        return
    
    analyzer = TransitionAnalyzer()
    features = analyzer.analyze(text)
    
    print("="*60)
    print("TRANSITION FEATURES")
    print("="*60)
    print(f"\nTotal transitions: {features.total_transitions}")
    print(f"Density: {features.transition_density:.1f} per 100 words")
    
    print("\nDistribution:")
    for cat in ["additive", "contrastive", "causal", "temporal", "exemplifying", "reformulating"]:
        rate = getattr(features, f"{cat}_rate")
        count = features.transition_counts.get(cat, 0)
        bar = "█" * int(rate * 30)
        print(f"  {cat:14s}: {rate:.2f} ({count:2d}) {bar}")
    
    print(f"\nDominant type: {features.dominant_type}")


if __name__ == "__main__":
    main()
