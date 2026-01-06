#!/usr/bin/env python3
"""
Epistemic Analyzer

Measures the "confidence topology" of text:
- How does the model hedge?
- Where do hedges cluster?
- What's the confidence arc (start to end)?
- What assertion patterns does it use?

These features are medium-trust (harder to spoof than vocabulary, easier than trajectory).

The 6 Epistemic Features:
1. hedge_density       - Overall rate of hedging language
2. hedge_clustering    - Are hedges bunched or spread evenly?
3. hedge_position_bias - Front-loaded vs back-loaded hedging
4. confidence_mean     - Average assertion strength
5. confidence_variance - How stable is confidence?
6. confidence_arc      - Does confidence increase or decrease through text?
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import Counter


# =============================================================================
# EPISTEMIC LEXICONS
# =============================================================================

HEDGE_MARKERS = {
    "uncertainty": [
        "perhaps", "maybe", "might", "could", "possibly", "likely", "unlikely",
        "probably", "presumably", "apparently", "seemingly", "arguably",
        "conceivably", "potentially", "plausibly",
    ],
    "softening": [
        "somewhat", "relatively", "fairly", "rather", "quite", "slightly",
        "moderately", "reasonably", "partially", "largely", "mostly",
        "generally", "typically", "usually", "often", "sometimes",
    ],
    "attribution": [
        "seems", "appears", "suggests", "indicates", "implies", "tends",
        "looks like", "sounds like", "feels like",
    ],
    "perspective": [
        "i think", "i believe", "in my view", "in my opinion", "i would say",
        "i suspect", "i imagine", "i suppose", "i assume", "i gather",
        "one could say", "one might argue", "it could be argued",
    ],
    "qualification": [
        "to some extent", "in a sense", "in some ways", "to a degree",
        "up to a point", "more or less", "kind of", "sort of",
        "in general", "for the most part", "by and large",
    ],
}

CONFIDENCE_MARKERS = {
    "high": [
        "certainly", "definitely", "absolutely", "clearly", "obviously",
        "undoubtedly", "unquestionably", "surely", "indeed", "of course",
        "without doubt", "no doubt", "must", "will", "always", "never",
        "proven", "established", "demonstrates", "proves", "confirms",
    ],
    "medium": [
        "should", "would", "can", "does", "is", "are", "shows", "indicates",
        "suggests", "supports", "evidence", "research shows", "studies show",
    ],
    "low": [
        "might", "may", "could", "possibly", "perhaps", "maybe",
        "it's possible", "there's a chance", "potentially",
    ],
}

ASSERTION_VERBS = {
    "strong": [
        "is", "are", "was", "were", "proves", "demonstrates", "shows",
        "confirms", "establishes", "reveals", "determines",
    ],
    "medium": [
        "indicates", "suggests", "implies", "supports", "points to",
        "appears", "seems", "tends",
    ],
    "weak": [
        "might", "may", "could", "possibly", "perhaps",
    ],
}


@dataclass
class EpistemicFeatures:
    """Epistemic signature of a text."""
    
    # Core features (6 dims)
    hedge_density: float = 0.0
    hedge_clustering: float = 0.0
    hedge_position_bias: float = 0.5  # 0=front-loaded, 1=back-loaded, 0.5=even
    confidence_mean: float = 0.5
    confidence_variance: float = 0.0
    confidence_arc: float = 0.0  # Positive=increasing, negative=decreasing
    
    # Diagnostic (not for classification)
    hedge_count: int = 0
    hedge_types: dict = None
    sentence_count: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Return the 6 core features as a vector."""
        return np.array([
            self.hedge_density,
            self.hedge_clustering,
            self.hedge_position_bias,
            self.confidence_mean,
            self.confidence_variance,
            self.confidence_arc,
        ])
    
    @property
    def feature_names(self) -> list[str]:
        return [
            "hedge_density",
            "hedge_clustering",
            "hedge_position_bias",
            "confidence_mean",
            "confidence_variance",
            "confidence_arc",
        ]


class EpistemicAnalyzer:
    """
    Analyzes the epistemic posture of text.
    
    Measures how a model expresses certainty and uncertainty.
    """
    
    def __init__(self):
        # Compile patterns for efficiency
        self._hedge_patterns = {}
        for category, words in HEDGE_MARKERS.items():
            pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
            self._hedge_patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        self._confidence_patterns = {}
        for level, words in CONFIDENCE_MARKERS.items():
            pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
            self._confidence_patterns[level] = re.compile(pattern, re.IGNORECASE)
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_hedges(self, text: str) -> list[tuple[int, str, str]]:
        """
        Find all hedges in text.
        
        Returns:
            List of (position, hedge_word, category)
        """
        hedges = []
        text_lower = text.lower()
        
        for category, pattern in self._hedge_patterns.items():
            for match in pattern.finditer(text_lower):
                hedges.append((match.start(), match.group(), category))
        
        # Sort by position
        hedges.sort(key=lambda x: x[0])
        return hedges
    
    def _compute_clustering(self, positions: list[float], n_bins: int = 10) -> float:
        """
        Compute clustering coefficient for positions.
        
        Returns:
            0 = evenly distributed, 1 = highly clustered
        """
        if len(positions) < 2:
            return 0.0
        
        # Bin positions
        bins = np.histogram(positions, bins=n_bins, range=(0, 1))[0]
        
        # Expected uniform distribution
        expected = len(positions) / n_bins
        
        # Chi-squared-like measure
        if expected > 0:
            deviation = np.sum((bins - expected) ** 2) / expected
            # Normalize to 0-1
            max_deviation = len(positions) * (n_bins - 1)  # All in one bin
            return min(1.0, deviation / max(max_deviation, 1))
        
        return 0.0
    
    def _score_sentence_confidence(self, sentence: str) -> float:
        """
        Score the confidence level of a sentence (0-1).
        
        Returns:
            0 = low confidence, 1 = high confidence
        """
        sentence_lower = sentence.lower()
        
        # Count markers at each level
        high_count = len(self._confidence_patterns["high"].findall(sentence_lower))
        medium_count = len(self._confidence_patterns["medium"].findall(sentence_lower))
        low_count = len(self._confidence_patterns["low"].findall(sentence_lower))
        
        # Also count hedges as negative confidence
        hedge_count = sum(
            len(pattern.findall(sentence_lower))
            for pattern in self._hedge_patterns.values()
        )
        
        # Weighted score
        total = high_count + medium_count + low_count + hedge_count
        if total == 0:
            return 0.5  # Neutral
        
        score = (high_count * 1.0 + medium_count * 0.6 + low_count * 0.3 - hedge_count * 0.3) / total
        return np.clip(score, 0, 1)
    
    def analyze(self, text: str) -> EpistemicFeatures:
        """
        Analyze epistemic features of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            EpistemicFeatures with 6 core metrics
        """
        text_length = len(text)
        words = text.split()
        n_words = len(words)
        
        if n_words < 5:
            return EpistemicFeatures(sentence_count=0)
        
        # Find hedges
        hedges = self._find_hedges(text)
        
        # Hedge density (per 100 words)
        hedge_density = len(hedges) / n_words * 100
        
        # Hedge positions (normalized 0-1)
        hedge_positions = [h[0] / text_length for h in hedges]
        
        # Hedge clustering
        hedge_clustering = self._compute_clustering(hedge_positions)
        
        # Hedge position bias (mean position)
        if hedge_positions:
            hedge_position_bias = np.mean(hedge_positions)
        else:
            hedge_position_bias = 0.5
        
        # Hedge types
        hedge_types = Counter(h[2] for h in hedges)
        
        # Sentence-level confidence
        sentences = self._split_sentences(text)
        n_sentences = len(sentences)
        
        if n_sentences == 0:
            return EpistemicFeatures(
                hedge_density=hedge_density,
                hedge_clustering=hedge_clustering,
                hedge_position_bias=hedge_position_bias,
                hedge_count=len(hedges),
                hedge_types=dict(hedge_types),
            )
        
        confidence_scores = [self._score_sentence_confidence(s) for s in sentences]
        
        confidence_mean = np.mean(confidence_scores)
        confidence_variance = np.var(confidence_scores) if len(confidence_scores) > 1 else 0.0
        
        # Confidence arc: linear regression slope
        if len(confidence_scores) >= 3:
            x = np.arange(len(confidence_scores))
            slope, _ = np.polyfit(x, confidence_scores, 1)
            # Normalize slope to roughly -1 to 1 range
            confidence_arc = slope * len(confidence_scores) / 2
            confidence_arc = np.clip(confidence_arc, -1, 1)
        else:
            confidence_arc = 0.0
        
        return EpistemicFeatures(
            hedge_density=float(hedge_density),
            hedge_clustering=float(hedge_clustering),
            hedge_position_bias=float(hedge_position_bias),
            confidence_mean=float(confidence_mean),
            confidence_variance=float(confidence_variance),
            confidence_arc=float(confidence_arc),
            hedge_count=len(hedges),
            hedge_types=dict(hedge_types),
            sentence_count=n_sentences,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the epistemic analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Epistemic analysis demo")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    
    args = parser.parse_args()
    
    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        # Demo texts with different epistemic postures
        
        print("="*60)
        print("EPISTEMIC ANALYSIS DEMO")
        print("="*60)
        
        analyzer = EpistemicAnalyzer()
        
        # High confidence text
        text_confident = """
        The evidence clearly demonstrates that climate change is accelerating. 
        Scientists have definitively proven the connection between CO2 and warming.
        This is unquestionably the greatest challenge of our time.
        We must act immediately. The data leaves no room for doubt.
        """
        
        # Hedged text
        text_hedged = """
        It seems likely that the proposed approach might work in some cases.
        Perhaps we could consider this as one possible solution, though it's 
        somewhat uncertain whether it would be effective. I think, generally 
        speaking, there could be merit to this idea, but I would say we should
        probably investigate further before drawing any conclusions.
        """
        
        # Mixed/arc text
        text_arc = """
        Initially, I wasn't sure about this approach. It seemed questionable
        at first glance. However, after reviewing the evidence, I began to see
        its merits. The data increasingly supports the hypothesis. Now I'm 
        confident this is the right direction. The conclusion is clear.
        """
        
        for name, text in [("Confident", text_confident), ("Hedged", text_hedged), ("Arc", text_arc)]:
            print(f"\n--- {name} Text ---")
            features = analyzer.analyze(text)
            
            print(f"  hedge_density:       {features.hedge_density:.2f}")
            print(f"  hedge_clustering:    {features.hedge_clustering:.2f}")
            print(f"  hedge_position_bias: {features.hedge_position_bias:.2f}")
            print(f"  confidence_mean:     {features.confidence_mean:.2f}")
            print(f"  confidence_variance: {features.confidence_variance:.3f}")
            print(f"  confidence_arc:      {features.confidence_arc:+.2f}")
            print(f"  hedge_types:         {features.hedge_types}")
        
        return
    
    analyzer = EpistemicAnalyzer()
    features = analyzer.analyze(text)
    
    print("="*60)
    print("EPISTEMIC FEATURES")
    print("="*60)
    print(f"\nHedge density:       {features.hedge_density:.2f} (per 100 words)")
    print(f"Hedge clustering:    {features.hedge_clustering:.2f} (0=spread, 1=bunched)")
    print(f"Hedge position bias: {features.hedge_position_bias:.2f} (0=front, 1=back)")
    print(f"Confidence mean:     {features.confidence_mean:.2f}")
    print(f"Confidence variance: {features.confidence_variance:.3f}")
    print(f"Confidence arc:      {features.confidence_arc:+.2f} (+=increasing, -=decreasing)")
    print(f"\nHedge types: {features.hedge_types}")
    print(f"Sentences analyzed: {features.sentence_count}")


if __name__ == "__main__":
    main()
