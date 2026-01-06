#!/usr/bin/env python3
"""
Syntactic Rhythm Analyzer

Measures the "breath" of text - the structural rhythm and pacing.

Models have distinct rhythms:
- Some prefer short, punchy clauses
- Others use long, complex structures
- Some oscillate between short and long (rhythm)
- Punctuation choices reveal training biases

The 7 Syntactic Features:
1. clause_length_mean    - Average clause length
2. clause_length_std     - Variation in clause length
3. clause_rhythm_autocorr - Short-long-short patterns (autocorrelation)
4. sentence_complexity   - Clauses per sentence
5. comma_density         - Comma usage rate
6. semicolon_rate        - Semicolon preference
7. parenthetical_rate    - Parenthetical/em-dash usage

These are low-medium trust (can be consciously adjusted, but hard to fake rhythm).
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import Counter


@dataclass
class SyntacticFeatures:
    """Syntactic rhythm signature."""
    
    # Core features (7 dims)
    clause_length_mean: float = 0.0
    clause_length_std: float = 0.0
    clause_rhythm_autocorr: float = 0.0
    sentence_complexity: float = 1.0  # Clauses per sentence
    comma_density: float = 0.0
    semicolon_rate: float = 0.0
    parenthetical_rate: float = 0.0
    
    # Diagnostic
    n_clauses: int = 0
    n_sentences: int = 0
    clause_lengths: list = None
    
    def to_vector(self) -> np.ndarray:
        """Return the 7 core features as a vector."""
        return np.array([
            self.clause_length_mean,
            self.clause_length_std,
            self.clause_rhythm_autocorr,
            self.sentence_complexity,
            self.comma_density,
            self.semicolon_rate,
            self.parenthetical_rate,
        ])
    
    @property
    def feature_names(self) -> list[str]:
        return [
            "clause_length_mean",
            "clause_length_std",
            "clause_rhythm_autocorr",
            "sentence_complexity",
            "comma_density",
            "semicolon_rate",
            "parenthetical_rate",
        ]


class SyntacticAnalyzer:
    """
    Analyzes syntactic rhythm and structure.
    
    Measures the "pacing" of text through structural patterns.
    """
    
    def __init__(self):
        pass
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1PERIOD ', text)
        text = re.sub(r'\b(e\.g|i\.e|etc)\.\s', r'\1PERIOD ', text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        sentences = [s.replace('PERIOD', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_clauses(self, sentence: str) -> list[str]:
        """
        Split a sentence into clauses.
        
        Uses punctuation and conjunctions as boundaries.
        """
        # Split on clause boundaries
        parts = re.split(
            r'(?<=[,;:])\s+|'           # After comma, semicolon, colon
            r'\s*[—–]\s*|'               # Em-dash, en-dash
            r'\s+(?:but|and|or|yet|so|because|although|while|if|when|since|unless)\s+',
            sentence,
            flags=re.IGNORECASE
        )
        
        # Filter empty and too-short clauses
        clauses = [p.strip() for p in parts if p and len(p.split()) >= 2]
        
        return clauses if clauses else [sentence]
    
    def _compute_autocorrelation(self, sequence: list[float], lag: int = 1) -> float:
        """
        Compute autocorrelation at given lag.
        
        High autocorrelation = similar adjacent values (monotonous rhythm)
        Low/negative = alternating values (varied rhythm)
        """
        if len(sequence) < lag + 2:
            return 0.0
        
        n = len(sequence)
        mean = np.mean(sequence)
        
        numerator = sum((sequence[i] - mean) * (sequence[i + lag] - mean) 
                       for i in range(n - lag))
        denominator = sum((x - mean) ** 2 for x in sequence)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _count_punctuation(self, text: str) -> dict:
        """Count punctuation marks."""
        return {
            "comma": text.count(','),
            "semicolon": text.count(';'),
            "colon": text.count(':'),
            "em_dash": text.count('—') + text.count('–'),
            "parenthesis": text.count('('),
            "period": text.count('.'),
            "question": text.count('?'),
            "exclamation": text.count('!'),
        }
    
    def analyze(self, text: str) -> SyntacticFeatures:
        """
        Analyze syntactic features of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SyntacticFeatures with 7 core metrics
        """
        words = text.split()
        n_words = len(words)
        
        if n_words < 10:
            return SyntacticFeatures()
        
        # Split into sentences and clauses
        sentences = self._split_sentences(text)
        n_sentences = len(sentences)
        
        all_clauses = []
        clauses_per_sentence = []
        
        for sentence in sentences:
            clauses = self._split_clauses(sentence)
            all_clauses.extend(clauses)
            clauses_per_sentence.append(len(clauses))
        
        n_clauses = len(all_clauses)
        
        if n_clauses == 0:
            return SyntacticFeatures(n_sentences=n_sentences)
        
        # Clause length statistics
        clause_lengths = [len(c.split()) for c in all_clauses]
        
        clause_length_mean = np.mean(clause_lengths)
        clause_length_std = np.std(clause_lengths) if len(clause_lengths) > 1 else 0.0
        
        # Rhythm autocorrelation (do lengths alternate or stay similar?)
        clause_rhythm_autocorr = self._compute_autocorrelation(clause_lengths, lag=1)
        
        # Sentence complexity (average clauses per sentence)
        sentence_complexity = n_clauses / max(n_sentences, 1)
        
        # Punctuation analysis
        punct_counts = self._count_punctuation(text)
        
        comma_density = punct_counts["comma"] / n_words * 100
        semicolon_rate = punct_counts["semicolon"] / max(n_sentences, 1)
        parenthetical_rate = (punct_counts["parenthesis"] + punct_counts["em_dash"]) / max(n_sentences, 1)
        
        return SyntacticFeatures(
            clause_length_mean=float(clause_length_mean),
            clause_length_std=float(clause_length_std),
            clause_rhythm_autocorr=float(clause_rhythm_autocorr),
            sentence_complexity=float(sentence_complexity),
            comma_density=float(comma_density),
            semicolon_rate=float(semicolon_rate),
            parenthetical_rate=float(parenthetical_rate),
            n_clauses=n_clauses,
            n_sentences=n_sentences,
            clause_lengths=clause_lengths,
        )


# =============================================================================
# RHYTHM VISUALIZATION
# =============================================================================

def visualize_rhythm(text: str) -> str:
    """Create ASCII visualization of clause length rhythm."""
    analyzer = SyntacticAnalyzer()
    features = analyzer.analyze(text)
    
    if not features.clause_lengths:
        return "No clauses detected"
    
    lengths = features.clause_lengths
    max_len = max(lengths)
    
    lines = []
    for i, length in enumerate(lengths):
        bar_width = int(length / max_len * 40)
        bar = "█" * bar_width
        lines.append(f"{i+1:3d} [{length:2d}] {bar}")
    
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the syntactic analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Syntactic rhythm analysis")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--visualize", action="store_true", help="Show rhythm visualization")
    
    args = parser.parse_args()
    
    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        print("="*60)
        print("SYNTACTIC RHYTHM ANALYSIS DEMO")
        print("="*60)
        
        analyzer = SyntacticAnalyzer()
        
        # Different rhythmic styles
        text_uniform = """
        The system processes data. The data is stored locally. The storage
        is encrypted. The encryption uses AES. The keys rotate daily.
        The rotation is automatic. The process is reliable.
        """
        
        text_varied = """
        Modern distributed systems—those sprawling architectures that power 
        everything from social media to financial trading—face an inherent 
        tension. They must be fast. And they must be consistent. But physics, 
        that stubborn discipline, refuses to allow both simultaneously across 
        arbitrary distances; hence the CAP theorem.
        """
        
        text_complex = """
        The implementation of microservices architecture, which has become 
        increasingly popular in enterprise software development over the past 
        decade, requires careful consideration of service boundaries (which 
        define the scope of each independent unit), communication patterns 
        (whether synchronous or asynchronous), and data management strategies 
        (including the challenging problem of distributed transactions).
        """
        
        for name, text in [("Uniform/Simple", text_uniform), 
                          ("Varied/Punchy", text_varied),
                          ("Complex/Nested", text_complex)]:
            print(f"\n--- {name} ---")
            features = analyzer.analyze(text)
            
            print(f"  Clause length:    {features.clause_length_mean:.1f} ± {features.clause_length_std:.1f} words")
            print(f"  Rhythm autocorr:  {features.clause_rhythm_autocorr:+.2f} ({'uniform' if features.clause_rhythm_autocorr > 0.3 else 'varied' if features.clause_rhythm_autocorr < -0.1 else 'neutral'})")
            print(f"  Complexity:       {features.sentence_complexity:.1f} clauses/sentence")
            print(f"  Comma density:    {features.comma_density:.1f} per 100 words")
            print(f"  Semicolon rate:   {features.semicolon_rate:.2f} per sentence")
            print(f"  Parentheticals:   {features.parenthetical_rate:.2f} per sentence")
            
            print(f"\n  Clause rhythm:")
            print(visualize_rhythm(text))
        
        return
    
    analyzer = SyntacticAnalyzer()
    features = analyzer.analyze(text)
    
    print("="*60)
    print("SYNTACTIC FEATURES")
    print("="*60)
    print(f"\nClauses: {features.n_clauses}")
    print(f"Sentences: {features.n_sentences}")
    print(f"\nClause length: {features.clause_length_mean:.1f} ± {features.clause_length_std:.1f} words")
    print(f"Rhythm autocorrelation: {features.clause_rhythm_autocorr:+.2f}")
    print(f"Sentence complexity: {features.sentence_complexity:.1f} clauses/sentence")
    print(f"\nPunctuation:")
    print(f"  Comma density: {features.comma_density:.1f} per 100 words")
    print(f"  Semicolon rate: {features.semicolon_rate:.2f} per sentence")
    print(f"  Parenthetical rate: {features.parenthetical_rate:.2f} per sentence")
    
    if args.visualize:
        print("\nClause Rhythm:")
        print(visualize_rhythm(text))


if __name__ == "__main__":
    main()
