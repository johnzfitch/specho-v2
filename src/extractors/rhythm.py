"""
Syntactic Rhythm Extractor (7D)

Structural pacing patterns.

Features (sorted by Cohen's d):
- clause_length_std (d=-0.30): AI has uniform clauses
- semicolon_rate (d=-0.30): Mixed signal
- clause_rhythm_autocorr (d=+0.25): AI has rhythmic pattern
- parenthetical_rate (d=-0.25): Humans use more
- clause_length_mean (d=+0.20): AI slightly longer
- comma_density (d=+0.20): AI uses more commas
- sentence_complexity (d=+0.15): AI is slightly complex
"""

import re
import math
from typing import Dict, List

from .base import BaseExtractor


class RhythmExtractor(BaseExtractor):
    """
    Syntactic rhythm features with NO external dependencies.

    Measures pacing patterns that differ between human and AI writing.
    Each feature is extracted independently with fault isolation.
    """

    group = "rhythm"
    dependencies = []  # Pure Python

    @property
    def feature_names(self) -> List[str]:
        return [
            "clause_length_mean",      # d=+0.20
            "clause_length_std",       # d=-0.30
            "clause_rhythm_autocorr",  # d=+0.25
            "sentence_complexity",     # d=+0.15
            "comma_density",           # d=+0.20
            "semicolon_rate",          # d=-0.30
            "parenthetical_rate",      # d=-0.25
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Safely extract a single feature with error handling."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract rhythm features with fault isolation."""
        features = {name: 0.0 for name in self.feature_names}

        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        word_count = max(len(words), 1)
        sentence_count = max(len(sentences), 1)

        if word_count < 10:
            return features

        # Clause approximation (using commas and semicolons as boundaries)
        clauses = re.split(r'[,;]', text)
        clauses = [c.strip() for c in clauses if c.strip() and len(c.split()) > 1]

        # 1-2. Clause length statistics (ISOLATED)
        def calc_clause_stats():
            if clauses:
                clause_lengths = [len(c.split()) for c in clauses]
                features["clause_length_mean"] = sum(clause_lengths) / len(clause_lengths)
                if len(clause_lengths) > 1:
                    mean = features["clause_length_mean"]
                    features["clause_length_std"] = math.sqrt(
                        sum((l - mean)**2 for l in clause_lengths) / len(clause_lengths)
                    )

        self._safe_extract("clause_stats", calc_clause_stats)

        # 3. Clause rhythm autocorrelation (ISOLATED)
        def calc_autocorr():
            if clauses and len(clauses) > 2:
                clause_lengths = [len(c.split()) for c in clauses]
                mean_c = sum(clause_lengths) / len(clause_lengths)
                var_c = sum((l - mean_c)**2 for l in clause_lengths) / len(clause_lengths)
                if var_c > 0:
                    autocov = sum(
                        (clause_lengths[i] - mean_c) * (clause_lengths[i+1] - mean_c)
                        for i in range(len(clause_lengths) - 1)
                    ) / (len(clause_lengths) - 1)
                    features["clause_rhythm_autocorr"] = autocov / var_c

        self._safe_extract("autocorr", calc_autocorr)

        # 4. Sentence complexity (clauses per sentence) (ISOLATED)
        def calc_complexity():
            features["sentence_complexity"] = len(clauses) / sentence_count

        self._safe_extract("complexity", calc_complexity)

        # 5. Comma density (per 100 words) (ISOLATED)
        def calc_comma():
            features["comma_density"] = text.count(',') / (word_count / 100)

        self._safe_extract("comma", calc_comma)

        # 6. Semicolon rate (per sentence) (ISOLATED)
        def calc_semicolon():
            features["semicolon_rate"] = text.count(';') / sentence_count

        self._safe_extract("semicolon", calc_semicolon)

        # 7. Parenthetical rate (ISOLATED)
        def calc_paren():
            paren_count = (text.count('(') + text.count(')')) / 2
            features["parenthetical_rate"] = paren_count / sentence_count

        self._safe_extract("paren", calc_paren)

        return features


# Standalone function
def extract_rhythm_features(text: str) -> Dict[str, float]:
    """Extract rhythm features (standalone function)."""
    extractor = RhythmExtractor()
    result = extractor.extract(text)
    return result.features
