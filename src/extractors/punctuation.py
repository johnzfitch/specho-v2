"""
Punctuation Pattern Extractor

Extracts features based on punctuation usage patterns.

Features (sorted by Cohen's d):
- question_rate (d=+1.14): Humans ask more questions
- em_dash_rate (d=+0.70): AI uses more em-dashes for parentheticals
- semicolon_rate (d=+0.65): AI uses more semicolons
- exclamation_rate (d=-0.55): Humans use more exclamations
- ellipsis_rate (d=-0.48): Humans use more trailing off...
- parenthetical_rate (d=+0.42): AI uses more parenthetical asides
- colon_rate (d=+0.38): AI uses more colons for lists/explanations
"""

import re
from typing import Dict, List

from .base import BaseExtractor


class PunctuationExtractor(BaseExtractor):
    """
    Punctuation pattern features with NO external dependencies.

    Punctuation choices reveal writing style - AI tends toward
    formal punctuation while humans are more expressive.
    """

    group = "punctuation"
    dependencies = []  # Pure regex - no dependencies

    @property
    def feature_names(self) -> List[str]:
        return [
            "question_rate",       # d=+1.14 CRITICAL
            "em_dash_rate",        # d=+0.70 HIGH
            "semicolon_rate",      # d=+0.65
            "exclamation_rate",    # d=-0.55
            "ellipsis_rate",       # d=-0.48
            "parenthetical_rate",  # d=+0.42
            "colon_rate",          # d=+0.38
            "comma_density",       # General measure
        ]

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract punctuation pattern features."""
        # Count sentences (approximate)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        word_count = len(text.split())
        if word_count < 10:
            return {name: 0.0 for name in self.feature_names}

        features = {}

        # 1. Question rate (d=+1.14) - CRITICAL
        # Questions per sentence
        question_count = text.count('?')
        features["question_rate"] = question_count / sentence_count

        # 2. Em-dash rate (d=+0.70)
        # Count both proper em-dashes and double-hyphens used as em-dashes
        em_dash_count = (
            text.count('—') +      # proper em-dash
            text.count('–') +      # en-dash often used as em-dash
            len(re.findall(r'--', text))  # double-hyphen
        )
        features["em_dash_rate"] = em_dash_count / sentence_count

        # 3. Semicolon rate (d=+0.65)
        semicolon_count = text.count(';')
        features["semicolon_rate"] = semicolon_count / sentence_count

        # 4. Exclamation rate (d=-0.55)
        exclamation_count = text.count('!')
        features["exclamation_rate"] = exclamation_count / sentence_count

        # 5. Ellipsis rate (d=-0.48)
        # Count actual ellipsis and triple dots
        ellipsis_count = (
            text.count('…') +
            len(re.findall(r'\.{3,}', text))
        )
        features["ellipsis_rate"] = ellipsis_count / sentence_count

        # 6. Parenthetical rate (d=+0.42)
        # Count parenthetical asides
        paren_count = text.count('(')
        features["parenthetical_rate"] = paren_count / sentence_count

        # 7. Colon rate (d=+0.38)
        # Exclude colons in times (12:30) and URLs
        colon_matches = re.findall(r'(?<!\d):(?!\d|//)', text)
        features["colon_rate"] = len(colon_matches) / sentence_count

        # 8. Comma density
        # Commas per 100 words
        comma_count = text.count(',')
        features["comma_density"] = (comma_count / word_count) * 100

        return features


# Standalone function
def extract_punctuation_features(text: str) -> Dict[str, float]:
    """Extract punctuation features (standalone function)."""
    extractor = PunctuationExtractor()
    result = extractor.extract(text)
    return result.features
