"""
Structural Feature Extractor (10D)

Document structure patterns.

Features (sorted by Cohen's d):
- sentence_length_cv (d=-1.47): AI sentences are uniform
- paragraph_length_cv (d=-1.10): AI paragraphs are uniform
- list_marker_rate (d=+0.93): AI loves bullet points
- sentence_length_std (d=-0.80): AI has low variation
- sentence_initial_entropy (d=-0.72): AI starts sentences same way
- header_rate (d=+0.60): AI uses more headers
- sentence_final_entropy (d=-0.55): AI ends sentences same way
- quote_rate (d=-0.35): Humans quote more
- sentence_length_mean (d=+0.30): AI slightly longer
- paragraph_count_norm (d=+0.25): AI uses more paragraphs
"""

import re
import math
from typing import Dict, List
from collections import Counter

from .base import BaseExtractor


def shannon_entropy(values: List) -> float:
    """Calculate Shannon entropy of a distribution."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def cv(values: List[float]) -> float:
    """Coefficient of variation with safety."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    std = math.sqrt(sum((x - mean)**2 for x in values) / len(values))
    return std / mean


class StructuralExtractor(BaseExtractor):
    """
    Document structure features with NO external dependencies.

    Each feature is extracted independently with fault isolation.
    """

    group = "structural"
    dependencies = []  # Pure Python

    @property
    def feature_names(self) -> List[str]:
        return [
            "sentence_length_cv",        # d=-1.47 CRITICAL
            "sentence_length_mean",      # d=+0.30
            "sentence_length_std",       # d=-0.80
            "paragraph_length_cv",       # d=-1.10 CRITICAL
            "paragraph_count_norm",      # d=+0.25
            "list_marker_rate",          # d=+0.93
            "header_rate",               # d=+0.60
            "sentence_initial_entropy",  # d=-0.72
            "sentence_final_entropy",    # d=-0.55
            "quote_rate",                # d=-0.35
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Safely extract a single feature with error handling."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract structural features with fault isolation."""
        features = {name: 0.0 for name in self.feature_names}

        # Basic parsing
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        lines = text.split('\n')

        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        if word_count < 10:
            return features

        # 1. Sentence length statistics (ISOLATED)
        def calc_sent_stats():
            sent_lengths = [len(s.split()) for s in sentences]
            if len(sent_lengths) > 1:
                features["sentence_length_cv"] = cv(sent_lengths)
                features["sentence_length_mean"] = sum(sent_lengths) / len(sent_lengths)
                mean = features["sentence_length_mean"]
                features["sentence_length_std"] = math.sqrt(
                    sum((l - mean)**2 for l in sent_lengths) / len(sent_lengths)
                )
            elif sent_lengths:
                features["sentence_length_mean"] = sent_lengths[0]

        self._safe_extract("sent_stats", calc_sent_stats)

        # 2. Paragraph length CV (ISOLATED)
        def calc_para_cv():
            if len(paragraphs) > 1:
                para_lengths = [len(p.split()) for p in paragraphs]
                features["paragraph_length_cv"] = cv(para_lengths)

        self._safe_extract("para_cv", calc_para_cv)

        # 3. Paragraph count normalized (ISOLATED)
        def calc_para_norm():
            features["paragraph_count_norm"] = len(paragraphs) / (word_count / 1000) if word_count > 0 else 0

        self._safe_extract("para_norm", calc_para_norm)

        # 4. List marker rate (ISOLATED)
        def calc_list_rate():
            list_markers = len(re.findall(r'(?:^|\n)\s*[-*•]\s|(?:^|\n)\s*\d+\.\s', text))
            features["list_marker_rate"] = list_markers / sentence_count

        self._safe_extract("list_rate", calc_list_rate)

        # 5. Header rate (ISOLATED)
        def calc_header_rate():
            headers = sum(1 for line in lines if line.strip() and (
                line.strip().startswith('#') or
                (len(line.strip().split()) <= 8 and line.strip().endswith(':')) or
                line.isupper()
            ))
            features["header_rate"] = headers / max(len(paragraphs), 1)

        self._safe_extract("header_rate", calc_header_rate)

        # 6. Sentence initial entropy (ISOLATED)
        def calc_initial_entropy():
            first_words = []
            for s in sentences:
                parts = s.split()
                if parts:
                    first_words.append(parts[0].lower().strip('"""\''))
            if first_words:
                features["sentence_initial_entropy"] = shannon_entropy(first_words)

        self._safe_extract("initial_entropy", calc_initial_entropy)

        # 7. Sentence final entropy (ISOLATED)
        def calc_final_entropy():
            last_words = []
            for s in sentences:
                parts = s.split()
                if parts:
                    last_words.append(parts[-1].lower().strip('"""\''))
            if last_words:
                features["sentence_final_entropy"] = shannon_entropy(last_words)

        self._safe_extract("final_entropy", calc_final_entropy)

        # 8. Quote rate (ISOLATED)
        def calc_quote_rate():
            quotes = len(re.findall(r'["\'].*?["\']', text))
            features["quote_rate"] = quotes / sentence_count

        self._safe_extract("quote_rate", calc_quote_rate)

        return features


# Standalone function
def extract_structural_features(text: str) -> Dict[str, float]:
    """Extract structural features (standalone function)."""
    extractor = StructuralExtractor()
    result = extractor.extract(text)
    return result.features
