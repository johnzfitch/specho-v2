"""
Lightweight Feature Extractor (10D)

NO DEPENDENCIES - works anywhere, <1ms extraction.

Features (sorted by Cohen's d):
- sentence_length_cv (d=-1.47): AI text has uniform sentence lengths
- paragraph_length_cv (d=-1.10): AI paragraphs are uniform
- type_token_ratio (d=-1.07): AI has more repetitive vocabulary
- compression_ratio (d=-0.99): AI text compresses better (patterns)
- list_marker_rate (d=+0.93): AI uses more bullet points
- hapax_legomena_ratio (d=-0.75): AI has fewer unique words
- sentence_initial_entropy (d=-0.72): AI starts sentences predictably
- avg_word_length (d=+0.68): AI uses slightly longer words
- punctuation_diversity (d=+0.55): AI uses more varied punctuation
- question_ratio (d=-0.42): Humans ask more questions
"""

import re
import math
import statistics
from collections import Counter
from typing import Dict, List

from .base import BaseExtractor


class LightweightExtractor(BaseExtractor):
    """
    10D lightweight features with NO external dependencies.

    These features achieve 98.6% accuracy alone because they measure
    continuous structural properties rather than sparse presence/absence.
    """

    group = "lightweight"
    dependencies = []  # None!

    @property
    def feature_names(self) -> List[str]:
        return [
            "sentence_length_cv",
            "paragraph_length_cv",
            "type_token_ratio",
            "compression_ratio",
            "list_marker_rate",
            "hapax_legomena_ratio",
            "sentence_initial_entropy",
            "avg_word_length",
            "punctuation_diversity",
            "question_ratio",
        ]

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract all 10 lightweight features."""
        features = {}

        # Basic tokenization
        words = text.split()
        word_count = len(words)

        if word_count < 5:
            return {name: 0.0 for name in self.feature_names}

        # Sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # 1. Sentence length CV (d=-1.47)
        if sentence_count > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = statistics.mean(sent_lengths)
            std_len = statistics.stdev(sent_lengths)
            features["sentence_length_cv"] = std_len / mean_len if mean_len > 0 else 0
        else:
            features["sentence_length_cv"] = 0.0

        # 2. Paragraph length CV (d=-1.10)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            mean_para = statistics.mean(para_lengths)
            std_para = statistics.stdev(para_lengths)
            features["paragraph_length_cv"] = std_para / mean_para if mean_para > 0 else 0
        else:
            features["paragraph_length_cv"] = 0.0

        # 3. Type-token ratio (d=-1.07)
        word_lower = [w.lower() for w in words]
        unique_words = set(word_lower)
        features["type_token_ratio"] = len(unique_words) / word_count

        # 4. Compression ratio (d=-0.99)
        try:
            import zlib
            text_bytes = text.encode('utf-8')
            compressed = zlib.compress(text_bytes)
            features["compression_ratio"] = len(compressed) / len(text_bytes)
        except:
            features["compression_ratio"] = 0.7

        # 5. List marker rate (d=+0.93)
        list_patterns = re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', text)
        features["list_marker_rate"] = len(list_patterns) / word_count if word_count > 0 else 0

        # 6. Hapax legomena ratio (d=-0.75)
        word_freq = Counter(word_lower)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        features["hapax_legomena_ratio"] = hapax / len(unique_words) if unique_words else 0

        # 7. Sentence initial entropy (d=-0.72)
        if sentences:
            first_words = [s.split()[0].lower() for s in sentences if s.split()]
            if first_words:
                freq = Counter(first_words)
                total = len(first_words)
                entropy = -sum(
                    (c / total) * math.log2(c / total)
                    for c in freq.values()
                )
                max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
                features["sentence_initial_entropy"] = entropy / max_entropy if max_entropy > 0 else 0
            else:
                features["sentence_initial_entropy"] = 0
        else:
            features["sentence_initial_entropy"] = 0

        # 8. Average word length (d=+0.68)
        alpha_words = [w for w in words if w.isalpha()]
        if alpha_words:
            features["avg_word_length"] = sum(len(w) for w in alpha_words) / len(alpha_words)
        else:
            features["avg_word_length"] = 0

        # 9. Punctuation diversity (d=+0.55)
        punct_chars = set(c for c in text if c in '.,;:!?-()[]{}"\'/—–')
        features["punctuation_diversity"] = len(punct_chars) / 15  # Normalize to ~0-1

        # 10. Question ratio (d=-0.42)
        question_count = text.count('?')
        features["question_ratio"] = question_count / sentence_count if sentence_count > 0 else 0

        return features


# Standalone function for backward compatibility
def extract_lightweight_features(text: str) -> Dict[str, float]:
    """Extract lightweight features (standalone function)."""
    extractor = LightweightExtractor()
    result = extractor.extract(text)
    return result.features
