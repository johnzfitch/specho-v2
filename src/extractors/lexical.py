"""
Lexical Feature Extractor (12D)

Word-level vocabulary patterns.

Features (sorted by Cohen's d):
- type_token_ratio (d=-1.07): AI reuses vocabulary
- hapax_legomena_ratio (d=-0.75): AI uses fewer unique words
- avg_word_length (d=+0.68): AI uses longer words
- rare_word_ratio (d=-0.60): Humans use rarer words
- long_word_ratio (d=+0.55): AI uses more long words
- dis_legomena_ratio (d=-0.55): AI has fewer dis legomena
- word_length_cv (d=-0.50): AI word lengths are regular
- formality_score (d=+0.50): AI is more formal
- word_length_std (d=-0.45): AI word lengths are uniform
- syllable_ratio (d=+0.40): AI uses more syllables
- abstract_ratio (d=+0.35): AI uses more abstract words
- concrete_ratio (d=-0.35): Humans use more concrete words
"""

import re
import math
from typing import Dict, List
from collections import Counter

from .base import BaseExtractor


# Top 1000 most common words
TOP_1000_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having', 'does',
    'did', 'doing', 'done', 'should', 'could', 'would', 'might', 'must', 'shall',
    'very', 'much', 'more', 'many', 'such', 'same', 'still', 'however', 'therefore',
}


def syllable_count(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjustments
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1

    return count


class LexicalExtractor(BaseExtractor):
    """
    Lexical vocabulary features with NO external dependencies.

    Measures word-level patterns that differ between
    human and AI writing.
    """

    group = "lexical"
    dependencies = []  # Pure Python

    @property
    def feature_names(self) -> List[str]:
        return [
            "type_token_ratio",      # d=-1.07
            "hapax_legomena_ratio",  # d=-0.75
            "dis_legomena_ratio",    # d=-0.55
            "avg_word_length",       # d=+0.68
            "word_length_std",       # d=-0.45
            "word_length_cv",        # d=-0.50
            "syllable_ratio",        # d=+0.40
            "long_word_ratio",       # d=+0.55
            "rare_word_ratio",       # d=-0.60
            "concrete_ratio",        # d=-0.35
            "abstract_ratio",        # d=+0.35
            "formality_score",       # d=+0.50
        ]

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract lexical features."""
        features = {}
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(words) < 10:
            return {name: 0.0 for name in self.feature_names}

        word_counts = Counter(words)
        N = len(words)
        V = len(word_counts)

        # Type-token ratio
        features["type_token_ratio"] = V / N

        # Hapax legomena ratio (words appearing once)
        V1 = sum(1 for n in word_counts.values() if n == 1)
        features["hapax_legomena_ratio"] = V1 / N

        # Dis legomena ratio (words appearing twice)
        V2 = sum(1 for n in word_counts.values() if n == 2)
        features["dis_legomena_ratio"] = V2 / N

        # Word length statistics
        lengths = [len(w) for w in words]
        features["avg_word_length"] = sum(lengths) / len(lengths)

        if len(lengths) > 1:
            mean_len = features["avg_word_length"]
            features["word_length_std"] = math.sqrt(sum((l - mean_len)**2 for l in lengths) / len(lengths))
            features["word_length_cv"] = features["word_length_std"] / mean_len if mean_len > 0 else 0
        else:
            features["word_length_std"] = 0.0
            features["word_length_cv"] = 0.0

        # Syllable ratio
        syllables = [syllable_count(w) for w in words]
        features["syllable_ratio"] = sum(syllables) / len(syllables)

        # Long word ratio (> 6 chars)
        features["long_word_ratio"] = sum(1 for w in words if len(w) > 6) / N

        # Rare word ratio (not in top 1000)
        features["rare_word_ratio"] = sum(1 for w in words if w not in TOP_1000_WORDS) / N

        # Abstract/Concrete (simplified heuristics based on word endings)
        abstract_endings = ('tion', 'ness', 'ity', 'ism', 'ment', 'ence', 'ance')
        features["abstract_ratio"] = sum(1 for w in words if w.endswith(abstract_endings)) / N
        features["concrete_ratio"] = 1 - features["abstract_ratio"]

        # Formality score (simplified: longer words and abstract terms = more formal)
        features["formality_score"] = (
            0.3 * (features["avg_word_length"] / 10) +
            0.3 * features["long_word_ratio"] +
            0.4 * features["abstract_ratio"]
        )

        return features


# Standalone function
def extract_lexical_features(text: str) -> Dict[str, float]:
    """Extract lexical features (standalone function)."""
    extractor = LexicalExtractor()
    result = extractor.extract(text)
    return result.features
