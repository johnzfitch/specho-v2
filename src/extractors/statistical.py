"""
Statistical Feature Extractor (14D)

Deep statistical analysis of text patterns.

Features (sorted by Cohen's d):
- compression_ratio (d=-0.99): AI text compresses better (more predictable)
- entropy_word (d=-0.70): AI has lower word entropy
- entropy_bigram (d=-0.68): AI has lower bigram entropy
- entropy_char (d=-0.65): AI has lower character entropy
- redundancy_rate (d=+0.60): AI is more redundant
- yule_k (d=-0.55): AI has lower lexical richness
- perplexity_approx (d=-0.55): AI has lower perplexity
- simpson_d (d=-0.50): AI has lower diversity
- burstiness_words (d=-0.50): AI word usage is regular
- honore_r (d=-0.48): AI has lower Honoré's R
- repetition_rate (d=+0.45): AI repeats more n-grams
- sichel_s (d=-0.42): AI has lower Sichel's S
- zipf_coefficient (d=-0.40): AI follows Zipf less naturally
- brunet_w (d=+0.38): AI has higher Brunet's W
"""

import re
import math
import gzip
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


class StatisticalExtractor(BaseExtractor):
    """
    Deep statistical features with NO external dependencies.

    Measures information-theoretic properties of text that
    differ systematically between human and AI writing.
    """

    group = "statistical"
    dependencies = []  # Pure Python

    @property
    def feature_names(self) -> List[str]:
        return [
            "compression_ratio",     # d=-0.99
            "entropy_char",          # d=-0.65
            "entropy_word",          # d=-0.70
            "entropy_bigram",        # d=-0.68
            "perplexity_approx",     # d=-0.55
            "redundancy_rate",       # d=+0.60
            "repetition_rate",       # d=+0.45
            "burstiness_words",      # d=-0.50
            "zipf_coefficient",      # d=-0.40
            "yule_k",                # d=-0.55
            "simpson_d",             # d=-0.50
            "honore_r",              # d=-0.48
            "sichel_s",              # d=-0.42
            "brunet_w",              # d=+0.38
        ]

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(words) < 10:
            return {name: 0.0 for name in self.feature_names}

        # Compression ratio
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > 0:
            compressed = gzip.compress(text_bytes, compresslevel=9)
            features["compression_ratio"] = len(compressed) / len(text_bytes)
        else:
            features["compression_ratio"] = 0.0

        # Character entropy
        features["entropy_char"] = shannon_entropy(list(text.lower()))

        # Word entropy
        features["entropy_word"] = shannon_entropy(words)

        # Bigram entropy
        if len(words) >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            features["entropy_bigram"] = shannon_entropy(bigrams)
        else:
            features["entropy_bigram"] = 0.0

        # Perplexity approximation
        if features["entropy_word"] > 0:
            features["perplexity_approx"] = 2 ** features["entropy_word"]
        else:
            features["perplexity_approx"] = 0.0

        # Redundancy rate
        max_entropy = math.log2(len(set(words))) if words else 0
        if max_entropy > 0:
            features["redundancy_rate"] = 1 - (features["entropy_word"] / max_entropy)
        else:
            features["redundancy_rate"] = 0.0

        # Repetition rate (repeated trigrams)
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated = sum(1 for c in trigram_counts.values() if c > 1)
            features["repetition_rate"] = repeated / len(trigrams) if trigrams else 0
        else:
            features["repetition_rate"] = 0.0

        # Word burstiness
        word_positions = {}
        for i, w in enumerate(words):
            if w not in word_positions:
                word_positions[w] = []
            word_positions[w].append(i)

        gaps = []
        for positions in word_positions.values():
            if len(positions) > 1:
                gaps.extend([positions[i+1] - positions[i] for i in range(len(positions)-1)])

        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            std_gap = math.sqrt(sum((g - mean_gap)**2 for g in gaps) / len(gaps))
            features["burstiness_words"] = std_gap / mean_gap if mean_gap > 0 else 0
        else:
            features["burstiness_words"] = 0.0

        # Vocabulary metrics
        word_counts = Counter(words)
        N = len(words)
        V = len(word_counts)

        if N > 0 and V > 0:
            # Zipf coefficient (simplified)
            freqs = sorted(word_counts.values(), reverse=True)
            if len(freqs) >= 10:
                ranks = list(range(1, len(freqs)+1))
                log_ranks = [math.log(r) for r in ranks[:100]]
                log_freqs = [math.log(f) for f in freqs[:100]]
                if len(log_ranks) > 1:
                    mean_r = sum(log_ranks) / len(log_ranks)
                    mean_f = sum(log_freqs) / len(log_freqs)
                    num = sum((r - mean_r) * (f - mean_f) for r, f in zip(log_ranks, log_freqs))
                    den = sum((r - mean_r)**2 for r in log_ranks)
                    features["zipf_coefficient"] = abs(num / den) if den > 0 else 0
                else:
                    features["zipf_coefficient"] = 0.0
            else:
                features["zipf_coefficient"] = 0.0

            # Yule's K
            freq_of_freq = Counter(word_counts.values())
            m1 = N
            m2 = sum(f * (r ** 2) for r, f in freq_of_freq.items())
            features["yule_k"] = 10000 * (m2 - m1) / (m1 ** 2) if m1 > 0 else 0

            # Simpson's D
            features["simpson_d"] = sum(n * (n - 1) for n in word_counts.values()) / (N * (N - 1)) if N > 1 else 0

            # Honoré's R
            V1 = sum(1 for n in word_counts.values() if n == 1)
            if V1 > 0 and V1 < V:
                features["honore_r"] = 100 * math.log(N) / (1 - V1/V)
            else:
                features["honore_r"] = 0.0

            # Sichel's S (dis legomena / V)
            V2 = sum(1 for n in word_counts.values() if n == 2)
            features["sichel_s"] = V2 / V if V > 0 else 0

            # Brunet's W
            features["brunet_w"] = N ** (V ** -0.165)
        else:
            features["zipf_coefficient"] = 0.0
            features["yule_k"] = 0.0
            features["simpson_d"] = 0.0
            features["honore_r"] = 0.0
            features["sichel_s"] = 0.0
            features["brunet_w"] = 0.0

        return features


# Standalone function
def extract_statistical_features(text: str) -> Dict[str, float]:
    """Extract statistical features (standalone function)."""
    extractor = StatisticalExtractor()
    result = extractor.extract(text)
    return result.features
