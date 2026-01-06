"""
Temporal Dynamics Extractor (4D)

Measures how text properties change over the course of the document.
AI text tends to be more stationary/predictable in its evolution.

Features (estimated Cohen's d):
- perplexity_slope (d~0.8-1.2): Rate of vocabulary complexity change
- lexical_density_drift (d~0.7-1.0): Content word ratio variance across text
- sentence_length_autocorr (d~0.6-0.9): Self-similarity in sentence lengths
- entropy_impulse_response (d~0.7-1.1): How fast entropy stabilizes

Tier 0: No external dependencies (numpy-like calculations in pure Python)
"""

import re
import math
from typing import Dict, List, Tuple
from collections import Counter

from .base import BaseExtractor


class DynamicsExtractor(BaseExtractor):
    """
    Extract temporal dynamics features.

    These features analyze how text properties evolve over the document.
    Human writing tends to be more "bursty" and non-stationary, while
    AI maintains more consistent statistical properties throughout.
    """

    group = "dynamics"
    dependencies = []  # Pure Python

    # Function words (closed class - determiners, prepositions, conjunctions)
    FUNCTION_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
        'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
        'who', 'whom', 'which', 'what', 'where', 'when', 'why', 'how', 'if',
        'then', 'so', 'than', 'just', 'only', 'also', 'very', 'too', 'not',
        'no', 'yes', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'once',
    }

    @property
    def feature_names(self) -> List[str]:
        return [
            "perplexity_slope",         # d~0.8-1.2
            "lexical_density_drift",    # d~0.7-1.0
            "sentence_length_autocorr", # d~0.6-0.9
            "entropy_impulse_response", # d~0.7-1.1
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _shannon_entropy(self, values: List) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not values:
            return 0.0
        counts = Counter(values)
        total = len(values)
        probs = [c / total for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _linear_regression_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope of linear regression."""
        n = len(x)
        if n < 2 or len(y) != n:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _autocorrelation_lag1(self, series: List[float]) -> float:
        """Calculate lag-1 autocorrelation."""
        n = len(series)
        if n < 3:
            return 0.0

        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / n

        if variance == 0:
            return 0.0

        covariance = sum(
            (series[i] - mean) * (series[i + 1] - mean)
            for i in range(n - 1)
        ) / (n - 1)

        return covariance / variance

    def _chunk_text(self, words: List[str], n_chunks: int = 5) -> List[List[str]]:
        """Split words into roughly equal chunks."""
        if not words or n_chunks < 1:
            return []

        chunk_size = max(1, len(words) // n_chunks)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)

        # Merge last small chunk if needed
        if len(chunks) > n_chunks and len(chunks[-1]) < chunk_size // 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract temporal dynamics features."""
        features = {name: 0.0 for name in self.feature_names}

        # Early return for short text
        if len(text) < 200:
            return features

        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        word_count = len(words)

        if word_count < 50:
            return features

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 5:
            return features

        # =====================================================================
        # Feature 1: Perplexity slope - ISOLATED
        # =====================================================================
        def calc_perplexity_slope():
            # Divide text into chunks and measure entropy trend
            chunks = self._chunk_text(words, n_chunks=5)
            if len(chunks) < 3:
                return

            chunk_entropies = []
            for chunk in chunks:
                entropy = self._shannon_entropy(chunk)
                chunk_entropies.append(entropy)

            # Calculate slope of entropy over chunks
            x_values = list(range(len(chunk_entropies)))
            slope = self._linear_regression_slope(x_values, chunk_entropies)

            # Normalize slope to reasonable range (-1 to 1)
            # Positive slope = increasing complexity (human-like)
            # Near-zero slope = consistent (AI-like)
            features["perplexity_slope"] = max(-1.0, min(1.0, slope))
        self._safe_extract("perplexity_slope", calc_perplexity_slope)

        # =====================================================================
        # Feature 2: Lexical density drift - ISOLATED
        # =====================================================================
        def calc_lexical_density_drift():
            # Lexical density = content words / total words
            chunks = self._chunk_text(words, n_chunks=5)
            if len(chunks) < 3:
                return

            densities = []
            for chunk in chunks:
                content_words = [w for w in chunk if w not in self.FUNCTION_WORDS]
                density = len(content_words) / len(chunk) if chunk else 0
                densities.append(density)

            # Calculate variance in density across chunks
            if len(densities) < 2:
                return

            mean_density = sum(densities) / len(densities)
            variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
            std_dev = math.sqrt(variance)

            # Higher variance = more drift = more human-like
            # Normalize to 0-1 range (typical std is 0.05-0.15)
            features["lexical_density_drift"] = min(std_dev * 5, 1.0)
        self._safe_extract("lexical_density_drift", calc_lexical_density_drift)

        # =====================================================================
        # Feature 3: Sentence length autocorrelation - ISOLATED
        # =====================================================================
        def calc_sent_autocorr():
            sent_lengths = [len(s.split()) for s in sentences]
            if len(sent_lengths) < 5:
                return

            autocorr = self._autocorrelation_lag1(sent_lengths)

            # AI tends to have higher autocorrelation (more predictable)
            # Human writing is more variable (lower autocorrelation)
            # Map to 0-1 where higher = more AI-like (higher autocorr)
            features["sentence_length_autocorr"] = max(0, min(1.0, (autocorr + 1) / 2))
        self._safe_extract("sent_autocorr", calc_sent_autocorr)

        # =====================================================================
        # Feature 4: Entropy impulse response - ISOLATED
        # =====================================================================
        def calc_entropy_impulse():
            # Measure how quickly entropy stabilizes with more text
            # Use cumulative windows of increasing size

            if word_count < 100:
                return

            # Sample at 10%, 25%, 50%, 75%, 100% of text
            sample_points = [0.1, 0.25, 0.5, 0.75, 1.0]
            entropies = []

            for frac in sample_points:
                n_words = int(word_count * frac)
                sample = words[:n_words]
                entropy = self._shannon_entropy(sample)
                # Normalize by max possible entropy for that sample size
                max_entropy = math.log2(len(set(sample))) if sample else 0
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                entropies.append(norm_entropy)

            # Measure how fast entropy stabilizes
            # Calculate the "settling time" - how quickly we reach near-final value
            if len(entropies) < 3:
                return

            final_entropy = entropies[-1]
            if final_entropy == 0:
                return

            # Find first point where we're within 90% of final
            settling_index = len(entropies) - 1
            for i, e in enumerate(entropies):
                if e >= 0.9 * final_entropy:
                    settling_index = i
                    break

            # Faster settling = more predictable = more AI-like
            # Normalize: 0 = settled at first sample, 1 = settled at last
            settling_score = settling_index / (len(entropies) - 1)

            # Invert so higher = faster settling = more AI-like
            features["entropy_impulse_response"] = 1.0 - settling_score
        self._safe_extract("entropy_impulse", calc_entropy_impulse)

        return features


# Standalone function
def extract_dynamics_features(text: str) -> Dict[str, float]:
    """Extract temporal dynamics features (standalone function)."""
    extractor = DynamicsExtractor()
    result = extractor.extract(text)
    return result.features
