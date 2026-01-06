"""
Fractal/Multiscale Extractor (3D)

Analyzes text at multiple scales to detect self-similarity patterns.
Human writing tends to have more fractal/irregular structure than AI.

Features (estimated Cohen's d):
- hurst_exponent (d~0.8-1.2): Long-range dependence measure (R/S analysis)
- multifractal_spectrum (d~0.7-1.0): Complexity variation across scales
- topic_decay_rate (d~0.6-0.9): How quickly topics fade from text

Tier 0: No external dependencies (pure Python R/S analysis)
"""

import re
import math
from typing import Dict, List, Tuple
from collections import Counter

from .base import BaseExtractor


class FractalExtractor(BaseExtractor):
    """
    Extract fractal/multiscale features.

    These features analyze how text properties scale across different
    "zoom levels". Human writing tends to have more irregular, fractal-like
    patterns while AI maintains more uniform structure across scales.
    """

    group = "fractal"
    dependencies = []  # Pure Python

    @property
    def feature_names(self) -> List[str]:
        return [
            "hurst_exponent",       # d~0.8-1.2
            "multifractal_spectrum", # d~0.7-1.0
            "topic_decay_rate",     # d~0.6-0.9
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _rescaled_range(self, series: List[float], n: int) -> float:
        """
        Calculate rescaled range R/S for a given window size.

        R/S = (max - min of cumulative deviation) / standard deviation
        """
        if n < 2 or len(series) < n:
            return 0.0

        # Use first n elements
        subseries = series[:n]
        mean = sum(subseries) / n

        # Calculate deviations from mean
        deviations = [x - mean for x in subseries]

        # Cumulative sum of deviations
        cumsum = []
        total = 0
        for d in deviations:
            total += d
            cumsum.append(total)

        # Range
        R = max(cumsum) - min(cumsum)

        # Standard deviation
        variance = sum((x - mean) ** 2 for x in subseries) / n
        S = math.sqrt(variance) if variance > 0 else 1e-10

        return R / S if S > 0 else 0.0

    def _calculate_hurst(self, series: List[float]) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H > 0.5: persistent/trending (typical of structured text)
        H = 0.5: random walk
        H < 0.5: anti-persistent/mean-reverting

        AI text tends to have H closer to 0.5 (more "random" structure)
        Human text often shows higher H (more persistent patterns)
        """
        n = len(series)
        if n < 20:
            return 0.5  # Default for short series

        # Calculate R/S for different window sizes
        # Use logarithmic spacing
        min_window = 10
        max_window = n // 2

        if max_window < min_window:
            return 0.5

        # Generate window sizes (at least 4 points for regression)
        window_sizes = []
        current = min_window
        while current <= max_window:
            window_sizes.append(current)
            current = int(current * 1.5)  # Logarithmic spacing

        if len(window_sizes) < 3:
            return 0.5

        # Calculate R/S for each window size
        log_n = []
        log_rs = []

        for window in window_sizes:
            rs = self._rescaled_range(series, window)
            if rs > 0:
                log_n.append(math.log(window))
                log_rs.append(math.log(rs))

        if len(log_n) < 3:
            return 0.5

        # Linear regression: log(R/S) = H * log(n) + c
        # Hurst exponent is the slope
        n_points = len(log_n)
        mean_x = sum(log_n) / n_points
        mean_y = sum(log_rs) / n_points

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_rs))
        denominator = sum((x - mean_x) ** 2 for x in log_n)

        if denominator == 0:
            return 0.5

        H = numerator / denominator

        # Clamp to valid range [0, 1]
        return max(0.0, min(1.0, H))

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract fractal/multiscale features."""
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

        # =====================================================================
        # Feature 1: Hurst exponent - ISOLATED
        # =====================================================================
        def calc_hurst():
            # Use word lengths as the time series
            word_lengths = [len(w) for w in words]
            H = self._calculate_hurst(word_lengths)
            features["hurst_exponent"] = H
        self._safe_extract("hurst", calc_hurst)

        # =====================================================================
        # Feature 2: Multifractal spectrum width - ISOLATED
        # =====================================================================
        def calc_multifractal():
            # Measure how complexity varies at different scales
            # Use sentence lengths at different aggregation levels

            if len(sentences) < 10:
                features["multifractal_spectrum"] = 0.5
                return

            sent_lengths = [len(s.split()) for s in sentences]

            # Calculate variance at different scales
            variances = []

            # Scale 1: individual sentences
            if len(sent_lengths) >= 2:
                mean1 = sum(sent_lengths) / len(sent_lengths)
                var1 = sum((x - mean1) ** 2 for x in sent_lengths) / len(sent_lengths)
                variances.append(var1)

            # Scale 2: pairs of sentences
            pairs = []
            for i in range(0, len(sent_lengths) - 1, 2):
                pairs.append(sent_lengths[i] + sent_lengths[i + 1])
            if len(pairs) >= 2:
                mean2 = sum(pairs) / len(pairs)
                var2 = sum((x - mean2) ** 2 for x in pairs) / len(pairs)
                variances.append(var2)

            # Scale 3: groups of 4
            quads = []
            for i in range(0, len(sent_lengths) - 3, 4):
                quads.append(sum(sent_lengths[i:i + 4]))
            if len(quads) >= 2:
                mean3 = sum(quads) / len(quads)
                var3 = sum((x - mean3) ** 2 for x in quads) / len(quads)
                variances.append(var3)

            if len(variances) < 2:
                features["multifractal_spectrum"] = 0.5
                return

            # Multifractal width: variation in how variance changes across scales
            # Normalize variances
            max_var = max(variances) if variances else 1
            norm_vars = [v / max_var if max_var > 0 else 0 for v in variances]

            # Calculate coefficient of variation of the normalized variances
            if len(norm_vars) >= 2:
                mean_nv = sum(norm_vars) / len(norm_vars)
                var_nv = sum((v - mean_nv) ** 2 for v in norm_vars) / len(norm_vars)
                cv = math.sqrt(var_nv) / mean_nv if mean_nv > 0 else 0

                # Higher CV = more multifractal = more human-like
                features["multifractal_spectrum"] = min(cv, 1.0)
            else:
                features["multifractal_spectrum"] = 0.5
        self._safe_extract("multifractal", calc_multifractal)

        # =====================================================================
        # Feature 3: Topic decay rate - ISOLATED
        # =====================================================================
        def calc_topic_decay():
            # Measure how quickly content words "decay" (stop appearing)
            # Human text often has faster topic decay (introduces then abandons topics)
            # AI text maintains more consistent vocabulary throughout

            if word_count < 50:
                features["topic_decay_rate"] = 0.5
                return

            # Track content words (longer words, not common function words)
            content_words = [w for w in words if len(w) >= 5]

            if len(content_words) < 20:
                features["topic_decay_rate"] = 0.5
                return

            # Divide text into thirds
            third = len(content_words) // 3
            first_third = set(content_words[:third])
            middle_third = set(content_words[third:2 * third])
            last_third = set(content_words[2 * third:])

            # Calculate survival rates
            # How many words from first third appear in middle third?
            if len(first_third) > 0:
                survival_1_to_2 = len(first_third & middle_third) / len(first_third)
            else:
                survival_1_to_2 = 1.0

            # How many words from first third appear in last third?
            if len(first_third) > 0:
                survival_1_to_3 = len(first_third & last_third) / len(first_third)
            else:
                survival_1_to_3 = 1.0

            # Calculate decay rate
            # Lower survival = faster decay = more human-like
            # Average the two survival rates and invert
            avg_survival = (survival_1_to_2 + survival_1_to_3) / 2
            decay_rate = 1.0 - avg_survival

            features["topic_decay_rate"] = decay_rate
        self._safe_extract("topic_decay", calc_topic_decay)

        return features


# Standalone function
def extract_fractal_features(text: str) -> Dict[str, float]:
    """Extract fractal/multiscale features (standalone function)."""
    extractor = FractalExtractor()
    result = extractor.extract(text)
    return result.features
