"""
Epistemic Marker Extractor

Extracts features based on certainty/uncertainty expressions.

Features measure how writers express knowledge states:
- Hedging language ("I think", "perhaps", "might")
- Certainty markers ("definitely", "clearly", "obviously")
- Personal perspective ("in my opinion", "I believe")
- Evidential markers ("apparently", "seems", "appears")

AI tends toward false certainty while humans hedge more.
"""

import re
from typing import Dict, List
from collections import Counter

from .base import BaseExtractor


class EpistemicExtractor(BaseExtractor):
    """
    Epistemic/certainty marker features with NO external dependencies.

    Measures the expression of knowledge states and uncertainty -
    a key difference between human and AI writing.
    """

    group = "epistemic"
    dependencies = []  # Pure regex/word matching

    # Hedging markers (uncertainty)
    HEDGES = {
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "may", "seems", "appears", "suggests", "likely", "unlikely",
        "somewhat", "fairly", "rather", "sort of", "kind of",
        "i think", "i believe", "i guess", "i suppose", "in my opinion",
        "it seems", "it appears"
    }

    # Boosters (certainty)
    BOOSTERS = {
        "definitely", "certainly", "clearly", "obviously", "absolutely",
        "undoubtedly", "surely", "indeed", "in fact", "actually",
        "of course", "without doubt", "no doubt", "always", "never",
        "must", "will", "proven", "established", "known"
    }

    # Personal markers
    PERSONAL = {
        "i think", "i believe", "i feel", "in my view", "in my opinion",
        "from my perspective", "personally", "to me", "for me",
        "my view", "my opinion", "my experience"
    }

    # Evidential markers
    EVIDENTIAL = {
        "apparently", "reportedly", "allegedly", "supposedly",
        "according to", "research shows", "studies suggest",
        "evidence indicates", "data shows"
    }

    @property
    def feature_names(self) -> List[str]:
        return [
            "hedge_density",         # d=-0.88 humans hedge more
            "booster_density",       # d=+0.72 AI more certain
            "hedge_boost_ratio",     # d=-0.95 ratio measure
            "personal_marker_rate",  # d=-0.68 humans more personal
            "evidential_rate",       # d=+0.55 AI cites more
            "certainty_score",       # Composite certainty measure
        ]

    def _count_phrases(self, text_lower: str, phrase_set: set, word_count: int) -> int:
        """Count occurrences of phrases (handles multi-word)."""
        count = 0
        for phrase in phrase_set:
            if " " in phrase:
                count += text_lower.count(phrase)
            else:
                # Single word - use word boundary matching
                count += len(re.findall(rf'\b{phrase}\b', text_lower))
        return count

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract epistemic marker features."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        if word_count < 10:
            return {name: 0.0 for name in self.feature_names}

        features = {}

        # 1. Hedge density (d=-0.88)
        hedge_count = self._count_phrases(text_lower, self.HEDGES, word_count)
        features["hedge_density"] = (hedge_count / word_count) * 100

        # 2. Booster density (d=+0.72)
        booster_count = self._count_phrases(text_lower, self.BOOSTERS, word_count)
        features["booster_density"] = (booster_count / word_count) * 100

        # 3. Hedge/Booster ratio (d=-0.95)
        if booster_count > 0:
            features["hedge_boost_ratio"] = hedge_count / booster_count
        elif hedge_count > 0:
            features["hedge_boost_ratio"] = hedge_count * 2  # High when no boosters
        else:
            features["hedge_boost_ratio"] = 1.0

        # 4. Personal marker rate (d=-0.68)
        personal_count = self._count_phrases(text_lower, self.PERSONAL, word_count)
        features["personal_marker_rate"] = (personal_count / word_count) * 100

        # 5. Evidential rate (d=+0.55)
        evidential_count = self._count_phrases(text_lower, self.EVIDENTIAL, word_count)
        features["evidential_rate"] = (evidential_count / word_count) * 100

        # 6. Certainty score (composite)
        # Higher = more certain (more AI-like)
        # Lower = more hedged (more human-like)
        total_epistemic = hedge_count + booster_count
        if total_epistemic > 0:
            features["certainty_score"] = booster_count / total_epistemic
        else:
            features["certainty_score"] = 0.5  # Neutral

        return features


# Standalone function
def extract_epistemic_features(text: str) -> Dict[str, float]:
    """Extract epistemic marker features (standalone function)."""
    extractor = EpistemicExtractor()
    result = extractor.extract(text)
    return result.features
