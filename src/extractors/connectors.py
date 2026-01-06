"""
Connector/Discourse Marker Extractor

Extracts features based on logical connectors and discourse markers.

Features (sorted by Cohen's d):
- and_but_ratio (d=-1.00): AI uses more "and", humans use more "but"
- however_rate (d=-0.80): Humans use "however" more for contrast
- therefore_rate (d=+0.75): AI uses more "therefore" for conclusions
- furthermore_rate (d=+0.68): AI uses more additive connectors
- because_rate (d=-0.62): Humans use more causal explanations
- contrast_marker_density (d=-0.55): Overall contrast word usage
"""

import re
from typing import Dict, List
from collections import Counter

from .base import BaseExtractor


class ConnectorExtractor(BaseExtractor):
    """
    Discourse connector features with NO external dependencies.

    Measures how writers connect ideas - AI tends toward additive
    connectors while humans use more contrastive ones.
    """

    group = "connectors"
    dependencies = []  # Pure regex - no dependencies

    # Connector word lists
    ADDITIVE = {
        "and", "also", "additionally", "furthermore", "moreover",
        "besides", "plus", "likewise", "similarly", "indeed"
    }

    CONTRASTIVE = {
        "but", "however", "although", "though", "yet", "nevertheless",
        "nonetheless", "whereas", "while", "conversely", "instead",
        "rather", "still", "on the other hand"
    }

    CAUSAL = {
        "because", "since", "therefore", "thus", "hence", "consequently",
        "as a result", "so", "accordingly", "due to", "owing to"
    }

    TEMPORAL = {
        "then", "next", "first", "finally", "meanwhile", "subsequently",
        "afterwards", "previously", "before", "after", "during"
    }

    @property
    def feature_names(self) -> List[str]:
        return [
            "and_but_ratio",          # d=-1.00
            "however_rate",           # d=-0.80
            "therefore_rate",         # d=+0.75
            "furthermore_rate",       # d=+0.68
            "because_rate",           # d=-0.62
            "contrast_marker_density", # d=-0.55
            "additive_density",       # AI tends higher
            "causal_density",         # Humans tend higher
        ]

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract connector features from text."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        if word_count < 10:
            return {name: 0.0 for name in self.feature_names}

        # Count specific connectors
        word_freq = Counter(words)

        # Clean word counting (strip punctuation for matching)
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        clean_freq = Counter(clean_words)

        features = {}

        # 1. And/But ratio (d=-1.00)
        and_count = clean_freq.get("and", 0)
        but_count = clean_freq.get("but", 0)
        if but_count > 0:
            features["and_but_ratio"] = and_count / but_count
        elif and_count > 0:
            features["and_but_ratio"] = and_count * 2  # High when no "but"
        else:
            features["and_but_ratio"] = 1.0

        # 2. However rate (d=-0.80)
        however_count = clean_freq.get("however", 0)
        features["however_rate"] = however_count / word_count * 100

        # 3. Therefore rate (d=+0.75)
        therefore_count = clean_freq.get("therefore", 0) + clean_freq.get("thus", 0)
        features["therefore_rate"] = therefore_count / word_count * 100

        # 4. Furthermore rate (d=+0.68)
        furthermore_count = (
            clean_freq.get("furthermore", 0) +
            clean_freq.get("moreover", 0) +
            clean_freq.get("additionally", 0)
        )
        features["furthermore_rate"] = furthermore_count / word_count * 100

        # 5. Because rate (d=-0.62)
        because_count = clean_freq.get("because", 0) + clean_freq.get("since", 0)
        features["because_rate"] = because_count / word_count * 100

        # 6. Contrast marker density (d=-0.55)
        contrast_count = sum(clean_freq.get(w, 0) for w in self.CONTRASTIVE if " " not in w)
        # Also check multi-word markers
        contrast_count += text_lower.count("on the other hand")
        features["contrast_marker_density"] = contrast_count / word_count * 100

        # 7. Additive density
        additive_count = sum(clean_freq.get(w, 0) for w in self.ADDITIVE)
        features["additive_density"] = additive_count / word_count * 100

        # 8. Causal density
        causal_count = sum(clean_freq.get(w, 0) for w in self.CAUSAL if " " not in w)
        causal_count += text_lower.count("as a result")
        causal_count += text_lower.count("due to")
        features["causal_density"] = causal_count / word_count * 100

        return features


# Standalone function
def extract_connector_features(text: str) -> Dict[str, float]:
    """Extract connector features (standalone function)."""
    extractor = ConnectorExtractor()
    result = extractor.extract(text)
    return result.features
