"""
TEMPLATE: New Feature Extractor

Copy this file, rename it, and implement your features.
Auto-discovery will find it automatically - no registry edits needed!

REQUIREMENTS:
1. Class must inherit from BaseExtractor
2. Must define: group (str), feature_names (List[str])
3. Must implement: _extract(text, **kwargs) -> Dict[str, float]

OPTIONAL:
- dependencies = ["spacy", "tiktoken", etc.] - for Tier 1/2 extractors
- Use _safe_extract() for fault isolation

TIERS:
- Tier 0: No dependencies, pure Python (<5ms)
- Tier 1: spaCy dependency (~50ms) - receives doc= kwarg from registry
- Tier 2: transformers/tiktoken (~100ms)
"""

from typing import Dict, List
from .base import BaseExtractor


class TemplateExtractor(BaseExtractor):
    """
    Description of what this extractor measures.

    Features (sorted by estimated Cohen's d):
    - feature_one (d~0.8): Description
    - feature_two (d~0.6): Description
    """

    # REQUIRED: Unique group name for this extractor
    group = "template"

    # OPTIONAL: Dependencies for Tier 1/2 extractors
    # dependencies = ["spacy"]  # Uncomment if needed

    @property
    def feature_names(self) -> List[str]:
        """REQUIRED: List of feature names this extractor produces."""
        return [
            "feature_one",
            "feature_two",
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """OPTIONAL: Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """
        REQUIRED: Extract features from text.

        Args:
            text: Input text to analyze
            **kwargs: May include 'doc' (spaCy Doc) if registry shares it

        Returns:
            Dict mapping feature names to float values
        """
        features = {name: 0.0 for name in self.feature_names}

        # Minimum text length check
        if len(text) < 50:
            return features

        # EXAMPLE: Tier 1 with shared spaCy doc
        # doc = kwargs.get('doc')
        # if doc is None and hasattr(self, 'nlp') and self.nlp:
        #     doc = self.nlp(text)

        # Feature 1 - ISOLATED
        def calc_feature_one():
            # Your feature calculation here
            features["feature_one"] = len(text) / 1000
        self._safe_extract("feature_one", calc_feature_one)

        # Feature 2 - ISOLATED
        def calc_feature_two():
            # Your feature calculation here
            features["feature_two"] = text.count(' ') / len(text)
        self._safe_extract("feature_two", calc_feature_two)

        return features


# OPTIONAL: Standalone function for backwards compatibility
def extract_template_features(text: str) -> Dict[str, float]:
    """Extract template features (standalone function)."""
    extractor = TemplateExtractor()
    result = extractor.extract(text)
    return result.features
