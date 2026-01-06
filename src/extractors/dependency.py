"""
Dependency Feature Extractor (6D)

Syntactic complexity from dependency parsing.

Features (sorted by Cohen's d):
- avg_dependency_distance (d=+0.60): Humans have longer deps
- max_dependency_distance (d=+0.50): Humans have longer deps
- passive_voice_rate (d=-0.50): AI uses more passive
- dependency_distance_std (d=+0.45): Humans have more variation
- tree_depth_mean (d=+0.40): Humans have deeper trees
- relative_clause_rate (d=-0.35): AI uses fewer rel clauses
"""

import math
from typing import Dict, List, Optional

from .base import BaseExtractor


def get_tree_depth(token, depth: int = 0) -> int:
    """Get depth of token in dependency tree."""
    if token.head == token:
        return depth
    return get_tree_depth(token.head, depth + 1)


class DependencyExtractor(BaseExtractor):
    """
    Dependency parsing features requiring spaCy.

    Each feature is extracted independently with fault isolation.
    """

    group = "dependency"
    dependencies = ["spacy"]

    def __init__(self, nlp=None):
        """
        Args:
            nlp: Optional pre-loaded spaCy model.
        """
        self._nlp = nlp
        self._loaded = nlp is not None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if not self._loaded:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
            self._loaded = True
        return self._nlp

    @property
    def feature_names(self) -> List[str]:
        return [
            "avg_dependency_distance",   # d=+0.60
            "max_dependency_distance",   # d=+0.50
            "dependency_distance_std",   # d=+0.45
            "tree_depth_mean",           # d=+0.40
            "passive_voice_rate",        # d=-0.50
            "relative_clause_rate",      # d=-0.35
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Safely extract a single feature with error handling."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, doc=None, **kwargs) -> Dict[str, float]:
        """Extract dependency features with fault isolation."""
        features = {name: 0.0 for name in self.feature_names}

        # Graceful degradation if spaCy unavailable
        if self.nlp is None:
            return features

        if doc is None:
            doc = self.nlp(text)

        tokens = list(doc)
        if len(tokens) < 5:
            return features

        # 1. Dependency distances (ISOLATED)
        def calc_distances():
            distances = [abs(t.i - t.head.i) for t in tokens if t.dep_ != 'ROOT']
            if distances:
                features["avg_dependency_distance"] = sum(distances) / len(distances)
                features["max_dependency_distance"] = float(max(distances))
                mean_d = features["avg_dependency_distance"]
                features["dependency_distance_std"] = math.sqrt(
                    sum((d - mean_d)**2 for d in distances) / len(distances)
                )

        self._safe_extract("distances", calc_distances)

        # 2. Tree depth (ISOLATED)
        def calc_tree_depth():
            depths = [get_tree_depth(t) for t in tokens]
            if depths:
                features["tree_depth_mean"] = sum(depths) / len(depths)

        self._safe_extract("tree_depth", calc_tree_depth)

        # 3. Passive voice rate (ISOLATED)
        def calc_passive():
            n_sent = len(list(doc.sents))
            passive_count = sum(1 for t in tokens if t.dep_ == 'nsubjpass')
            features["passive_voice_rate"] = passive_count / max(n_sent, 1)

        self._safe_extract("passive", calc_passive)

        # 4. Relative clause rate (ISOLATED)
        def calc_relcl():
            n_sent = len(list(doc.sents))
            relcl_count = sum(1 for t in tokens if t.dep_ == 'relcl')
            features["relative_clause_rate"] = relcl_count / max(n_sent, 1)

        self._safe_extract("relcl", calc_relcl)

        return features


# Standalone function
def extract_dependency_features(text: str, nlp=None) -> Dict[str, float]:
    """Extract dependency features (standalone function)."""
    extractor = DependencyExtractor(nlp=nlp)
    result = extractor.extract(text)
    return result.features
