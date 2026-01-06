"""
POS Distribution Extractor (12D)

Part-of-speech based features with highest discriminative power.

Features (sorted by Cohen's d):
- coord_subord_ratio (d=-1.70): CCONJ/SCONJ ratio - AI uses more coordinating
- adj_noun_ratio (d=-1.60): AI uses fewer adjectives relative to nouns
- pos_bigram_entropy (d=-0.70): AI has less syntax variety
- pos_trigram_entropy (d=-0.65): AI has less syntax variety
- adv_verb_ratio (d=-0.60): AI under-modifies
- noun_verb_ratio (d=+0.50): AI is entity-heavy
- pron_ratio (d=-0.45): AI uses fewer pronouns
- adj_ratio (d=-0.40): AI uses fewer adjectives
- adv_ratio (d=-0.35): AI uses fewer adverbs
- propn_ratio (d=-0.30): AI uses fewer names
- noun_ratio (d=+0.30): AI slightly noun-heavy
- verb_ratio (d=-0.25): AI slightly verb-light
"""

import math
from typing import Dict, List, Optional
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


class POSDistributionExtractor(BaseExtractor):
    """
    Full 12D POS-based features requiring spaCy.

    Each feature is extracted independently with fault isolation.
    """

    group = "pos"
    dependencies = ["spacy"]

    def __init__(self, nlp=None):
        """
        Args:
            nlp: Optional pre-loaded spaCy model. If None, loads on first use.
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
            # High signal (critical)
            "coord_subord_ratio",  # d=-1.70 CRITICAL
            "adj_noun_ratio",      # d=-1.60 CRITICAL
            # Entropy features
            "pos_bigram_entropy",  # d=-0.70
            "pos_trigram_entropy", # d=-0.65
            # Ratio features
            "adv_verb_ratio",      # d=-0.60
            "noun_verb_ratio",     # d=+0.50
            # Basic ratios
            "noun_ratio",          # d=+0.30
            "verb_ratio",          # d=-0.25
            "adj_ratio",           # d=-0.40
            "adv_ratio",           # d=-0.35
            "pron_ratio",          # d=-0.45
            "propn_ratio",         # d=-0.30
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Safely extract a single feature with error handling."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, doc=None, **kwargs) -> Dict[str, float]:
        """Extract POS distribution features with fault isolation."""
        features = {name: 0.0 for name in self.feature_names}

        # Graceful degradation if spaCy unavailable
        if self.nlp is None:
            return features

        if doc is None:
            doc = self.nlp(text)

        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if len(tokens) < 5:
            return features

        n = len(tokens)
        pos_counts = Counter(t.pos_ for t in tokens)

        # Get counts
        nouns = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        verbs = pos_counts.get('VERB', 0)
        adjs = pos_counts.get('ADJ', 0)
        advs = pos_counts.get('ADV', 0)
        cconj = pos_counts.get('CCONJ', 0)
        sconj = pos_counts.get('SCONJ', 0)

        # 1. Coord/Subord ratio (ISOLATED)
        def calc_coord_subord():
            if sconj > 0:
                features["coord_subord_ratio"] = cconj / sconj
            elif cconj > 0:
                features["coord_subord_ratio"] = cconj * 2
            else:
                features["coord_subord_ratio"] = 1.0

        self._safe_extract("coord_subord", calc_coord_subord)

        # 2. Adj/Noun ratio (ISOLATED)
        def calc_adj_noun():
            features["adj_noun_ratio"] = adjs / max(nouns, 1)

        self._safe_extract("adj_noun", calc_adj_noun)

        # 3. POS bigram entropy (ISOLATED)
        def calc_bigram_entropy():
            if len(tokens) >= 2:
                bigrams = [(tokens[i].pos_, tokens[i+1].pos_) for i in range(len(tokens)-1)]
                features["pos_bigram_entropy"] = shannon_entropy(bigrams)

        self._safe_extract("bigram_entropy", calc_bigram_entropy)

        # 4. POS trigram entropy (ISOLATED)
        def calc_trigram_entropy():
            if len(tokens) >= 3:
                trigrams = [(tokens[i].pos_, tokens[i+1].pos_, tokens[i+2].pos_)
                           for i in range(len(tokens)-2)]
                features["pos_trigram_entropy"] = shannon_entropy(trigrams)

        self._safe_extract("trigram_entropy", calc_trigram_entropy)

        # 5. Adv/Verb ratio (ISOLATED)
        def calc_adv_verb():
            features["adv_verb_ratio"] = advs / max(verbs, 1)

        self._safe_extract("adv_verb", calc_adv_verb)

        # 6. Noun/Verb ratio (ISOLATED)
        def calc_noun_verb():
            features["noun_verb_ratio"] = nouns / max(verbs, 1)

        self._safe_extract("noun_verb", calc_noun_verb)

        # 7-12. Basic ratios (ISOLATED)
        def calc_ratios():
            features["noun_ratio"] = nouns / n
            features["verb_ratio"] = verbs / n
            features["adj_ratio"] = adjs / n
            features["adv_ratio"] = advs / n
            features["pron_ratio"] = pos_counts.get('PRON', 0) / n
            features["propn_ratio"] = pos_counts.get('PROPN', 0) / max(nouns, 1)

        self._safe_extract("ratios", calc_ratios)

        return features


# Standalone function for backward compatibility
def extract_pos_features(text: str, nlp=None) -> Dict[str, float]:
    """Extract POS distribution features (standalone function)."""
    extractor = POSDistributionExtractor(nlp=nlp)
    result = extractor.extract(text)
    return result.features
