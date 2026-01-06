"""
Information Density Extractor (4D)

Measures semantic compression and information packing in text.
AI tends to have more uniform information density across sentences.

Features (estimated Cohen's d):
- propositional_density (d~0.7-1.0): Claims/propositions per sentence
- named_entity_velocity (d~0.8-1.2): Rate of new entity introduction
- adverbial_stuffing (d~0.6-0.9): Modifier-to-content ratio
- specific_number_rate (d~0.7-1.1): Concrete numeric references

Tier 1: Requires spaCy for NER and POS tagging
"""

import re
import math
from typing import Dict, List, Optional
from collections import Counter

from .base import BaseExtractor


class InformationExtractor(BaseExtractor):
    """
    Extract information density features.

    These features measure how information is packed and distributed
    in text. Human writing tends to have more variable information
    density, while AI maintains more consistent levels.
    """

    group = "information"
    dependencies = ["spacy"]

    def __init__(self):
        self._nlp = None
        self._loaded = False

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if not self._loaded:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self._loaded = True
            except Exception:
                self._nlp = None
                self._loaded = True
        return self._nlp

    @property
    def feature_names(self) -> List[str]:
        return [
            "propositional_density",  # d~0.7-1.0
            "named_entity_velocity",  # d~0.8-1.2
            "adverbial_stuffing",     # d~0.6-0.9
            "specific_number_rate",   # d~0.7-1.1
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract information density features."""
        features = {name: 0.0 for name in self.feature_names}

        # Early return for short text
        if len(text) < 100:
            return features

        # Check if spaCy is available
        if self.nlp is None:
            return features

        # Process text with spaCy
        doc = self.nlp(text)

        sentences = list(doc.sents)
        if len(sentences) < 3:
            return features

        words = [token for token in doc if token.is_alpha]
        word_count = len(words)

        if word_count < 20:
            return features

        # =====================================================================
        # Feature 1: Propositional density - ISOLATED
        # =====================================================================
        def calc_propositional():
            # Count verb phrases with arguments as proxy for propositions
            # A proposition roughly = a claim/statement

            proposition_count = 0
            for sent in sentences:
                # Count main verbs (not auxiliaries)
                verbs = [token for token in sent
                        if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass")]

                for verb in verbs:
                    # Count verb's arguments (subject, object, etc.)
                    args = [child for child in verb.children
                           if child.dep_ in ("nsubj", "nsubjpass", "dobj", "iobj",
                                            "attr", "ccomp", "xcomp", "acomp")]
                    if args:
                        proposition_count += 1

            # Normalize by sentence count
            if len(sentences) > 0:
                density = proposition_count / len(sentences)
                # Typical range: 1-3 propositions per sentence
                features["propositional_density"] = min(density / 3.0, 1.0)
        self._safe_extract("propositional", calc_propositional)

        # =====================================================================
        # Feature 2: Named entity velocity - ISOLATED
        # =====================================================================
        def calc_entity_velocity():
            # Measure rate at which new entities are introduced
            # Human writing often front-loads entities, AI distributes more evenly

            entities = [(ent.text.lower(), ent.start) for ent in doc.ents]
            if not entities:
                features["named_entity_velocity"] = 0.5
                return

            # Track first occurrence position of each unique entity
            entity_first_pos = {}
            for ent_text, pos in entities:
                if ent_text not in entity_first_pos:
                    entity_first_pos[ent_text] = pos

            if len(entity_first_pos) < 2:
                features["named_entity_velocity"] = 0.5
                return

            # Calculate what fraction of entities are introduced in first half
            total_tokens = len(doc)
            halfway = total_tokens // 2

            early_entities = sum(1 for pos in entity_first_pos.values() if pos < halfway)
            total_unique = len(entity_first_pos)

            # Human text tends to introduce more entities early
            # AI tends to distribute more evenly
            early_ratio = early_entities / total_unique

            # Score: deviation from 0.5 (even distribution)
            # Higher deviation = more human-like
            features["named_entity_velocity"] = abs(early_ratio - 0.5) * 2
        self._safe_extract("entity_velocity", calc_entity_velocity)

        # =====================================================================
        # Feature 3: Adverbial stuffing - ISOLATED
        # =====================================================================
        def calc_adverbial():
            # Count modifiers (adverbs, adjectives) vs content words (nouns, verbs)

            modifiers = [t for t in doc if t.pos_ in ("ADV", "ADJ")]
            content = [t for t in doc if t.pos_ in ("NOUN", "VERB", "PROPN")]

            modifier_count = len(modifiers)
            content_count = len(content)

            if content_count == 0:
                features["adverbial_stuffing"] = 0.5
                return

            # Ratio of modifiers to content words
            ratio = modifier_count / content_count

            # Typical ratio is 0.2-0.5
            # AI tends to use more modifiers ("very important", "highly significant")
            # Normalize to 0-1 range (0.5 ratio → score of 1.0)
            features["adverbial_stuffing"] = min(ratio * 2, 1.0)
        self._safe_extract("adverbial", calc_adverbial)

        # =====================================================================
        # Feature 4: Specific number rate - ISOLATED
        # =====================================================================
        def calc_numbers():
            # Count specific numeric references
            # Human text (especially informal) has fewer specific numbers
            # AI tends to include more precise figures

            # Find numbers using regex (more robust than just NUM tokens)
            number_patterns = [
                r'\b\d+\.?\d*%\b',      # Percentages: 50%, 3.5%
                r'\b\d{1,3}(,\d{3})+\b', # Large numbers with commas
                r'\b\d+/\d+\b',          # Fractions: 1/2, 3/4
                r'\b\d{4}\b',            # Years: 2024
                r'\b\d+\.\d+\b',         # Decimals: 3.14
                r'\$\d+',                # Currency: $50
                r'\b\d+\s?(million|billion|thousand|hundred)\b',  # Word numbers
            ]

            number_count = 0
            for pattern in number_patterns:
                number_count += len(re.findall(pattern, text, re.IGNORECASE))

            # Also count NUM tokens from spaCy
            num_tokens = [t for t in doc if t.pos_ == "NUM" and t.text.isdigit()]
            number_count += len(num_tokens)

            # Normalize per 100 words
            if word_count > 0:
                rate = (number_count * 100) / word_count
                # Typical: 0-5 numbers per 100 words
                features["specific_number_rate"] = min(rate / 5.0, 1.0)
        self._safe_extract("numbers", calc_numbers)

        return features


# Standalone function
def extract_information_features(text: str) -> Dict[str, float]:
    """Extract information density features (standalone function)."""
    extractor = InformationExtractor()
    result = extractor.extract(text)
    return result.features
