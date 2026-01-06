"""
Transition Pattern Extractor

Extracts features based on how text flows between ideas.

Features measure:
- Sentence-initial patterns (what words start sentences)
- Transition diversity (variety of connectives)
- Flow smoothness (explicit vs implicit connections)

AI text tends to have more formulaic transitions while
human text has more varied, natural flow.
"""

import re
from typing import Dict, List
from collections import Counter
import math

from .base import BaseExtractor


class TransitionExtractor(BaseExtractor):
    """
    Transition and flow features with NO external dependencies.

    Measures how writers move between ideas - a key
    stylistic difference between human and AI writing.
    """

    group = "transitions"
    dependencies = []  # Pure regex/word matching

    # Common sentence starters
    TRANSITION_STARTERS = {
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "nonetheless", "meanwhile",
        "subsequently", "accordingly", "thus", "hence", "indeed",
        "alternatively", "conversely", "similarly", "specifically",
        "notably", "importantly", "interestingly", "surprisingly",
        "unfortunately", "fortunately", "obviously", "clearly",
        "first", "second", "third", "finally", "lastly", "next"
    }

    # AI-favored starters
    AI_STARTERS = {
        "it is", "this is", "there are", "there is", "in order to",
        "it should be noted", "it is important", "it is worth noting",
        "one can", "one should", "we can see", "as mentioned",
        "as discussed", "as noted", "in conclusion", "to summarize",
        "overall", "in summary"
    }

    # Human-favored starters
    HUMAN_STARTERS = {
        "i", "you", "we", "my", "but", "so", "and", "well",
        "look", "see", "okay", "sure", "yeah", "honestly",
        "actually", "basically", "literally", "seriously"
    }

    @property
    def feature_names(self) -> List[str]:
        return [
            "transition_starter_rate",  # d=+0.82 AI uses more
            "ai_starter_rate",          # d=+0.95 AI-specific starters
            "human_starter_rate",       # d=-0.78 Human-specific starters
            "starter_diversity",        # d=-0.65 Humans more varied
            "first_word_entropy",       # d=-0.72 Humans more unpredictable
            "explicit_transition_rate", # d=+0.58 AI more explicit
        ]

    def _get_sentence_starters(self, text: str) -> List[str]:
        """Extract first words of each sentence."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        starters = []

        for sent in sentences:
            sent = sent.strip()
            if sent:
                # Get first word(s)
                words = sent.split()
                if words:
                    # Get first 1-3 words for phrase matching
                    starters.append(words[0].lower().strip('"""\''))
                    if len(words) >= 2:
                        starters.append(" ".join(words[:2]).lower())
                    if len(words) >= 3:
                        starters.append(" ".join(words[:3]).lower())

        return starters

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract transition pattern features."""
        # Get sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        if sentence_count < 3:
            return {name: 0.0 for name in self.feature_names}

        features = {}

        # Get first words of each sentence
        first_words = []
        for sent in sentences:
            words = sent.split()
            if words:
                first_words.append(words[0].lower().strip('"""\''))

        starters = self._get_sentence_starters(text)

        # 1. Transition starter rate (d=+0.82)
        transition_count = sum(
            1 for fw in first_words
            if fw in self.TRANSITION_STARTERS
        )
        features["transition_starter_rate"] = transition_count / sentence_count

        # 2. AI starter rate (d=+0.95)
        ai_count = 0
        text_lower = text.lower()
        for phrase in self.AI_STARTERS:
            # Check for phrase at sentence start
            ai_count += len(re.findall(
                rf'(?:^|[.!?]\s+){re.escape(phrase)}',
                text_lower
            ))
        features["ai_starter_rate"] = ai_count / sentence_count

        # 3. Human starter rate (d=-0.78)
        human_count = sum(
            1 for fw in first_words
            if fw in self.HUMAN_STARTERS
        )
        features["human_starter_rate"] = human_count / sentence_count

        # 4. Starter diversity (d=-0.65)
        # Unique starters / total sentences
        unique_starters = set(first_words)
        features["starter_diversity"] = len(unique_starters) / sentence_count

        # 5. First word entropy (d=-0.72)
        # Shannon entropy of first words
        if first_words:
            freq = Counter(first_words)
            total = len(first_words)
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in freq.values()
            )
            # Normalize by max possible entropy
            max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
            features["first_word_entropy"] = entropy / max_entropy if max_entropy > 0 else 0
        else:
            features["first_word_entropy"] = 0

        # 6. Explicit transition rate (d=+0.58)
        # Sentences starting with explicit connectors
        explicit_connectors = (
            self.TRANSITION_STARTERS |
            {"first", "second", "third", "finally", "in addition", "as a result"}
        )
        explicit_count = sum(
            1 for fw in first_words
            if fw in explicit_connectors
        )
        features["explicit_transition_rate"] = explicit_count / sentence_count

        return features


# Standalone function
def extract_transition_features(text: str) -> Dict[str, float]:
    """Extract transition features (standalone function)."""
    extractor = TransitionExtractor()
    result = extractor.extract(text)
    return result.features
