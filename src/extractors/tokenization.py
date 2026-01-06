"""
Tokenization Artifacts Extractor (4D)

Detects patterns related to BPE/tokenizer behavior in LLM outputs.
AI text reflects the tokenizer's vocabulary and splitting patterns.

Features (estimated Cohen's d):
- token_fragmentation (d~0.7-1.1): Avg subword tokens per word
- strange_token_split (d~0.6-0.9): Unusual tokenization boundaries
- whitespace_token_bias (d~0.5-0.8): Leading space token frequency
- glitch_token_sensitivity (d~0.6-1.0): Avoidance of problematic tokens

Tier 2: Requires tiktoken (optional, graceful degradation without it)
"""

import re
from typing import Dict, List, Optional
from collections import Counter

from .base import BaseExtractor


class TokenizationExtractor(BaseExtractor):
    """
    Extract tokenization artifact features.

    These features detect patterns that emerge from how LLMs tokenize
    text using BPE (Byte Pair Encoding). AI-generated text may show
    different tokenization patterns than human text.
    """

    group = "tokenization"
    dependencies = ["tiktoken"]

    # Known "glitch" tokens that LLMs tend to avoid or handle strangely
    GLITCH_TOKENS = [
        'SolidGoldMagworthy', 'rawdownload', 'rawdownloadclone',
        'reportprint', 'embedreportprint', 'cloneembedreportprint',
        'TheNitrome', 'TheNitromeFan', 'GradlentGrailpoints',
        'PsjMQKWGnE', 'Smartstocks', 'ForgeModLoader',
        'RandomRedditorWithNo', 'exaborations', 'attRot',
    ]

    # Common morpheme boundaries (prefixes/suffixes)
    MORPHEME_PREFIXES = ['un', 're', 'pre', 'dis', 'mis', 'non', 'anti', 'over', 'under']
    MORPHEME_SUFFIXES = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment', 'able', 'ible']

    def __init__(self):
        self._encoder = None
        self._loaded = False

    @property
    def encoder(self):
        """Lazy-load tiktoken encoder."""
        if not self._loaded:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
                self._loaded = True
            except Exception:
                self._encoder = None
                self._loaded = True
        return self._encoder

    @property
    def feature_names(self) -> List[str]:
        return [
            "token_fragmentation",    # d~0.7-1.1
            "strange_token_split",    # d~0.6-0.9
            "whitespace_token_bias",  # d~0.5-0.8
            "glitch_token_sensitivity", # d~0.6-1.0
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract tokenization artifact features."""
        features = {name: 0.0 for name in self.feature_names}

        if len(text) < 50:
            return features

        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count = len(words)

        if word_count < 10:
            return features

        # Feature 1: Token fragmentation - ISOLATED
        def calc_fragmentation():
            if self.encoder is None:
                features["token_fragmentation"] = 0.5
                return

            total_tokens = 0
            for word in words:
                tokens = self.encoder.encode(word)
                total_tokens += len(tokens)

            avg_tokens_per_word = total_tokens / word_count if word_count > 0 else 1
            features["token_fragmentation"] = min((avg_tokens_per_word - 1) / 2, 1.0)
        self._safe_extract("fragmentation", calc_fragmentation)

        # Feature 2: Strange token splits - ISOLATED
        def calc_strange_splits():
            if self.encoder is None:
                features["strange_token_split"] = 0.5
                return

            strange_count = 0
            for word in words:
                if len(word) < 4:
                    continue
                tokens = self.encoder.encode(word)
                if len(tokens) > 1:
                    decoded = [self.encoder.decode([t]) for t in tokens]
                    for i, part in enumerate(decoded[:-1]):
                        clean = part.replace(' ', '').replace('Ġ', '')
                        if len(clean) >= 2:
                            breaks_prefix = any(clean.endswith(p[:len(clean)]) for p in self.MORPHEME_PREFIXES if len(p) > len(clean))
                            breaks_suffix = any(clean.startswith(s[-len(clean):]) for s in self.MORPHEME_SUFFIXES if len(s) > len(clean))
                            if breaks_prefix or breaks_suffix:
                                strange_count += 1

            rate = strange_count / word_count if word_count > 0 else 0
            features["strange_token_split"] = min(rate * 10, 1.0)
        self._safe_extract("strange_splits", calc_strange_splits)

        # Feature 3: Whitespace token bias - ISOLATED
        def calc_whitespace_bias():
            if self.encoder is None:
                features["whitespace_token_bias"] = 0.5
                return

            tokens = self.encoder.encode(text)
            if not tokens:
                return

            space_prefix_count = 0
            for token_id in tokens:
                decoded = self.encoder.decode([token_id])
                if decoded.startswith(' ') or decoded.startswith('Ġ'):
                    space_prefix_count += 1

            ratio = space_prefix_count / len(tokens)
            features["whitespace_token_bias"] = ratio
        self._safe_extract("whitespace_bias", calc_whitespace_bias)

        # Feature 4: Glitch token sensitivity - ISOLATED
        def calc_glitch_sensitivity():
            text_lower = text.lower()
            glitch_present = sum(1 for g in self.GLITCH_TOKENS if g.lower() in text_lower)
            features["glitch_token_sensitivity"] = 1.0 if glitch_present == 0 else max(0, 1.0 - glitch_present * 0.2)
        self._safe_extract("glitch_sensitivity", calc_glitch_sensitivity)

        return features


def extract_tokenization_features(text: str) -> Dict[str, float]:
    """Extract tokenization artifact features (standalone function)."""
    extractor = TokenizationExtractor()
    result = extractor.extract(text)
    return result.features
