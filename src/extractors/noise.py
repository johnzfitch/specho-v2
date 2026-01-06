"""
Noise/Error Plane Extractor (3D)

Detects imperfection patterns that distinguish human from AI text.
Humans make characteristic "errors" that AI systematically avoids.

Features (estimated Cohen's d):
- idiosyncrasy_rate (d~1.0-1.5): Unusual word choices, rare words, neologisms
- formatting_entropy (d~0.8-1.2): Irregularity in whitespace, punctuation patterns
- dead_end_rate (d~0.7-1.0): Topics/entities introduced but not developed

Tier 0: No external dependencies (regex + frequency lists)
"""

import re
import math
from typing import Dict, List, Set
from collections import Counter

from .base import BaseExtractor


class NoiseExtractor(BaseExtractor):
    """
    Extract noise/imperfection features.

    These features capture the "messiness" of human writing that AI
    tends to smooth over. Humans introduce orphan topics, use rare
    words idiosyncratically, and have inconsistent formatting.
    """

    group = "noise"
    dependencies = []  # Pure Python

    # Top 5000 most common English words (abbreviated for efficiency)
    # AI tends to stay within common vocabulary more consistently
    COMMON_WORDS = {
        # Articles, pronouns, prepositions (top ~200)
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
        'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'this', 'that',
        'these', 'those', 'who', 'whom', 'which', 'what', 'whose', 'where',
        'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
        'there', 'then', 'once', 'if', 'unless', 'until', 'while', 'although',
        'because', 'since', 'before', 'after', 'above', 'below', 'between',
        'under', 'over', 'through', 'during', 'about', 'against', 'into',
        'out', 'up', 'down', 'off', 'again', 'further', 'always', 'never',
        # Common verbs
        'get', 'got', 'make', 'made', 'go', 'went', 'gone', 'come', 'came',
        'take', 'took', 'taken', 'see', 'saw', 'seen', 'know', 'knew', 'known',
        'think', 'thought', 'want', 'wanted', 'use', 'using', 'find', 'found',
        'give', 'gave', 'given', 'tell', 'told', 'say', 'said', 'work', 'seem',
        'feel', 'felt', 'try', 'tried', 'leave', 'left', 'call', 'called',
        'keep', 'kept', 'let', 'begin', 'began', 'begun', 'show', 'showed',
        'hear', 'heard', 'play', 'run', 'ran', 'move', 'live', 'believe',
        'hold', 'held', 'bring', 'brought', 'happen', 'write', 'wrote',
        'written', 'provide', 'sit', 'sat', 'stand', 'stood', 'lose', 'lost',
        'pay', 'paid', 'meet', 'met', 'include', 'continue', 'set', 'learn',
        'change', 'lead', 'led', 'understand', 'understood', 'watch', 'follow',
        'stop', 'create', 'speak', 'spoke', 'spoken', 'read', 'allow', 'add',
        'spend', 'spent', 'grow', 'grew', 'grown', 'open', 'walk', 'win', 'won',
        'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'bought',
        'wait', 'serve', 'die', 'died', 'send', 'sent', 'expect', 'build',
        'built', 'stay', 'fall', 'fell', 'fallen', 'cut', 'reach', 'kill',
        'remain', 'suggest', 'raise', 'pass', 'sell', 'sold', 'require',
        'report', 'decide', 'pull', 'break', 'broke', 'broken',
        # Common nouns
        'time', 'year', 'people', 'way', 'day', 'man', 'men', 'woman', 'women',
        'child', 'children', 'world', 'life', 'hand', 'part', 'place', 'case',
        'week', 'company', 'system', 'program', 'question', 'work', 'government',
        'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area',
        'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book',
        'eye', 'job', 'word', 'business', 'issue', 'side', 'kind', 'head',
        'house', 'service', 'friend', 'father', 'power', 'hour', 'game',
        'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name',
        'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
        'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health',
        'person', 'art', 'war', 'history', 'party', 'result', 'change',
        'morning', 'reason', 'research', 'girl', 'guy', 'moment', 'air',
        'teacher', 'force', 'education',
        # Common adjectives
        'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own',
        'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
        'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
        'free', 'sure', 'real', 'full', 'special', 'easy', 'clear', 'recent',
        'certain', 'personal', 'open', 'red', 'difficult', 'available', 'likely',
        'short', 'single', 'medical', 'current', 'wrong', 'private', 'past',
        'foreign', 'fine', 'common', 'poor', 'natural', 'significant', 'similar',
        'hot', 'dead', 'central', 'happy', 'serious', 'ready', 'simple', 'left',
        'physical', 'general', 'environmental', 'financial', 'blue', 'democratic',
        'dark', 'various', 'entire', 'close', 'legal', 'religious', 'cold',
        'final', 'main', 'green', 'nice', 'huge', 'popular', 'traditional',
        'cultural',
    }

    @property
    def feature_names(self) -> List[str]:
        return [
            "idiosyncrasy_rate",     # d~1.0-1.5
            "formatting_entropy",    # d~0.8-1.2
            "dead_end_rate",         # d~0.7-1.0
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

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract noise/imperfection features."""
        features = {name: 0.0 for name in self.feature_names}

        # Early return for short text
        if len(text) < 100:
            return features

        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        word_count = len(words)

        if word_count < 20:
            return features

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # =====================================================================
        # Feature 1: Idiosyncrasy rate - ISOLATED
        # =====================================================================
        def calc_idiosyncrasy():
            # Count words not in common vocabulary
            uncommon_count = 0
            for word in words:
                if len(word) > 2 and word not in self.COMMON_WORDS:
                    uncommon_count += 1

            # Also check for very rare patterns:
            # - Words with unusual letter combinations
            # - Very long words (15+ chars)
            # - Words with repeated unusual patterns
            unusual_patterns = 0
            for word in words:
                if len(word) > 15:
                    unusual_patterns += 1
                # Check for unusual letter sequences (q not followed by u, etc.)
                if re.search(r'q[^u]|[^aeiou]{5,}|(.)\1{2,}', word):
                    unusual_patterns += 1

            # Combine metrics
            uncommon_rate = uncommon_count / word_count
            unusual_rate = unusual_patterns / word_count

            # Weight uncommon words more (they're the main signal)
            features["idiosyncrasy_rate"] = (uncommon_rate * 0.7) + (unusual_rate * 0.3)
        self._safe_extract("idiosyncrasy", calc_idiosyncrasy)

        # =====================================================================
        # Feature 2: Formatting entropy - ISOLATED
        # =====================================================================
        def calc_formatting_entropy():
            # Analyze whitespace patterns
            whitespace_runs = re.findall(r'[ \t]+', text)
            ws_lengths = [len(ws) for ws in whitespace_runs]
            ws_entropy = self._shannon_entropy(ws_lengths) if ws_lengths else 0

            # Analyze punctuation sequences
            punct_sequences = re.findall(r'[.,;:!?\'"()\[\]{}]+', text)
            punct_entropy = self._shannon_entropy(punct_sequences)

            # Analyze quote style consistency
            single_quotes = len(re.findall(r"'", text))
            double_quotes = len(re.findall(r'"', text))
            smart_quotes = len(re.findall(r'[""''"]', text))
            quote_types = []
            if single_quotes > 0:
                quote_types.append('single')
            if double_quotes > 0:
                quote_types.append('double')
            if smart_quotes > 0:
                quote_types.append('smart')
            quote_diversity = len(quote_types) / 3.0

            # Analyze sentence-ending punctuation variety
            sentence_ends = re.findall(r'[.!?]+', text)
            end_entropy = self._shannon_entropy(sentence_ends)

            # Combine into overall formatting entropy
            # Higher entropy = more variation = more human-like
            # Normalize each component to roughly 0-1 range
            normalized_ws = min(ws_entropy / 2.0, 1.0)
            normalized_punct = min(punct_entropy / 3.0, 1.0)
            normalized_end = min(end_entropy / 1.5, 1.0)

            features["formatting_entropy"] = (
                normalized_ws * 0.3 +
                normalized_punct * 0.3 +
                quote_diversity * 0.2 +
                normalized_end * 0.2
            )
        self._safe_extract("formatting_entropy", calc_formatting_entropy)

        # =====================================================================
        # Feature 3: Dead end rate (orphan topics) - ISOLATED
        # =====================================================================
        def calc_dead_ends():
            # Extract potential topics (capitalized words that aren't sentence starters)
            # and noun-like words (simplified without POS tagger)

            # Find capitalized words not at sentence start
            potential_entities = []
            for sent in sentences:
                sent_words = sent.split()
                if len(sent_words) > 1:
                    # Skip first word, look for capitalized words
                    for word in sent_words[1:]:
                        clean_word = re.sub(r'[^\w]', '', word)
                        if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                            potential_entities.append(clean_word.lower())

            # Also track content words (nouns tend to be longer, end in certain suffixes)
            content_words = []
            for word in words:
                # Heuristic for content words: 4+ chars, not a common function word
                if len(word) >= 4 and word not in self.COMMON_WORDS:
                    content_words.append(word)

            # Count entities/content words that appear only once (orphans)
            entity_counts = Counter(potential_entities)
            content_counts = Counter(content_words)

            orphan_entities = sum(1 for count in entity_counts.values() if count == 1)
            orphan_content = sum(1 for count in content_counts.values() if count == 1)

            total_entities = len(entity_counts)
            total_content = len(content_counts)

            # Calculate orphan rates
            entity_orphan_rate = orphan_entities / total_entities if total_entities > 0 else 0
            content_orphan_rate = orphan_content / total_content if total_content > 0 else 0

            # Combine (entities are stronger signal)
            features["dead_end_rate"] = (entity_orphan_rate * 0.6) + (content_orphan_rate * 0.4)
        self._safe_extract("dead_ends", calc_dead_ends)

        return features


# Standalone function
def extract_noise_features(text: str) -> Dict[str, float]:
    """Extract noise/imperfection features (standalone function)."""
    extractor = NoiseExtractor()
    result = extractor.extract(text)
    return result.features
