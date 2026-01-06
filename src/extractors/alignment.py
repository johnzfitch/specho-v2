"""
Alignment/RLHF Artifact Extractor (8D)

Detects fingerprints left by RLHF training in AI-generated text.
Based on 2024-2025 arxiv literature on sycophancy, stylometric detection,
and LLM linguistic markers.

Features (estimated Cohen's d - requires empirical validation):
- benevolence_score (d~0.8-1.2): AI trained to be helpful/harmless
- neutrality_bias (d~0.6-0.9): AI avoids strong positions
- moralizing_coda (d~0.7-1.0): AI adds ethical concluding statements
- syllogism_stability (d~0.5-0.8): AI maintains logical consistency
- sycophancy_index (d~0.9-1.3): AI validates/affirms user excessively
- lexical_tellscore (d~1.0-1.5): Overuse of specific "AI words" (delve, crucial, etc.)
- uniformity_score (d~0.8-1.2): Low variance in sentence structure (burstiness inverse)
- imperfection_absence (d~0.6-1.0): Lack of typos, informal markers, human errors

Tier 0: No external dependencies (regex + word lists only)
"""

import re
import math
from typing import Dict, List
from collections import Counter

from .base import BaseExtractor


class AlignmentExtractor(BaseExtractor):
    """
    Extract RLHF/alignment artifact features.

    These features capture patterns that emerge from reinforcement
    learning from human feedback (RLHF) training, which systematically
    shapes AI output in detectable ways.
    """

    group = "alignment"
    dependencies = []  # Pure Python

    # =========================================================================
    # LEXICAL MARKERS - Words overused by LLMs (from 2024-2025 marker word research)
    # =========================================================================

    # High-signal AI marker words (frequently flagged in detection literature)
    AI_MARKER_WORDS_STRONG = [
        'delve', 'delving', 'delved',
        'tapestry', 'multifaceted', 'holistic',
        'nuanced', 'nuances',
        'comprehensive', 'comprehensively',
        'crucial', 'crucially',
        'pivotal', 'pivotally',
        'intricate', 'intricacies',
        'landscape', 'landscapes',  # metaphorical use
        'realm', 'realms',
        'foster', 'fostering', 'fosters',
        'leverage', 'leveraging', 'leverages',
        'utilize', 'utilizing', 'utilizes', 'utilized',
        'facilitate', 'facilitating', 'facilitates',
        'enhance', 'enhancing', 'enhances',
        'robust', 'robustly',
        'streamline', 'streamlining',
        'optimize', 'optimizing', 'optimizes',
        'underscore', 'underscores', 'underscoring',
        'noteworthy',
        'commendable',
        'meticulous', 'meticulously',
        'invaluable',
        'indispensable',
        'paramount',
        'encompasses', 'encompassing',
        'elevate', 'elevating', 'elevates',
        'aligns', 'aligning',
        'synergy', 'synergies',
        'bolster', 'bolstering',
    ]

    # Medium-signal markers (common but less distinctive)
    AI_MARKER_WORDS_MEDIUM = [
        'furthermore', 'moreover', 'additionally',
        'subsequently', 'consequently',
        'nevertheless', 'nonetheless',
        'significantly',
        'essentially',
        'fundamentally',
        'inherently',
        'ultimately',
        'arguably',
        'notably',
        'particularly',
        'specifically',
        'accordingly',
        'thereby',
        'thus',
        'hence',
        'whereas',
        'whereby',
        'wherein',
    ]

    # =========================================================================
    # HEDGE PATTERNS - Uncertainty/caution markers
    # =========================================================================

    HEDGE_PATTERNS = [
        r'\b(might|may|could|perhaps|possibly|potentially)\b',
        r'\b(generally|typically|usually|often|sometimes)\b',
        r'\b(tends? to|seems? to|appears? to)\b',
        r'\b(in (some|many|most) cases)\b',
        r"\b(it'?s (worth|important) (to )?(note|mention|consider|remember))\b",
        r'\b(keep in mind)\b',
        r'\b(that (being )?said)\b',
        r'\b(to some (extent|degree))\b',
        r'\b(in a sense)\b',
        r'\b(more or less)\b',
    ]

    # =========================================================================
    # BENEVOLENT/HELPFUL PATTERNS
    # =========================================================================

    BENEVOLENT_PATTERNS = [
        r"\b(i'?d be happy to|happy to help)\b",
        r'\b(let me (help|explain|clarify|show|know))\b',
        r"\b(here'?s (how|what|a|the))\b",
        r'\b(i can (help|assist|explain|provide))\b',
        r'\b(feel free to (ask|reach out|let me know|contact))\b',
        r'\b(hope (this|that) helps?)\b',
        r'\b(if you (have|need) (any )?(more |further )?(questions|help|assistance|clarification))\b',
        r"\b(don'?t hesitate to)\b",
        r'\b(glad to (help|assist))\b',
        r'\b(i understand (your|the|that))\b',
        r"\b(that'?s a (great|good|fair|valid|interesting) (question|point))\b",
        r'\b(thank you for (sharing|asking|your))\b',
    ]

    # =========================================================================
    # SYCOPHANTIC PATTERNS
    # =========================================================================

    SYCOPHANCY_PATTERNS = [
        r'\b(great|excellent|good|wonderful|fantastic|brilliant) (question|point|observation|insight)\b',
        r"\b(you'?re (absolutely|completely|totally|entirely) right)\b",
        r"\b(that'?s (a |an )?(great|excellent|good|valid|fair|excellent|insightful|thoughtful) (point|question|observation|perspective))\b",
        r'\b(i (completely |totally |absolutely |entirely )?(agree|understand|see your point))\b',
        r'\b(you make (a |an )?(great|excellent|good|valid|compelling|important) point)\b',
        r"\b(that'?s (very |really |quite )?(insightful|thoughtful|interesting|perceptive|astute))\b",
        r'\b(absolutely|definitely|certainly|exactly)[,!]?\s',
        r'\b(you raise (a |an )?(important|valid|good|excellent|great) (point|concern|issue))\b',
        r"\b(i couldn'?t agree more)\b",
        r'\b(well said)\b',
        r'\b(spot on)\b',
    ]

    # =========================================================================
    # MORALIZING/ETHICAL CONCLUSION PATTERNS
    # =========================================================================

    MORALIZING_PATTERNS = [
        r"\b(it'?s (important|crucial|essential|vital|critical) (to|that))\b",
        r'\b(we (should|must|need to) (always |)?(remember|consider|keep in mind|be mindful))\b',
        r'\b(ethic(s|al|ally)|moral(ly)?)\b',
        r'\b(responsib(ility|le|ly))\b',
        r'\b(ultimately[,]? (what matters|the key|it comes down to|the most important))\b',
        r'\b(at the end of the day)\b',
        r'\b(in conclusion|to conclude|to sum(marize| up)|in summary)\b',
        r'\b(the (key|important|main|critical) (thing|point|takeaway) (is|here))\b',
        r'\b(above all)\b',
        r'\b(first and foremost)\b',
        r'\b(bear in mind)\b',
        r'\b(it bears (mentioning|noting|emphasizing))\b',
    ]

    # =========================================================================
    # NEUTRALITY/BALANCE PATTERNS
    # =========================================================================

    NEUTRALITY_PATTERNS = [
        r'\b(on (the )?one hand.*on (the )?other( hand)?)\b',
        r'\b(there are (valid )?(arguments|points) (on both|for (and against|both)))\b',
        r'\b(it (depends|varies))\b',
        r'\b(pros and cons)\b',
        r'\b(advantages and disadvantages)\b',
        r'\b(both (sides|perspectives|views|arguments))\b',
        r'\b(some (people |)?(argue|believe|think|say).*while others)\b',
        r'\b(from (one|a certain|another) perspective)\b',
        r"\b(there'?s no (one[- ]size[- ]fits[- ]all|simple|easy) answer)\b",
        r"\b(it'?s (complex|complicated|nuanced|multifaceted))\b",
        r'\b(balancing|balance between)\b',
        r'\b(trade-?offs?)\b',
        r'\b(on balance)\b',
    ]

    # =========================================================================
    # LOGICAL STRUCTURE PATTERNS
    # =========================================================================

    PREMISE_MARKERS = [
        r'\b(because|since|given that|as|considering)\b',
        r'\b(first(ly)?|second(ly)?|third(ly)?|finally)\b',
        r'\b(the (reason|fact) (is|that))\b',
        r'\b(for (one|another) thing)\b',
        r'\b(to begin with)\b',
    ]

    CONCLUSION_MARKERS = [
        r'\b(therefore|thus|hence|consequently|accordingly)\b',
        r'\b(so[,]? (we can|it follows|this means|this suggests))\b',
        r'\b(this (shows|demonstrates|proves|indicates|suggests|implies))\b',
        r'\b(in conclusion|to conclude)\b',
        r'\b(as a result)\b',
        r'\b(it follows that)\b',
    ]

    # =========================================================================
    # HUMAN IMPERFECTION MARKERS (absence suggests AI)
    # =========================================================================

    # Patterns that indicate human writing (typos, contractions, informal markers)
    HUMAN_MARKERS = [
        # Informal contractions/slang
        r'\b(gonna|wanna|gotta|kinda|sorta|dunno|lemme|gimme)\b',
        r'\b(lol|lmao|rofl|haha|hahaha|heh|wtf|omg|omfg|idk|imo|imho|tbh|ngl|smh|fwiw)\b',
        r'\b(yeah|yep|nope|yup|nah|uh|um|uhh|umm|hmm|mhm)\b',
        # Expressive punctuation
        r'\.{3,}',  # Ellipsis with 3+ dots
        r'!{2,}',   # Multiple exclamation marks
        r'\?{2,}',  # Multiple question marks
        r'[!?]{2,}',  # Mixed punctuation
        # Text speak
        r'\bu\b',   # "u" for "you"
        r'\bur\b',  # "ur" for "your/you're"
        r'\br\b',   # "r" for "are"
        r'\bw/\b',  # "w/" for "with"
        r'\bb/c\b', # "b/c" for "because"
        r'\bpls\b', # "pls" for "please"
        r'\bthx\b', # "thx" for "thanks"
        # Informal words (less common in AI)
        r'\b(ok|okay|alright|anyways|anyhoo)\b',
        r'\b(stuff|thingy|whatnot|whatchamacallit)\b',
        r'\b(kinda|sorta|prolly|probs)\b',
        # Filler/hedge words humans overuse
        r'\b(like,|so,|well,|anyway,|basically,)\b',
        # Dropped g patterns (must have apostrophe to be safe)
        r"\b\w+in'\b",  # walkin', talkin', etc.
    ]

    # Common typo patterns
    TYPO_PATTERNS = [
        r'\b(teh|hte|taht|waht|adn|nad|jsut|wiht|thier|becuase|recieve|seperate)\b',
        r'\b\w*([a-z])\1{3,}\w*\b',  # 4+ repeated letters (soooo, realllly)
        r' {2,}',   # Multiple spaces (not newlines)
        r',,+',     # Multiple commas
    ]

    @property
    def feature_names(self) -> List[str]:
        return [
            "benevolence_score",     # d~0.8-1.2
            "neutrality_bias",       # d~0.6-0.9
            "moralizing_coda",       # d~0.7-1.0
            "syllogism_stability",   # d~0.5-0.8
            "sycophancy_index",      # d~0.9-1.3
            "lexical_tellscore",     # d~1.0-1.5
            "uniformity_score",      # d~0.8-1.2
            "imperfection_absence",  # d~0.6-1.0
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count total matches across multiple patterns."""
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _count_word_matches(self, words: List[str], wordlist: List[str]) -> int:
        """Count occurrences of words from a list."""
        wordset = set(w.lower() for w in wordlist)
        return sum(1 for w in words if w.lower() in wordset)

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate population variance."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation (std/mean)."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = self._calculate_variance(values)
        return math.sqrt(variance) / mean

    # Domain-specific weight presets for composite scoring
    WEIGHT_PRESETS = {
        # For formal text (news, academic, professional)
        "formal": {
            "benevolence_score": 0.00,
            "neutrality_bias": 0.00,
            "moralizing_coda": 0.10,
            "syllogism_stability": 0.00,
            "sycophancy_index": 0.00,
            "lexical_tellscore": 0.50,
            "uniformity_score": 0.40,
            "imperfection_absence": 0.00,
        },
        # For conversational text (chat, social media, forums)
        "conversational": {
            "benevolence_score": 0.15,
            "neutrality_bias": 0.05,
            "moralizing_coda": 0.08,
            "syllogism_stability": 0.02,
            "sycophancy_index": 0.15,
            "lexical_tellscore": 0.25,
            "uniformity_score": 0.15,
            "imperfection_absence": 0.15,
        },
        # General/unknown domain
        "general": {
            "benevolence_score": 0.08,
            "neutrality_bias": 0.05,
            "moralizing_coda": 0.12,
            "syllogism_stability": 0.02,
            "sycophancy_index": 0.03,
            "lexical_tellscore": 0.35,
            "uniformity_score": 0.30,
            "imperfection_absence": 0.05,
        },
    }

    # Normalizers to map raw scores to 0-1 range
    NORMALIZERS = {
        "benevolence_score": 0.5,
        "neutrality_bias": 0.5,
        "moralizing_coda": 0.5,
        "syllogism_stability": 0.1,
        "sycophancy_index": 0.2,
        "lexical_tellscore": 3.0,
        "uniformity_score": 1.0,
        "imperfection_absence": 1.0,
    }

    # Domain-specific thresholds (calibrated on benchmark)
    THRESHOLDS = {
        "formal": 0.11,
        "conversational": 0.20,
        "general": 0.15,
    }

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract RLHF/alignment artifact features."""
        features = {name: 0.0 for name in self.feature_names}

        # Early return for short text
        if len(text) < 50:
            return features

        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        word_count = len(words)

        if word_count < 10:
            return features

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences) if sentences else 1

        # =====================================================================
        # Feature 1: Benevolence score - ISOLATED
        # =====================================================================
        def calc_benevolence():
            benevolent_matches = self._count_pattern_matches(text_lower, self.BENEVOLENT_PATTERNS)
            hedge_matches = self._count_pattern_matches(text_lower, self.HEDGE_PATTERNS)
            # Combine helpfulness and hedging (both are RLHF artifacts)
            # Weight hedging lower as it's more common in human text too
            total_matches = benevolent_matches + (hedge_matches * 0.3)
            # Normalize by sentence count
            raw_score = total_matches / sentence_count
            features["benevolence_score"] = min(raw_score, 2.0)
        self._safe_extract("benevolence", calc_benevolence)

        # =====================================================================
        # Feature 2: Neutrality bias - ISOLATED
        # =====================================================================
        def calc_neutrality():
            neutrality_matches = self._count_pattern_matches(text_lower, self.NEUTRALITY_PATTERNS)
            # Normalize by word count (per 100 words)
            raw_score = (neutrality_matches * 100) / word_count
            features["neutrality_bias"] = min(raw_score, 5.0)
        self._safe_extract("neutrality", calc_neutrality)

        # =====================================================================
        # Feature 3: Moralizing coda - ISOLATED
        # =====================================================================
        def calc_moralizing():
            # Check last 25% of text for moralizing patterns (refined from 20%)
            text_len = len(text)
            coda_start = int(text_len * 0.75)
            coda_text = text[coda_start:].lower()

            moralizing_in_coda = self._count_pattern_matches(coda_text, self.MORALIZING_PATTERNS)
            moralizing_total = self._count_pattern_matches(text_lower, self.MORALIZING_PATTERNS)

            # Score based on concentration at end
            if moralizing_total > 0:
                coda_concentration = moralizing_in_coda / moralizing_total
            else:
                coda_concentration = 0.0

            # Weight presence and concentration
            presence_score = min(moralizing_total / sentence_count, 1.0)
            # Higher weight on concentration (AI tends to moralize at end)
            features["moralizing_coda"] = (presence_score * 0.4) + (coda_concentration * 0.6)
        self._safe_extract("moralizing", calc_moralizing)

        # =====================================================================
        # Feature 4: Syllogism stability - ISOLATED
        # =====================================================================
        def calc_syllogism():
            premise_count = self._count_pattern_matches(text_lower, self.PREMISE_MARKERS)
            conclusion_count = self._count_pattern_matches(text_lower, self.CONCLUSION_MARKERS)

            total_logical = premise_count + conclusion_count
            if total_logical > 0 and premise_count > 0 and conclusion_count > 0:
                # AI tends to have balanced premise-conclusion structure
                ratio = min(premise_count, conclusion_count) / max(premise_count, conclusion_count)
                density = min(total_logical / sentence_count, 1.0)
                features["syllogism_stability"] = ratio * density
            else:
                features["syllogism_stability"] = 0.0
        self._safe_extract("syllogism", calc_syllogism)

        # =====================================================================
        # Feature 5: Sycophancy index - ISOLATED
        # =====================================================================
        def calc_sycophancy():
            sycophancy_matches = self._count_pattern_matches(text_lower, self.SYCOPHANCY_PATTERNS)
            # Normalize by sentence count
            raw_score = sycophancy_matches / sentence_count
            features["sycophancy_index"] = min(raw_score, 2.0)
        self._safe_extract("sycophancy", calc_sycophancy)

        # =====================================================================
        # Feature 6: Lexical tell score - ISOLATED
        # =====================================================================
        def calc_lexical_tells():
            strong_matches = self._count_word_matches(words, self.AI_MARKER_WORDS_STRONG)
            medium_matches = self._count_word_matches(words, self.AI_MARKER_WORDS_MEDIUM)

            # Weight strong markers higher
            weighted_count = (strong_matches * 2.0) + (medium_matches * 0.5)
            # Normalize per 100 words
            raw_score = (weighted_count * 100) / word_count
            features["lexical_tellscore"] = min(raw_score, 10.0)
        self._safe_extract("lexical_tells", calc_lexical_tells)

        # =====================================================================
        # Feature 7: Uniformity score (inverse burstiness) - ISOLATED
        # =====================================================================
        def calc_uniformity():
            if len(sentences) < 3:
                features["uniformity_score"] = 0.5  # Neutral for short text
                return

            # Sentence length variance (AI tends to be more uniform)
            sent_lengths = [len(s.split()) for s in sentences]
            sent_cv = self._calculate_cv(sent_lengths)

            # Word length variance within text
            word_lengths = [len(w) for w in words]
            word_cv = self._calculate_cv(word_lengths)

            # Paragraph/sentence start diversity
            # AI often starts sentences similarly
            first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
            unique_starts = len(set(first_words))
            start_diversity = unique_starts / len(sentences) if sentences else 1.0

            # Lower CV = more uniform = more AI-like
            # We invert so higher score = more AI-like
            sent_uniformity = max(0, 1.0 - sent_cv)  # CV typically 0.3-0.8
            word_uniformity = max(0, 1.0 - word_cv * 2)  # Scale word CV
            start_uniformity = 1.0 - start_diversity

            # Combine metrics
            features["uniformity_score"] = (
                sent_uniformity * 0.5 +
                word_uniformity * 0.2 +
                start_uniformity * 0.3
            )
        self._safe_extract("uniformity", calc_uniformity)

        # =====================================================================
        # Feature 8: Imperfection absence - ISOLATED
        # =====================================================================
        def calc_imperfection():
            # Count human markers (their absence suggests AI)
            human_matches = self._count_pattern_matches(text, self.HUMAN_MARKERS)
            typo_matches = self._count_pattern_matches(text, self.TYPO_PATTERNS)

            total_imperfections = human_matches + typo_matches

            # Normalize per 100 words
            imperfection_rate = (total_imperfections * 100) / word_count

            # Invert: high score = few imperfections = AI-like
            # Typical human text might have 2-8 markers per 100 words
            # Score of 1.0 = no imperfections (very AI-like)
            # Score of 0.0 = many imperfections (human-like)
            features["imperfection_absence"] = max(0, min(1.0, 1.0 - (imperfection_rate / 5.0)))
        self._safe_extract("imperfection", calc_imperfection)

        return features


    def get_composite_score(self, features: Dict[str, float], domain: str = "general") -> float:
        """
        Calculate a composite AI probability score.

        Args:
            features: Feature dict from extract()
            domain: "general", "formal" (news/academic), or "conversational"

        Returns a value between 0 and 1, where higher = more likely AI.
        """
        weights = self.WEIGHT_PRESETS.get(domain, self.WEIGHT_PRESETS["general"])

        score = 0.0
        for name, weight in weights.items():
            if weight > 0:
                normalized = min(features.get(name, 0) / self.NORMALIZERS[name], 1.0)
                score += normalized * weight

        return min(score, 1.0)


# Standalone function
def extract_alignment_features(text: str) -> Dict[str, float]:
    """Extract RLHF/alignment artifact features (standalone function)."""
    extractor = AlignmentExtractor()
    result = extractor.extract(text)
    return result.features
