"""
Complete 89D Fingerprint Extractor
==================================

Extracts EVERY possible feature for AI text detection.

Categories:
- STAT:   Statistical (14D)      - compression, entropy, vocabulary metrics
- LEX:    Lexical (12D)          - word-level patterns
- PUNCT:  Punctuation (8D)       - punctuation usage
- STRUCT: Structural (10D)       - document structure
- CONN:   Connectors (8D)        - conjunction patterns
- POS:    Part-of-Speech (12D)   - grammatical categories
- DEP:    Dependency (6D)        - syntactic complexity
- TRAJ:   Trajectory (5D)        - semantic path geometry
- ECHO:   Echo (15D)             - clause boundary patterns

Total: 89 dimensions (primary)
Extended: 108 dimensions (with EPIS, TRANS, RHYTHM)

Usage:
    extractor = CompleteExtractor(tier=0)  # No dependencies
    extractor = CompleteExtractor(tier=2)  # With spaCy
    extractor = CompleteExtractor(tier=3)  # Full system
    
    fp = extractor.extract(text)
    vector = fp.to_vector()  # numpy array
"""

import re
import math
import gzip
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

# Common words for rare word detection
TOP_1000_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having', 'does',
    'did', 'doing', 'done', 'should', 'could', 'would', 'might', 'must', 'shall',
    'very', 'much', 'more', 'many', 'such', 'same', 'still', 'however', 'therefore',
}

# Hedge words
HEDGE_WORDS = {
    'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'may',
    'seemingly', 'apparently', 'somewhat', 'rather', 'quite', 'fairly',
    'kind of', 'sort of', 'a bit', 'a little', 'to some extent',
    'it seems', 'it appears', 'i think', 'i believe', 'i suppose',
    'in my opinion', 'arguably', 'presumably', 'allegedly',
}

# Transition words by category
TRANSITIONS = {
    'additive': ['moreover', 'furthermore', 'additionally', 'also', 'besides', 'in addition'],
    'contrastive': ['however', 'nevertheless', 'nonetheless', 'although', 'though', 'yet', 'but'],
    'causal': ['therefore', 'thus', 'hence', 'consequently', 'because', 'since', 'as a result'],
    'temporal': ['then', 'next', 'afterward', 'meanwhile', 'subsequently', 'finally', 'later'],
    'exemplifying': ['for example', 'for instance', 'such as', 'specifically', 'namely'],
    'reformulating': ['in other words', 'that is', 'i.e.', 'namely', 'to put it differently'],
}


# =============================================================================
# FINGERPRINT DATACLASS
# =============================================================================

@dataclass
class CompleteFingerprint:
    """
    Complete 89D (primary) / 108D (extended) fingerprint.
    """
    
    # Metadata
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    extraction_tier: int = 0
    extraction_time_ms: float = 0.0
    
    # STAT: Statistical Features (14D)
    stat_compression_ratio: float = 0.0
    stat_entropy_char: float = 0.0
    stat_entropy_word: float = 0.0
    stat_entropy_bigram: float = 0.0
    stat_perplexity_approx: float = 0.0
    stat_redundancy_rate: float = 0.0
    stat_repetition_rate: float = 0.0
    stat_burstiness_words: float = 0.0
    stat_zipf_coefficient: float = 0.0
    stat_yule_k: float = 0.0
    stat_simpson_d: float = 0.0
    stat_honore_r: float = 0.0
    stat_sichel_s: float = 0.0
    stat_brunet_w: float = 0.0
    
    # LEX: Lexical Features (12D)
    lex_type_token_ratio: float = 0.0
    lex_hapax_legomena_ratio: float = 0.0
    lex_dis_legomena_ratio: float = 0.0
    lex_avg_word_length: float = 0.0
    lex_word_length_std: float = 0.0
    lex_word_length_cv: float = 0.0
    lex_syllable_ratio: float = 0.0
    lex_long_word_ratio: float = 0.0
    lex_rare_word_ratio: float = 0.0
    lex_concrete_ratio: float = 0.0
    lex_abstract_ratio: float = 0.0
    lex_formality_score: float = 0.0
    
    # PUNCT: Punctuation Features (8D)
    punct_question_rate: float = 0.0
    punct_exclamation_rate: float = 0.0
    punct_em_dash_rate: float = 0.0
    punct_semicolon_rate: float = 0.0
    punct_colon_rate: float = 0.0
    punct_parenthetical_rate: float = 0.0
    punct_ellipsis_rate: float = 0.0
    punct_entropy: float = 0.0
    
    # STRUCT: Structural Features (10D)
    struct_sentence_length_cv: float = 0.0
    struct_sentence_length_mean: float = 0.0
    struct_sentence_length_std: float = 0.0
    struct_paragraph_length_cv: float = 0.0
    struct_paragraph_count_norm: float = 0.0
    struct_list_marker_rate: float = 0.0
    struct_header_rate: float = 0.0
    struct_sentence_initial_entropy: float = 0.0
    struct_sentence_final_entropy: float = 0.0
    struct_quote_rate: float = 0.0
    
    # CONN: Connector Features (8D)
    conn_coord_subord_ratio: float = 0.0
    conn_and_but_ratio: float = 0.0
    conn_coord_conj_rate: float = 0.0
    conn_subord_conj_rate: float = 0.0
    conn_however_rate: float = 0.0
    conn_therefore_rate: float = 0.0
    conn_moreover_rate: float = 0.0
    conn_furthermore_rate: float = 0.0
    
    # POS: Part-of-Speech Features (12D)
    pos_noun_ratio: float = 0.0
    pos_verb_ratio: float = 0.0
    pos_adj_ratio: float = 0.0
    pos_adv_ratio: float = 0.0
    pos_pron_ratio: float = 0.0
    pos_det_ratio: float = 0.0
    pos_adj_noun_ratio: float = 0.0
    pos_adv_verb_ratio: float = 0.0
    pos_noun_verb_ratio: float = 0.0
    pos_propn_ratio: float = 0.0
    pos_bigram_entropy: float = 0.0
    pos_trigram_entropy: float = 0.0
    
    # DEP: Dependency Features (6D)
    dep_avg_distance: float = 0.0
    dep_max_distance: float = 0.0
    dep_distance_std: float = 0.0
    dep_tree_depth_mean: float = 0.0
    dep_passive_voice_rate: float = 0.0
    dep_relative_clause_rate: float = 0.0
    
    # TRAJ: Trajectory Features (5D)
    traj_concept_jump_mean: float = 0.0
    traj_concept_jump_variance: float = 0.0
    traj_path_tortuosity: float = 1.0
    traj_turning_angle_mean: float = 0.0
    traj_return_rate: float = 0.0
    
    # ECHO: Echo Features (15D)
    echo_phonetic_mean: float = 0.0
    echo_phonetic_std: float = 0.0
    echo_phonetic_max: float = 0.0
    echo_structural_mean: float = 0.0
    echo_structural_std: float = 0.0
    echo_structural_max: float = 0.0
    echo_semantic_mean: float = 0.0
    echo_semantic_std: float = 0.0
    echo_semantic_max: float = 0.0
    echo_cooccurrence_rate: float = 0.0
    echo_geometric_mean: float = 0.0
    echo_overall_mean: float = 0.0
    echo_overall_std: float = 0.0
    echo_overall_max: float = 0.0
    echo_burstiness: float = 0.0
    
    # Extended: EPIS (6D)
    epis_hedge_density: float = 0.0
    epis_hedge_clustering: float = 0.0
    epis_hedge_position_bias: float = 0.5
    epis_confidence_mean: float = 0.5
    epis_confidence_variance: float = 0.0
    epis_confidence_arc: float = 0.0
    
    # Extended: TRANS (6D)
    trans_additive_rate: float = 0.0
    trans_contrastive_rate: float = 0.0
    trans_causal_rate: float = 0.0
    trans_temporal_rate: float = 0.0
    trans_exemplifying_rate: float = 0.0
    trans_reformulating_rate: float = 0.0
    
    # Extended: RHYTHM (7D)
    rhythm_clause_length_mean: float = 0.0
    rhythm_clause_length_std: float = 0.0
    rhythm_autocorr: float = 0.0
    rhythm_sentence_complexity: float = 1.0
    rhythm_comma_density: float = 0.0
    rhythm_semicolon_rate: float = 0.0
    rhythm_parenthetical_rate: float = 0.0
    
    def to_vector_89d(self) -> np.ndarray:
        """Primary 89D vector."""
        return np.array([
            # STAT (14)
            self.stat_compression_ratio, self.stat_entropy_char, self.stat_entropy_word,
            self.stat_entropy_bigram, self.stat_perplexity_approx, self.stat_redundancy_rate,
            self.stat_repetition_rate, self.stat_burstiness_words, self.stat_zipf_coefficient,
            self.stat_yule_k, self.stat_simpson_d, self.stat_honore_r, self.stat_sichel_s,
            self.stat_brunet_w,
            # LEX (12)
            self.lex_type_token_ratio, self.lex_hapax_legomena_ratio, self.lex_dis_legomena_ratio,
            self.lex_avg_word_length, self.lex_word_length_std, self.lex_word_length_cv,
            self.lex_syllable_ratio, self.lex_long_word_ratio, self.lex_rare_word_ratio,
            self.lex_concrete_ratio, self.lex_abstract_ratio, self.lex_formality_score,
            # PUNCT (8)
            self.punct_question_rate, self.punct_exclamation_rate, self.punct_em_dash_rate,
            self.punct_semicolon_rate, self.punct_colon_rate, self.punct_parenthetical_rate,
            self.punct_ellipsis_rate, self.punct_entropy,
            # STRUCT (10)
            self.struct_sentence_length_cv, self.struct_sentence_length_mean,
            self.struct_sentence_length_std, self.struct_paragraph_length_cv,
            self.struct_paragraph_count_norm, self.struct_list_marker_rate,
            self.struct_header_rate, self.struct_sentence_initial_entropy,
            self.struct_sentence_final_entropy, self.struct_quote_rate,
            # CONN (8)
            self.conn_coord_subord_ratio, self.conn_and_but_ratio, self.conn_coord_conj_rate,
            self.conn_subord_conj_rate, self.conn_however_rate, self.conn_therefore_rate,
            self.conn_moreover_rate, self.conn_furthermore_rate,
            # POS (12)
            self.pos_noun_ratio, self.pos_verb_ratio, self.pos_adj_ratio, self.pos_adv_ratio,
            self.pos_pron_ratio, self.pos_det_ratio, self.pos_adj_noun_ratio,
            self.pos_adv_verb_ratio, self.pos_noun_verb_ratio, self.pos_propn_ratio,
            self.pos_bigram_entropy, self.pos_trigram_entropy,
            # DEP (6)
            self.dep_avg_distance, self.dep_max_distance, self.dep_distance_std,
            self.dep_tree_depth_mean, self.dep_passive_voice_rate, self.dep_relative_clause_rate,
            # TRAJ (5)
            self.traj_concept_jump_mean, self.traj_concept_jump_variance,
            self.traj_path_tortuosity, self.traj_turning_angle_mean, self.traj_return_rate,
            # ECHO (15)
            self.echo_phonetic_mean, self.echo_phonetic_std, self.echo_phonetic_max,
            self.echo_structural_mean, self.echo_structural_std, self.echo_structural_max,
            self.echo_semantic_mean, self.echo_semantic_std, self.echo_semantic_max,
            self.echo_cooccurrence_rate, self.echo_geometric_mean, self.echo_overall_mean,
            self.echo_overall_std, self.echo_overall_max, self.echo_burstiness,
        ])
    
    def to_vector_extended(self) -> np.ndarray:
        """Extended 108D vector."""
        base = self.to_vector_89d()
        extended = np.array([
            # EPIS (6)
            self.epis_hedge_density, self.epis_hedge_clustering, self.epis_hedge_position_bias,
            self.epis_confidence_mean, self.epis_confidence_variance, self.epis_confidence_arc,
            # TRANS (6)
            self.trans_additive_rate, self.trans_contrastive_rate, self.trans_causal_rate,
            self.trans_temporal_rate, self.trans_exemplifying_rate, self.trans_reformulating_rate,
            # RHYTHM (7)
            self.rhythm_clause_length_mean, self.rhythm_clause_length_std, self.rhythm_autocorr,
            self.rhythm_sentence_complexity, self.rhythm_comma_density, self.rhythm_semicolon_rate,
            self.rhythm_parenthetical_rate,
        ])
        return np.concatenate([base, extended])
    
    def to_vector_tier0(self) -> np.ndarray:
        """Tier 0: 44D (STAT + LEX + PUNCT + STRUCT)"""
        return np.array([
            # STAT (14)
            self.stat_compression_ratio, self.stat_entropy_char, self.stat_entropy_word,
            self.stat_entropy_bigram, self.stat_perplexity_approx, self.stat_redundancy_rate,
            self.stat_repetition_rate, self.stat_burstiness_words, self.stat_zipf_coefficient,
            self.stat_yule_k, self.stat_simpson_d, self.stat_honore_r, self.stat_sichel_s,
            self.stat_brunet_w,
            # LEX (12)
            self.lex_type_token_ratio, self.lex_hapax_legomena_ratio, self.lex_dis_legomena_ratio,
            self.lex_avg_word_length, self.lex_word_length_std, self.lex_word_length_cv,
            self.lex_syllable_ratio, self.lex_long_word_ratio, self.lex_rare_word_ratio,
            self.lex_concrete_ratio, self.lex_abstract_ratio, self.lex_formality_score,
            # PUNCT (8)
            self.punct_question_rate, self.punct_exclamation_rate, self.punct_em_dash_rate,
            self.punct_semicolon_rate, self.punct_colon_rate, self.punct_parenthetical_rate,
            self.punct_ellipsis_rate, self.punct_entropy,
            # STRUCT (10)
            self.struct_sentence_length_cv, self.struct_sentence_length_mean,
            self.struct_sentence_length_std, self.struct_paragraph_length_cv,
            self.struct_paragraph_count_norm, self.struct_list_marker_rate,
            self.struct_header_rate, self.struct_sentence_initial_entropy,
            self.struct_sentence_final_entropy, self.struct_quote_rate,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @staticmethod
    def feature_names_89d() -> List[str]:
        """Feature names for 89D vector."""
        return [
            # STAT
            'stat_compression_ratio', 'stat_entropy_char', 'stat_entropy_word',
            'stat_entropy_bigram', 'stat_perplexity_approx', 'stat_redundancy_rate',
            'stat_repetition_rate', 'stat_burstiness_words', 'stat_zipf_coefficient',
            'stat_yule_k', 'stat_simpson_d', 'stat_honore_r', 'stat_sichel_s', 'stat_brunet_w',
            # LEX
            'lex_type_token_ratio', 'lex_hapax_legomena_ratio', 'lex_dis_legomena_ratio',
            'lex_avg_word_length', 'lex_word_length_std', 'lex_word_length_cv',
            'lex_syllable_ratio', 'lex_long_word_ratio', 'lex_rare_word_ratio',
            'lex_concrete_ratio', 'lex_abstract_ratio', 'lex_formality_score',
            # PUNCT
            'punct_question_rate', 'punct_exclamation_rate', 'punct_em_dash_rate',
            'punct_semicolon_rate', 'punct_colon_rate', 'punct_parenthetical_rate',
            'punct_ellipsis_rate', 'punct_entropy',
            # STRUCT
            'struct_sentence_length_cv', 'struct_sentence_length_mean',
            'struct_sentence_length_std', 'struct_paragraph_length_cv',
            'struct_paragraph_count_norm', 'struct_list_marker_rate',
            'struct_header_rate', 'struct_sentence_initial_entropy',
            'struct_sentence_final_entropy', 'struct_quote_rate',
            # CONN
            'conn_coord_subord_ratio', 'conn_and_but_ratio', 'conn_coord_conj_rate',
            'conn_subord_conj_rate', 'conn_however_rate', 'conn_therefore_rate',
            'conn_moreover_rate', 'conn_furthermore_rate',
            # POS
            'pos_noun_ratio', 'pos_verb_ratio', 'pos_adj_ratio', 'pos_adv_ratio',
            'pos_pron_ratio', 'pos_det_ratio', 'pos_adj_noun_ratio',
            'pos_adv_verb_ratio', 'pos_noun_verb_ratio', 'pos_propn_ratio',
            'pos_bigram_entropy', 'pos_trigram_entropy',
            # DEP
            'dep_avg_distance', 'dep_max_distance', 'dep_distance_std',
            'dep_tree_depth_mean', 'dep_passive_voice_rate', 'dep_relative_clause_rate',
            # TRAJ
            'traj_concept_jump_mean', 'traj_concept_jump_variance',
            'traj_path_tortuosity', 'traj_turning_angle_mean', 'traj_return_rate',
            # ECHO
            'echo_phonetic_mean', 'echo_phonetic_std', 'echo_phonetic_max',
            'echo_structural_mean', 'echo_structural_std', 'echo_structural_max',
            'echo_semantic_mean', 'echo_semantic_std', 'echo_semantic_max',
            'echo_cooccurrence_rate', 'echo_geometric_mean', 'echo_overall_mean',
            'echo_overall_std', 'echo_overall_max', 'echo_burstiness',
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def shannon_entropy(values: List) -> float:
    """Calculate Shannon entropy of a distribution."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def syllable_count(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    # Adjustments
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1
    
    return count


def cv(values: List[float]) -> float:
    """Coefficient of variation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    std = math.sqrt(sum((x - mean)**2 for x in values) / len(values))
    return std / mean


def get_tree_depth(token, depth=0) -> int:
    """Get depth of token in dependency tree."""
    if token.head == token:
        return depth
    return get_tree_depth(token.head, depth + 1)


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class CompleteExtractor:
    """
    Complete 89D/108D feature extractor.
    
    Tiers:
        0: Pure Python, no dependencies (44D)
        1: With connector word lists (52D)
        2: With spaCy (70D)
        3: Full system with embeddings (89D+)
    """
    
    def __init__(self, tier: int = 0, spacy_model: str = "en_core_web_sm"):
        """
        Initialize extractor.
        
        Args:
            tier: Feature tier (0-3)
            spacy_model: spaCy model name for tier 2+
        """
        self.tier = tier
        self.nlp = None
        self.embedder = None
        
        if tier >= 2:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                print(f"[CompleteExtractor] Loaded spaCy: {spacy_model}")
            except Exception as e:
                print(f"[CompleteExtractor] spaCy unavailable: {e}")
                self.tier = min(self.tier, 1)
        
        if tier >= 3:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[CompleteExtractor] Loaded sentence-transformers")
            except Exception as e:
                print(f"[CompleteExtractor] Embeddings unavailable: {e}")
    
    def extract(self, text: str) -> CompleteFingerprint:
        """Extract complete fingerprint."""
        import time
        start = time.time()
        
        fp = CompleteFingerprint()
        
        # Basic text processing
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        fp.text_length = len(text)
        fp.word_count = len(words)
        fp.sentence_count = max(len(sentences), 1)
        fp.extraction_tier = self.tier
        
        # Tier 0: Statistical + Lexical + Punctuation + Structural
        self._extract_stat(fp, text, words)
        self._extract_lex(fp, text, words)
        self._extract_punct(fp, text, sentences)
        self._extract_struct(fp, text, words, sentences, paragraphs)
        
        # Tier 1: Connectors
        if self.tier >= 1:
            self._extract_conn(fp, text, sentences)
        
        # Tier 2: POS + Dependency (requires spaCy)
        if self.tier >= 2 and self.nlp:
            doc = self.nlp(text)
            self._extract_pos(fp, doc)
            self._extract_dep(fp, doc)
        
        # Tier 3: Trajectory + Echo (requires embeddings)
        if self.tier >= 3:
            self._extract_traj(fp, text, sentences)
            self._extract_echo(fp, text)
        
        # Extended features (always extract what we can)
        self._extract_epis(fp, text, sentences)
        self._extract_trans(fp, text, sentences)
        self._extract_rhythm(fp, text, sentences)
        
        fp.extraction_time_ms = (time.time() - start) * 1000
        return fp
    
    # =========================================================================
    # STAT: Statistical Features (14D)
    # =========================================================================
    
    def _extract_stat(self, fp: CompleteFingerprint, text: str, words: List[str]):
        """Extract statistical features."""
        
        # Compression ratio
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > 0:
            compressed = gzip.compress(text_bytes, compresslevel=9)
            fp.stat_compression_ratio = len(compressed) / len(text_bytes)
        
        # Character entropy
        fp.stat_entropy_char = shannon_entropy(list(text.lower()))
        
        # Word entropy
        fp.stat_entropy_word = shannon_entropy(words)
        
        # Bigram entropy
        if len(words) >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            fp.stat_entropy_bigram = shannon_entropy(bigrams)
        
        # Perplexity approximation
        if fp.stat_entropy_word > 0:
            fp.stat_perplexity_approx = 2 ** fp.stat_entropy_word
        
        # Redundancy rate
        max_entropy = math.log2(len(set(words))) if words else 0
        if max_entropy > 0:
            fp.stat_redundancy_rate = 1 - (fp.stat_entropy_word / max_entropy)
        
        # Repetition rate (repeated trigrams)
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated = sum(1 for c in trigram_counts.values() if c > 1)
            fp.stat_repetition_rate = repeated / len(trigrams) if trigrams else 0
        
        # Word burstiness
        if words:
            word_positions = {}
            for i, w in enumerate(words):
                if w not in word_positions:
                    word_positions[w] = []
                word_positions[w].append(i)
            
            gaps = []
            for positions in word_positions.values():
                if len(positions) > 1:
                    gaps.extend([positions[i+1] - positions[i] for i in range(len(positions)-1)])
            
            if gaps:
                mean_gap = sum(gaps) / len(gaps)
                std_gap = math.sqrt(sum((g - mean_gap)**2 for g in gaps) / len(gaps))
                fp.stat_burstiness_words = std_gap / mean_gap if mean_gap > 0 else 0
        
        # Vocabulary metrics
        word_counts = Counter(words)
        N = len(words)
        V = len(word_counts)
        
        if N > 0 and V > 0:
            # Zipf coefficient (simplified)
            freqs = sorted(word_counts.values(), reverse=True)
            if len(freqs) >= 10:
                # Log-log slope of rank vs frequency
                ranks = list(range(1, len(freqs)+1))
                log_ranks = [math.log(r) for r in ranks[:100]]
                log_freqs = [math.log(f) for f in freqs[:100]]
                if len(log_ranks) > 1:
                    mean_r = sum(log_ranks) / len(log_ranks)
                    mean_f = sum(log_freqs) / len(log_freqs)
                    num = sum((r - mean_r) * (f - mean_f) for r, f in zip(log_ranks, log_freqs))
                    den = sum((r - mean_r)**2 for r in log_ranks)
                    fp.stat_zipf_coefficient = abs(num / den) if den > 0 else 0
            
            # Yule's K
            freq_of_freq = Counter(word_counts.values())
            m1 = N
            m2 = sum(f * (r ** 2) for r, f in freq_of_freq.items())
            fp.stat_yule_k = 10000 * (m2 - m1) / (m1 ** 2) if m1 > 0 else 0
            
            # Simpson's D
            fp.stat_simpson_d = sum(n * (n - 1) for n in word_counts.values()) / (N * (N - 1)) if N > 1 else 0
            
            # Honoré's R
            V1 = sum(1 for n in word_counts.values() if n == 1)
            if V1 > 0 and V1 < V:
                fp.stat_honore_r = 100 * math.log(N) / (1 - V1/V)
            
            # Sichel's S (dis legomena / V)
            V2 = sum(1 for n in word_counts.values() if n == 2)
            fp.stat_sichel_s = V2 / V if V > 0 else 0
            
            # Brunet's W
            if N > 0 and V > 0:
                fp.stat_brunet_w = N ** (V ** -0.165)
    
    # =========================================================================
    # LEX: Lexical Features (12D)
    # =========================================================================
    
    def _extract_lex(self, fp: CompleteFingerprint, text: str, words: List[str]):
        """Extract lexical features."""
        if not words:
            return
        
        word_counts = Counter(words)
        N = len(words)
        V = len(word_counts)
        
        # Type-token ratio
        fp.lex_type_token_ratio = V / N
        
        # Hapax legomena ratio
        V1 = sum(1 for n in word_counts.values() if n == 1)
        fp.lex_hapax_legomena_ratio = V1 / N
        
        # Dis legomena ratio
        V2 = sum(1 for n in word_counts.values() if n == 2)
        fp.lex_dis_legomena_ratio = V2 / N
        
        # Word length statistics
        lengths = [len(w) for w in words]
        fp.lex_avg_word_length = sum(lengths) / len(lengths)
        
        if len(lengths) > 1:
            mean_len = fp.lex_avg_word_length
            fp.lex_word_length_std = math.sqrt(sum((l - mean_len)**2 for l in lengths) / len(lengths))
            fp.lex_word_length_cv = fp.lex_word_length_std / mean_len if mean_len > 0 else 0
        
        # Syllable ratio
        syllables = [syllable_count(w) for w in words]
        fp.lex_syllable_ratio = sum(syllables) / len(syllables)
        
        # Long word ratio
        fp.lex_long_word_ratio = sum(1 for w in words if len(w) > 6) / N
        
        # Rare word ratio
        fp.lex_rare_word_ratio = sum(1 for w in words if w not in TOP_1000_WORDS) / N
        
        # Concrete/abstract (simplified heuristics)
        # Concrete: common nouns, specific words
        # Abstract: -tion, -ness, -ity endings
        abstract_endings = ('tion', 'ness', 'ity', 'ism', 'ment')
        fp.lex_abstract_ratio = sum(1 for w in words if w.endswith(abstract_endings)) / N
        fp.lex_concrete_ratio = 1 - fp.lex_abstract_ratio  # Simplified
        
        # Formality (simplified: more nouns/adj = more formal)
        # Would need POS tags for accurate version
        fp.lex_formality_score = 0.5  # Placeholder for tier 0
    
    # =========================================================================
    # PUNCT: Punctuation Features (8D)
    # =========================================================================
    
    def _extract_punct(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract punctuation features."""
        n_sent = max(len(sentences), 1)
        n_words = max(len(text.split()), 1)
        
        # Question rate
        fp.punct_question_rate = text.count('?') / n_sent
        
        # Exclamation rate
        fp.punct_exclamation_rate = text.count('!') / n_sent
        
        # Em-dash rate
        em_dashes = text.count('—') + text.count('--') + text.count(' - ')
        fp.punct_em_dash_rate = em_dashes / n_words
        
        # Semicolon rate
        fp.punct_semicolon_rate = text.count(';') / n_sent
        
        # Colon rate
        fp.punct_colon_rate = text.count(':') / n_sent
        
        # Parenthetical rate
        fp.punct_parenthetical_rate = (text.count('(') + text.count(')')) / 2 / n_sent
        
        # Ellipsis rate
        fp.punct_ellipsis_rate = (text.count('...') + text.count('…')) / n_sent
        
        # Punctuation entropy
        punct_chars = re.findall(r'[.!?,;:\-\(\)\[\]"\'…—]', text)
        fp.punct_entropy = shannon_entropy(punct_chars)
    
    # =========================================================================
    # STRUCT: Structural Features (10D)
    # =========================================================================
    
    def _extract_struct(self, fp: CompleteFingerprint, text: str, words: List[str], 
                        sentences: List[str], paragraphs: List[str]):
        """Extract structural features."""
        
        # Sentence length statistics
        sent_lengths = [len(s.split()) for s in sentences]
        if sent_lengths:
            fp.struct_sentence_length_mean = sum(sent_lengths) / len(sent_lengths)
            if len(sent_lengths) > 1:
                mean = fp.struct_sentence_length_mean
                fp.struct_sentence_length_std = math.sqrt(
                    sum((l - mean)**2 for l in sent_lengths) / len(sent_lengths)
                )
                fp.struct_sentence_length_cv = cv(sent_lengths)
        
        # Paragraph length CV
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            fp.struct_paragraph_length_cv = cv(para_lengths)
        
        # Paragraph count normalized
        n_words = max(len(words), 1)
        fp.struct_paragraph_count_norm = len(paragraphs) / (n_words / 1000)
        
        # List marker rate
        list_markers = len(re.findall(r'(?:^|\n)\s*[-*•]\s|(?:^|\n)\s*\d+\.\s', text))
        fp.struct_list_marker_rate = list_markers / max(len(sentences), 1)
        
        # Header rate (lines that look like headers)
        lines = text.split('\n')
        headers = sum(1 for line in lines if line.strip() and 
                     (line.strip().startswith('#') or 
                      (len(line.strip().split()) <= 8 and line.strip().endswith(':')) or
                      line.isupper()))
        fp.struct_header_rate = headers / max(len(paragraphs), 1)
        
        # Sentence initial entropy
        if sentences:
            first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
            fp.struct_sentence_initial_entropy = shannon_entropy(first_words)
        
        # Sentence final entropy
        if sentences:
            last_words = [s.split()[-1].lower() if s.split() else '' for s in sentences]
            fp.struct_sentence_final_entropy = shannon_entropy(last_words)
        
        # Quote rate
        quotes = len(re.findall(r'["\'].*?["\']', text))
        fp.struct_quote_rate = quotes / max(len(sentences), 1)
    
    # =========================================================================
    # CONN: Connector Features (8D)
    # =========================================================================
    
    def _extract_conn(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract connector features."""
        text_lower = text.lower()
        n_sent = max(len(sentences), 1)
        
        # Coordinating conjunctions
        coord_conj = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
        coord_count = sum(len(re.findall(rf'\b{c}\b', text_lower)) for c in coord_conj)
        
        # Subordinating conjunctions
        subord_conj = ['because', 'although', 'while', 'if', 'when', 'unless', 'since',
                       'before', 'after', 'whereas', 'though', 'until', 'whenever',
                       'wherever', 'whether', 'as if', 'even though', 'so that']
        subord_count = sum(len(re.findall(rf'\b{c}\b', text_lower)) for c in subord_conj)
        
        # Coord/subord ratio
        fp.conn_coord_subord_ratio = coord_count / max(subord_count, 1)
        
        # And/but ratio
        and_count = len(re.findall(r'\band\b', text_lower))
        but_count = len(re.findall(r'\bbut\b', text_lower))
        fp.conn_and_but_ratio = and_count / max(but_count, 1)
        
        # Rates
        fp.conn_coord_conj_rate = coord_count / n_sent
        fp.conn_subord_conj_rate = subord_count / n_sent
        fp.conn_however_rate = len(re.findall(r'\bhowever\b', text_lower)) / n_sent
        fp.conn_therefore_rate = len(re.findall(r'\btherefore\b', text_lower)) / n_sent
        fp.conn_moreover_rate = len(re.findall(r'\bmoreover\b', text_lower)) / n_sent
        fp.conn_furthermore_rate = len(re.findall(r'\bfurthermore\b', text_lower)) / n_sent
    
    # =========================================================================
    # POS: Part-of-Speech Features (12D)
    # =========================================================================
    
    def _extract_pos(self, fp: CompleteFingerprint, doc):
        """Extract POS features from spaCy doc."""
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if not tokens:
            return
        
        n = len(tokens)
        pos_counts = Counter(t.pos_ for t in tokens)
        
        # Basic ratios
        fp.pos_noun_ratio = (pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)) / n
        fp.pos_verb_ratio = pos_counts.get('VERB', 0) / n
        fp.pos_adj_ratio = pos_counts.get('ADJ', 0) / n
        fp.pos_adv_ratio = pos_counts.get('ADV', 0) / n
        fp.pos_pron_ratio = pos_counts.get('PRON', 0) / n
        fp.pos_det_ratio = pos_counts.get('DET', 0) / n
        
        # Derived ratios
        nouns = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        verbs = pos_counts.get('VERB', 0)
        adjs = pos_counts.get('ADJ', 0)
        advs = pos_counts.get('ADV', 0)
        
        fp.pos_adj_noun_ratio = adjs / max(nouns, 1)
        fp.pos_adv_verb_ratio = advs / max(verbs, 1)
        fp.pos_noun_verb_ratio = nouns / max(verbs, 1)
        fp.pos_propn_ratio = pos_counts.get('PROPN', 0) / max(nouns, 1)
        
        # POS bigram entropy
        if len(tokens) >= 2:
            bigrams = [(tokens[i].pos_, tokens[i+1].pos_) for i in range(len(tokens)-1)]
            fp.pos_bigram_entropy = shannon_entropy(bigrams)
        
        # POS trigram entropy
        if len(tokens) >= 3:
            trigrams = [(tokens[i].pos_, tokens[i+1].pos_, tokens[i+2].pos_) 
                       for i in range(len(tokens)-2)]
            fp.pos_trigram_entropy = shannon_entropy(trigrams)
        
        # Update formality score with real POS data
        fp.lex_formality_score = (
            (nouns + adjs + pos_counts.get('ADP', 0)) - 
            (pos_counts.get('PRON', 0) + verbs + advs)
        ) / n
    
    # =========================================================================
    # DEP: Dependency Features (6D)
    # =========================================================================
    
    def _extract_dep(self, fp: CompleteFingerprint, doc):
        """Extract dependency features from spaCy doc."""
        tokens = list(doc)
        if not tokens:
            return
        
        # Dependency distances
        distances = [abs(t.i - t.head.i) for t in tokens if t.dep_ != 'ROOT']
        if distances:
            fp.dep_avg_distance = sum(distances) / len(distances)
            fp.dep_max_distance = max(distances)
            mean_d = fp.dep_avg_distance
            fp.dep_distance_std = math.sqrt(
                sum((d - mean_d)**2 for d in distances) / len(distances)
            )
        
        # Tree depth
        depths = [get_tree_depth(t) for t in tokens]
        if depths:
            fp.dep_tree_depth_mean = sum(depths) / len(depths)
        
        # Passive voice
        n_sent = len(list(doc.sents))
        passive_count = sum(1 for t in tokens if t.dep_ == 'nsubjpass')
        fp.dep_passive_voice_rate = passive_count / max(n_sent, 1)
        
        # Relative clauses
        relcl_count = sum(1 for t in tokens if t.dep_ == 'relcl')
        fp.dep_relative_clause_rate = relcl_count / max(n_sent, 1)
    
    # =========================================================================
    # TRAJ: Trajectory Features (5D)
    # =========================================================================
    
    def _extract_traj(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract trajectory features using embeddings."""
        if not self.embedder or len(sentences) < 3:
            return
        
        try:
            # Get embeddings
            embeddings = self.embedder.encode(sentences)
            
            # Compute jumps
            jumps = []
            for i in range(len(embeddings) - 1):
                dist = np.linalg.norm(embeddings[i+1] - embeddings[i])
                jumps.append(dist)
            
            if jumps:
                fp.traj_concept_jump_mean = float(np.mean(jumps))
                fp.traj_concept_jump_variance = float(np.var(jumps))
            
            # Path tortuosity
            total_path = sum(jumps) if jumps else 0
            straight_line = np.linalg.norm(embeddings[-1] - embeddings[0])
            fp.traj_path_tortuosity = total_path / max(straight_line, 0.001)
            
            # Turning angles
            if len(embeddings) >= 3:
                angles = []
                for i in range(len(embeddings) - 2):
                    v1 = embeddings[i+1] - embeddings[i]
                    v2 = embeddings[i+2] - embeddings[i+1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
                fp.traj_turning_angle_mean = float(np.mean(angles))
            
            # Return rate
            n_returns = 0
            threshold = fp.traj_concept_jump_mean * 0.5 if fp.traj_concept_jump_mean > 0 else 0.1
            for i in range(len(embeddings)):
                for j in range(i + 2, len(embeddings)):
                    if np.linalg.norm(embeddings[j] - embeddings[i]) < threshold:
                        n_returns += 1
                        break
            fp.traj_return_rate = n_returns / max(len(embeddings), 1)
            
        except Exception as e:
            print(f"[TRAJ] Error: {e}")
    
    # =========================================================================
    # ECHO: Echo Features (15D)
    # =========================================================================
    
    def _extract_echo(self, fp: CompleteFingerprint, text: str):
        """Extract echo features."""
        # Simplified echo extraction without full SpecHO infrastructure
        # Full version would use clause boundary detection
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return
        
        try:
            # Phonetic similarity (simplified: shared word endings)
            phonetic_scores = []
            for i in range(len(sentences) - 1):
                words1 = sentences[i].lower().split()
                words2 = sentences[i+1].lower().split()
                if words1 and words2:
                    endings1 = set(w[-3:] for w in words1 if len(w) >= 3)
                    endings2 = set(w[-3:] for w in words2 if len(w) >= 3)
                    overlap = len(endings1 & endings2) / max(len(endings1 | endings2), 1)
                    phonetic_scores.append(overlap)
            
            if phonetic_scores:
                fp.echo_phonetic_mean = float(np.mean(phonetic_scores))
                fp.echo_phonetic_std = float(np.std(phonetic_scores))
                fp.echo_phonetic_max = float(max(phonetic_scores))
            
            # Structural similarity (simplified: sentence length similarity)
            struct_scores = []
            for i in range(len(sentences) - 1):
                len1 = len(sentences[i].split())
                len2 = len(sentences[i+1].split())
                similarity = 1 - abs(len1 - len2) / max(len1 + len2, 1)
                struct_scores.append(similarity)
            
            if struct_scores:
                fp.echo_structural_mean = float(np.mean(struct_scores))
                fp.echo_structural_std = float(np.std(struct_scores))
                fp.echo_structural_max = float(max(struct_scores))
            
            # Semantic similarity (requires embeddings)
            if self.embedder:
                embeddings = self.embedder.encode(sentences)
                semantic_scores = []
                for i in range(len(embeddings) - 1):
                    cos_sim = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                    )
                    semantic_scores.append(cos_sim)
                
                if semantic_scores:
                    fp.echo_semantic_mean = float(np.mean(semantic_scores))
                    fp.echo_semantic_std = float(np.std(semantic_scores))
                    fp.echo_semantic_max = float(max(semantic_scores))
            
            # Combined metrics
            all_scores = phonetic_scores + struct_scores
            if all_scores:
                fp.echo_overall_mean = float(np.mean(all_scores))
                fp.echo_overall_std = float(np.std(all_scores))
                fp.echo_overall_max = float(max(all_scores))
                
                # Burstiness
                if fp.echo_overall_mean > 0:
                    fp.echo_burstiness = fp.echo_overall_std / fp.echo_overall_mean
                
                # Geometric mean
                fp.echo_geometric_mean = float(np.exp(np.mean(np.log(np.array(all_scores) + 1e-8))))
                
                # Cooccurrence (all three high)
                if phonetic_scores and struct_scores:
                    threshold = 0.3
                    cooccur = sum(1 for p, s in zip(phonetic_scores, struct_scores) 
                                 if p > threshold and s > threshold)
                    fp.echo_cooccurrence_rate = cooccur / len(phonetic_scores)
                    
        except Exception as e:
            print(f"[ECHO] Error: {e}")
    
    # =========================================================================
    # Extended: EPIS (6D)
    # =========================================================================
    
    def _extract_epis(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract epistemic features."""
        text_lower = text.lower()
        n_sent = max(len(sentences), 1)
        
        # Hedge detection
        hedge_count = 0
        hedge_positions = []
        
        for hedge in HEDGE_WORDS:
            matches = list(re.finditer(rf'\b{re.escape(hedge)}\b', text_lower))
            hedge_count += len(matches)
            hedge_positions.extend(m.start() / max(len(text), 1) for m in matches)
        
        fp.epis_hedge_density = hedge_count / n_sent
        
        # Hedge clustering (Gini coefficient of positions)
        if len(hedge_positions) > 1:
            sorted_pos = sorted(hedge_positions)
            n = len(sorted_pos)
            cumsum = sum((i + 1) * p for i, p in enumerate(sorted_pos))
            fp.epis_hedge_clustering = (2 * cumsum) / (n * sum(sorted_pos)) - (n + 1) / n if sum(sorted_pos) > 0 else 0
        
        # Position bias
        if hedge_positions:
            fp.epis_hedge_position_bias = sum(hedge_positions) / len(hedge_positions)
        
        # Confidence (simplified: absence of hedges)
        fp.epis_confidence_mean = 1 - min(fp.epis_hedge_density, 1)
    
    # =========================================================================
    # Extended: TRANS (6D)
    # =========================================================================
    
    def _extract_trans(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract transition features."""
        text_lower = text.lower()
        n_sent = max(len(sentences), 1)
        
        for category, words in TRANSITIONS.items():
            count = sum(len(re.findall(rf'\b{re.escape(w)}\b', text_lower)) for w in words)
            rate = count / n_sent
            setattr(fp, f'trans_{category}_rate', rate)
    
    # =========================================================================
    # Extended: RHYTHM (7D)
    # =========================================================================
    
    def _extract_rhythm(self, fp: CompleteFingerprint, text: str, sentences: List[str]):
        """Extract rhythm features."""
        n_words = max(len(text.split()), 1)
        n_sent = max(len(sentences), 1)
        
        # Clause approximation (using commas and semicolons as boundaries)
        clauses = re.split(r'[,;]', text)
        clauses = [c.strip() for c in clauses if c.strip()]
        
        if clauses:
            clause_lengths = [len(c.split()) for c in clauses]
            fp.rhythm_clause_length_mean = sum(clause_lengths) / len(clause_lengths)
            if len(clause_lengths) > 1:
                mean = fp.rhythm_clause_length_mean
                fp.rhythm_clause_length_std = math.sqrt(
                    sum((l - mean)**2 for l in clause_lengths) / len(clause_lengths)
                )
                
                # Autocorrelation
                if len(clause_lengths) > 2:
                    mean_c = sum(clause_lengths) / len(clause_lengths)
                    var_c = sum((l - mean_c)**2 for l in clause_lengths) / len(clause_lengths)
                    if var_c > 0:
                        autocov = sum(
                            (clause_lengths[i] - mean_c) * (clause_lengths[i+1] - mean_c)
                            for i in range(len(clause_lengths) - 1)
                        ) / (len(clause_lengths) - 1)
                        fp.rhythm_autocorr = autocov / var_c
        
        # Sentence complexity (clauses per sentence)
        fp.rhythm_sentence_complexity = len(clauses) / n_sent
        
        # Punctuation densities
        fp.rhythm_comma_density = text.count(',') / (n_words / 100)
        fp.rhythm_semicolon_rate = text.count(';') / n_sent
        fp.rhythm_parenthetical_rate = (text.count('(') + text.count(')')) / 2 / n_sent


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test the complete extractor."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Complete 89D Feature Extractor')
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('-f', '--file', help='Read from file')
    parser.add_argument('-t', '--tier', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Feature tier (0=basic, 3=full)')
    parser.add_argument('--json', action='store_true', help='JSON output')
    args = parser.parse_args()
    
    # Get text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Demo
        print("=" * 70)
        print("COMPLETE 89D EXTRACTOR DEMO")
        print("=" * 70)
        
        extractor = CompleteExtractor(tier=0)
        
        ai_text = """
        First, we must examine the underlying assumptions and evaluate the evidence.
        Moreover, it is important to note that these considerations are significant.
        Additionally, the methodology ensures rigor and validity. Furthermore, we
        can synthesize our findings and present a comprehensive conclusion.
        """
        
        human_text = """
        So I've been thinking about this—what if we tried something different?
        It's weird, but the whole thing doesn't add up. Like, on one hand you've
        got X, but then there's evidence for Y. I don't know. Maybe I'm overthinking?
        """
        
        print("\n--- AI TEXT (Tier 0: 44D) ---")
        fp_ai = extractor.extract(ai_text)
        print(f"Extraction time: {fp_ai.extraction_time_ms:.1f}ms")
        print(f"Sentence length CV: {fp_ai.struct_sentence_length_cv:.3f}")
        print(f"Compression ratio: {fp_ai.stat_compression_ratio:.3f}")
        print(f"And/but ratio: {fp_ai.conn_and_but_ratio:.2f}")
        print(f"Question rate: {fp_ai.punct_question_rate:.2f}")
        
        print("\n--- HUMAN TEXT (Tier 0: 44D) ---")
        fp_human = extractor.extract(human_text)
        print(f"Extraction time: {fp_human.extraction_time_ms:.1f}ms")
        print(f"Sentence length CV: {fp_human.struct_sentence_length_cv:.3f}")
        print(f"Compression ratio: {fp_human.stat_compression_ratio:.3f}")
        print(f"And/but ratio: {fp_human.conn_and_but_ratio:.2f}")
        print(f"Question rate: {fp_human.punct_question_rate:.2f}")
        
        print(f"\nVector size (tier 0): {len(fp_human.to_vector_tier0())}D")
        print(f"Vector size (full): {len(fp_human.to_vector_89d())}D")
        print(f"Vector size (extended): {len(fp_human.to_vector_extended())}D")
        
        return
    
    # Extract
    extractor = CompleteExtractor(tier=args.tier)
    fp = extractor.extract(text)
    
    if args.json:
        print(json.dumps(fp.to_dict(), indent=2))
    else:
        print(f"Extraction tier: {fp.extraction_tier}")
        print(f"Extraction time: {fp.extraction_time_ms:.1f}ms")
        print(f"Text length: {fp.text_length} chars, {fp.word_count} words")
        print(f"\nVector (tier 0): {len(fp.to_vector_tier0())}D")
        print(f"Vector (89D): {len(fp.to_vector_89d())}D")
        print(f"Vector (extended): {len(fp.to_vector_extended())}D")


if __name__ == '__main__':
    main()
