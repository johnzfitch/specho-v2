"""
SpecHO Micro-Feature Extractor

Extracts granular linguistic features from EXISTING specho infrastructure:
- POS tags (already computed by POSTagger)
- Clause pairs (already computed by PairRulesEngine)  
- Dependency info (already computed by DependencyParser)

NO NEW SPACY CALLS - just mining what we already have.

These features capture low-level syntactic patterns that:
1. Can't be prompted away (they're implicit in model training data)
2. Are model-specific (each model learned from different corpora)
3. Are orthogonal to our current statistical features
"""

from collections import Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import re


@dataclass
class Token:
    """Simplified token for standalone testing. Use specHO.models.Token in production."""
    text: str
    pos_tag: str
    phonetic: str = ""
    is_content_word: bool = False
    syllable_count: int = 0


@dataclass
class ClausePair:
    """Simplified clause pair. Use specHO.models.ClausePair in production."""
    pair_type: str  # "punctuation", "conjunction", "transition"


# =============================================================================
# CATEGORY 1: POS DISTRIBUTION FEATURES
# =============================================================================

def extract_pos_ratios(tokens: List[Token]) -> Dict[str, float]:
    """
    Extract POS distribution ratios from already-tagged tokens.
    
    These capture what TYPES of words a model prefers.
    
    Hypothesis:
    - GPT-4: balanced, journalistic prose
    - Claude: more adjectives (nuanced description)
    - Llama: more verb-heavy (action-focused)
    - Humans: vary wildly by individual/genre
    """
    if not tokens:
        return {}
    
    pos_counts = Counter(t.pos_tag for t in tokens if t.pos_tag)
    total = len([t for t in tokens if t.pos_tag])
    
    if total == 0:
        return {}
    
    # Core POS categories
    nouns = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    verbs = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
    adjs = pos_counts.get('ADJ', 0)
    advs = pos_counts.get('ADV', 0)
    preps = pos_counts.get('ADP', 0)
    dets = pos_counts.get('DET', 0)
    prons = pos_counts.get('PRON', 0)
    cconjs = pos_counts.get('CCONJ', 0)  # and, but, or
    sconjs = pos_counts.get('SCONJ', 0)  # because, although, if
    
    # Avoid division by zero
    safe_nouns = nouns if nouns > 0 else 1
    safe_verbs = verbs if verbs > 0 else 1
    
    return {
        # Raw ratios (per total tokens)
        'pos_noun_ratio': nouns / total,
        'pos_verb_ratio': verbs / total,
        'pos_adj_ratio': adjs / total,
        'pos_adv_ratio': advs / total,
        'pos_prep_ratio': preps / total,
        'pos_det_ratio': dets / total,
        'pos_pron_ratio': prons / total,
        'pos_cconj_ratio': cconjs / total,
        'pos_sconj_ratio': sconjs / total,
        
        # Derived ratios (relationships)
        'pos_noun_verb_ratio': nouns / safe_verbs,
        'pos_adj_noun_ratio': adjs / safe_nouns,
        'pos_adv_verb_ratio': advs / safe_verbs,
        'pos_pron_noun_ratio': prons / safe_nouns,
        'pos_content_function_ratio': (nouns + verbs + adjs + advs) / (dets + preps + prons + 1),
    }


# =============================================================================
# CATEGORY 2: POS SEQUENCE FEATURES (BIGRAMS)
# =============================================================================

def extract_pos_bigrams(tokens: List[Token]) -> Dict[str, float]:
    """
    Extract POS bigram patterns from already-tagged tokens.
    
    These capture HOW words are combined syntactically.
    
    Hypothesis:
    - Models learn syntactic templates from training data
    - Different models prefer different modification patterns
    - ADJ-NOUN vs DET-ADJ-NOUN rates vary by model
    - These are LOW-LEVEL patterns hard to style-prompt away
    """
    if len(tokens) < 2:
        return {}
    
    # Build bigrams
    pos_tags = [t.pos_tag for t in tokens if t.pos_tag]
    if len(pos_tags) < 2:
        return {}
    
    bigrams = [(pos_tags[i], pos_tags[i+1]) for i in range(len(pos_tags)-1)]
    bigram_counts = Counter(bigrams)
    total_bigrams = len(bigrams)
    
    # Count words (exclude punctuation for normalization)
    word_count = sum(1 for t in tokens if t.pos_tag and t.pos_tag not in ['PUNCT', 'SPACE', 'SYM'])
    if word_count == 0:
        word_count = 1
    
    # Key patterns to track
    patterns = {
        # Modification patterns
        'bigram_adj_noun': bigram_counts.get(('ADJ', 'NOUN'), 0),
        'bigram_adv_verb': bigram_counts.get(('ADV', 'VERB'), 0),
        'bigram_verb_adv': bigram_counts.get(('VERB', 'ADV'), 0),
        'bigram_adv_adj': bigram_counts.get(('ADV', 'ADJ'), 0),
        
        # Determiner patterns
        'bigram_det_noun': bigram_counts.get(('DET', 'NOUN'), 0),
        'bigram_det_adj': bigram_counts.get(('DET', 'ADJ'), 0),
        
        # Noun patterns
        'bigram_noun_noun': bigram_counts.get(('NOUN', 'NOUN'), 0),  # Compounds
        'bigram_noun_verb': bigram_counts.get(('NOUN', 'VERB'), 0),
        
        # Verb patterns
        'bigram_verb_noun': bigram_counts.get(('VERB', 'NOUN'), 0),
        'bigram_aux_verb': bigram_counts.get(('AUX', 'VERB'), 0),
        
        # Preposition patterns
        'bigram_prep_noun': bigram_counts.get(('ADP', 'NOUN'), 0),
        'bigram_prep_det': bigram_counts.get(('ADP', 'DET'), 0),
        'bigram_verb_prep': bigram_counts.get(('VERB', 'ADP'), 0),
        
        # Conjunction patterns
        'bigram_cconj_noun': bigram_counts.get(('CCONJ', 'NOUN'), 0),
        'bigram_cconj_verb': bigram_counts.get(('CCONJ', 'VERB'), 0),
        'bigram_cconj_det': bigram_counts.get(('CCONJ', 'DET'), 0),
    }
    
    # Normalize per 100 words
    result = {f'{k}_rate': (v / word_count * 100) for k, v in patterns.items()}
    
    # Add bigram entropy (diversity of syntax)
    if total_bigrams > 0:
        probs = [c / total_bigrams for c in bigram_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        # Normalize by max possible entropy
        max_entropy = np.log2(total_bigrams) if total_bigrams > 1 else 1
        result['bigram_entropy_norm'] = entropy / max_entropy if max_entropy > 0 else 0
    else:
        result['bigram_entropy_norm'] = 0
    
    return result


# =============================================================================
# CATEGORY 3: CLAUSE PAIR FEATURES
# =============================================================================

def extract_clause_pair_features(pairs: List[ClausePair], sentence_count: int = 1) -> Dict[str, float]:
    """
    Extract clause pairing features from already-identified pairs.
    
    These capture how clauses are connected in the text.
    
    Hypothesis:
    - AI models overuse certain transitions (However, Therefore)
    - Humans use semicolons more than AI
    - Conjunction patterns (and vs but vs or) vary by model
    """
    if not pairs:
        return {
            'pair_punctuation_rate': 0,
            'pair_conjunction_rate': 0,
            'pair_transition_rate': 0,
            'pairs_per_sentence': 0,
        }
    
    pair_types = Counter(p.pair_type for p in pairs)
    total_pairs = len(pairs)
    safe_sentences = max(sentence_count, 1)
    
    return {
        # Distribution of pair types
        'pair_punctuation_rate': pair_types.get('punctuation', 0) / total_pairs,
        'pair_conjunction_rate': pair_types.get('conjunction', 0) / total_pairs,
        'pair_transition_rate': pair_types.get('transition', 0) / total_pairs,
        
        # Absolute rates
        'pairs_per_sentence': total_pairs / safe_sentences,
        'punctuation_pairs_per_sentence': pair_types.get('punctuation', 0) / safe_sentences,
        'conjunction_pairs_per_sentence': pair_types.get('conjunction', 0) / safe_sentences,
        'transition_pairs_per_sentence': pair_types.get('transition', 0) / safe_sentences,
    }


# =============================================================================
# CATEGORY 4: SPECIFIC CONJUNCTION/TRANSITION FEATURES
# =============================================================================

def extract_connector_features(text: str, sentence_count: int = 1) -> Dict[str, float]:
    """
    Extract specific conjunction and transition word features.
    
    Uses simple regex on raw text - no spacy needed.
    
    Hypothesis:
    - AI LOVES "However," at sentence start
    - AI overuses "and" chains
    - Humans use more varied connectors
    - Each model has favorite transitions from training data
    """
    text_lower = text.lower()
    safe_sentences = max(sentence_count, 1)
    word_count = len(text.split())
    safe_words = max(word_count, 1)
    
    # Coordinating conjunctions
    and_count = len(re.findall(r'\band\b', text_lower))
    but_count = len(re.findall(r'\bbut\b', text_lower))
    or_count = len(re.findall(r'\bor\b', text_lower))
    
    # Subordinating conjunctions
    because_count = len(re.findall(r'\bbecause\b', text_lower))
    although_count = len(re.findall(r'\balthough\b', text_lower))
    while_count = len(re.findall(r'\bwhile\b', text_lower))
    if_count = len(re.findall(r'\bif\b', text_lower))
    
    # Sentence-initial transitions (the AI telltale)
    however_initial = len(re.findall(r'(?:^|\. )However[,\s]', text, re.IGNORECASE))
    therefore_initial = len(re.findall(r'(?:^|\. )Therefore[,\s]', text, re.IGNORECASE))
    thus_initial = len(re.findall(r'(?:^|\. )Thus[,\s]', text, re.IGNORECASE))
    moreover_initial = len(re.findall(r'(?:^|\. )Moreover[,\s]', text, re.IGNORECASE))
    furthermore_initial = len(re.findall(r'(?:^|\. )Furthermore[,\s]', text, re.IGNORECASE))
    additionally_initial = len(re.findall(r'(?:^|\. )Additionally[,\s]', text, re.IGNORECASE))
    consequently_initial = len(re.findall(r'(?:^|\. )Consequently[,\s]', text, re.IGNORECASE))
    
    # In contrast, On the other hand, etc.
    in_contrast_initial = len(re.findall(r'(?:^|\. )In contrast[,\s]', text, re.IGNORECASE))
    on_other_hand = len(re.findall(r'on the other hand', text_lower))
    
    total_initial_transitions = (however_initial + therefore_initial + thus_initial + 
                                  moreover_initial + furthermore_initial + additionally_initial +
                                  consequently_initial + in_contrast_initial)
    
    return {
        # Coordinating conjunction rates (per sentence)
        'conn_and_rate': and_count / safe_sentences,
        'conn_but_rate': but_count / safe_sentences,
        'conn_or_rate': or_count / safe_sentences,
        
        # Subordinating conjunction rates
        'conn_because_rate': because_count / safe_sentences,
        'conn_although_rate': although_count / safe_sentences,
        'conn_while_rate': while_count / safe_sentences,
        'conn_if_rate': if_count / safe_sentences,
        
        # Sentence-initial transition rates (THE AI TELL)
        'trans_however_rate': however_initial / safe_sentences,
        'trans_therefore_rate': therefore_initial / safe_sentences,
        'trans_thus_rate': thus_initial / safe_sentences,
        'trans_moreover_rate': moreover_initial / safe_sentences,
        'trans_furthermore_rate': furthermore_initial / safe_sentences,
        'trans_additionally_rate': additionally_initial / safe_sentences,
        
        # Aggregate
        'trans_initial_total_rate': total_initial_transitions / safe_sentences,
        
        # Ratios
        'conn_and_but_ratio': and_count / (but_count + 1),  # Additive vs contrastive
        'conn_coord_subord_ratio': (and_count + but_count + or_count) / (because_count + although_count + while_count + if_count + 1),
    }


# =============================================================================
# CATEGORY 5: PUNCTUATION FEATURES
# =============================================================================

def extract_punctuation_features(text: str, sentence_count: int = 1) -> Dict[str, float]:
    """
    Extract punctuation usage features.
    
    Hypothesis:
    - Semicolons are underused by AI models
    - Em-dashes are a human style marker
    - AI uses fewer parentheticals
    - Colon usage varies by model
    """
    safe_sentences = max(sentence_count, 1)
    word_count = len(text.split())
    safe_words = max(word_count, 100)  # per 100 words
    
    # Count punctuation
    semicolons = text.count(';')
    colons = text.count(':')
    em_dashes = text.count('—') + text.count('--')  # Both forms
    parentheses = text.count('(')  # Opening parens = pairs
    ellipses = text.count('...') + text.count('…')
    exclamations = text.count('!')
    questions = text.count('?')
    
    return {
        # Per-sentence rates
        'punct_semicolon_rate': semicolons / safe_sentences,
        'punct_colon_rate': colons / safe_sentences,
        'punct_em_dash_rate': em_dashes / safe_sentences,
        'punct_parenthetical_rate': parentheses / safe_sentences,
        'punct_ellipsis_rate': ellipses / safe_sentences,
        'punct_exclamation_rate': exclamations / safe_sentences,
        'punct_question_rate': questions / safe_sentences,
        
        # Per-100-words rates
        'punct_semicolon_per_100': semicolons / safe_words * 100,
        'punct_em_dash_per_100': em_dashes / safe_words * 100,
    }


# =============================================================================
# CATEGORY 6: CLAUSE STRUCTURE FEATURES
# =============================================================================

def extract_clause_structure_features(clauses: List[dict]) -> Dict[str, float]:
    """
    Extract clause structure features.
    
    Takes list of clause dicts with 'clause_type' and 'tokens' keys.
    
    Hypothesis:
    - AI prefers medium complexity (2-3 clauses)
    - Humans vary wildly in clause structure
    - Independent/dependent ratio differs by model
    """
    if not clauses:
        return {
            'clause_count': 0,
            'clause_main_ratio': 0,
            'clause_subordinate_ratio': 0,
            'clause_length_mean': 0,
            'clause_length_cv': 0,
        }
    
    clause_types = Counter(c.get('clause_type', 'unknown') for c in clauses)
    clause_lengths = [len(c.get('tokens', [])) for c in clauses]
    
    total_clauses = len(clauses)
    main_count = clause_types.get('main', 0) + clause_types.get('independent', 0)
    subordinate_count = clause_types.get('subordinate', 0) + clause_types.get('dependent', 0)
    
    mean_length = np.mean(clause_lengths) if clause_lengths else 0
    std_length = np.std(clause_lengths) if len(clause_lengths) > 1 else 0
    cv_length = std_length / mean_length if mean_length > 0 else 0
    
    return {
        'clause_count': total_clauses,
        'clause_main_ratio': main_count / total_clauses,
        'clause_subordinate_ratio': subordinate_count / total_clauses,
        'clause_main_subordinate_ratio': main_count / (subordinate_count + 1),
        'clause_length_mean': mean_length,
        'clause_length_std': std_length,
        'clause_length_cv': cv_length,
        'clause_length_max': max(clause_lengths) if clause_lengths else 0,
        'clause_length_min': min(clause_lengths) if clause_lengths else 0,
    }


# =============================================================================
# MASTER EXTRACTOR
# =============================================================================

def extract_all_micro_features(
    text: str,
    tokens: Optional[List[Token]] = None,
    clause_pairs: Optional[List[ClausePair]] = None,
    clauses: Optional[List[dict]] = None,
) -> Dict[str, float]:
    """
    Extract all micro-features from text and/or pre-computed linguistic data.
    
    If tokens/clause_pairs/clauses are provided (from specho pipeline), uses them.
    Otherwise, falls back to text-only features.
    
    Returns dict of all micro-features for fingerprinting.
    """
    # Count sentences
    sentence_count = max(1, len(re.findall(r'[.!?]+', text)))
    
    features = {}
    
    # Category 1: POS ratios (needs tokens)
    if tokens:
        features.update(extract_pos_ratios(tokens))
        features.update(extract_pos_bigrams(tokens))
    
    # Category 2: Clause pair features (needs pairs)
    if clause_pairs:
        features.update(extract_clause_pair_features(clause_pairs, sentence_count))
    
    # Category 3: Connector features (text only)
    features.update(extract_connector_features(text, sentence_count))
    
    # Category 4: Punctuation features (text only)
    features.update(extract_punctuation_features(text, sentence_count))
    
    # Category 5: Clause structure (needs clauses)
    if clauses:
        features.update(extract_clause_structure_features(clauses))
    
    return features


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test with sample texts
    
    ai_text = """The rapid advancement of artificial intelligence has transformed numerous industries, 
    from healthcare to finance. Machine learning algorithms now power recommendation systems, 
    autonomous vehicles, and natural language processing applications. However, these technologies 
    raise important ethical considerations. Therefore, society must carefully evaluate the implications. 
    Additionally, regulatory frameworks need to be developed."""
    
    human_text = """So I was thinking about this AI stuff the other day—you know how it goes. 
    Everyone's talking about it; my neighbor won't shut up about ChatGPT. But here's the thing: 
    I'm not sure anyone really gets what's happening? Like, it's changing everything and nothing 
    at the same time. Wild, right?"""
    
    print("=" * 60)
    print("AI-GENERATED TEXT FEATURES")
    print("=" * 60)
    ai_features = extract_all_micro_features(ai_text)
    for k, v in sorted(ai_features.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("HUMAN TEXT FEATURES")
    print("=" * 60)
    human_features = extract_all_micro_features(human_text)
    for k, v in sorted(human_features.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("KEY DIFFERENCES (AI - Human)")
    print("=" * 60)
    for k in sorted(ai_features.keys()):
        if k in human_features:
            diff = ai_features[k] - human_features[k]
            if abs(diff) > 0.1:
                print(f"  {k}: {diff:+.4f}")
