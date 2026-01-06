"""
SpecHO Feature Extraction v2
============================

New orthogonal features for AI text detection and model fingerprinting.
Developed 2026-01-03.

Results Summary (n=464 samples, 6 models + human):
- Binary Classification: 98.6% accuracy, 92.9% human recognition
- Model ID: 74.6% accuracy (7-class)
- Best features by Cohen's d:
    - sentence_length_cv:    d=-1.47 (SLAM DUNK)
    - paragraph_length_cv:   d=-1.10 (SLAM DUNK)
    - type_token_ratio:      d=-1.07 (SLAM DUNK)
    - compression_ratio:     d=-0.99 (SLAM DUNK)
    - list_marker_rate:      d=+0.93 (SLAM DUNK)

Usage:
    from specho_features_v2 import extract_all_features, SpecHOClassifier
    
    features = extract_all_features(text)
    classifier = SpecHOClassifier()
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(text)
"""

import gzip
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

# Optional imports for Tier 2 features
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# =============================================================================
# TIER 1: High probability, trivial compute
# =============================================================================

def compression_ratio(text: str) -> float:
    """
    Measure text compressibility. Lower = more predictable = more AI-like.
    
    AI text compresses ~15-20% better than human text due to more
    predictable token sequences.
    
    Cohen's d: -0.99 (SLAM DUNK)
    
    Args:
        text: Input text
        
    Returns:
        Ratio of compressed size to original size (0.0-1.0)
    """
    encoded = text.encode('utf-8')
    if len(encoded) == 0:
        return 0.0
    compressed = gzip.compress(encoded, compresslevel=9)
    return len(compressed) / len(encoded)


def hapax_rate(text: str) -> float:
    """
    Proportion of words appearing exactly once (hapax legomena).
    Higher = more unique vocabulary = more human-like.
    
    AI vocabulary is normalized by training, humans use more rare words.
    
    Cohen's d: -0.54 (USEFUL)
    
    Args:
        text: Input text
        
    Returns:
        Ratio of hapax words to total words
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) == 0:
        return 0.0
    word_counts = Counter(words)
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    return hapax / len(words)


def sentence_initial_entropy(text: str) -> float:
    """
    Entropy of sentence-starting words. Higher = more varied starters.
    
    AI overuses "The" as sentence opener. Humans more varied.
    
    Cohen's d: -0.49 (WEAK)
    
    Args:
        text: Input text
        
    Returns:
        Normalized entropy (0.0-1.0)
    """
    sentences = re.split(r'[.!?]+', text)
    first_words = []
    for s in sentences:
        words = s.strip().split()
        if words:
            first_words.append(words[0].lower())
    
    if len(first_words) < 2:
        return 0.0
    
    counts = Counter(first_words)
    total = len(first_words)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    max_entropy = math.log2(total) if total > 1 else 1
    return entropy / max_entropy if max_entropy > 0 else 0


# Top 20 English function words
FUNCTION_WORDS = [
    'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for',
    'on', 'with', 'as', 'it', 'be', 'at', 'by', 'this', 'from', 'or'
]


def function_word_variance(text: str) -> float:
    """
    Coefficient of variation in function word usage.
    Lower = tighter distribution = more AI-like.
    
    AI has normalized function word patterns from training.
    
    Cohen's d: -0.51 (USEFUL)
    
    Args:
        text: Input text
        
    Returns:
        CV of function word frequencies
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) == 0:
        return 0.0
    
    total = len(words)
    freqs = [words.count(fw) / total for fw in FUNCTION_WORDS]
    
    mean_freq = np.mean(freqs)
    if mean_freq == 0:
        return 0.0
    
    return np.std(freqs) / mean_freq


# =============================================================================
# TIER 2: Strong signal, heavier compute
# =============================================================================

def _get_tree_depth(token) -> int:
    """Recursively get max depth from a spacy token."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_get_tree_depth(c) for c in children)


def dependency_tree_stats(text: str, nlp=None) -> Dict[str, float]:
    """
    Dependency parse tree depth statistics.
    
    AI tends toward flatter syntax (easier to generate).
    Humans nest deeper.
    
    Cohen's d: ~0.5 (MEDIUM) - varies by model
    
    Args:
        text: Input text
        nlp: Optional spacy model (will load en_core_web_sm if not provided)
        
    Returns:
        Dict with max_depth, mean_depth, depth_variance
    """
    if not SPACY_AVAILABLE:
        return {'max_depth': 0, 'mean_depth': 0, 'depth_variance': 0}
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
        nlp.max_length = 50000
    
    try:
        doc = nlp(text[:10000])  # Limit for speed
        depths = []
        for sent in doc.sents:
            root = [t for t in sent if t.head == t]
            if root:
                depths.append(_get_tree_depth(root[0]))
        
        if not depths:
            return {'max_depth': 0, 'mean_depth': 0, 'depth_variance': 0}
        
        return {
            'max_depth': float(max(depths)),
            'mean_depth': float(np.mean(depths)),
            'depth_variance': float(np.var(depths))
        }
    except Exception:
        return {'max_depth': 0, 'mean_depth': 0, 'depth_variance': 0}


def inter_sentence_similarity_decay(text: str, nlp=None) -> Dict[str, float]:
    """
    Measure semantic similarity decay between sentences at different lags.
    
    AI maintains tighter coherence across sentences.
    Humans have more varied topic drift.
    
    Cohen's d: 0.6-0.8 (USEFUL to LARGE)
    
    Args:
        text: Input text
        nlp: Optional spacy model
        
    Returns:
        Dict with decay_k1, decay_k2, decay_k3 (similarity at lag 1,2,3)
    """
    if not SPACY_AVAILABLE:
        return {'decay_k1': 0, 'decay_k2': 0, 'decay_k3': 0}
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
        nlp.max_length = 50000
    
    try:
        doc = nlp(text[:10000])
        sents = [s for s in doc.sents if len(s) > 3]
        
        if len(sents) < 4:
            return {'decay_k1': 0, 'decay_k2': 0, 'decay_k3': 0}
        
        vectors = [sent.vector for sent in sents if sent.has_vector]
        
        if len(vectors) < 4:
            return {'decay_k1': 0, 'decay_k2': 0, 'decay_k3': 0}
        
        def cosine_sim(v1, v2):
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        
        sims_k1, sims_k2, sims_k3 = [], [], []
        for i in range(len(vectors) - 3):
            sims_k1.append(cosine_sim(vectors[i], vectors[i+1]))
            sims_k2.append(cosine_sim(vectors[i], vectors[i+2]))
            sims_k3.append(cosine_sim(vectors[i], vectors[i+3]))
        
        return {
            'decay_k1': float(np.mean(sims_k1)) if sims_k1 else 0,
            'decay_k2': float(np.mean(sims_k2)) if sims_k2 else 0,
            'decay_k3': float(np.mean(sims_k3)) if sims_k3 else 0
        }
    except Exception:
        return {'decay_k1': 0, 'decay_k2': 0, 'decay_k3': 0}


# =============================================================================
# TIER 3: Worth checking, mixed results
# =============================================================================

def punctuation_entropy(text: str) -> float:
    """
    Entropy of punctuation usage. Higher = more varied punctuation.
    
    AI often has MORE varied punctuation (uses more types consistently).
    
    Cohen's d: +0.72 (USEFUL)
    
    Args:
        text: Input text
        
    Returns:
        Entropy in bits
    """
    punct = re.findall(r'[.,;:!?\-\(\)\[\]"\'…—–]', text)
    if len(punct) < 5:
        return 0.0
    
    counts = Counter(punct)
    total = len(punct)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy


def paragraph_length_variance(text: str) -> float:
    """
    Coefficient of variation in paragraph lengths.
    Lower = more uniform = more AI-like.
    
    AI produces uniform paragraphs. Humans are irregular.
    
    Cohen's d: -1.10 (SLAM DUNK)
    
    Args:
        text: Input text
        
    Returns:
        CV of paragraph word counts
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    if len(paragraphs) < 2:
        return 0.0
    
    lengths = [len(p.split()) for p in paragraphs]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0
    
    return float(np.std(lengths) / mean_len)


def sentence_length_variance(text: str) -> float:
    """
    Coefficient of variation in sentence lengths.
    Lower = more uniform = more AI-like.
    
    AI produces uniform sentence lengths. Humans vary wildly.
    
    Cohen's d: -1.47 (SLAM DUNK) - BEST NEW FEATURE
    
    Args:
        text: Input text
        
    Returns:
        CV of sentence word counts
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0
    
    return float(np.std(lengths) / mean_len)


def list_marker_rate(text: str) -> float:
    """
    Proportion of lines that are list items.
    Higher = more lists = more AI-like.
    
    AI LOVES bullet points. GPT-4o uses 21x more than humans.
    
    Cohen's d: +0.93 (SLAM DUNK)
    
    Args:
        text: Input text
        
    Returns:
        Ratio of list lines to total lines
    """
    lines = text.split('\n')
    if not lines:
        return 0.0
    
    list_markers = 0
    for line in lines:
        stripped = line.strip()
        # Bullet points: -, *, •
        if re.match(r'^[\-\*\•]', stripped):
            list_markers += 1
        # Numbered lists: 1. or 1)
        elif re.match(r'^\d+[\.\)]', stripped):
            list_markers += 1
    
    return list_markers / len(lines)


def type_token_ratio(text: str) -> float:
    """
    Ratio of unique words to total words (lexical diversity).
    Lower = more repetitive = more AI-like.
    
    AI vocabulary is normalized by training.
    
    Cohen's d: -1.07 (SLAM DUNK)
    
    Args:
        text: Input text
        
    Returns:
        TTR (0.0-1.0)
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def em_dash_rate(text: str) -> float:
    """
    Em-dash usage per 100 words.
    Most AI avoids em-dashes. GPT-4o is exception.
    
    Cohen's d: -0.44 (WEAK)
    
    Args:
        text: Input text
        
    Returns:
        Em-dashes per 100 words
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    em_dashes = text.count('—') + text.count('--') + text.count('–')
    return em_dashes / words * 100


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

# Feature names for the 10-feature lightweight extractor
LIGHTWEIGHT_FEATURE_NAMES = [
    'compression_ratio',
    'hapax_rate', 
    'sentence_initial_entropy',
    'function_word_cv',
    'punctuation_entropy',
    'paragraph_length_cv',
    'sentence_length_cv',
    'list_marker_rate',
    'type_token_ratio',
    'em_dash_rate',
]

# Feature names for full extraction (includes Tier 2)
FULL_FEATURE_NAMES = LIGHTWEIGHT_FEATURE_NAMES + [
    'max_depth',
    'mean_depth', 
    'depth_variance',
    'decay_k1',
    'decay_k2',
    'decay_k3',
]


def extract_lightweight_features(text: str) -> Dict[str, float]:
    """
    Extract 10 lightweight features (no dependencies, sub-ms).
    
    These alone achieve 98.6% binary accuracy.
    
    Args:
        text: Input text
        
    Returns:
        Dict of feature name -> value
    """
    return {
        'compression_ratio': compression_ratio(text),
        'hapax_rate': hapax_rate(text),
        'sentence_initial_entropy': sentence_initial_entropy(text),
        'function_word_cv': function_word_variance(text),
        'punctuation_entropy': punctuation_entropy(text),
        'paragraph_length_cv': paragraph_length_variance(text),
        'sentence_length_cv': sentence_length_variance(text),
        'list_marker_rate': list_marker_rate(text),
        'type_token_ratio': type_token_ratio(text),
        'em_dash_rate': em_dash_rate(text),
    }


def extract_lightweight_features_array(text: str) -> np.ndarray:
    """
    Extract lightweight features as numpy array (for sklearn).
    
    Args:
        text: Input text
        
    Returns:
        1D numpy array of features
    """
    features = extract_lightweight_features(text)
    return np.array([features[name] for name in LIGHTWEIGHT_FEATURE_NAMES])


def extract_all_features(text: str, nlp=None) -> Dict[str, float]:
    """
    Extract all 16 features (includes spacy-dependent Tier 2).
    
    Args:
        text: Input text
        nlp: Optional spacy model
        
    Returns:
        Dict of feature name -> value
    """
    features = extract_lightweight_features(text)
    
    # Add Tier 2 features
    depth_stats = dependency_tree_stats(text, nlp)
    decay_stats = inter_sentence_similarity_decay(text, nlp)
    
    features.update(depth_stats)
    features.update(decay_stats)
    
    return features


def extract_all_features_array(text: str, nlp=None) -> np.ndarray:
    """
    Extract all features as numpy array.
    
    Args:
        text: Input text
        nlp: Optional spacy model
        
    Returns:
        1D numpy array of features
    """
    features = extract_all_features(text, nlp)
    return np.array([features[name] for name in FULL_FEATURE_NAMES])


# =============================================================================
# CLASSIFIER
# =============================================================================

class SpecHOClassifier:
    """
    AI text detector using the new feature set.
    
    Usage:
        classifier = SpecHOClassifier()
        
        # Train
        X = [extract_lightweight_features_array(t) for t in texts]
        y = [0 if human else 1 for ...]
        classifier.fit(X, y)
        
        # Predict
        result = classifier.predict_text("Some text to analyze")
        print(result['label'], result['confidence'])
    """
    
    def __init__(self, n_estimators: int = 100, use_full_features: bool = False):
        """
        Args:
            n_estimators: Number of trees in random forest
            use_full_features: If True, use 16 features (requires spacy)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.use_full_features = use_full_features
        self.is_fitted = False
        self.nlp = None
        
        if use_full_features and SPACY_AVAILABLE:
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
            self.nlp.max_length = 50000
    
    def _extract(self, text: str) -> np.ndarray:
        """Extract features from text."""
        if self.use_full_features:
            return extract_all_features_array(text, self.nlp)
        return extract_lightweight_features_array(text)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=human, 1=AI)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def fit_texts(self, texts: List[str], labels: List[int]):
        """
        Fit from raw texts.
        
        Args:
            texts: List of text strings
            labels: List of labels (0=human, 1=AI)
        """
        X = np.array([self._extract(t) for t in texts])
        return self.fit(X, np.array(labels))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for feature matrix."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)
    
    def predict_text(self, text: str) -> Dict:
        """
        Predict for a single text with confidence scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with 'label', 'confidence', 'human_prob', 'ai_prob', 'features'
        """
        features = self._extract(text)
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        proba = self.clf.predict_proba(X_scaled)[0]
        pred = self.clf.predict(X_scaled)[0]
        
        return {
            'label': 'AI' if pred == 1 else 'HUMAN',
            'confidence': float(max(proba)),
            'human_prob': float(proba[0]),
            'ai_prob': float(proba[1]),
            'features': dict(zip(
                FULL_FEATURE_NAMES if self.use_full_features else LIGHTWEIGHT_FEATURE_NAMES,
                features
            ))
        }
    
    def feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importances sorted by importance."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted.")
        
        names = FULL_FEATURE_NAMES if self.use_full_features else LIGHTWEIGHT_FEATURE_NAMES
        importances = list(zip(names, self.clf.feature_importances_))
        return sorted(importances, key=lambda x: -x[1])


# =============================================================================
# TIERED CLASSIFIER (Fast path + slow verification)
# =============================================================================

class TieredSpecHOClassifier:
    """
    Two-tier classifier for production use.
    
    Tier 1 (fast): Lightweight features only
    Tier 2 (slow): Full features if Tier 1 is uncertain
    
    Usage:
        classifier = TieredSpecHOClassifier()
        classifier.fit(texts_train, labels_train)
        result = classifier.predict_text(text)
    """
    
    def __init__(self, uncertainty_threshold: float = 0.3):
        """
        Args:
            uncertainty_threshold: If |prob - 0.5| < threshold, use Tier 2
        """
        self.tier1 = SpecHOClassifier(use_full_features=False)
        self.tier2 = SpecHOClassifier(use_full_features=True) if SPACY_AVAILABLE else None
        self.uncertainty_threshold = uncertainty_threshold
    
    def fit_texts(self, texts: List[str], labels: List[int]):
        """Fit both tiers."""
        self.tier1.fit_texts(texts, labels)
        if self.tier2:
            self.tier2.fit_texts(texts, labels)
        return self
    
    def predict_text(self, text: str) -> Dict:
        """
        Predict with tiered approach.
        
        Returns dict with additional 'tier_used' field.
        """
        result = self.tier1.predict_text(text)
        result['tier_used'] = 1
        
        # Check if uncertain
        uncertainty = abs(result['ai_prob'] - 0.5)
        if uncertainty < self.uncertainty_threshold and self.tier2:
            result = self.tier2.predict_text(text)
            result['tier_used'] = 2
        
        return result


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == '__main__':
    # Demo usage
    human_text = """
    I've been thinking about this problem for weeks now, and honestly? 
    It's driving me a bit crazy. The thing is—and I know this sounds weird—
    but sometimes the best solutions come when you're not even trying.
    Like yesterday, I was just making coffee and BAM. It hit me.
    """
    
    ai_text = """
    The problem at hand requires careful consideration of multiple factors.
    First, we must examine the underlying assumptions. Second, we need to
    evaluate the available evidence. Third, we should consider alternative
    approaches. Finally, we can synthesize our findings into a coherent
    conclusion that addresses the original question.
    """
    
    print("=" * 60)
    print("SpecHO Features v2 - Demo")
    print("=" * 60)
    
    print("\n--- Human Text Features ---")
    h_features = extract_lightweight_features(human_text)
    for name, value in h_features.items():
        print(f"  {name:25}: {value:.4f}")
    
    print("\n--- AI Text Features ---")
    a_features = extract_lightweight_features(ai_text)
    for name, value in a_features.items():
        print(f"  {name:25}: {value:.4f}")
    
    print("\n--- Feature Deltas (AI - Human) ---")
    for name in h_features:
        delta = a_features[name] - h_features[name]
        direction = "↑ AI higher" if delta > 0 else "↓ AI lower"
        print(f"  {name:25}: {delta:+.4f} ({direction})")
