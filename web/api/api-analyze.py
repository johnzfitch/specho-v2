#!/usr/bin/env python3
"""
SpecHO Server API - Full Analysis Endpoint

Reads JSON from stdin, outputs JSON to stdout.
Called by PHP via proc_open.

Input modes:
  - "full": Extract all features (Tier 1 + Tier 2)
  - "tier2_only": Extract only Tier 2 features (embeddings)

Usage:
  echo '{"text": "...", "mode": "full"}' | python api/analyze.py
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the full specho module
try:
    from specho_features_v2 import (
        extract_lightweight_features,
        extract_all_features,
        SpecHOClassifier,
        LIGHTWEIGHT_FEATURE_NAMES,
    )
    SPECHO_AVAILABLE = True
except ImportError:
    SPECHO_AVAILABLE = False

# Try to load pre-trained model
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'specho_model.pkl'
classifier = None

if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, 'rb') as f:
            classifier = pickle.load(f)
    except Exception:
        pass


# Reference distributions for heuristic scoring
REFERENCE_STATS = {
    'compression_ratio': {'human': 0.564, 'ai': 0.473, 'dir': -1, 'weight': 0.188},
    'sentence_length_cv': {'human': 0.822, 'ai': 0.422, 'dir': -1, 'weight': 0.267},
    'paragraph_length_cv': {'human': 0.847, 'ai': 0.493, 'dir': -1, 'weight': 0.119},
    'list_marker_rate': {'human': 0.010, 'ai': 0.107, 'dir': 1, 'weight': 0.110},
    'type_token_ratio': {'human': 0.596, 'ai': 0.481, 'dir': -1, 'weight': 0.085},
    'punctuation_entropy': {'human': 1.590, 'ai': 1.982, 'dir': 1, 'weight': 0.068},
    'hapax_rate': {'human': 0.417, 'ai': 0.355, 'dir': -1, 'weight': 0.049},
    'sentence_initial_entropy': {'human': 0.860, 'ai': 0.777, 'dir': -1, 'weight': 0.059},
    'function_word_cv': {'human': 1.420, 'ai': 1.217, 'dir': -1, 'weight': 0.027},
    'em_dash_rate': {'human': 0.283, 'ai': 0.091, 'dir': -1, 'weight': 0.029},
}

# Model fingerprint reference (from training)
MODEL_FINGERPRINTS = {
    'HUMAN': {'sentence_length_cv': 0.822, 'compression_ratio': 0.564, 'list_marker_rate': 0.010},
    'GPT_4-o': {'sentence_length_cv': 0.507, 'compression_ratio': 0.481, 'list_marker_rate': 0.214},
    'gemma-2-9b': {'sentence_length_cv': 0.462, 'compression_ratio': 0.504, 'list_marker_rate': 0.158},
    'mistral-7B': {'sentence_length_cv': 0.408, 'compression_ratio': 0.462, 'list_marker_rate': 0.022},
    'qwen-2-72B': {'sentence_length_cv': 0.385, 'compression_ratio': 0.459, 'list_marker_rate': 0.013},
    'llama-8B': {'sentence_length_cv': 0.372, 'compression_ratio': 0.462, 'list_marker_rate': 0.126},
    'yi-large': {'sentence_length_cv': 0.394, 'compression_ratio': 0.469, 'list_marker_rate': 0.108},
}


def score_heuristic(features: dict) -> dict:
    """Score features using reference distributions."""
    ai_score = 0.0
    total_weight = 0.0
    feature_signals = {}
    
    for name, value in features.items():
        if name.startswith('_') or name not in REFERENCE_STATS:
            continue
        
        ref = REFERENCE_STATS[name]
        human_mean = ref['human']
        ai_mean = ref['ai']
        direction = ref['dir']
        weight = ref['weight']
        
        if abs(ai_mean - human_mean) > 0.001:
            if direction < 0:
                signal = (human_mean - value) / (human_mean - ai_mean)
            else:
                signal = (value - human_mean) / (ai_mean - human_mean)
            signal = max(0.0, min(1.0, signal))
        else:
            signal = 0.5
        
        feature_signals[name] = signal
        ai_score += signal * weight
        total_weight += weight
    
    ai_prob = ai_score / total_weight if total_weight > 0 else 0.5
    
    if ai_prob > 0.6:
        label = 'AI'
    elif ai_prob < 0.4:
        label = 'HUMAN'
    else:
        label = 'UNCERTAIN'
    
    return {
        'label': label,
        'confidence': abs(ai_prob - 0.5) * 2,
        'human_prob': 1 - ai_prob,
        'ai_prob': ai_prob,
        'feature_signals': feature_signals,
    }


def estimate_model_probabilities(features: dict) -> dict:
    """Estimate which model likely generated the text."""
    probs = {}
    
    # Simple distance-based estimation
    for model, fingerprint in MODEL_FINGERPRINTS.items():
        distance = 0
        count = 0
        
        for feat, ref_val in fingerprint.items():
            if feat in features:
                diff = abs(features[feat] - ref_val)
                distance += diff
                count += 1
        
        if count > 0:
            avg_distance = distance / count
            # Convert distance to probability-like score
            probs[model] = max(0, 1 - avg_distance * 2)
    
    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    
    return probs


def analyze_full(text: str) -> dict:
    """Full analysis with all features."""
    if SPECHO_AVAILABLE:
        features = extract_lightweight_features(text)
        
        # Try to add Tier 2 features
        try:
            full_features = extract_all_features(text)
            features.update({k: v for k, v in full_features.items() if k not in features})
            tier = 2
        except Exception:
            tier = 1
    else:
        # Fallback: inline lightweight extraction
        features = extract_lightweight_inline(text)
        tier = 1
    
    # Score
    result = score_heuristic(features)
    result['features'] = features
    result['tier'] = tier
    
    # Model probabilities
    result['model_probabilities'] = estimate_model_probabilities(features)
    
    # Use trained classifier if available
    if classifier is not None:
        try:
            clf_result = classifier.predict_text(text)
            result['label'] = clf_result['label']
            result['confidence'] = clf_result['confidence']
            result['human_prob'] = clf_result['human_prob']
            result['ai_prob'] = clf_result['ai_prob']
            result['classifier_used'] = True
        except Exception:
            result['classifier_used'] = False
    
    return result


def analyze_tier2_only(text: str, client_features: dict) -> dict:
    """Add Tier 2 features to client-computed Tier 1."""
    features = client_features.copy()
    
    if SPECHO_AVAILABLE:
        try:
            full_features = extract_all_features(text)
            # Add only Tier 2 features
            tier2_keys = ['path_tortuosity', 'semantic_mean', 'phonetic_mean', 
                         'turning_angle_mean', 'concept_jump_mean']
            for k in tier2_keys:
                if k in full_features:
                    features[k] = full_features[k]
            tier = 2
        except Exception:
            tier = 1
    else:
        tier = 1
    
    result = score_heuristic(features)
    result['features'] = features
    result['tier'] = tier
    result['model_probabilities'] = estimate_model_probabilities(features)
    
    return result


def extract_lightweight_inline(text: str) -> dict:
    """Inline lightweight feature extraction (no imports needed)."""
    import gzip
    import re
    import math
    from collections import Counter
    
    features = {}
    
    # Compression ratio
    encoded = text.encode('utf-8')
    if len(encoded) > 0:
        compressed = gzip.compress(encoded, compresslevel=9)
        features['compression_ratio'] = len(compressed) / len(encoded)
    else:
        features['compression_ratio'] = 0
    
    # Words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Hapax rate
    if words:
        counts = Counter(words)
        hapax = sum(1 for c in counts.values() if c == 1)
        features['hapax_rate'] = hapax / len(words)
        features['type_token_ratio'] = len(counts) / len(words)
    else:
        features['hapax_rate'] = 0
        features['type_token_ratio'] = 0
    
    # Sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # Sentence length CV
    if len(sentences) >= 2:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        if mean_len > 0:
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            features['sentence_length_cv'] = (variance ** 0.5) / mean_len
        else:
            features['sentence_length_cv'] = 0
    else:
        features['sentence_length_cv'] = 0
    
    # Sentence initial entropy
    first_words = []
    for s in sentences:
        ws = s.split()
        if ws:
            first_words.append(ws[0].lower())
    
    if len(first_words) >= 2:
        fw_counts = Counter(first_words)
        total = len(first_words)
        entropy = -sum((c/total) * math.log2(c/total) for c in fw_counts.values())
        max_ent = math.log2(total)
        features['sentence_initial_entropy'] = entropy / max_ent if max_ent > 0 else 0
    else:
        features['sentence_initial_entropy'] = 0
    
    # Paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    if len(paragraphs) >= 2:
        p_lengths = [len(p.split()) for p in paragraphs]
        p_mean = sum(p_lengths) / len(p_lengths)
        if p_mean > 0:
            p_var = sum((l - p_mean) ** 2 for l in p_lengths) / len(p_lengths)
            features['paragraph_length_cv'] = (p_var ** 0.5) / p_mean
        else:
            features['paragraph_length_cv'] = 0
    else:
        features['paragraph_length_cv'] = 0
    
    # List markers
    lines = text.split('\n')
    list_count = sum(1 for l in lines if re.match(r'^\s*[\-\*\•]', l) or re.match(r'^\s*\d+[\.\)]', l))
    features['list_marker_rate'] = list_count / len(lines) if lines else 0
    
    # Punctuation entropy
    punct = re.findall(r'[.,;:!?\-\(\)\[\]"\'…—–]', text)
    if len(punct) >= 5:
        p_counts = Counter(punct)
        p_total = len(punct)
        features['punctuation_entropy'] = -sum((c/p_total) * math.log2(c/p_total) for c in p_counts.values())
    else:
        features['punctuation_entropy'] = 0
    
    # Function word CV
    function_words = ['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for',
                      'on', 'with', 'as', 'it', 'be', 'at', 'by', 'this', 'from', 'or']
    if words:
        total_words = len(words)
        freqs = [words.count(fw) / total_words for fw in function_words]
        fw_mean = sum(freqs) / len(freqs)
        if fw_mean > 0:
            fw_var = sum((f - fw_mean) ** 2 for f in freqs) / len(freqs)
            features['function_word_cv'] = (fw_var ** 0.5) / fw_mean
        else:
            features['function_word_cv'] = 0
    else:
        features['function_word_cv'] = 0
    
    # Em-dash rate
    word_count = len(text.split())
    if word_count > 0:
        em_dashes = text.count('—') + text.count('–') + text.count('--')
        features['em_dash_rate'] = (em_dashes / word_count) * 100
    else:
        features['em_dash_rate'] = 0
    
    return features


def main():
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON: {e}'}))
        return
    
    text = input_data.get('text', '')
    mode = input_data.get('mode', 'full')
    client_features = input_data.get('client_features', {})
    
    if not text:
        print(json.dumps({'error': 'No text provided'}))
        return
    
    try:
        if mode == 'tier2_only' and client_features:
            result = analyze_tier2_only(text, client_features)
        else:
            result = analyze_full(text)
        
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        result = convert(result)
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()
