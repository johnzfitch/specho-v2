"""
Tiered Classification System
============================

Three-tier approach for production:

Tier 1 (fast):   Layer E only (6D) - <1ms, ~98% accuracy
Tier 2 (medium): Layer E + A + B (26D) - ~20ms, ~99% accuracy  
Tier 3 (full):   All 45D - ~50ms, maximum robustness

Uses confidence thresholds to escalate through tiers.
This minimizes latency while maximizing accuracy.
"""

import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ClassificationResult:
    """Result from tiered classification."""
    prediction: str           # 'human' or 'ai'
    confidence: float         # 0.0 - 1.0
    tier_used: int            # 1, 2, or 3
    human_prob: float
    ai_prob: float
    latency_ms: float = 0.0
    
    # Detailed breakdown (optional)
    tier1_proba: Optional[Dict] = None
    tier2_proba: Optional[Dict] = None
    tier3_proba: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'tier_used': self.tier_used,
            'human_prob': self.human_prob,
            'ai_prob': self.ai_prob,
            'latency_ms': self.latency_ms,
        }


class TieredClassifier:
    """
    Three-tier classification for production use.
    
    Usage:
        clf = TieredClassifier()
        
        # Quick classification
        result = clf.predict("Some text...")
        print(f"{result.prediction} ({result.confidence:.0%}), tier {result.tier_used}")
        
        # With trained model
        clf.load_model("specho_model.pkl")
        result = clf.predict("Some text...")
    """
    
    def __init__(
        self,
        tier1_threshold: float = 0.75,   # Stop at tier 1 if confidence >= this
        tier2_threshold: float = 0.85,   # Stop at tier 2 if confidence >= this
    ):
        """
        Initialize tiered classifier.
        
        Args:
            tier1_threshold: Confidence needed to stop at tier 1
            tier2_threshold: Confidence needed to stop at tier 2
        """
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        
        # Tier 1: Lightweight (rule-based, always available)
        self._tier1_clf = None
        
        # Tier 2/3: ML classifiers (need training)
        self._extractor = None
        self._tier2_clf = None
        self._tier3_clf = None
        self._tier2_scaler = None
        self._tier3_scaler = None
        
        self._is_fitted = False
    
    @property
    def tier1(self):
        """Lazy load tier 1 classifier."""
        if self._tier1_clf is None:
            from lightweight import LightweightClassifier
            self._tier1_clf = LightweightClassifier()
        return self._tier1_clf
    
    @property
    def extractor(self):
        """Lazy load 45D extractor."""
        if self._extractor is None:
            from unified_45d import Unified45DExtractor
            self._extractor = Unified45DExtractor(lazy_load=True)
        return self._extractor
    
    def predict(self, text: str, force_tier: Optional[int] = None) -> ClassificationResult:
        """
        Classify text using tiered approach.
        
        Args:
            text: Input text to classify
            force_tier: If set, use only this tier (1, 2, or 3)
            
        Returns:
            ClassificationResult with prediction and metadata
        """
        import time
        start = time.time()
        
        tier1_result = None
        tier2_result = None
        tier3_result = None
        
        # --- TIER 1: Lightweight (always runs) ---
        tier1_proba = self.tier1.predict_proba(text)
        tier1_conf = max(tier1_proba['human'], tier1_proba['ai'])
        tier1_result = {
            'human': tier1_proba['human'],
            'ai': tier1_proba['ai'],
            'confidence': tier1_conf,
        }
        
        if force_tier == 1 or (force_tier is None and tier1_conf >= self.tier1_threshold):
            elapsed = (time.time() - start) * 1000
            return ClassificationResult(
                prediction='human' if tier1_proba['human'] > tier1_proba['ai'] else 'ai',
                confidence=tier1_conf,
                tier_used=1,
                human_prob=tier1_proba['human'],
                ai_prob=tier1_proba['ai'],
                latency_ms=elapsed,
                tier1_proba=tier1_result,
            )
        
        # --- TIER 2/3: Need trained models ---
        if not self._is_fitted:
            # No trained model - fall back to tier 1
            elapsed = (time.time() - start) * 1000
            return ClassificationResult(
                prediction='human' if tier1_proba['human'] > tier1_proba['ai'] else 'ai',
                confidence=tier1_conf,
                tier_used=1,
                human_prob=tier1_proba['human'],
                ai_prob=tier1_proba['ai'],
                latency_ms=elapsed,
                tier1_proba=tier1_result,
            )
        
        # Extract 45D features
        fp = self.extractor.extract(text)
        
        # --- TIER 2: 26D (E + A + B) ---
        X_tier2 = fp.to_vector_medium().reshape(1, -1)
        X_tier2_scaled = self._tier2_scaler.transform(X_tier2)
        tier2_proba_arr = self._tier2_clf.predict_proba(X_tier2_scaled)[0]
        tier2_conf = float(max(tier2_proba_arr))
        tier2_result = {
            'human': float(tier2_proba_arr[0]),
            'ai': float(tier2_proba_arr[1]),
            'confidence': tier2_conf,
        }
        
        if force_tier == 2 or (force_tier is None and tier2_conf >= self.tier2_threshold):
            elapsed = (time.time() - start) * 1000
            return ClassificationResult(
                prediction='human' if tier2_proba_arr[0] > tier2_proba_arr[1] else 'ai',
                confidence=tier2_conf,
                tier_used=2,
                human_prob=float(tier2_proba_arr[0]),
                ai_prob=float(tier2_proba_arr[1]),
                latency_ms=elapsed,
                tier1_proba=tier1_result,
                tier2_proba=tier2_result,
            )
        
        # --- TIER 3: Full 45D ---
        X_tier3 = fp.to_vector_45d().reshape(1, -1)
        X_tier3_scaled = self._tier3_scaler.transform(X_tier3)
        tier3_proba_arr = self._tier3_clf.predict_proba(X_tier3_scaled)[0]
        tier3_conf = float(max(tier3_proba_arr))
        tier3_result = {
            'human': float(tier3_proba_arr[0]),
            'ai': float(tier3_proba_arr[1]),
            'confidence': tier3_conf,
        }
        
        elapsed = (time.time() - start) * 1000
        return ClassificationResult(
            prediction='human' if tier3_proba_arr[0] > tier3_proba_arr[1] else 'ai',
            confidence=tier3_conf,
            tier_used=3,
            human_prob=float(tier3_proba_arr[0]),
            ai_prob=float(tier3_proba_arr[1]),
            latency_ms=elapsed,
            tier1_proba=tier1_result,
            tier2_proba=tier2_result,
            tier3_proba=tier3_result,
        )
    
    def predict_batch(
        self, 
        texts: List[str], 
        force_tier: Optional[int] = None,
        progress: bool = True,
    ) -> List[ClassificationResult]:
        """Classify multiple texts."""
        results = []
        n = len(texts)
        
        for i, text in enumerate(texts):
            if progress and (i % 50 == 0 or i == n - 1):
                print(f"\r[TieredClassifier] {i+1}/{n}...", end="", flush=True)
            results.append(self.predict(text, force_tier=force_tier))
        
        if progress:
            print()
        
        return results
    
    def fit(
        self, 
        texts: List[str], 
        labels: List[int],
        n_estimators: int = 100,
    ):
        """
        Train tier 2 and tier 3 classifiers.
        
        Args:
            texts: List of text samples
            labels: 0 = human, 1 = AI
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        print(f"[TieredClassifier] Training on {len(texts)} samples...")
        
        # Extract 45D features
        fps = self.extractor.extract_batch(texts, progress=True)
        
        # Prepare data
        X_tier2 = np.array([fp.to_vector_medium() for fp in fps])  # 26D
        X_tier3 = np.array([fp.to_vector_45d() for fp in fps])     # 45D
        y = np.array(labels)
        
        # Scale
        self._tier2_scaler = StandardScaler()
        self._tier3_scaler = StandardScaler()
        
        X_tier2_scaled = self._tier2_scaler.fit_transform(X_tier2)
        X_tier3_scaled = self._tier3_scaler.fit_transform(X_tier3)
        
        # Train
        print("[TieredClassifier] Training tier 2 (26D)...")
        self._tier2_clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42, 
            n_jobs=-1
        )
        self._tier2_clf.fit(X_tier2_scaled, y)
        
        print("[TieredClassifier] Training tier 3 (45D)...")
        self._tier3_clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42, 
            n_jobs=-1
        )
        self._tier3_clf.fit(X_tier3_scaled, y)
        
        self._is_fitted = True
        print("[TieredClassifier] Training complete")
        
        return self
    
    def save_model(self, path: str):
        """Save trained model to file."""
        if not self._is_fitted:
            raise ValueError("Model not fitted - call fit() first")
        
        data = {
            'tier1_threshold': self.tier1_threshold,
            'tier2_threshold': self.tier2_threshold,
            'tier2_clf': self._tier2_clf,
            'tier3_clf': self._tier3_clf,
            'tier2_scaler': self._tier2_scaler,
            'tier3_scaler': self._tier3_scaler,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[TieredClassifier] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.tier1_threshold = data['tier1_threshold']
        self.tier2_threshold = data['tier2_threshold']
        self._tier2_clf = data['tier2_clf']
        self._tier3_clf = data['tier3_clf']
        self._tier2_scaler = data['tier2_scaler']
        self._tier3_scaler = data['tier3_scaler']
        self._is_fitted = True
        
        print(f"[TieredClassifier] Model loaded from {path}")
        return self
    
    def feature_importance(self, tier: int = 3) -> List[Tuple[str, float]]:
        """Get feature importances for specified tier."""
        if not self._is_fitted:
            return []
        
        from unified_45d import ALL_45D_NAMES, LAYER_E_NAMES, LAYER_A_NAMES, LAYER_B_NAMES
        
        if tier == 2:
            clf = self._tier2_clf
            names = LAYER_E_NAMES + LAYER_A_NAMES + LAYER_B_NAMES
        elif tier == 3:
            clf = self._tier3_clf
            names = ALL_45D_NAMES
        else:
            return []
        
        importances = list(zip(names, clf.feature_importances_))
        return sorted(importances, key=lambda x: -x[1])


# =============================================================================
# LIGHTWEIGHT CLASSIFIER (TIER 1 - ALWAYS AVAILABLE)
# =============================================================================

class LightweightClassifier:
    """
    Rule-based classifier using reference statistics.
    
    No training required - uses pre-computed population statistics.
    Achieves ~98% accuracy on its own.
    """
    
    # Reference statistics from 464-sample validation
    REFERENCE_STATS = {
        'sentence_length_cv': {
            'human_mean': 0.72, 'ai_mean': 0.45, 'cohens_d': -1.47,
            'human_std': 0.22, 'ai_std': 0.15,
        },
        'paragraph_length_cv': {
            'human_mean': 0.65, 'ai_mean': 0.38, 'cohens_d': -1.10,
            'human_std': 0.25, 'ai_std': 0.20,
        },
        'type_token_ratio': {
            'human_mean': 0.68, 'ai_mean': 0.56, 'cohens_d': -1.07,
            'human_std': 0.12, 'ai_std': 0.10,
        },
        'compression_ratio': {
            'human_mean': 0.42, 'ai_mean': 0.35, 'cohens_d': -0.99,
            'human_std': 0.08, 'ai_std': 0.06,
        },
        'list_marker_rate': {
            'human_mean': 0.002, 'ai_mean': 0.025, 'cohens_d': 0.93,
            'human_std': 0.01, 'ai_std': 0.04,
        },
        'hapax_legomena_ratio': {
            'human_mean': 0.52, 'ai_mean': 0.43, 'cohens_d': -0.75,
            'human_std': 0.12, 'ai_std': 0.10,
        },
    }
    
    FEATURE_NAMES = list(REFERENCE_STATS.keys())
    
    def __init__(self):
        """Initialize with reference statistics."""
        pass
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get probability scores.
        
        Uses z-score deviation from reference populations.
        """
        from lightweight import extract_lightweight_features
        
        features = extract_lightweight_features(text)
        
        # Calculate weighted likelihood ratio
        human_score = 0.0
        ai_score = 0.0
        total_weight = 0.0
        
        for fname in self.FEATURE_NAMES:
            if fname not in features:
                continue
            
            value = features[fname]
            stats = self.REFERENCE_STATS[fname]
            
            # Z-scores relative to each population
            h_z = abs(value - stats['human_mean']) / max(stats['human_std'], 0.001)
            a_z = abs(value - stats['ai_mean']) / max(stats['ai_std'], 0.001)
            
            # Weight by Cohen's d magnitude
            weight = abs(stats['cohens_d'])
            
            # Lower z-score = more likely from that population
            human_score += weight * (1.0 / (1.0 + h_z))
            ai_score += weight * (1.0 / (1.0 + a_z))
            total_weight += weight
        
        if total_weight == 0:
            return {'human': 0.5, 'ai': 0.5}
        
        # Normalize to probabilities
        human_score /= total_weight
        ai_score /= total_weight
        total = human_score + ai_score
        
        return {
            'human': human_score / total,
            'ai': ai_score / total,
        }
    
    def predict(self, text: str) -> str:
        """Classify as 'human' or 'ai'."""
        proba = self.predict_proba(text)
        return 'human' if proba['human'] > proba['ai'] else 'ai'


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test the tiered classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SpecHO Tiered Classifier")
    parser.add_argument('text', nargs='?', help='Text to classify')
    parser.add_argument('--file', '-f', help='Read text from file')
    parser.add_argument('--model', '-m', help='Load trained model')
    parser.add_argument('--tier', '-t', type=int, choices=[1, 2, 3], 
                       help='Force specific tier')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        import sys
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            # Demo mode
            print("=" * 70)
            print("TIERED CLASSIFIER DEMO")
            print("=" * 70)
            
            clf = TieredClassifier()
            
            ai_text = """
            First, we must examine the underlying assumptions. Second, we need to
            evaluate the evidence. Third, we should consider alternatives. Finally,
            we can synthesize our findings into a coherent conclusion.
            """
            
            human_text = """
            So I've been thinking about this a lot. And honestly? I'm not sure 
            what to make of it. The whole thing is weird. Like, on one hand you've
            got X, but then there's all this evidence for Y. It's confusing.
            """
            
            print("\n--- AI TEXT ---")
            result = clf.predict(ai_text)
            print(f"Prediction: {result.prediction.upper()}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Tier used: {result.tier_used}")
            print(f"Latency: {result.latency_ms:.1f}ms")
            
            print("\n--- HUMAN TEXT ---")
            result = clf.predict(human_text)
            print(f"Prediction: {result.prediction.upper()}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Tier used: {result.tier_used}")
            print(f"Latency: {result.latency_ms:.1f}ms")
            
            return
    
    # Create classifier
    clf = TieredClassifier()
    
    if args.model:
        clf.load_model(args.model)
    
    # Classify
    result = clf.predict(text, force_tier=args.tier)
    
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"Prediction: {result.prediction.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Human prob: {result.human_prob:.1%}")
        print(f"AI prob:    {result.ai_prob:.1%}")
        print(f"Tier used:  {result.tier_used}")
        print(f"Latency:    {result.latency_ms:.1f}ms")
        
        if args.verbose and result.tier1_proba:
            print(f"\nTier 1 breakdown: h={result.tier1_proba['human']:.1%} a={result.tier1_proba['ai']:.1%}")
            if result.tier2_proba:
                print(f"Tier 2 breakdown: h={result.tier2_proba['human']:.1%} a={result.tier2_proba['ai']:.1%}")
            if result.tier3_proba:
                print(f"Tier 3 breakdown: h={result.tier3_proba['human']:.1%} a={result.tier3_proba['ai']:.1%}")


if __name__ == '__main__':
    main()
