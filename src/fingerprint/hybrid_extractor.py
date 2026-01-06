"""
SpecHO Hybrid Extractor (18D)
=============================

Combines the best of both worlds:
- 10D Lightweight: Structural uniformity features (fast, no zeros)
- 8D from 39D: Best cognitive/semantic features

Total: 18D features achieving >99% expected accuracy.

Architecture:
    [10D Lightweight]          [8D Selected from 39D]
    sentence_length_cv         path_tortuosity (d=-1.72)
    paragraph_length_cv        additive_rate (d=-0.89)
    type_token_ratio           temporal_rate (d=+0.68)
    compression_ratio          return_rate
    list_marker_rate           confidence_arc
    hapax_legomena_ratio       hedge_density
    sentence_initial_entropy   semantic_max
    avg_word_length            phonetic_max
    punctuation_diversity
    question_ratio
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from lightweight import (
    extract_lightweight_features,
    LIGHTWEIGHT_FEATURE_NAMES,
    LightweightClassifier
)


# Selected 8D features from 39D (highest signal)
SELECTED_39D_FEATURES = [
    'path_tortuosity',    # d=-1.72, standout trajectory feature
    'additive_rate',      # d=-0.89, AI overuses "moreover", "additionally"
    'temporal_rate',      # d=+0.68, human uses more time references
    'return_rate',        # trajectory: revisiting concepts
    'confidence_arc',     # epistemic: confidence trajectory
    'hedge_density',      # epistemic: hedging patterns
    'semantic_max',       # echo: peak semantic similarity
    'phonetic_max',       # echo: peak phonetic similarity
]

# Full hybrid feature names
HYBRID_FEATURE_NAMES = LIGHTWEIGHT_FEATURE_NAMES + SELECTED_39D_FEATURES


@dataclass
class HybridFingerprint:
    """18D hybrid fingerprint."""
    sample_id: str
    source: str
    text_length: int
    
    lightweight: Dict[str, float]  # 10D
    selected_39d: Dict[str, float]  # 8D
    
    extraction_time_ms: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Get full 18D vector."""
        light_vec = [self.lightweight.get(n, 0.0) for n in LIGHTWEIGHT_FEATURE_NAMES]
        sel_vec = [self.selected_39d.get(n, 0.0) for n in SELECTED_39D_FEATURES]
        return np.array(light_vec + sel_vec)
    
    def to_lightweight_vector(self) -> np.ndarray:
        """Get just 10D lightweight."""
        return np.array([self.lightweight.get(n, 0.0) for n in LIGHTWEIGHT_FEATURE_NAMES])
    
    @property
    def feature_names(self) -> List[str]:
        return HYBRID_FEATURE_NAMES


class HybridExtractor:
    """
    18D Hybrid feature extractor.
    
    Combines 10D lightweight (fast, structural) with 8D best from 39D (cognitive).
    
    Usage:
        extractor = HybridExtractor()
        fp = extractor.extract(text, sample_id="001", source="HUMAN")
        vector = fp.to_vector()  # 18D numpy array
    """
    
    def __init__(self, use_39d: bool = True):
        """
        Args:
            use_39d: If True, include 8D from 39D. If False, lightweight only.
        """
        self.use_39d = use_39d
        self.cognitive_extractor = None
        
        if use_39d:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / 'fingerprint_39d'))
                from extractor import Fingerprint39DExtractor
                self.cognitive_extractor = Fingerprint39DExtractor()
                print("[HybridExtractor] Loaded 39D extractor")
            except Exception as e:
                print(f"[HybridExtractor] 39D unavailable: {e}")
                self.use_39d = False
        
        status = f"18D hybrid" if self.use_39d else "10D lightweight only"
        print(f"[HybridExtractor] Mode: {status}")
    
    def extract(
        self,
        text: str,
        sample_id: str = "",
        source: str = "unknown"
    ) -> HybridFingerprint:
        """
        Extract 18D hybrid fingerprint.
        
        Args:
            text: Input text
            sample_id: Unique identifier
            source: Source label
            
        Returns:
            HybridFingerprint with both feature sets
        """
        import time
        start = time.time()
        
        # 10D lightweight (always)
        lightweight = extract_lightweight_features(text)
        
        # 8D from 39D (if available)
        selected_39d = {}
        if self.use_39d and self.cognitive_extractor:
            try:
                full_fp = self.cognitive_extractor.extract(text)
                for fname in SELECTED_39D_FEATURES:
                    selected_39d[fname] = full_fp.fingerprint.get(fname, 0.0)
            except Exception as e:
                print(f"[HybridExtractor] 39D extraction failed: {e}")
                selected_39d = {n: 0.0 for n in SELECTED_39D_FEATURES}
        else:
            selected_39d = {n: 0.0 for n in SELECTED_39D_FEATURES}
        
        elapsed = (time.time() - start) * 1000
        
        return HybridFingerprint(
            sample_id=sample_id,
            source=source,
            text_length=len(text),
            lightweight=lightweight,
            selected_39d=selected_39d,
            extraction_time_ms=elapsed
        )
    
    def extract_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        progress: bool = True
    ) -> List[HybridFingerprint]:
        """Extract from multiple texts."""
        n = len(texts)
        sample_ids = sample_ids or [f"sample_{i}" for i in range(n)]
        sources = sources or ["unknown"] * n
        
        results = []
        for i, (text, sid, src) in enumerate(zip(texts, sample_ids, sources)):
            if progress and (i % 100 == 0 or i == n - 1):
                print(f"\r[HybridExtractor] {i+1}/{n}...", end="", flush=True)
            results.append(self.extract(text, sid, src))
        
        if progress:
            print()
        
        return results


class HybridClassifier:
    """
    Classifier using 18D hybrid features.
    
    Uses tiered approach:
    1. Fast path: 10D lightweight for confident predictions
    2. Slow path: Full 18D for uncertain cases
    """
    
    def __init__(self, uncertainty_threshold: float = 0.25):
        """
        Args:
            uncertainty_threshold: If |prob - 0.5| < threshold, use full 18D
        """
        self.extractor = HybridExtractor(use_39d=True)
        self.lightweight_classifier = LightweightClassifier()
        self.uncertainty_threshold = uncertainty_threshold
        
        # Full classifier (needs training)
        self.full_classifier = None
        self.scaler = None
        self.is_fitted = False
    
    def predict_proba(self, text: str, force_full: bool = False) -> Dict[str, float]:
        """
        Get probability scores.
        
        Args:
            text: Input text
            force_full: If True, always use 18D features
            
        Returns:
            {'human': 0.xx, 'ai': 0.xx, 'tier_used': 1 or 2}
        """
        # Fast path first
        light_proba = self.lightweight_classifier.predict_proba(text)
        
        uncertainty = abs(light_proba['ai'] - 0.5)
        
        if not force_full and uncertainty >= self.uncertainty_threshold:
            return {
                'human': light_proba['human'],
                'ai': light_proba['ai'],
                'tier_used': 1
            }
        
        # Slow path for uncertain cases
        if self.is_fitted and self.full_classifier:
            fp = self.extractor.extract(text)
            vec = fp.to_vector().reshape(1, -1)
            vec_scaled = self.scaler.transform(vec)
            proba = self.full_classifier.predict_proba(vec_scaled)[0]
            return {
                'human': float(proba[0]),
                'ai': float(proba[1]),
                'tier_used': 2
            }
        else:
            # Fall back to lightweight if not trained
            return {
                'human': light_proba['human'],
                'ai': light_proba['ai'],
                'tier_used': 1
            }
    
    def predict(self, text: str) -> str:
        """Classify as 'human' or 'ai'."""
        proba = self.predict_proba(text)
        return 'human' if proba['human'] > proba['ai'] else 'ai'
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the full 18D classifier.
        
        Args:
            texts: List of text samples
            labels: 0 = human, 1 = AI
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Extract features
        print(f"[HybridClassifier] Extracting features from {len(texts)} samples...")
        fps = self.extractor.extract_batch(texts, progress=True)
        X = np.array([fp.to_vector() for fp in fps])
        y = np.array(labels)
        
        # Scale and fit
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.full_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.full_classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"[HybridClassifier] Training complete")
        return self
    
    def feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importances if fitted."""
        if not self.is_fitted:
            return []
        
        importances = list(zip(HYBRID_FEATURE_NAMES, self.full_classifier.feature_importances_))
        return sorted(importances, key=lambda x: -x[1])
    
    def analyze(self, text: str, verbose: bool = False) -> Dict:
        """
        Full analysis with feature breakdown.
        """
        fp = self.extractor.extract(text)
        proba = self.predict_proba(text, force_full=True)
        
        prediction = 'human' if proba['human'] > proba['ai'] else 'ai'
        confidence = max(proba['human'], proba['ai'])
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'human_prob': proba['human'],
            'ai_prob': proba['ai'],
            'tier_used': proba['tier_used'],
            'lightweight_features': fp.lightweight,
            'selected_39d_features': fp.selected_39d,
            'extraction_time_ms': fp.extraction_time_ms,
        }
        
        if verbose:
            result['full_vector'] = fp.to_vector().tolist()
        
        return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """Quick test."""
    print("=" * 70)
    print("HYBRID EXTRACTOR TEST (18D = 10D + 8D)")
    print("=" * 70)
    
    test_ai = """
    First, we must examine the underlying assumptions. Second, we need to
    evaluate the available evidence. Third, we should consider alternative
    approaches. Finally, we can synthesize our findings into a coherent
    conclusion. This methodology ensures comprehensive analysis.
    """
    
    test_human = """
    So I've been thinking about this a lot lately. And honestly? I'm not
    sure what to make of it. The whole thing is... weird, I guess. Like,
    on one hand you've got people saying X, but then on the other hand
    there's all this evidence for Y. It's confusing.
    """
    
    # Test lightweight only (fast)
    extractor_light = HybridExtractor(use_39d=False)
    
    print("\n--- LIGHTWEIGHT ONLY (10D) ---")
    fp1 = extractor_light.extract(test_ai, source="AI")
    print(f"AI sample: {fp1.extraction_time_ms:.1f}ms")
    print(f"Vector shape: {fp1.to_vector().shape}")
    
    fp2 = extractor_light.extract(test_human, source="HUMAN")
    print(f"Human sample: {fp2.extraction_time_ms:.1f}ms")
    
    # Test classifier
    print("\n--- CLASSIFIER TEST ---")
    classifier = HybridClassifier()
    
    result_ai = classifier.analyze(test_ai)
    print(f"AI text → {result_ai['prediction'].upper()} ({result_ai['confidence']:.1%})")
    
    result_human = classifier.analyze(test_human)
    print(f"Human text → {result_human['prediction'].upper()} ({result_human['confidence']:.1%})")


if __name__ == '__main__':
    main()
