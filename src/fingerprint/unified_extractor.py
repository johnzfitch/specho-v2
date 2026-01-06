"""
SpecHO Unified Feature Extractor
================================

Combines three feature sets:
1. Lightweight structural (10 features) - NEW, high signal
2. Cognitive fingerprint (24 features) - Original 39D subset
3. SpecHO echo features (15 features) - Original semantic analysis

Total: 49 features

The key insight is that features 1 and 3 have LOW zero rates,
while features 2 (cognitive) have HIGH zero rates in human text.
This caused false positives when relying only on cognitive features.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Import the lightweight features
sys.path.insert(0, str(Path(__file__).parent))
try:
    from lightweight import (
        extract_lightweight_features,
        LIGHTWEIGHT_FEATURE_NAMES,
        LightweightClassifier
    )
    LIGHTWEIGHT_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_AVAILABLE = False
    LIGHTWEIGHT_FEATURE_NAMES = []

# Import original 24D cognitive features
sys.path.insert(0, str(Path(__file__).parent.parent / 'fingerprint_39d'))
try:
    from extractor import Fingerprint39DExtractor
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False

# Import SpecHO echo features
sys.path.insert(0, str(Path(__file__).parent.parent / 'specHO_core'))
try:
    from detector import SpecHODetector
    ECHO_AVAILABLE = True
except ImportError:
    ECHO_AVAILABLE = False


# Feature group definitions
COGNITIVE_FEATURE_NAMES = [
    # Trajectory (5)
    'concept_jump_mean', 'concept_jump_variance', 'path_tortuosity',
    'turning_angle_mean', 'return_rate',
    # Epistemic (6)
    'hedge_density', 'hedge_clustering', 'hedge_position_bias',
    'confidence_mean', 'confidence_variance', 'confidence_arc',
    # Syntactic (7)
    'clause_length_mean', 'clause_length_std', 'clause_rhythm_autocorr',
    'sentence_complexity', 'comma_density', 'semicolon_rate', 'parenthetical_rate',
    # Transitions (6)
    'additive_rate', 'contrastive_rate', 'causal_rate',
    'temporal_rate', 'exemplifying_rate', 'reformulating_rate',
]

ECHO_FEATURE_NAMES = [
    # Phonetic (3)
    'phonetic_mean', 'phonetic_std', 'phonetic_max',
    # Structural (3)
    'structural_mean', 'structural_std', 'structural_max',
    # Semantic (3)
    'semantic_mean', 'semantic_std', 'semantic_max',
    # Co-occurrence (2)
    'cooccurrence_rate', 'geometric_mean',
    # Distribution (4)
    'overall_mean', 'overall_std', 'overall_max', 'burstiness',
]

ALL_FEATURE_NAMES = (
    LIGHTWEIGHT_FEATURE_NAMES +
    COGNITIVE_FEATURE_NAMES +
    ECHO_FEATURE_NAMES
)


@dataclass
class UnifiedFingerprint:
    """Complete fingerprint across all feature sets."""
    sample_id: str
    source: str
    text_length: int
    
    # Feature vectors
    lightweight: Dict[str, float]
    cognitive: Dict[str, float]
    echo: Dict[str, float]
    
    # Metadata
    extraction_time_ms: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_vector(self) -> np.ndarray:
        """Get full 49D feature vector."""
        values = []
        for name in LIGHTWEIGHT_FEATURE_NAMES:
            values.append(self.lightweight.get(name, 0.0))
        for name in COGNITIVE_FEATURE_NAMES:
            values.append(self.cognitive.get(name, 0.0))
        for name in ECHO_FEATURE_NAMES:
            values.append(self.echo.get(name, 0.0))
        return np.array(values)
    
    def to_lightweight_vector(self) -> np.ndarray:
        """Get just the 10D lightweight vector."""
        return np.array([self.lightweight.get(n, 0.0) for n in LIGHTWEIGHT_FEATURE_NAMES])
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'sample_id': self.sample_id,
            'source': self.source,
            'text_length': self.text_length,
            'lightweight': self.lightweight,
            'cognitive': self.cognitive,
            'echo': self.echo,
            'extraction_time_ms': self.extraction_time_ms,
            'errors': self.errors,
            'full_vector': self.to_vector().tolist(),
        }
    
    def zero_count(self) -> Dict[str, int]:
        """Count zeros in each feature group."""
        light_zeros = sum(1 for v in self.lightweight.values() if v == 0)
        cog_zeros = sum(1 for v in self.cognitive.values() if v == 0)
        echo_zeros = sum(1 for v in self.echo.values() if v == 0)
        return {
            'lightweight': light_zeros,
            'cognitive': cog_zeros,
            'echo': echo_zeros,
            'total': light_zeros + cog_zeros + echo_zeros
        }


class UnifiedExtractor:
    """
    Extract all features from text.
    
    Usage:
        extractor = UnifiedExtractor()
        fp = extractor.extract("Some text...", sample_id="001", source="HUMAN")
        
        # Get feature vectors
        vec_10d = fp.to_lightweight_vector()  # Just lightweight
        vec_49d = fp.to_vector()  # Full features
    """
    
    def __init__(self, use_cognitive: bool = True, use_echo: bool = True):
        """
        Args:
            use_cognitive: Include 24D cognitive features
            use_echo: Include 15D SpecHO echo features
        """
        self.use_cognitive = use_cognitive and COGNITIVE_AVAILABLE
        self.use_echo = use_echo and ECHO_AVAILABLE
        
        # Initialize extractors
        self.cognitive_extractor = None
        self.echo_detector = None
        
        if self.use_cognitive:
            try:
                self.cognitive_extractor = Fingerprint39DExtractor()
            except Exception as e:
                print(f"[UnifiedExtractor] Cognitive extractor unavailable: {e}")
                self.use_cognitive = False
        
        if self.use_echo:
            try:
                self.echo_detector = SpecHODetector()
            except Exception as e:
                print(f"[UnifiedExtractor] Echo detector unavailable: {e}")
                self.use_echo = False
        
        # Report status
        status = []
        status.append(f"Lightweight: {'✓' if LIGHTWEIGHT_AVAILABLE else '✗'}")
        status.append(f"Cognitive: {'✓' if self.use_cognitive else '✗'}")
        status.append(f"Echo: {'✓' if self.use_echo else '✗'}")
        print(f"[UnifiedExtractor] {', '.join(status)}")
    
    def extract(
        self, 
        text: str, 
        sample_id: str = "",
        source: str = "unknown"
    ) -> UnifiedFingerprint:
        """
        Extract all features from text.
        
        Args:
            text: Input text
            sample_id: Unique identifier
            source: Source label (HUMAN, GPT-4o, etc.)
            
        Returns:
            UnifiedFingerprint with all feature sets
        """
        import time
        start = time.time()
        errors = []
        
        # 1. Lightweight features (always available)
        if LIGHTWEIGHT_AVAILABLE:
            try:
                lightweight = extract_lightweight_features(text)
            except Exception as e:
                errors.append(f"lightweight: {e}")
                lightweight = {n: 0.0 for n in LIGHTWEIGHT_FEATURE_NAMES}
        else:
            lightweight = {n: 0.0 for n in LIGHTWEIGHT_FEATURE_NAMES}
        
        # 2. Cognitive features
        if self.use_cognitive and self.cognitive_extractor:
            try:
                cog_fp = self.cognitive_extractor.extract(text)
                cognitive = cog_fp.fingerprint
            except Exception as e:
                errors.append(f"cognitive: {e}")
                cognitive = {n: 0.0 for n in COGNITIVE_FEATURE_NAMES}
        else:
            cognitive = {n: 0.0 for n in COGNITIVE_FEATURE_NAMES}
        
        # 3. Echo features
        if self.use_echo and self.echo_detector:
            try:
                result = self.echo_detector.analyze(text)
                echo = self._extract_echo_features(result)
            except Exception as e:
                errors.append(f"echo: {e}")
                echo = {n: 0.0 for n in ECHO_FEATURE_NAMES}
        else:
            echo = {n: 0.0 for n in ECHO_FEATURE_NAMES}
        
        elapsed = (time.time() - start) * 1000
        
        return UnifiedFingerprint(
            sample_id=sample_id,
            source=source,
            text_length=len(text),
            lightweight=lightweight,
            cognitive=cognitive,
            echo=echo,
            extraction_time_ms=elapsed,
            errors=errors if errors else None
        )
    
    def _extract_echo_features(self, result) -> Dict[str, float]:
        """Extract 15D echo features from SpecHO analysis result."""
        features = {}
        
        # Try to get raw pair scores
        pair_scores = getattr(result, 'pair_scores', [])
        
        if pair_scores:
            phonetic = [p.phonetic_score for p in pair_scores if hasattr(p, 'phonetic_score')]
            structural = [p.structural_score for p in pair_scores if hasattr(p, 'structural_score')]
            semantic = [p.semantic_score for p in pair_scores if hasattr(p, 'semantic_score')]
            overall = [p.combined_score for p in pair_scores if hasattr(p, 'combined_score')]
            
            # Phonetic
            features['phonetic_mean'] = np.mean(phonetic) if phonetic else 0.0
            features['phonetic_std'] = np.std(phonetic) if phonetic else 0.0
            features['phonetic_max'] = max(phonetic) if phonetic else 0.0
            
            # Structural
            features['structural_mean'] = np.mean(structural) if structural else 0.0
            features['structural_std'] = np.std(structural) if structural else 0.0
            features['structural_max'] = max(structural) if structural else 0.0
            
            # Semantic
            features['semantic_mean'] = np.mean(semantic) if semantic else 0.0
            features['semantic_std'] = np.std(semantic) if semantic else 0.0
            features['semantic_max'] = max(semantic) if semantic else 0.0
            
            # Overall
            features['overall_mean'] = np.mean(overall) if overall else 0.0
            features['overall_std'] = np.std(overall) if overall else 0.0
            features['overall_max'] = max(overall) if overall else 0.0
            
            # Burstiness
            if features['overall_mean'] > 0:
                features['burstiness'] = features['overall_std'] / features['overall_mean']
            else:
                features['burstiness'] = 0.0
        else:
            # Fallback: use top-level scores from result
            features['phonetic_mean'] = getattr(result, 'phonetic_score', 0.0)
            features['phonetic_std'] = 0.0
            features['phonetic_max'] = features['phonetic_mean']
            
            features['structural_mean'] = getattr(result, 'structural_score', 0.0)
            features['structural_std'] = 0.0
            features['structural_max'] = features['structural_mean']
            
            features['semantic_mean'] = getattr(result, 'semantic_score', 0.0)
            features['semantic_std'] = 0.0
            features['semantic_max'] = features['semantic_mean']
            
            features['overall_mean'] = getattr(result, 'final_score', 0.0)
            features['overall_std'] = 0.0
            features['overall_max'] = features['overall_mean']
            features['burstiness'] = 0.0
        
        # Co-occurrence (from result if available)
        features['cooccurrence_rate'] = getattr(result, 'cooccurrence_rate', 0.0)
        features['geometric_mean'] = getattr(result, 'geometric_mean', 0.0)
        
        return features
    
    def extract_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        progress: bool = True
    ) -> List[UnifiedFingerprint]:
        """
        Extract features from multiple texts.
        
        Args:
            texts: List of texts
            sample_ids: Optional list of IDs
            sources: Optional list of source labels
            progress: Show progress bar
            
        Returns:
            List of UnifiedFingerprint objects
        """
        n = len(texts)
        sample_ids = sample_ids or [f"sample_{i}" for i in range(n)]
        sources = sources or ["unknown"] * n
        
        results = []
        for i, (text, sid, src) in enumerate(zip(texts, sample_ids, sources)):
            if progress and (i % 100 == 0 or i == n - 1):
                print(f"\r[UnifiedExtractor] Processing {i+1}/{n}...", end="", flush=True)
            
            fp = self.extract(text, sample_id=sid, source=src)
            results.append(fp)
        
        if progress:
            print()
        
        return results


def analyze_zero_distribution(fingerprints: List[UnifiedFingerprint]) -> Dict:
    """
    Analyze zero value distribution across feature groups.
    
    This is key to understanding why the 24D cognitive features
    caused false positives - they have high zero rates in human text.
    """
    by_source = {}
    
    for fp in fingerprints:
        src = fp.source
        if src not in by_source:
            by_source[src] = {
                'count': 0,
                'lightweight_zeros': 0,
                'cognitive_zeros': 0,
                'echo_zeros': 0,
            }
        
        zeros = fp.zero_count()
        by_source[src]['count'] += 1
        by_source[src]['lightweight_zeros'] += zeros['lightweight']
        by_source[src]['cognitive_zeros'] += zeros['cognitive']
        by_source[src]['echo_zeros'] += zeros['echo']
    
    # Calculate averages
    for src, data in by_source.items():
        n = data['count']
        data['avg_lightweight_zeros'] = data['lightweight_zeros'] / n
        data['avg_cognitive_zeros'] = data['cognitive_zeros'] / n
        data['avg_echo_zeros'] = data['echo_zeros'] / n
    
    return by_source


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    # Quick test
    test_text = """
    First, we must examine the underlying assumptions. Second, we need to
    evaluate the available evidence. Third, we should consider alternative
    approaches. Finally, we can synthesize our findings into a coherent
    conclusion that addresses the original question.
    """
    
    print("=" * 70)
    print("UNIFIED EXTRACTOR TEST")
    print("=" * 70)
    
    extractor = UnifiedExtractor(use_cognitive=False, use_echo=False)
    fp = extractor.extract(test_text, sample_id="test_001", source="AI")
    
    print(f"\nSample ID: {fp.sample_id}")
    print(f"Source: {fp.source}")
    print(f"Text length: {fp.text_length}")
    print(f"Extraction time: {fp.extraction_time_ms:.1f}ms")
    
    print("\nLightweight features:")
    for name, value in fp.lightweight.items():
        print(f"  {name:30}: {value:.4f}")
    
    zeros = fp.zero_count()
    print(f"\nZero counts: {zeros}")
    
    print(f"\nFull vector shape: {fp.to_vector().shape}")
