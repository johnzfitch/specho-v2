"""
Layer B: SpecHO Echo Features (15D)
===================================

Extracts echo patterns at clause boundaries:
- Phonetic: Sound rhythm similarities (3D)
- Structural: POS/syntax patterns (3D)
- Semantic: Meaning bridging (3D)
- Co-occurrence: Multi-dimension alignment (2D)
- Distribution: Aggregate statistics (3D)
- Burstiness: Human signature (1D)

These features have LOW zero rates - critical for avoiding false positives.
The original 24D system was missing these, causing 27.6% false positive rate.
"""

import sys
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path


# Feature names for this layer
LAYER_B_FEATURE_NAMES = [
    "phonetic_mean", "phonetic_std", "phonetic_max",
    "structural_mean", "structural_std", "structural_max",
    "semantic_mean", "semantic_std", "semantic_max",
    "cooccurrence_rate", "geometric_mean",
    "overall_mean", "overall_std", "overall_max",
    "burstiness",
]


@dataclass
class EchoFeatures:
    """15D SpecHO echo feature vector."""
    
    # Phonetic (3)
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    phonetic_max: float = 0.0
    
    # Structural (3)
    structural_mean: float = 0.0
    structural_std: float = 0.0
    structural_max: float = 0.0
    
    # Semantic (3)
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    semantic_max: float = 0.0
    
    # Co-occurrence (2)
    cooccurrence_rate: float = 0.0
    geometric_mean: float = 0.0
    
    # Distribution (3)
    overall_mean: float = 0.0
    overall_std: float = 0.0
    overall_max: float = 0.0
    
    # Human signature (1)
    burstiness: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 15D numpy array."""
        return np.array([
            self.phonetic_mean, self.phonetic_std, self.phonetic_max,
            self.structural_mean, self.structural_std, self.structural_max,
            self.semantic_mean, self.semantic_std, self.semantic_max,
            self.cooccurrence_rate, self.geometric_mean,
            self.overall_mean, self.overall_std, self.overall_max,
            self.burstiness,
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def feature_names(self) -> List[str]:
        return LAYER_B_FEATURE_NAMES
    
    def zero_count(self) -> int:
        """Count zero values."""
        return int(np.sum(self.to_vector() == 0))


class EchoExtractor:
    """
    Extract 15D SpecHO echo features.
    
    Usage:
        extractor = EchoExtractor()
        features = extractor.extract(text)
        vec = features.to_vector()  # 15D numpy array
    """
    
    def __init__(
        self, 
        semantic_model: str = "all-MiniLM-L6-v2",
        specHO_path: Optional[str] = None
    ):
        """
        Initialize the echo extractor.
        
        Args:
            semantic_model: Sentence transformer model for semantic similarity
            specHO_path: Path to specHO_core directory (auto-detected if None)
        """
        self.detector = None
        self.semantic_model = semantic_model
        
        # Try to load SpecHO detector
        self._init_detector(specHO_path)
    
    def _init_detector(self, specHO_path: Optional[str] = None):
        """Initialize the SpecHO detector."""
        # Find src directory (where detector.py lives)
        search_paths = [
            specHO_path,
            str(Path(__file__).parent.parent),  # src/
            str(Path(__file__).parent.parent.parent / 'src'),  # ~/dev/specho-v2/src
        ]
        
        for path in search_paths:
            if path and Path(path).exists():
                sys.path.insert(0, str(Path(path).parent))
                break
        
        try:
            from detector import SpecHODetector
            
            # CRITICAL: Pass the semantic model path to fix fallback mode
            self.detector = SpecHODetector(
                semantic_model_path=self.semantic_model
            )
            print(f"[EchoExtractor] SpecHO detector initialized with {self.semantic_model}")
            
        except ImportError as e:
            print(f"[EchoExtractor] SpecHO not available: {e}")
            print("[EchoExtractor] Using fallback extraction")
            self.detector = None
        except Exception as e:
            print(f"[EchoExtractor] Error initializing detector: {e}")
            self.detector = None
    
    def extract(self, text: str) -> EchoFeatures:
        """
        Extract echo features from text.
        
        Args:
            text: Input text (should have multiple clauses/sentences)
            
        Returns:
            EchoFeatures with 15 dimensions
        """
        if self.detector is None:
            return self._extract_fallback(text)
        
        try:
            result = self.detector.analyze(text)
            return self._extract_from_result(result)
        except Exception as e:
            print(f"[EchoExtractor] Extraction error: {e}")
            return self._extract_fallback(text)
    
    def _extract_from_result(self, result) -> EchoFeatures:
        """Extract features from SpecHO analysis result."""
        # Get pair-level scores
        pairs = getattr(result, 'pair_analyses', [])
        
        if not pairs or len(pairs) == 0:
            # No pairs found - return minimal features based on overall scores
            return EchoFeatures(
                phonetic_mean=getattr(result, 'phonetic_score', 0.5),
                structural_mean=getattr(result, 'structural_score', 0.5),
                semantic_mean=getattr(result, 'semantic_score', 0.5),
                overall_mean=getattr(result, 'final_score', 0.5),
            )
        
        # Extract per-dimension scores
        phonetic = [getattr(p, 'phonetic_score', 0.5) for p in pairs]
        structural = [getattr(p, 'structural_score', 0.5) for p in pairs]
        semantic = [getattr(p, 'semantic_score', 0.5) for p in pairs]
        
        # Combined scores (average of three dimensions)
        combined = [(p + s + m) / 3 for p, s, m in 
                    zip(phonetic, structural, semantic)]
        
        # Co-occurrence: all three dimensions above threshold
        threshold = 0.6
        cooccur = sum(1 for p, s, m in zip(phonetic, structural, semantic)
                     if p > threshold and s > threshold and m > threshold)
        
        # Geometric mean of three dimensions
        geometric = []
        for p, s, m in zip(phonetic, structural, semantic):
            # Avoid log(0) issues
            product = max(0.001, p) * max(0.001, s) * max(0.001, m)
            geometric.append(pow(product, 1/3))
        
        # Burstiness: coefficient of variation (human signature)
        mean_score = np.mean(combined) if combined else 0
        std_score = np.std(combined) if len(combined) > 1 else 0
        burstiness = std_score / mean_score if mean_score > 0 else 0
        
        return EchoFeatures(
            # Phonetic
            phonetic_mean=float(np.mean(phonetic)),
            phonetic_std=float(np.std(phonetic)) if len(phonetic) > 1 else 0.0,
            phonetic_max=float(max(phonetic)),
            
            # Structural
            structural_mean=float(np.mean(structural)),
            structural_std=float(np.std(structural)) if len(structural) > 1 else 0.0,
            structural_max=float(max(structural)),
            
            # Semantic
            semantic_mean=float(np.mean(semantic)),
            semantic_std=float(np.std(semantic)) if len(semantic) > 1 else 0.0,
            semantic_max=float(max(semantic)),
            
            # Co-occurrence
            cooccurrence_rate=float(cooccur / len(pairs)),
            geometric_mean=float(np.mean(geometric)),
            
            # Distribution
            overall_mean=float(mean_score),
            overall_std=float(std_score),
            overall_max=float(max(combined)),
            
            # Human signature
            burstiness=float(burstiness),
        )
    
    def _extract_fallback(self, text: str) -> EchoFeatures:
        """
        Fallback extraction using basic text analysis.
        
        Used when SpecHO detector is not available.
        Still produces reasonable features for structural patterns.
        """
        import re
        from collections import Counter
        
        # Split into sentences/clauses
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return EchoFeatures()
        
        # Basic phonetic: syllable patterns (approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiou'
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)
        
        phonetic_scores = []
        for i in range(len(sentences) - 1):
            words1 = sentences[i].split()
            words2 = sentences[i + 1].split()
            if words1 and words2:
                # Compare syllable patterns at boundary
                end_syl = count_syllables(words1[-1]) if words1 else 0
                start_syl = count_syllables(words2[0]) if words2 else 0
                # Similarity based on rhythm match
                score = 1.0 - abs(end_syl - start_syl) / max(end_syl, start_syl, 1)
                phonetic_scores.append(score)
        
        # Basic structural: sentence length patterns
        structural_scores = []
        lengths = [len(s.split()) for s in sentences]
        for i in range(len(lengths) - 1):
            # Similarity based on length ratio
            ratio = min(lengths[i], lengths[i + 1]) / max(lengths[i], lengths[i + 1], 1)
            structural_scores.append(ratio)
        
        # Basic semantic: word overlap
        semantic_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                semantic_scores.append(overlap)
        
        # Combine scores
        if not phonetic_scores:
            phonetic_scores = [0.5]
        if not structural_scores:
            structural_scores = [0.5]
        if not semantic_scores:
            semantic_scores = [0.5]
        
        combined = [(p + s + m) / 3 for p, s, m in 
                    zip(phonetic_scores, structural_scores, semantic_scores)]
        
        mean_score = np.mean(combined)
        std_score = np.std(combined) if len(combined) > 1 else 0
        
        return EchoFeatures(
            phonetic_mean=float(np.mean(phonetic_scores)),
            phonetic_std=float(np.std(phonetic_scores)) if len(phonetic_scores) > 1 else 0.0,
            phonetic_max=float(max(phonetic_scores)),
            
            structural_mean=float(np.mean(structural_scores)),
            structural_std=float(np.std(structural_scores)) if len(structural_scores) > 1 else 0.0,
            structural_max=float(max(structural_scores)),
            
            semantic_mean=float(np.mean(semantic_scores)),
            semantic_std=float(np.std(semantic_scores)) if len(semantic_scores) > 1 else 0.0,
            semantic_max=float(max(semantic_scores)),
            
            cooccurrence_rate=0.0,  # Can't compute without thresholds
            geometric_mean=float(np.mean([
                pow(max(0.001, p * s * m), 1/3) 
                for p, s, m in zip(phonetic_scores, structural_scores, semantic_scores)
            ])),
            
            overall_mean=float(mean_score),
            overall_std=float(std_score),
            overall_max=float(max(combined)),
            
            burstiness=float(std_score / mean_score) if mean_score > 0 else 0.0,
        )


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    """Test the echo extractor."""
    print("=" * 70)
    print("LAYER B: ECHO EXTRACTOR TEST")
    print("=" * 70)
    
    extractor = EchoExtractor()
    
    # Test on AI-like text
    ai_text = """
    First, we must examine the underlying assumptions. Second, we need to
    evaluate the available evidence. Third, we should consider alternative
    approaches. Finally, we can synthesize our findings into a coherent
    conclusion that addresses the original question.
    """
    
    print("\n--- AI-LIKE TEXT ---")
    features = extractor.extract(ai_text)
    print(f"Phonetic:   μ={features.phonetic_mean:.3f} σ={features.phonetic_std:.3f} max={features.phonetic_max:.3f}")
    print(f"Structural: μ={features.structural_mean:.3f} σ={features.structural_std:.3f} max={features.structural_max:.3f}")
    print(f"Semantic:   μ={features.semantic_mean:.3f} σ={features.semantic_std:.3f} max={features.semantic_max:.3f}")
    print(f"Cooccur:    rate={features.cooccurrence_rate:.3f} geo={features.geometric_mean:.3f}")
    print(f"Overall:    μ={features.overall_mean:.3f} σ={features.overall_std:.3f} max={features.overall_max:.3f}")
    print(f"Burstiness: {features.burstiness:.3f}")
    print(f"Zero count: {features.zero_count()}/15")
    
    # Test on human-like text
    human_text = """
    So I was thinking about this yesterday—you know how it is. Sometimes 
    ideas just come to you out of nowhere. Anyway, the point is... wait, 
    what was I saying? Oh right! The whole thing about patterns. It's weird
    how they show up everywhere once you start looking.
    """
    
    print("\n--- HUMAN-LIKE TEXT ---")
    features = extractor.extract(human_text)
    print(f"Phonetic:   μ={features.phonetic_mean:.3f} σ={features.phonetic_std:.3f} max={features.phonetic_max:.3f}")
    print(f"Structural: μ={features.structural_mean:.3f} σ={features.structural_std:.3f} max={features.structural_max:.3f}")
    print(f"Semantic:   μ={features.semantic_mean:.3f} σ={features.semantic_std:.3f} max={features.semantic_max:.3f}")
    print(f"Cooccur:    rate={features.cooccurrence_rate:.3f} geo={features.geometric_mean:.3f}")
    print(f"Overall:    μ={features.overall_mean:.3f} σ={features.overall_std:.3f} max={features.overall_max:.3f}")
    print(f"Burstiness: {features.burstiness:.3f}")
    print(f"Zero count: {features.zero_count()}/15")
    
    print("\n--- VECTOR OUTPUT ---")
    vec = features.to_vector()
    print(f"Shape: {vec.shape}")
    print(f"Vector: {vec}")


if __name__ == '__main__':
    main()
