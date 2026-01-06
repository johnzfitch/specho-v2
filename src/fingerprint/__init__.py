"""
SpecHO 45-Dimensional Cognitive Fingerprint
============================================

Complete AI text detection system with tiered classification.

ARCHITECTURE:

Layer A (High Trust):        5D  - Trajectory geometry
Layer B (Medium-High Trust): 15D - SpecHO echo patterns [RESTORED]
Layer C (Medium Trust):      12D - Epistemic + Transitions
Layer D (Lower Trust):       7D  - Syntactic rhythm
Layer E (New):               6D  - Lightweight structural [BEST DISCRIMINATORS]

Total: 45 dimensions

TIERED CLASSIFICATION:

Tier 1 (fast):   Layer E only (6D) - <1ms, ~98% accuracy
Tier 2 (medium): Layer E + A + B (26D) - ~20ms, ~99% accuracy  
Tier 3 (full):   All 45D - ~50ms, maximum robustness

QUICK START:

    from fingerprint_39d import TieredClassifier
    
    clf = TieredClassifier()
    result = clf.predict("Your text here")
    print(f"{result.prediction} ({result.confidence:.0%})")

FULL EXTRACTION:

    from fingerprint_39d import Unified45DExtractor
    
    extractor = Unified45DExtractor()
    fp = extractor.extract(text)
    
    vec_fast = fp.to_vector_fast()    # 7D
    vec_full = fp.to_vector_45d()     # 45D
"""

# Core extractors
from .trajectory import TrajectoryAnalyzer, TrajectoryFeatures
from .layer_b_echo import EchoExtractor, EchoFeatures
from .epistemic import EpistemicAnalyzer, EpistemicFeatures
from .transitions import TransitionAnalyzer, TransitionFeatures
from .syntactic import SyntacticAnalyzer, SyntacticFeatures
from .lightweight import (
    extract_lightweight_features, 
    LightweightClassifier,
    LIGHTWEIGHT_FEATURE_NAMES,
)

# Unified systems
from .unified_45d import (
    Unified45DExtractor,
    Fingerprint45D,
    ALL_45D_NAMES,
    ALL_39D_NAMES,
    LAYER_A_NAMES,
    LAYER_B_NAMES,
    LAYER_C_EPISTEMIC_NAMES,
    LAYER_C_TRANSITION_NAMES,
    LAYER_D_NAMES,
    LAYER_E_NAMES,
)

from .tiered_classifier import TieredClassifier, ClassificationResult

# Legacy (for backwards compatibility)
from .cognitive import CognitiveExtractor, CognitiveFingerprint

__all__ = [
    # Primary interfaces
    "TieredClassifier",
    "ClassificationResult",
    "Unified45DExtractor",
    "Fingerprint45D",
    "LightweightClassifier",
    
    # Layer extractors
    "TrajectoryAnalyzer",
    "TrajectoryFeatures",
    "EchoExtractor",
    "EchoFeatures",
    "EpistemicAnalyzer",
    "EpistemicFeatures",
    "TransitionAnalyzer",
    "TransitionFeatures",
    "SyntacticAnalyzer",
    "SyntacticFeatures",
    
    # Utilities
    "extract_lightweight_features",
    
    # Constants
    "ALL_45D_NAMES",
    "ALL_39D_NAMES",
    "LAYER_A_NAMES",
    "LAYER_B_NAMES",
    "LAYER_C_EPISTEMIC_NAMES",
    "LAYER_C_TRANSITION_NAMES",
    "LAYER_D_NAMES",
    "LAYER_E_NAMES",
    "LIGHTWEIGHT_FEATURE_NAMES",
    
    # Legacy
    "CognitiveExtractor",
    "CognitiveFingerprint",
]
