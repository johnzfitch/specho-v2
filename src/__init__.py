"""
SpecHO v2 Unified AI Text Detection System

Complete model fingerprinting with auto-storage and verification.

Main Classes:
- SpecHO: Integrated interface (analyze, verify, identify)
- SpecHOUnified: Low-level 77D fingerprint extractor
- FingerprintStore: Auto-accumulating reference database
- UnifiedFingerprint: Complete fingerprint data structure

NEW: Modular Extractor Architecture
- get_registry(): Access all feature extractors
- Registry allows testing features individually, by group, or all at once
- 44 features across 6 groups (lightweight, connectors, punctuation, epistemic, transitions, pos)

Quick Start (Extractors):
    from src.extractors import get_registry

    registry = get_registry()
    result = registry.extract_all("Your text here")
    print(result.features)  # All 44 features

    # By group
    lightweight = registry.extract_group("lightweight", text)

    # Single feature
    value = registry.extract_feature("sentence_length_cv", text)

Quick Start (Fingerprinting):
    from src import SpecHO

    specho = SpecHO("data/fingerprints")
    result = specho.analyze("Some text", model_id="claude-sonnet-4")
    verification = specho.verify("Unknown text", claimed_model="claude-sonnet-4")
"""

from .specho_unified import (
    SpecHOUnified,
    UnifiedFingerprint,
    CognitiveFingerprint39D,
    MicroDistributions,
    PairTypeStats,
    ClauseTypeStats,
    ZoneStats,
    EchoDistribution,
)

from .fingerprint_store import (
    FingerprintStore,
    ModelProfile,
    StoredFingerprint,
)

from .specHO_integrated import (
    SpecHO,
    AnalysisResult,
    VerificationResult,
    TrustLevel,
    analyze,
    verify,
    identify,
    get_specho,
)

# Modular extractors
from .extractors import (
    get_registry,
    reset_registry,
    ExtractorRegistry,
    BaseExtractor,
    ExtractorResult,
    CombinedResult,
)

__version__ = "2.0.0"
__all__ = [
    # High-level API
    "SpecHO",
    "analyze",
    "verify",
    "identify",
    "get_specho",

    # Results
    "AnalysisResult",
    "VerificationResult",
    "TrustLevel",

    # Low-level extractor
    "SpecHOUnified",
    "UnifiedFingerprint",
    "CognitiveFingerprint39D",
    "MicroDistributions",

    # Storage
    "FingerprintStore",
    "ModelProfile",
    "StoredFingerprint",

    # Data structures
    "PairTypeStats",
    "ClauseTypeStats",
    "ZoneStats",
    "EchoDistribution",

    # Modular extractors (NEW)
    "get_registry",
    "reset_registry",
    "ExtractorRegistry",
    "BaseExtractor",
    "ExtractorResult",
    "CombinedResult",
]
