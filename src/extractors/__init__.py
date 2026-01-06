"""
SpecHO v2 Modular Extractor Architecture

Each extractor is independent and can be:
- Run alone: extractor.extract(text)
- Run with group: registry.extract_group("pos", text)
- Run all: registry.extract_all(text)

If one breaks, it doesn't take down the others.

Quick Start:
    from src.extractors import get_registry

    # Get all features
    registry = get_registry()
    result = registry.extract_all(text)
    print(result.features)

    # Get one group
    result = registry.extract_group("lightweight", text)

    # Get single feature
    value = registry.extract_feature("sentence_length_cv", text)

    # Test all extractors
    report = registry.test_all()
"""

from .base import BaseExtractor, ExtractorResult
from .registry import ExtractorRegistry, get_registry, reset_registry, CombinedResult

# Lazy imports for individual extractors
__all__ = [
    # Core
    "BaseExtractor",
    "ExtractorResult",
    "ExtractorRegistry",
    "CombinedResult",
    "get_registry",
    "reset_registry",
    # Extractors (importable individually)
    "LightweightExtractor",
    "POSDistributionExtractor",
    "ConnectorExtractor",
    "PunctuationExtractor",
    "EpistemicExtractor",
    "TransitionExtractor",
]


def __getattr__(name):
    """Lazy load extractors on demand."""
    if name == "LightweightExtractor":
        from .lightweight import LightweightExtractor
        return LightweightExtractor
    elif name == "POSDistributionExtractor":
        from .pos_distribution import POSDistributionExtractor
        return POSDistributionExtractor
    elif name == "ConnectorExtractor":
        from .connectors import ConnectorExtractor
        return ConnectorExtractor
    elif name == "PunctuationExtractor":
        from .punctuation import PunctuationExtractor
        return PunctuationExtractor
    elif name == "EpistemicExtractor":
        from .epistemic import EpistemicExtractor
        return EpistemicExtractor
    elif name == "TransitionExtractor":
        from .transitions import TransitionExtractor
        return TransitionExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
