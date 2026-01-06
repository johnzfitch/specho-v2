"""
Extractor Registry

Manages all feature extractors and provides unified access:
- extract_all(text) -> all features
- extract_group(group, text) -> one group
- extract_feature(name, text) -> single feature
- test_all() -> diagnostic report

AUTO-DISCOVERY:
New extractors are automatically discovered if they:
1. Are in src/extractors/*.py
2. Inherit from BaseExtractor
3. Have group and feature_names defined

Future agents: just create a new file following the template!
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
import numpy as np

from .base import BaseExtractor, ExtractorResult


@dataclass
class CombinedResult:
    """Result from multiple extractors."""

    features: Dict[str, float] = field(default_factory=dict)
    by_group: Dict[str, ExtractorResult] = field(default_factory=dict)
    total_time_ms: float = 0.0
    n_features: int = 0
    n_failed: int = 0
    errors: List[str] = field(default_factory=list)

    def to_vector(self, feature_order: List[str] = None) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        if feature_order is None:
            feature_order = sorted(self.features.keys())
        return np.array([self.features.get(f, 0.0) for f in feature_order])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "by_group": {k: v.to_dict() for k, v in self.by_group.items()},
            "total_time_ms": self.total_time_ms,
            "n_features": self.n_features,
            "n_failed": self.n_failed,
            "errors": self.errors,
        }


class ExtractorRegistry:
    """
    Central registry for all feature extractors.

    Usage:
        registry = ExtractorRegistry()
        registry.register(LightweightExtractor())
        registry.register(POSExtractor())

        # Extract everything
        result = registry.extract_all(text)

        # Extract one group
        result = registry.extract_group("pos", text)

        # Test all extractors
        report = registry.test_all()
    """

    def __init__(self):
        self._extractors: Dict[str, BaseExtractor] = {}
        self._by_group: Dict[str, List[str]] = {}
        self._feature_to_extractor: Dict[str, str] = {}
        self._shared_nlp = None
        self._nlp_loaded = False

    def _get_shared_nlp(self):
        """Lazy-load shared spaCy model for all extractors."""
        if not self._nlp_loaded:
            try:
                import spacy
                self._shared_nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._shared_nlp = None
            self._nlp_loaded = True
        return self._shared_nlp

    def register(self, extractor: BaseExtractor) -> "ExtractorRegistry":
        """
        Register an extractor.

        Returns self for chaining:
            registry.register(A()).register(B()).register(C())
        """
        name = extractor.__class__.__name__
        group = extractor.group

        self._extractors[name] = extractor

        # Track by group
        if group not in self._by_group:
            self._by_group[group] = []
        self._by_group[group].append(name)

        # Track feature -> extractor mapping
        for feature in extractor.feature_names:
            self._feature_to_extractor[feature] = name

        return self

    def get_extractor(self, name: str) -> Optional[BaseExtractor]:
        """Get extractor by class name."""
        return self._extractors.get(name)

    def get_group_extractors(self, group: str) -> List[BaseExtractor]:
        """Get all extractors in a group."""
        names = self._by_group.get(group, [])
        return [self._extractors[n] for n in names]

    @property
    def groups(self) -> List[str]:
        """List of all groups."""
        return list(self._by_group.keys())

    @property
    def all_feature_names(self) -> List[str]:
        """List of all feature names across all extractors."""
        return list(self._feature_to_extractor.keys())

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return len(self._feature_to_extractor)

    def extract_all(self, text: str, **kwargs) -> CombinedResult:
        """
        Run all extractors and combine results.

        Failed extractors don't stop the others.
        Shares spaCy doc between extractors for performance.
        """
        result = CombinedResult()

        # Pre-compute shared spaCy doc for all spaCy-dependent extractors
        doc = None
        has_spacy_deps = any(
            "spacy" in ext.dependencies
            for ext in self._extractors.values()
        )
        if has_spacy_deps:
            nlp = self._get_shared_nlp()
            if nlp is not None:
                doc = nlp(text)

        # Pass shared doc to all extractors
        extract_kwargs = {**kwargs}
        if doc is not None:
            extract_kwargs["doc"] = doc

        for name, extractor in self._extractors.items():
            ext_result = extractor.extract(text, **extract_kwargs)

            result.by_group[name] = ext_result
            result.features.update(ext_result.features)
            result.total_time_ms += ext_result.extraction_time_ms

            if not ext_result.success:
                result.n_failed += 1
                result.errors.append(f"{name}: {ext_result.error}")

        result.n_features = len(result.features)
        return result

    def extract_group(self, group: str, text: str, **kwargs) -> CombinedResult:
        """Extract features from one group only."""
        result = CombinedResult()

        extractors = self.get_group_extractors(group)
        if not extractors:
            result.errors.append(f"Unknown group: {group}")
            return result

        # Pre-compute shared spaCy doc if any extractor needs it
        doc = None
        has_spacy_deps = any("spacy" in ext.dependencies for ext in extractors)
        if has_spacy_deps:
            nlp = self._get_shared_nlp()
            if nlp is not None:
                doc = nlp(text)

        extract_kwargs = {**kwargs}
        if doc is not None:
            extract_kwargs["doc"] = doc

        for extractor in extractors:
            ext_result = extractor.extract(text, **extract_kwargs)

            result.by_group[extractor.__class__.__name__] = ext_result
            result.features.update(ext_result.features)
            result.total_time_ms += ext_result.extraction_time_ms

            if not ext_result.success:
                result.n_failed += 1
                result.errors.append(f"{extractor.__class__.__name__}: {ext_result.error}")

        result.n_features = len(result.features)
        return result

    def extract_feature(self, feature_name: str, text: str, **kwargs) -> float:
        """Extract a single feature by name."""
        extractor_name = self._feature_to_extractor.get(feature_name)
        if not extractor_name:
            return 0.0

        extractor = self._extractors[extractor_name]
        result = extractor.extract(text, **kwargs)
        return result.features.get(feature_name, 0.0)

    def test_all(self, text: str = None) -> Dict[str, Any]:
        """
        Test all extractors and return diagnostic report.

        Useful for debugging which extractors are working.
        """
        results = {}

        for name, extractor in self._extractors.items():
            results[name] = extractor.test(text)

        # Summary
        working = sum(1 for r in results.values() if r["result"]["success"])
        failed = len(results) - working

        return {
            "summary": {
                "total_extractors": len(results),
                "working": working,
                "failed": failed,
                "total_features": self.n_features,
                "groups": self.groups,
            },
            "extractors": results,
        }

    def status(self) -> str:
        """Human-readable status string."""
        lines = [
            "SpecHO v2 Extractor Registry",
            "=" * 40,
            f"Total extractors: {len(self._extractors)}",
            f"Total features: {self.n_features}",
            "",
            "Groups:",
        ]

        for group in sorted(self._by_group.keys()):
            extractors = self._by_group[group]
            n_features = sum(
                self._extractors[e].n_features for e in extractors
            )
            lines.append(f"  {group}: {n_features}D ({len(extractors)} extractors)")

        return "\n".join(lines)


# Global registry instance
_registry: Optional[ExtractorRegistry] = None


def _discover_extractors(quiet: bool = True) -> List[Type[BaseExtractor]]:
    """
    Auto-discover all extractor classes in the extractors package.

    Scans all .py files in src/extractors/ and finds classes that:
    1. Inherit from BaseExtractor
    2. Are not BaseExtractor itself
    3. Have valid group and feature_names

    Returns:
        List of extractor classes (not instances)
    """
    discovered = []
    package_dir = Path(__file__).parent

    # Skip these files (not extractors)
    skip_files = {'__init__', 'base', 'registry'}

    # Determine the correct package name based on how we're being imported
    # Could be "extractors" or "src.extractors"
    package_name = __name__.rsplit('.', 1)[0]  # "src.extractors" -> "src.extractors"

    for py_file in package_dir.glob("*.py"):
        module_name = py_file.stem
        if module_name in skip_files or module_name.startswith('_'):
            continue

        try:
            # Import the module using the correct package path
            full_module = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module)

            # Find extractor classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseExtractor) and
                    attr is not BaseExtractor and
                    hasattr(attr, 'group') and
                    hasattr(attr, 'feature_names')):
                    discovered.append(attr)
                    if not quiet:
                        print(f"[Registry] Discovered: {attr.__name__} ({attr.group})")

        except Exception as e:
            if not quiet:
                print(f"[Registry] Failed to import {module_name}: {e}")

    return discovered


def get_registry(quiet: bool = True, auto_discover: bool = True) -> ExtractorRegistry:
    """
    Get or create the global registry with all extractors registered.

    Args:
        quiet: If True, suppress import warnings. If False, print them.
        auto_discover: If True, auto-discover extractors from files.

    Returns:
        ExtractorRegistry with all available extractors registered.

    AUTO-DISCOVERY MODE (default):
        Scans src/extractors/*.py for BaseExtractor subclasses.
        Just drop in a new extractor file and it works!

    Feature Tiers:
        Tier 0 (Zero deps, <5ms): statistical, lexical, structural, punctuation,
                                  connectors, epistemic, transitions, rhythm,
                                  lightweight, alignment, noise, dynamics, fractal
        Tier 1 (spaCy, ~50ms): pos_distribution, dependency, information, topology
        Tier 2 (transformers/tiktoken, ~100ms): trajectory, echo, tokenization
    """
    global _registry

    if _registry is None:
        _registry = ExtractorRegistry()

        if auto_discover:
            # AUTO-DISCOVERY: Find and register all extractors automatically
            extractor_classes = _discover_extractors(quiet=quiet)
            for cls in extractor_classes:
                try:
                    _registry.register(cls())
                    if not quiet:
                        print(f"[Registry] Registered: {cls.__name__}")
                except Exception as e:
                    if not quiet:
                        print(f"[Registry] Failed to register {cls.__name__}: {e}")

            if not quiet:
                print(f"[Registry] Auto-discovered {len(extractor_classes)} extractors, "
                      f"{_registry.n_features} features")
            return _registry

        # MANUAL FALLBACK: Original explicit registration
        # (kept for backwards compatibility)

        # =====================================================================
        # TIER 0: Zero dependencies, <5ms total
        # =====================================================================

        # Statistical (14D)
        try:
            from .statistical import StatisticalExtractor
            _registry.register(StatisticalExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping statistical: {e}")

        # Lexical (12D)
        try:
            from .lexical import LexicalExtractor
            _registry.register(LexicalExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping lexical: {e}")

        # Structural (10D)
        try:
            from .structural import StructuralExtractor
            _registry.register(StructuralExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping structural: {e}")

        # Punctuation (8D)
        try:
            from .punctuation import PunctuationExtractor
            _registry.register(PunctuationExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping punctuation: {e}")

        # Connectors (8D)
        try:
            from .connectors import ConnectorExtractor
            _registry.register(ConnectorExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping connectors: {e}")

        # Epistemic (6D)
        try:
            from .epistemic import EpistemicExtractor
            _registry.register(EpistemicExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping epistemic: {e}")

        # Transitions (6D)
        try:
            from .transitions import TransitionExtractor
            _registry.register(TransitionExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping transitions: {e}")

        # Rhythm (7D)
        try:
            from .rhythm import RhythmExtractor
            _registry.register(RhythmExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping rhythm: {e}")

        # Lightweight (10D) - original lightweight features
        try:
            from .lightweight import LightweightExtractor
            _registry.register(LightweightExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping lightweight: {e}")

        # Alignment/RLHF (8D) - NEW: RLHF artifact detection
        try:
            from .alignment import AlignmentExtractor
            _registry.register(AlignmentExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping alignment: {e}")

        # Noise (3D) - NEW: Human imperfection patterns
        try:
            from .noise import NoiseExtractor
            _registry.register(NoiseExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping noise: {e}")

        # Dynamics (4D) - NEW: Temporal text evolution
        try:
            from .dynamics import DynamicsExtractor
            _registry.register(DynamicsExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping dynamics: {e}")

        # Fractal (3D) - NEW: Multiscale analysis
        try:
            from .fractal import FractalExtractor
            _registry.register(FractalExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping fractal: {e}")

        # =====================================================================
        # TIER 1: spaCy dependency, ~50ms
        # =====================================================================

        # POS Distribution (12D)
        try:
            from .pos_distribution import POSDistributionExtractor
            _registry.register(POSDistributionExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping pos_distribution: {e}")

        # Dependency (6D)
        try:
            from .dependency import DependencyExtractor
            _registry.register(DependencyExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping dependency: {e}")

        # Information (4D) - NEW: Information density
        try:
            from .information import InformationExtractor
            _registry.register(InformationExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping information: {e}")

        # Topology (3D) - NEW: Concept graph structure
        try:
            from .topology import TopologyExtractor
            _registry.register(TopologyExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping topology: {e}")

        # =====================================================================
        # TIER 2: sentence-transformers/tiktoken dependency, ~100ms
        # =====================================================================

        # Trajectory (5D) - HIGHEST SIGNAL: path_tortuosity d=+1.72
        try:
            from .trajectory import TrajectoryExtractor
            _registry.register(TrajectoryExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping trajectory: {e}")

        # Echo (15D) - Core SpecHO system
        try:
            from .echo import EchoExtractor
            _registry.register(EchoExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping echo: {e}")

        # Tokenization (4D) - NEW: BPE tokenizer artifacts
        try:
            from .tokenization import TokenizationExtractor
            _registry.register(TokenizationExtractor())
        except ImportError as e:
            if not quiet:
                print(f"[Registry] Skipping tokenization: {e}")

    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
