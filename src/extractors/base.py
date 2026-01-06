"""
Base Extractor Interface

All feature extractors inherit from BaseExtractor and implement:
- extract(text) -> Dict[str, float]
- feature_names -> List[str]
- group -> str (e.g., "lightweight", "pos", "connectors")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time


@dataclass
class ExtractorResult:
    """Result from a single extractor."""

    group: str
    features: Dict[str, float]
    extraction_time_ms: float
    success: bool = True
    error: Optional[str] = None

    def __getitem__(self, key: str) -> float:
        return self.features.get(key, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        # Ensure all floats are Python native for JSON serialization
        return {
            "group": self.group,
            "features": {k: float(v) for k, v in self.features.items()},
            "extraction_time_ms": float(self.extraction_time_ms),
            "success": self.success,
            "error": self.error,
        }


class BaseExtractor(ABC):
    """
    Base class for all feature extractors.

    Each extractor:
    - Belongs to a group (e.g., "lightweight", "pos", "echo")
    - Extracts a set of named features
    - Can fail gracefully without affecting others
    - Reports its own timing and status
    """

    # Subclasses must define these
    group: str = "unknown"
    dependencies: List[str] = []  # e.g., ["spacy", "sentence-transformers"]

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names this extractor produces."""
        pass

    @property
    def n_features(self) -> int:
        """Number of features this extractor produces."""
        return len(self.feature_names)

    @abstractmethod
    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """
        Internal extraction method. Subclasses implement this.

        Returns:
            Dict mapping feature names to values.
        """
        pass

    def extract(self, text: str, **kwargs) -> ExtractorResult:
        """
        Extract features with error handling and timing.

        This is the public API - always returns ExtractorResult,
        even on failure.
        """
        start = time.perf_counter()

        try:
            features = self._extract(text, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            return ExtractorResult(
                group=self.group,
                features=features,
                extraction_time_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000

            # Return zeros on failure, but mark as failed
            features = {name: 0.0 for name in self.feature_names}

            return ExtractorResult(
                group=self.group,
                features=features,
                extraction_time_ms=elapsed,
                success=False,
                error=str(e),
            )

    def test(self, text: str = None) -> Dict[str, Any]:
        """
        Test this extractor on sample text.

        Returns diagnostic info for debugging.
        """
        if text is None:
            text = """
            First, we should consider the underlying assumptions. However, this
            approach has some limitations. The evidence suggests that alternative
            methods might be more effective? Let me explain why.

            Furthermore, the data shows—quite clearly—that the traditional view
            needs revision. Because of this, we propose a new framework.
            """

        result = self.extract(text)

        return {
            "extractor": self.__class__.__name__,
            "group": self.group,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "dependencies": self.dependencies,
            "result": result.to_dict(),
            "sample_values": {k: f"{v:.4f}" for k, v in result.features.items()},
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} group={self.group} features={self.n_features}>"
