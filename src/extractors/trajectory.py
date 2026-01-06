"""
Trajectory Feature Extractor (5D)

Semantic path through embedding space.

Features (sorted by Cohen's d):
- path_tortuosity (d=+1.72): AI meanders more (HIGHEST SIGNAL!)
- return_rate (d=-0.42): Humans revisit concepts
- turning_angle_mean (d=+0.30): AI makes sharper turns
- concept_jump_variance (d=+0.15): AI is metronomic
- concept_jump_mean (d=+0.09): AI takes regular steps
"""

import re
from typing import Dict, List, Optional

from .base import BaseExtractor


class TrajectoryExtractor(BaseExtractor):
    """
    Trajectory features requiring sentence-transformers.

    Measures how the text moves through semantic space -
    AI tends to meander while humans are more direct.

    Each feature is extracted independently with fault isolation.
    """

    group = "trajectory"
    dependencies = ["sentence-transformers"]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model to use.
        """
        self._model_name = model_name
        self._embedder = None
        self._loaded = False

    @property
    def embedder(self):
        """Lazy-load sentence transformer model."""
        if not self._loaded:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._model_name)
                self._loaded = True
            except Exception as e:
                raise ImportError(f"sentence-transformers not available: {e}")
        return self._embedder

    @property
    def feature_names(self) -> List[str]:
        return [
            "path_tortuosity",       # d=+1.72 HIGHEST SIGNAL!
            "concept_jump_mean",     # d=+0.09
            "concept_jump_variance", # d=+0.15
            "turning_angle_mean",    # d=+0.30
            "return_rate",           # d=-0.42
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Safely extract a single feature with error handling."""
        try:
            return func()
        except Exception:
            return default

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract trajectory features with fault isolation."""
        import numpy as np

        features = {name: 0.0 for name in self.feature_names}
        features["path_tortuosity"] = 1.0  # Default neutral

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if len(sentences) < 3:
            return features

        # Get embeddings
        try:
            embeddings = self.embedder.encode(sentences)
        except Exception:
            return features

        # 1. Compute jumps between consecutive sentences (ISOLATED)
        def calc_jumps():
            jumps = []
            for i in range(len(embeddings) - 1):
                dist = np.linalg.norm(embeddings[i+1] - embeddings[i])
                jumps.append(dist)
            return jumps

        jumps = self._safe_extract("jumps", calc_jumps, [])
        if not jumps:
            return features

        # 2. Concept jump mean (ISOLATED)
        def calc_jump_mean():
            features["concept_jump_mean"] = float(np.mean(jumps))

        self._safe_extract("jump_mean", calc_jump_mean)

        # 3. Concept jump variance (ISOLATED)
        def calc_jump_var():
            features["concept_jump_variance"] = float(np.var(jumps))

        self._safe_extract("jump_var", calc_jump_var)

        # 4. Path tortuosity (ISOLATED) - HIGHEST SIGNAL FEATURE
        def calc_tortuosity():
            total_path = sum(jumps)
            straight_line = np.linalg.norm(embeddings[-1] - embeddings[0])
            features["path_tortuosity"] = total_path / max(straight_line, 0.001)

        self._safe_extract("tortuosity", calc_tortuosity)

        # 5. Turning angles (ISOLATED)
        def calc_angles():
            if len(embeddings) >= 3:
                angles = []
                for i in range(len(embeddings) - 2):
                    v1 = embeddings[i+1] - embeddings[i]
                    v2 = embeddings[i+2] - embeddings[i+1]
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angles.append(angle)
                if angles:
                    features["turning_angle_mean"] = float(np.mean(angles))

        self._safe_extract("angles", calc_angles)

        # 6. Return rate (ISOLATED)
        def calc_return_rate():
            threshold = features["concept_jump_mean"] * 0.5 if features["concept_jump_mean"] > 0 else 0.1
            n_returns = 0
            for i in range(len(embeddings)):
                for j in range(i + 2, len(embeddings)):
                    if np.linalg.norm(embeddings[j] - embeddings[i]) < threshold:
                        n_returns += 1
                        break
            features["return_rate"] = n_returns / max(len(embeddings), 1)

        self._safe_extract("return_rate", calc_return_rate)

        return features


# Standalone function
def extract_trajectory_features(text: str, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, float]:
    """Extract trajectory features (standalone function)."""
    extractor = TrajectoryExtractor(model_name=model_name)
    result = extractor.extract(text)
    return result.features
