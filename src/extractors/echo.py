"""
Echo Feature Extractor (15D)

Clause boundary similarity patterns - the core SpecHO system.

Features (sorted by Cohen's d):
- semantic_max (d=-0.91): Strongest discriminator - peak semantic echo
- semantic_mean (d=+0.65): AI has smooth transitions
- structural_max (d=+0.60): AI has higher structural peaks
- structural_mean (d=+0.55): AI has parallel structure
- phonetic_max (d=+0.50): AI has higher phonetic peaks
- overall_max (d=+0.50): AI has higher combined peaks
- semantic_std (d=-0.50): AI has less semantic variation
- echo_burstiness (d=-0.45): Humans have echo bursts
- phonetic_mean (d=+0.45): AI has uniform sound
- overall_mean (d=+0.45): AI is more uniform overall
- structural_std (d=-0.45): AI has less structural variation
- cooccurrence_rate (d=+0.40): AI aligns all three dimensions
- phonetic_std (d=-0.40): AI has less phonetic variation
- overall_std (d=-0.40): AI has less overall variation
- geometric_mean (d=+0.35): AI has higher combined echo
"""

import re
from typing import Dict, List, Optional

from .base import BaseExtractor


class EchoExtractor(BaseExtractor):
    """
    Echo features measuring clause boundary similarity.

    The Echo Rule: AI text exhibits stronger echoes (phonetic,
    structural, semantic) at clause boundaries than human text.

    Requires sentence-transformers for semantic similarity.
    Each feature is extracted independently with fault isolation.
    """

    group = "echo"
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
            # Phonetic echo (3D)
            "phonetic_mean",        # d=+0.45
            "phonetic_std",         # d=-0.40
            "phonetic_max",         # d=+0.50
            # Structural echo (3D)
            "structural_mean",      # d=+0.55
            "structural_std",       # d=-0.45
            "structural_max",       # d=+0.60
            # Semantic echo (3D)
            "semantic_mean",        # d=+0.65
            "semantic_std",         # d=-0.50
            "semantic_max",         # d=-0.91 STRONGEST!
            # Combined metrics (6D)
            "cooccurrence_rate",    # d=+0.40
            "geometric_mean",       # d=+0.35
            "overall_mean",         # d=+0.45
            "overall_std",          # d=-0.40
            "overall_max",          # d=+0.50
            "echo_burstiness",      # d=-0.45
        ]

    def _safe_extract(self, name: str, func, default=None):
        """Safely extract with error handling."""
        try:
            return func()
        except Exception:
            return default if default is not None else 0.0

    def _get_word_endings(self, text: str, n: int = 3) -> set:
        """Get word endings for phonetic similarity."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return set(w[-n:] for w in words if len(w) >= n)

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract echo features with fault isolation."""
        import numpy as np

        features = {name: 0.0 for name in self.feature_names}

        # Split into sentences (clause approximation)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if len(sentences) < 2:
            return features

        # =====================================================================
        # PHONETIC ECHO (simplified: word ending similarity)
        # =====================================================================
        def calc_phonetic():
            phonetic_scores = []
            for i in range(len(sentences) - 1):
                endings1 = self._get_word_endings(sentences[i])
                endings2 = self._get_word_endings(sentences[i+1])
                if endings1 and endings2:
                    overlap = len(endings1 & endings2) / max(len(endings1 | endings2), 1)
                    phonetic_scores.append(overlap)
            return phonetic_scores

        phonetic_scores = self._safe_extract("phonetic", calc_phonetic, [])

        def set_phonetic():
            if phonetic_scores:
                features["phonetic_mean"] = float(np.mean(phonetic_scores))
                features["phonetic_std"] = float(np.std(phonetic_scores))
                features["phonetic_max"] = float(max(phonetic_scores))

        self._safe_extract("set_phonetic", set_phonetic)

        # =====================================================================
        # STRUCTURAL ECHO (sentence length/structure similarity)
        # =====================================================================
        def calc_structural():
            struct_scores = []
            for i in range(len(sentences) - 1):
                len1 = len(sentences[i].split())
                len2 = len(sentences[i+1].split())
                # Similarity based on length ratio
                similarity = 1 - abs(len1 - len2) / max(len1 + len2, 1)
                struct_scores.append(similarity)
            return struct_scores

        struct_scores = self._safe_extract("structural", calc_structural, [])

        def set_structural():
            if struct_scores:
                features["structural_mean"] = float(np.mean(struct_scores))
                features["structural_std"] = float(np.std(struct_scores))
                features["structural_max"] = float(max(struct_scores))

        self._safe_extract("set_structural", set_structural)

        # =====================================================================
        # SEMANTIC ECHO (embedding cosine similarity)
        # =====================================================================
        semantic_scores = []

        def calc_semantic():
            nonlocal semantic_scores
            embeddings = self.embedder.encode(sentences)
            for i in range(len(embeddings) - 1):
                cos_sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                )
                semantic_scores.append(float(cos_sim))

        self._safe_extract("semantic", calc_semantic)

        def set_semantic():
            if semantic_scores:
                features["semantic_mean"] = float(np.mean(semantic_scores))
                features["semantic_std"] = float(np.std(semantic_scores))
                features["semantic_max"] = float(max(semantic_scores))

        self._safe_extract("set_semantic", set_semantic)

        # =====================================================================
        # COMBINED METRICS
        # =====================================================================
        def calc_combined():
            all_scores = []
            if phonetic_scores:
                all_scores.extend(phonetic_scores)
            if struct_scores:
                all_scores.extend(struct_scores)
            if semantic_scores:
                all_scores.extend(semantic_scores)

            if all_scores:
                features["overall_mean"] = float(np.mean(all_scores))
                features["overall_std"] = float(np.std(all_scores))
                features["overall_max"] = float(max(all_scores))

                # Burstiness
                if features["overall_mean"] > 0:
                    features["echo_burstiness"] = features["overall_std"] / features["overall_mean"]

                # Geometric mean (avoid log of 0)
                safe_scores = [max(s, 1e-8) for s in all_scores]
                features["geometric_mean"] = float(np.exp(np.mean(np.log(safe_scores))))

            # Cooccurrence rate (all three dimensions high together)
            if phonetic_scores and struct_scores and semantic_scores:
                threshold = 0.3
                min_len = min(len(phonetic_scores), len(struct_scores), len(semantic_scores))
                cooccur = 0
                for i in range(min_len):
                    if (phonetic_scores[i] > threshold and
                        struct_scores[i] > threshold and
                        (len(semantic_scores) > i and semantic_scores[i] > threshold)):
                        cooccur += 1
                features["cooccurrence_rate"] = cooccur / max(min_len, 1)

        self._safe_extract("combined", calc_combined)

        return features


# Standalone function
def extract_echo_features(text: str, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, float]:
    """Extract echo features (standalone function)."""
    extractor = EchoExtractor(model_name=model_name)
    result = extractor.extract(text)
    return result.features
