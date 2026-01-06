"""Semantic echo analysis using word embeddings.

This module implements the semantic dimension of the Echo Rule watermark
detection. It measures semantic similarity between clause zones using
word embeddings (Word2Vec, GloVe via gensim, or Sentence Transformers).

Tier: 1 (MVP)
Task: 4.3
Dependencies: Task 1.1 (Token dataclass)
"""

from typing import List, Union
import numpy as np
from specHO.models import Token


class SemanticEchoAnalyzer:
    """Analyzes semantic similarity between clause zones using word embeddings.

    Uses embeddings to represent each zone, then calculates cosine similarity.
    This captures whether the zones express semantically related concepts, even
    when phonetic and structural similarities are low.

    Tier 1 Implementation:
    - Pre-trained embeddings via gensim (Word2Vec/GloVe) or Sentence Transformers
    - Mean pooling across zone tokens (for gensim models)
    - Direct sentence encoding (for Sentence Transformers)
    - Cosine similarity mapped to [0,1] range
    - Fallback to 0.5 (neutral) if embeddings unavailable

    Attributes:
        model: Embedding model (gensim KeyedVectors or SentenceTransformer)
               None if embeddings could not be loaded
        model_type: Type of model ('gensim' or 'sentence_transformer')
    """

    DEFAULT_PATHS = [
        "data/embeddings/glove.6B.100d.txt",
        "data/embeddings/glove.6B.300d.txt",
        "~/.specho/embeddings/glove.6B.100d.txt",
        "~/.cache/specho/glove.6B.100d.txt",
    ]

    def __init__(self, model_path: str = None, model_type: str = None):
        """Initialize semantic analyzer with word embeddings.

        Args:
            model_path: Path to gensim-compatible embeddings file, OR
                       name of Sentence Transformer model (e.g., 'all-MiniLM-L6-v2')
                       If None, searches DEFAULT_PATHS then falls back to 0.5
            model_type: Type of model ('gensim', 'sentence_transformer')
                       If None, auto-detects based on model_path format
        """
        self.model = None
        self.model_type = None
        self.model_path = model_path
        self._vocab_size = 0

        if model_path:
            # Auto-detect or use specified model type
            if model_type:
                self.model_type = model_type
            elif '/' not in model_path and '\\' not in model_path and not model_path.endswith('.txt'):
                self.model_type = 'sentence_transformer'
            else:
                self.model_type = 'gensim'

            self._load_model(model_path)
        else:
            # Try default paths
            self._try_default_paths()

        # Warn if in fallback mode
        if self.model is None:
            import sys
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "SemanticEchoAnalyzer: No embeddings loaded - Running in FALLBACK MODE "
                "(all scores = 0.5). Detection accuracy severely limited! "
                f"Install with: uv pip install sentence-transformers"
            )

    def _try_default_paths(self):
        """Try loading embeddings from common default locations."""
        from pathlib import Path
        import logging
        logger = logging.getLogger(__name__)

        # First, try to load a Sentence Transformer model as fallback
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Attempting to load default Sentence Transformer model (all-MiniLM-L6-v2)...")
            self.model_type = 'sentence_transformer'
            self._load_model('all-MiniLM-L6-v2')
            if self.model is not None:
                logger.info("Successfully loaded default Sentence Transformer model")
                return
        except ImportError:
            logger.debug("sentence-transformers not available, trying local embeddings...")
        except Exception as e:
            logger.debug(f"Failed to load default Sentence Transformer: {e}")

        # Fall back to local embeddings paths
        for path_str in self.DEFAULT_PATHS:
            path = Path(path_str).expanduser()
            if path.exists():
                logger.info(f"Found embeddings at {path}")
                self.model_type = 'gensim'
                self._load_model(str(path))
                if self.model is not None:
                    return

    def _load_model(self, model_path: str):
        """Load the embedding model.

        Args:
            model_path: Path to embeddings file or model name
        """
        try:
            if self.model_type == 'sentence_transformer':
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_path)
                self._vocab_size = -1  # Transformer handles any vocab
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Loaded SentenceTransformer: {model_path}")

            else:  # gensim
                from gensim.models import KeyedVectors

                # Auto-detect binary format
                if model_path.endswith('.bin'):
                    self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
                else:
                    self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)

                self._vocab_size = len(self.model.key_to_index)
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Loaded {self._vocab_size:,} word vectors from {model_path}")

        except ImportError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: uv pip install gensim  OR  uv pip install sentence-transformers")
            self.model = None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load embeddings model: {e}")
            self.model = None

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """Calculate semantic similarity between two clause zones.

        Computes mean-pooled embeddings for each zone and returns cosine
        similarity mapped to [0,1] range. Falls back to 0.5 if embeddings
        are unavailable or zones are empty.

        Args:
            zone_a: List of tokens from terminal zone (clause A)
            zone_b: List of tokens from initial zone (clause B)

        Returns:
            Float in [0.0, 1.0] representing semantic similarity:
            - 0.0: semantically opposite or completely unrelated
            - 0.5: neutral (no clear semantic relationship) or fallback
            - 1.0: semantically identical or highly related

        Edge Cases:
            - Empty zones: returns 0.0
            - No embeddings available: returns 0.5 (neutral)
            - Tokens not in vocabulary: skipped (uses available tokens only)
            - All tokens OOV: returns 0.5 (neutral)
        """
        # Edge case: empty zones
        if not zone_a or not zone_b:
            return 0.0

        # Fallback: no embeddings available
        if self.model is None:
            return 0.5

        # Get zone vectors
        vec_a = self._get_zone_vector(zone_a)
        vec_b = self._get_zone_vector(zone_b)

        # Handle cases where no embeddings found
        if vec_a is None or vec_b is None:
            return 0.5

        # Calculate cosine similarity
        similarity = self._calculate_cosine_similarity(vec_a, vec_b)

        return similarity

    def _get_zone_vector(self, zone: List[Token]) -> np.ndarray:
        """Compute embedding vector for a zone.

        For gensim models: Uses mean-pooled word embeddings.
        For Sentence Transformers: Encodes the full text sequence.

        Args:
            zone: List of tokens to embed

        Returns:
            Embedding vector, or None if no tokens have embeddings
        """
        if self.model_type == 'sentence_transformer':
            # Use Sentence Transformer to encode the full text
            text = ' '.join(token.text for token in zone)
            if not text.strip():
                return None
            return self.model.encode(text, convert_to_numpy=True)

        elif self.model_type == 'gensim':
            # Use gensim word embeddings with mean pooling
            vectors = []

            for token in zone:
                # Get lowercase version for embedding lookup
                word = token.text.lower()

                try:
                    if word in self.model:
                        vectors.append(self.model[word])
                except Exception:
                    # Token not in vocabulary, skip it
                    continue

            # No embeddings found
            if not vectors:
                return None

            # Mean pooling
            return np.mean(vectors, axis=0)

        else:
            # Unknown model type
            return None

    def _calculate_cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> float:
        """Calculate cosine similarity and map to [0,1] range.

        Args:
            vec_a: Embedding vector for zone A
            vec_b: Embedding vector for zone B

        Returns:
            Similarity score in [0,1] range
        """
        # Cosine similarity formula: dot(a,b) / (||a|| * ||b||)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.5

        cosine_sim = dot_product / (norm_a * norm_b)

        # Map from [-1, 1] to [0, 1]: (1 + cos) / 2
        similarity = (1.0 + cosine_sim) / 2.0

        # Clip to [0,1] range (handle floating point errors)
        return np.clip(similarity, 0.0, 1.0)

    @property
    def is_loaded(self) -> bool:
        """Check if embedding model is loaded (not in fallback mode).

        Returns:
            True if model is loaded and ready to use, False if in fallback mode
        """
        return self.model is not None

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size of the loaded model.

        Returns:
            Number of words in vocabulary, or -1 for transformer models (unlimited)
        """
        return self._vocab_size
