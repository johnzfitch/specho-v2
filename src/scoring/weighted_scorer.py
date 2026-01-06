"""Weighted scoring for echo analysis results.

This module implements the WeightedScorer class which combines phonetic,
structural, and semantic similarity scores into a single pair score using
configurable weights.

Tier 1 uses simple weighted summation with equal weights and treats missing
data (NaN) as zero. The output is clipped to [0,1] to ensure valid probability
ranges for downstream statistical validation.

Tier: 1 (MVP)
Task: 5.1
Dependencies: Task 1.1 (models.py), Task 1.2 (config.py)
"""

import numpy as np
from typing import Dict
from ..models import EchoScore
from ..config import ScoringConfig, load_config


class WeightedScorer:
    """Combines individual echo scores into weighted pair scores.

    The WeightedScorer takes an EchoScore (containing phonetic, structural,
    and semantic similarity scores) and combines them using configurable
    weights to produce a single score representing the overall "echo strength"
    between two clauses.

    Tier 1 Implementation:
    - Simple weighted sum: w_p * phonetic + w_s * structural + w_sem * semantic
    - Fixed equal weights: {phonetic: 0.33, structural: 0.33, semantic: 0.34}
    - NaN handling: treat as 0.0 (missing data strategy = "zero")
    - Output clipping: ensure result is in [0,1] range

    Examples:
        >>> scorer = WeightedScorer()
        >>> echo_score = EchoScore(phonetic_score=0.8, structural_score=0.6,
        ...                        semantic_score=0.7, combined_score=0.0)
        >>> pair_score = scorer.calculate_pair_score(echo_score)
        >>> print(f"{pair_score:.3f}")
        0.700

        >>> # With custom weights
        >>> scorer = WeightedScorer(weights={'phonetic': 0.5, 'structural': 0.3,
        ...                                   'semantic': 0.2})
        >>> pair_score = scorer.calculate_pair_score(echo_score)
        >>> print(f"{pair_score:.3f}")
        0.690

        >>> # Handles NaN values
        >>> echo_score = EchoScore(phonetic_score=0.8, structural_score=np.nan,
        ...                        semantic_score=0.7, combined_score=0.0)
        >>> pair_score = scorer.calculate_pair_score(echo_score)
        >>> print(f"{pair_score:.3f}")
        0.500

    Attributes:
        weights: Dictionary mapping score types to their weights
        config: ScoringConfig object containing weight values
    """

    def __init__(self, weights: Dict[str, float] = None, config: ScoringConfig = None):
        """Initialize the weighted scorer with configuration.

        Args:
            weights: Optional dictionary of weights. If provided, overrides config.
                    Keys: 'phonetic', 'structural', 'semantic'
            config: Optional ScoringConfig object. If None, loads "simple" profile.

        Raises:
            ValueError: If weights don't sum to approximately 1.0 (within 0.01 tolerance)
        """
        # Load configuration
        if config is None:
            full_config = load_config("simple")
            self.config = full_config.scoring
        else:
            self.config = config

        # Use provided weights or extract from config
        if weights is not None:
            self.weights = weights
        else:
            self.weights = {
                'phonetic': self.config.phonetic_weight,
                'structural': self.config.structural_weight,
                'semantic': self.config.semantic_weight
            }

        # Validate weights sum to 1.0 (with tolerance for floating point)
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0 (got {total:.3f}). "
                f"Weights: {self.weights}"
            )

    def calculate_pair_score(self, echo_score: EchoScore,
                            weights: Dict[str, float] = None) -> float:
        """Calculate weighted score for a clause pair.

        Combines phonetic, structural, and semantic scores using weighted
        summation. This is the core scoring algorithm for Tier 1.

        Algorithm:
            1. Extract individual scores from EchoScore
            2. Handle NaN values by replacing with 0.0 (Tier 1 strategy)
            3. Apply weights: score = w_p*phonetic + w_s*structural + w_sem*semantic
            4. Clip result to [0,1] range

        Args:
            echo_score: EchoScore object containing individual similarity scores
            weights: Optional weights to use instead of instance weights

        Returns:
            Float in [0,1] representing overall echo strength

        Examples:
            >>> scorer = WeightedScorer()
            >>> echo = EchoScore(0.8, 0.6, 0.7, 0.0)
            >>> scorer.calculate_pair_score(echo)
            0.7

            >>> # NaN handling
            >>> echo = EchoScore(0.8, np.nan, 0.7, 0.0)
            >>> scorer.calculate_pair_score(echo)
            0.5
        """
        # Use instance weights if not provided
        if weights is None:
            weights = self.weights

        # Extract scores and handle NaN (Tier 1: replace with 0.0)
        phonetic = echo_score.phonetic_score
        structural = echo_score.structural_score
        semantic = echo_score.semantic_score

        # NaN handling: treat as 0.0 (missing_data_strategy = "zero")
        if np.isnan(phonetic):
            phonetic = 0.0
        if np.isnan(structural):
            structural = 0.0
        if np.isnan(semantic):
            semantic = 0.0

        # Calculate weighted sum
        weighted_sum = (
            weights['phonetic'] * phonetic +
            weights['structural'] * structural +
            weights['semantic'] * semantic
        )

        # Clip to [0,1] range (ensures valid probability range)
        result = np.clip(weighted_sum, 0.0, 1.0)

        return float(result)

    def get_weights(self) -> Dict[str, float]:
        """Return current weights configuration.

        Useful for debugging and logging which weights are being used.

        Returns:
            Dictionary of current weights

        Examples:
            >>> scorer = WeightedScorer()
            >>> weights = scorer.get_weights()
            >>> weights['phonetic']
            0.33
        """
        return self.weights.copy()
