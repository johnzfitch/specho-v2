"""Document-level score aggregation from pair scores.

This module implements the DocumentAggregator class which combines multiple
clause pair scores into a single document-level score representing overall
echo strength across the entire text.

Tier 1 uses simple mean aggregation. This establishes a baseline before
measuring whether more sophisticated aggregation (e.g., trimmed mean, outlier
removal) is needed in Tier 2.

Tier: 1 (MVP)
Task: 5.2
Dependencies: None (uses standard library only)
"""

import statistics
import warnings
from typing import List


class DocumentAggregator:
    """Aggregates clause pair scores into document-level score.

    The DocumentAggregator takes a list of pair scores (from WeightedScorer)
    and combines them into a single document-level score. This represents the
    overall strength of the Echo Rule watermark signal across the entire text.

    Tier 1 Implementation:
    - Simple arithmetic mean of all pair scores
    - Returns 0.0 if no pairs (empty document or no valid pairs)
    - Emits warning for empty input
    - No outlier removal or weighting

    The mean aggregation assumes all clause pairs contribute equally to the
    document score. Tier 2 will add strategies for handling outliers and
    weighting pairs by confidence.

    Examples:
        >>> aggregator = DocumentAggregator()
        >>> pair_scores = [0.7, 0.8, 0.6, 0.75]
        >>> doc_score = aggregator.aggregate_scores(pair_scores)
        >>> print(f"{doc_score:.3f}")
        0.713

        >>> # Empty input returns 0.0 with warning
        >>> doc_score = aggregator.aggregate_scores([])
        >>> print(doc_score)
        0.0

        >>> # Single pair score
        >>> doc_score = aggregator.aggregate_scores([0.65])
        >>> print(doc_score)
        0.65

    Attributes:
        None (stateless for Tier 1)
    """

    def __init__(self):
        """Initialize the document aggregator.

        Tier 1 aggregator is stateless and requires no configuration.
        Configuration will be added in Tier 2 for strategy selection.
        """
        pass

    def aggregate_scores(self, pair_scores: List[float]) -> float:
        """Aggregate pair scores into document-level score.

        Calculates the arithmetic mean of all pair scores. This represents
        the average echo strength across all clause pairs in the document.

        Algorithm (Tier 1):
            1. Check for empty input → return 0.0 with warning
            2. Calculate mean using statistics.mean()
            3. Return mean as document score

        Args:
            pair_scores: List of float scores from WeightedScorer, one per clause pair.
                        Expected range: [0,1] for each score.

        Returns:
            Float representing document-level echo score in [0,1] range.
            Returns 0.0 if input is empty.

        Warnings:
            Issues UserWarning if pair_scores is empty, indicating no clause
            pairs were available for scoring (could indicate preprocessing issue
            or very short document).

        Examples:
            >>> aggregator = DocumentAggregator()

            >>> # Typical case: multiple pairs
            >>> scores = [0.7, 0.8, 0.6, 0.75, 0.65]
            >>> aggregator.aggregate_scores(scores)
            0.7

            >>> # Edge case: single pair
            >>> aggregator.aggregate_scores([0.82])
            0.82

            >>> # Edge case: all same scores
            >>> aggregator.aggregate_scores([0.5, 0.5, 0.5])
            0.5

            >>> # Edge case: empty list
            >>> aggregator.aggregate_scores([])
            0.0
        """
        # Handle empty input
        if not pair_scores:
            warnings.warn(
                "DocumentAggregator received empty pair_scores list. "
                "Returning 0.0. This may indicate no valid clause pairs "
                "were found in the document.",
                UserWarning,
                stacklevel=2
            )
            return 0.0

        # Calculate simple mean (Tier 1 algorithm)
        document_score = statistics.mean(pair_scores)

        return document_score

    def get_statistics(self, pair_scores: List[float]) -> dict:
        """Calculate descriptive statistics for pair scores.

        This is a utility method useful for debugging and understanding score
        distributions. Not required by Tier 1 spec but helpful for validation.

        Args:
            pair_scores: List of float scores

        Returns:
            Dictionary containing:
                - n_pairs: Number of pairs
                - mean: Arithmetic mean
                - median: Median value (if available)
                - min: Minimum score
                - max: Maximum score
                - stdev: Standard deviation (if n >= 2)

        Examples:
            >>> aggregator = DocumentAggregator()
            >>> scores = [0.5, 0.7, 0.6, 0.8, 0.55]
            >>> stats = aggregator.get_statistics(scores)
            >>> stats['n_pairs']
            5
            >>> stats['mean']
            0.63
        """
        if not pair_scores:
            return {
                'n_pairs': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'stdev': 0.0
            }

        stats = {
            'n_pairs': len(pair_scores),
            'mean': statistics.mean(pair_scores),
            'median': statistics.median(pair_scores),
            'min': min(pair_scores),
            'max': max(pair_scores)
        }

        # Standard deviation requires at least 2 values
        if len(pair_scores) >= 2:
            stats['stdev'] = statistics.stdev(pair_scores)
        else:
            stats['stdev'] = 0.0

        return stats
