"""Scoring pipeline orchestrator.

This module implements the ScoringModule class which orchestrates the complete
scoring workflow from echo scores to final document score.

The pipeline chains WeightedScorer (pair-level scoring) and DocumentAggregator
(document-level aggregation) to provide a single entry point for scoring.

Tier 1 is a simple orchestrator with no additional logic beyond calling the
two component classes in sequence.

Tier: 1 (MVP)
Task: 5.3
Dependencies: Task 5.1 (WeightedScorer), Task 5.2 (DocumentAggregator)
"""

from typing import List
from ..models import EchoScore
from .weighted_scorer import WeightedScorer
from .aggregator import DocumentAggregator


class ScoringModule:
    """Orchestrates weighted scoring and aggregation pipeline.

    The ScoringModule provides a unified interface for converting echo scores
    into a final document-level score. It chains two components:
    1. WeightedScorer: Converts each EchoScore -> pair score (float)
    2. DocumentAggregator: Converts List[pair scores] -> document score (float)

    Tier 1 Implementation:
    - Simple orchestration with no additional logic
    - Uses default configurations for WeightedScorer and DocumentAggregator
    - Delegates all scoring logic to component classes
    - Stateless operation

    This orchestrator pattern keeps the scoring pipeline modular and testable
    while providing a clean API for downstream consumers (SpecHODetector).

    Examples:
        >>> scorer = ScoringModule()
        >>> echo_scores = [
        ...     EchoScore(phonetic_score=0.8, structural_score=0.6,
        ...               semantic_score=0.7, combined_score=0.0),
        ...     EchoScore(phonetic_score=0.75, structural_score=0.65,
        ...               semantic_score=0.72, combined_score=0.0)
        ... ]
        >>> doc_score = scorer.score_document(echo_scores)
        >>> print(f"{doc_score:.3f}")
        0.707

        >>> # Empty input returns 0.0
        >>> doc_score = scorer.score_document([])
        >>> print(doc_score)
        0.0

    Attributes:
        weighted_scorer: WeightedScorer instance for pair-level scoring
        aggregator: DocumentAggregator instance for document-level aggregation
    """

    def __init__(self):
        """Initialize the scoring pipeline with default components.

        Tier 1 uses default configurations for both WeightedScorer and
        DocumentAggregator. Configuration support will be added in Tier 2.
        """
        self.weighted_scorer = WeightedScorer()
        self.aggregator = DocumentAggregator()

    def score_document(self, echo_scores: List[EchoScore]) -> float:
        """Convert echo scores to final document-level score.

        This is the main entry point for the scoring pipeline. It performs
        the complete scoring workflow:
        1. Convert each EchoScore to a pair score using WeightedScorer
        2. Update the EchoScore objects with their combined score
        3. Aggregate all pair scores into document score using DocumentAggregator
        4. Return the final document score

        Algorithm (Tier 1):
            1. For each EchoScore, call weighted_scorer.calculate_pair_score()
            2. Update echo_score.combined_score with the result
            3. Collect all pair scores into a list
            4. Call aggregator.aggregate_scores() on the list
            5. Return the document score

        Args:
            echo_scores: List of EchoScore objects, one per clause pair.
                        Each contains phonetic, structural, and semantic scores.

        Returns:
            float: Document-level score in range [0,1]
                  - 0.0 indicates no echo signal (human/natural text)
                  - Higher values indicate stronger echo signal (watermarked)
                  - Returns 0.0 for empty input (delegated to aggregator)

        Examples:
            >>> scorer = ScoringModule()
            >>> echo_scores = [
            ...     EchoScore(0.8, 0.6, 0.7, 0.0),
            ...     EchoScore(0.75, 0.65, 0.72, 0.0)
            ... ]
            >>> doc_score = scorer.score_document(echo_scores)
            >>> 0.0 <= doc_score <= 1.0
            True
        """
        # Step 1: Convert each EchoScore to pair score and update the object
        pair_scores = []
        for echo_score in echo_scores:
            score = self.weighted_scorer.calculate_pair_score(echo_score)
            echo_score.combined_score = score
            pair_scores.append(score)

        # Step 2: Aggregate pair scores into document score using DocumentAggregator
        document_score = self.aggregator.aggregate_scores(pair_scores)

        # Step 3: Return final document score
        return document_score
