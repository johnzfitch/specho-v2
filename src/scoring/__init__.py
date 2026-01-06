"""Scoring module for SpecHO watermark detection.

This module implements weighted scoring and aggregation of echo scores.
It combines individual similarity metrics (phonetic, structural, semantic)
into pair scores and aggregates pair scores into document-level scores.

Components:
    - WeightedScorer: Combines echo scores using weighted summation
    - DocumentAggregator: Aggregates pair scores into document score (Task 5.2)
    - ScoringModule: Pipeline orchestrator (Task 5.3)

Tier: 1 (MVP)
Phase: Component 4 - Scoring
Tasks: 5.1, 5.2, 5.3
"""

from .weighted_scorer import WeightedScorer
from .aggregator import DocumentAggregator
from .pipeline import ScoringModule

__all__ = ['WeightedScorer', 'DocumentAggregator', 'ScoringModule']
