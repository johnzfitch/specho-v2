#!/usr/bin/env python3
"""
Cognitive Fingerprint Extractor for PACT Corpus

Combines SpecHO with additional cognitive dimensions:
- Trajectory (5 dims): Semantic navigation geometry
- Epistemic (6 dims): Hedging and confidence patterns
- Transitions (6 dims): How ideas connect
- Syntactic (7 dims): Structural rhythm
- SpecHO Extended (11 dims): Additional echo statistics

Total: 39 dimensions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
import sys
from pathlib import Path

from .trajectory import TrajectoryAnalyzer
from .epistemic import EpistemicAnalyzer
from .transitions import TransitionAnalyzer
from .syntactic import SyntacticAnalyzer


@dataclass
class CognitiveFingerprint:
    """Complete 39-dimensional cognitive fingerprint."""

    # === LAYER A: TRAJECTORY (5 dims) ===
    concept_jump_mean: float = 0.0
    concept_jump_variance: float = 0.0
    path_tortuosity: float = 1.0
    turning_angle_mean: float = 0.0
    return_rate: float = 0.0

    # === LAYER B: SPECIO EXTENDED (15 dims) ===
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    phonetic_max: float = 0.0
    structural_mean: float = 0.0
    structural_std: float = 0.0
    structural_max: float = 0.0
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    semantic_max: float = 0.0
    cooccurrence_rate: float = 0.0
    geometric_mean: float = 0.0
    overall_mean: float = 0.0
    overall_std: float = 0.0
    overall_max: float = 0.0
    burstiness: float = 0.0

    # === LAYER C: EPISTEMIC (6 dims) ===
    hedge_density: float = 0.0
    hedge_clustering: float = 0.0
    hedge_position_bias: float = 0.5
    confidence_mean: float = 0.5
    confidence_variance: float = 0.0
    confidence_arc: float = 0.0

    # === LAYER C: TRANSITIONS (6 dims) ===
    additive_rate: float = 0.0
    contrastive_rate: float = 0.0
    causal_rate: float = 0.0
    temporal_rate: float = 0.0
    exemplifying_rate: float = 0.0
    reformulating_rate: float = 0.0

    # === LAYER D: SYNTACTIC (7 dims) ===
    clause_length_mean: float = 0.0
    clause_length_std: float = 0.0
    clause_rhythm_autocorr: float = 0.0
    sentence_complexity: float = 1.0
    comma_density: float = 0.0
    semicolon_rate: float = 0.0
    parenthetical_rate: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            # Layer A
            "concept_jump_mean": self.concept_jump_mean,
            "concept_jump_variance": self.concept_jump_variance,
            "path_tortuosity": self.path_tortuosity,
            "turning_angle_mean": self.turning_angle_mean,
            "return_rate": self.return_rate,
            # Layer B
            "phonetic_mean": self.phonetic_mean,
            "phonetic_std": self.phonetic_std,
            "phonetic_max": self.phonetic_max,
            "structural_mean": self.structural_mean,
            "structural_std": self.structural_std,
            "structural_max": self.structural_max,
            "semantic_mean": self.semantic_mean,
            "semantic_std": self.semantic_std,
            "semantic_max": self.semantic_max,
            "cooccurrence_rate": self.cooccurrence_rate,
            "geometric_mean": self.geometric_mean,
            "overall_mean": self.overall_mean,
            "overall_std": self.overall_std,
            "overall_max": self.overall_max,
            "burstiness": self.burstiness,
            # Layer C
            "hedge_density": self.hedge_density,
            "hedge_clustering": self.hedge_clustering,
            "hedge_position_bias": self.hedge_position_bias,
            "confidence_mean": self.confidence_mean,
            "confidence_variance": self.confidence_variance,
            "confidence_arc": self.confidence_arc,
            "additive_rate": self.additive_rate,
            "contrastive_rate": self.contrastive_rate,
            "causal_rate": self.causal_rate,
            "temporal_rate": self.temporal_rate,
            "exemplifying_rate": self.exemplifying_rate,
            "reformulating_rate": self.reformulating_rate,
            # Layer D
            "clause_length_mean": self.clause_length_mean,
            "clause_length_std": self.clause_length_std,
            "clause_rhythm_autocorr": self.clause_rhythm_autocorr,
            "sentence_complexity": self.sentence_complexity,
            "comma_density": self.comma_density,
            "semicolon_rate": self.semicolon_rate,
            "parenthetical_rate": self.parenthetical_rate,
        }


class CognitiveExtractor:
    """
    Extracts 39-dimensional cognitive fingerprints.

    Integrates with SpecHO for complete feature extraction.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize all analyzers."""
        self.trajectory_analyzer = TrajectoryAnalyzer(embedding_model)
        self.epistemic_analyzer = EpistemicAnalyzer()
        self.transition_analyzer = TransitionAnalyzer()
        self.syntactic_analyzer = SyntacticAnalyzer()

    def extract_from_specho(
        self,
        text: str,
        echo_scores: list,
    ) -> CognitiveFingerprint:
        """
        Extract full fingerprint using existing SpecHO echo scores.

        Args:
            text: Text to analyze
            echo_scores: List of EchoScore objects from SpecHO

        Returns:
            CognitiveFingerprint with all 39 features
        """
        # Layer A: Trajectory
        trajectory = self.trajectory_analyzer.analyze(text)

        # Layer B: Extended SpecHO statistics
        specho_extended = self._extract_specho_extended(echo_scores)

        # Layer C: Epistemic
        epistemic = self.epistemic_analyzer.analyze(text)

        # Layer C: Transitions
        transitions = self.transition_analyzer.analyze(text)

        # Layer D: Syntactic
        syntactic = self.syntactic_analyzer.analyze(text)

        return CognitiveFingerprint(
            # Layer A
            concept_jump_mean=trajectory.concept_jump_mean,
            concept_jump_variance=trajectory.concept_jump_variance,
            path_tortuosity=trajectory.path_tortuosity,
            turning_angle_mean=trajectory.turning_angle_mean,
            return_rate=trajectory.return_rate,
            # Layer B
            **specho_extended,
            # Layer C
            hedge_density=epistemic.hedge_density,
            hedge_clustering=epistemic.hedge_clustering,
            hedge_position_bias=epistemic.hedge_position_bias,
            confidence_mean=epistemic.confidence_mean,
            confidence_variance=epistemic.confidence_variance,
            confidence_arc=epistemic.confidence_arc,
            additive_rate=transitions.additive_rate,
            contrastive_rate=transitions.contrastive_rate,
            causal_rate=transitions.causal_rate,
            temporal_rate=transitions.temporal_rate,
            exemplifying_rate=transitions.exemplifying_rate,
            reformulating_rate=transitions.reformulating_rate,
            # Layer D
            clause_length_mean=syntactic.clause_length_mean,
            clause_length_std=syntactic.clause_length_std,
            clause_rhythm_autocorr=syntactic.clause_rhythm_autocorr,
            sentence_complexity=syntactic.sentence_complexity,
            comma_density=syntactic.comma_density,
            semicolon_rate=syntactic.semicolon_rate,
            parenthetical_rate=syntactic.parenthetical_rate,
        )

    def _extract_specho_extended(self, echo_scores: list) -> Dict:
        """
        Extract extended SpecHO statistics (15 dimensions).

        Args:
            echo_scores: List of EchoScore objects

        Returns:
            Dictionary with 15 SpecHO features
        """
        if not echo_scores:
            return {k: 0.0 for k in [
                "phonetic_mean", "phonetic_std", "phonetic_max",
                "structural_mean", "structural_std", "structural_max",
                "semantic_mean", "semantic_std", "semantic_max",
                "cooccurrence_rate", "geometric_mean",
                "overall_mean", "overall_std", "overall_max", "burstiness",
            ]}

        phonetic = [s.phonetic_score for s in echo_scores]
        structural = [s.structural_score for s in echo_scores]
        semantic = [s.semantic_score for s in echo_scores]
        combined = [(p + s + m) / 3 for p, s, m in zip(phonetic, structural, semantic)]

        # Cooccurrence: all three dimensions high simultaneously
        threshold = 0.6
        cooccur = sum(1 for p, s, m in zip(phonetic, structural, semantic)
                     if p > threshold and s > threshold and m > threshold)

        # Geometric mean: root of product
        geometric = [pow(max(0.001, p * s * m), 1/3)
                    for p, s, m in zip(phonetic, structural, semantic)]

        mean_score = float(np.mean(combined))
        std_score = float(np.std(combined)) if len(combined) > 1 else 0.0

        return {
            "phonetic_mean": float(np.mean(phonetic)),
            "phonetic_std": float(np.std(phonetic)) if len(phonetic) > 1 else 0.0,
            "phonetic_max": float(np.max(phonetic)),
            "structural_mean": float(np.mean(structural)),
            "structural_std": float(np.std(structural)) if len(structural) > 1 else 0.0,
            "structural_max": float(np.max(structural)),
            "semantic_mean": float(np.mean(semantic)),
            "semantic_std": float(np.std(semantic)) if len(semantic) > 1 else 0.0,
            "semantic_max": float(np.max(semantic)),
            "cooccurrence_rate": cooccur / len(echo_scores),
            "geometric_mean": float(np.mean(geometric)),
            "overall_mean": mean_score,
            "overall_std": std_score,
            "overall_max": float(np.max(combined)),
            "burstiness": std_score / mean_score if mean_score > 0 else 0.0,
        }
