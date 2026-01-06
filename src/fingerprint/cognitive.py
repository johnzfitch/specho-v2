#!/usr/bin/env python3
"""
Cognitive Fingerprint Extractor

Combines all analyzers into a complete cognitive signature:
- Original SpecHO (15 dims): Echo patterns at clause boundaries
- Trajectory (5 dims): Semantic navigation geometry
- Epistemic (6 dims): Hedging and confidence patterns
- Transitions (6 dims): How ideas connect
- Syntactic (7 dims): Structural rhythm

Total: 39 dimensions (layered by trust level)

Layer A (High Trust): Trajectory - hard to spoof without changing weights
Layer B (Medium-High Trust): Original SpecHO - clause-level patterns
Layer C (Medium Trust): Epistemic + Transitions - discourse-level choices
Layer D (Lower Trust): Syntactic - can be consciously adjusted

Usage:
    from cognitive import CognitiveExtractor
    
    extractor = CognitiveExtractor()
    fingerprint = extractor.extract(text)
    
    # Different trust layers
    high_trust = fingerprint.to_vector_layer_a()    # 5 dims
    full = fingerprint.to_vector_full()              # 39 dims
"""

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict
from pathlib import Path
import json
import time
import sys

# Import analyzers
sys.path.insert(0, str(Path(__file__).parent))

from trajectory import TrajectoryAnalyzer, TrajectoryFeatures
from epistemic import EpistemicAnalyzer, EpistemicFeatures
from transitions import TransitionAnalyzer, TransitionFeatures
from syntactic import SyntacticAnalyzer, SyntacticFeatures


@dataclass
class CognitiveFingerprint:
    """
    Complete cognitive fingerprint (39 dimensions).
    
    Layered by trust level for flexible verification.
    """
    
    # Identity
    sample_id: str = ""
    source_model: str = ""
    
    # === LAYER A: TRAJECTORY (5 dims) - HIGH TRUST ===
    concept_jump_mean: float = 0.0
    concept_jump_variance: float = 0.0
    path_tortuosity: float = 1.0
    turning_angle_mean: float = 0.0
    return_rate: float = 0.0
    
    # === LAYER B: ORIGINAL SPECIO (15 dims) - MEDIUM-HIGH TRUST ===
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
    
    # === LAYER C: EPISTEMIC (6 dims) - MEDIUM TRUST ===
    hedge_density: float = 0.0
    hedge_clustering: float = 0.0
    hedge_position_bias: float = 0.5
    confidence_mean: float = 0.5
    confidence_variance: float = 0.0
    confidence_arc: float = 0.0
    
    # === LAYER C: TRANSITIONS (6 dims) - MEDIUM TRUST ===
    additive_rate: float = 0.0
    contrastive_rate: float = 0.0
    causal_rate: float = 0.0
    temporal_rate: float = 0.0
    exemplifying_rate: float = 0.0
    reformulating_rate: float = 0.0
    
    # === LAYER D: SYNTACTIC (7 dims) - LOWER TRUST ===
    clause_length_mean: float = 0.0
    clause_length_std: float = 0.0
    clause_rhythm_autocorr: float = 0.0
    sentence_complexity: float = 1.0
    comma_density: float = 0.0
    semicolon_rate: float = 0.0
    parenthetical_rate: float = 0.0
    
    # Metadata
    n_clauses: int = 0
    n_sentences: int = 0
    word_count: int = 0
    extraction_time_ms: float = 0.0
    
    def to_vector_layer_a(self) -> np.ndarray:
        """Layer A only: Trajectory (5 dims) - highest trust."""
        return np.array([
            self.concept_jump_mean,
            self.concept_jump_variance,
            self.path_tortuosity,
            self.turning_angle_mean,
            self.return_rate,
        ])
    
    def to_vector_layer_ab(self) -> np.ndarray:
        """Layers A+B: Trajectory + SpecHO (20 dims)."""
        return np.concatenate([
            self.to_vector_layer_a(),
            np.array([
                self.phonetic_mean, self.phonetic_std, self.phonetic_max,
                self.structural_mean, self.structural_std, self.structural_max,
                self.semantic_mean, self.semantic_std, self.semantic_max,
                self.cooccurrence_rate, self.geometric_mean,
                self.overall_mean, self.overall_std, self.overall_max,
                self.burstiness,
            ])
        ])
    
    def to_vector_layer_abc(self) -> np.ndarray:
        """Layers A+B+C: +Epistemic +Transitions (32 dims)."""
        return np.concatenate([
            self.to_vector_layer_ab(),
            np.array([
                self.hedge_density, self.hedge_clustering,
                self.hedge_position_bias, self.confidence_mean,
                self.confidence_variance, self.confidence_arc,
            ]),
            np.array([
                self.additive_rate, self.contrastive_rate,
                self.causal_rate, self.temporal_rate,
                self.exemplifying_rate, self.reformulating_rate,
            ])
        ])
    
    def to_vector_full(self) -> np.ndarray:
        """All layers: Full 39-dim fingerprint."""
        return np.concatenate([
            self.to_vector_layer_abc(),
            np.array([
                self.clause_length_mean, self.clause_length_std,
                self.clause_rhythm_autocorr, self.sentence_complexity,
                self.comma_density, self.semicolon_rate,
                self.parenthetical_rate,
            ])
        ])
    
    @property
    def feature_names_full(self) -> list[str]:
        """Names of all 39 features."""
        return [
            # Layer A: Trajectory (5)
            "concept_jump_mean", "concept_jump_variance",
            "path_tortuosity", "turning_angle_mean", "return_rate",
            # Layer B: SpecHO (15)
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max", "burstiness",
            # Layer C: Epistemic (6)
            "hedge_density", "hedge_clustering", "hedge_position_bias",
            "confidence_mean", "confidence_variance", "confidence_arc",
            # Layer C: Transitions (6)
            "additive_rate", "contrastive_rate", "causal_rate",
            "temporal_rate", "exemplifying_rate", "reformulating_rate",
            # Layer D: Syntactic (7)
            "clause_length_mean", "clause_length_std", "clause_rhythm_autocorr",
            "sentence_complexity", "comma_density", "semicolon_rate",
            "parenthetical_rate",
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class CognitiveExtractor:
    """
    Extracts complete cognitive fingerprints.
    
    Combines:
    - TrajectoryAnalyzer (semantic path geometry)
    - SpecHO (clause echo patterns) [optional]
    - EpistemicAnalyzer (hedging/confidence)
    - TransitionAnalyzer (connective patterns)
    - SyntacticAnalyzer (structural rhythm)
    """
    
    def __init__(
        self,
        specHO_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize all analyzers.
        
        Args:
            specHO_path: Path to SpecHO installation (optional)
            embedding_model: Sentence transformer model for trajectory
        """
        print("[CognitiveExtractor] Initializing analyzers...")
        
        self.trajectory_analyzer = TrajectoryAnalyzer(embedding_model)
        self.epistemic_analyzer = EpistemicAnalyzer()
        self.transition_analyzer = TransitionAnalyzer()
        self.syntactic_analyzer = SyntacticAnalyzer()
        
        self.specHO_detector = None
        self._init_specHO(specHO_path)
        
        print("[CognitiveExtractor] Ready")
    
    def _init_specHO(self, path: str = None):
        """Initialize SpecHO detector if available."""
        if path:
            sys.path.insert(0, path)
        
        try:
            from specHO.detector import SpecHODetector
            self.specHO_detector = SpecHODetector(
                semantic_model_path="all-MiniLM-L6-v2"
            )
            print("[CognitiveExtractor] SpecHO: ✓")
        except ImportError:
            print("[CognitiveExtractor] SpecHO: ✗ (using mock)")
            self.specHO_detector = None
    
    def _extract_specHO(self, text: str) -> dict:
        """Extract SpecHO features (Layer B)."""
        if self.specHO_detector is None:
            # Mock extraction - deterministic based on text
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            base = rng.uniform(0.4, 0.6)
            std = rng.uniform(0.08, 0.15)
            
            return {
                "phonetic_mean": base + rng.normal(0, 0.02),
                "phonetic_std": std,
                "phonetic_max": base + std * 1.5,
                "structural_mean": base + rng.normal(0, 0.02),
                "structural_std": std,
                "structural_max": base + std * 1.5,
                "semantic_mean": base + rng.normal(0, 0.02),
                "semantic_std": std,
                "semantic_max": base + std * 1.5,
                "cooccurrence_rate": rng.uniform(0.1, 0.4),
                "geometric_mean": base * 0.95,
                "overall_mean": base,
                "overall_std": std,
                "overall_max": base + std * 1.5,
                "burstiness": std / base if base > 0 else 0,
            }
        
        result = self.specHO_detector.analyze(text)
        
        if not hasattr(result, 'pair_analyses') or not result.pair_analyses:
            return {k: 0.5 for k in [
                "phonetic_mean", "phonetic_std", "phonetic_max",
                "structural_mean", "structural_std", "structural_max",
                "semantic_mean", "semantic_std", "semantic_max",
                "cooccurrence_rate", "geometric_mean",
                "overall_mean", "overall_std", "overall_max", "burstiness",
            ]}
        
        pairs = result.pair_analyses
        
        phonetic = [p.phonetic_score for p in pairs]
        structural = [p.structural_score for p in pairs]
        semantic = [p.semantic_score for p in pairs]
        combined = [(p + s + m) / 3 for p, s, m in zip(phonetic, structural, semantic)]
        
        threshold = 0.6
        cooccur = sum(1 for p, s, m in zip(phonetic, structural, semantic)
                     if p > threshold and s > threshold and m > threshold)
        
        geometric = [pow(max(0.001, p * s * m), 1/3) 
                    for p, s, m in zip(phonetic, structural, semantic)]
        
        mean_score = np.mean(combined)
        std_score = np.std(combined) if len(combined) > 1 else 0.0
        
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
            "cooccurrence_rate": cooccur / len(pairs),
            "geometric_mean": float(np.mean(geometric)),
            "overall_mean": float(mean_score),
            "overall_std": float(std_score),
            "overall_max": float(np.max(combined)),
            "burstiness": float(std_score / mean_score) if mean_score > 0 else 0.0,
        }
    
    def extract(
        self,
        text: str,
        sample_id: str = "",
        source_model: str = "",
    ) -> CognitiveFingerprint:
        """
        Extract complete cognitive fingerprint.
        
        Args:
            text: Text to analyze
            sample_id: Sample identifier
            source_model: Model that generated the text
            
        Returns:
            CognitiveFingerprint with all 39 features
        """
        start = time.time()
        
        words = text.split()
        n_words = len(words)
        
        # Layer A: Trajectory
        trajectory = self.trajectory_analyzer.analyze(text)
        
        # Layer B: SpecHO
        specHO = self._extract_specHO(text)
        
        # Layer C: Epistemic
        epistemic = self.epistemic_analyzer.analyze(text)
        
        # Layer C: Transitions
        transitions = self.transition_analyzer.analyze(text)
        
        # Layer D: Syntactic
        syntactic = self.syntactic_analyzer.analyze(text)
        
        elapsed = (time.time() - start) * 1000
        
        return CognitiveFingerprint(
            sample_id=sample_id,
            source_model=source_model,
            
            # Layer A: Trajectory
            concept_jump_mean=trajectory.concept_jump_mean,
            concept_jump_variance=trajectory.concept_jump_variance,
            path_tortuosity=trajectory.path_tortuosity,
            turning_angle_mean=trajectory.turning_angle_mean,
            return_rate=trajectory.return_rate,
            
            # Layer B: SpecHO
            phonetic_mean=specHO["phonetic_mean"],
            phonetic_std=specHO["phonetic_std"],
            phonetic_max=specHO["phonetic_max"],
            structural_mean=specHO["structural_mean"],
            structural_std=specHO["structural_std"],
            structural_max=specHO["structural_max"],
            semantic_mean=specHO["semantic_mean"],
            semantic_std=specHO["semantic_std"],
            semantic_max=specHO["semantic_max"],
            cooccurrence_rate=specHO["cooccurrence_rate"],
            geometric_mean=specHO["geometric_mean"],
            overall_mean=specHO["overall_mean"],
            overall_std=specHO["overall_std"],
            overall_max=specHO["overall_max"],
            burstiness=specHO["burstiness"],
            
            # Layer C: Epistemic
            hedge_density=epistemic.hedge_density,
            hedge_clustering=epistemic.hedge_clustering,
            hedge_position_bias=epistemic.hedge_position_bias,
            confidence_mean=epistemic.confidence_mean,
            confidence_variance=epistemic.confidence_variance,
            confidence_arc=epistemic.confidence_arc,
            
            # Layer C: Transitions
            additive_rate=transitions.additive_rate,
            contrastive_rate=transitions.contrastive_rate,
            causal_rate=transitions.causal_rate,
            temporal_rate=transitions.temporal_rate,
            exemplifying_rate=transitions.exemplifying_rate,
            reformulating_rate=transitions.reformulating_rate,
            
            # Layer D: Syntactic
            clause_length_mean=syntactic.clause_length_mean,
            clause_length_std=syntactic.clause_length_std,
            clause_rhythm_autocorr=syntactic.clause_rhythm_autocorr,
            sentence_complexity=syntactic.sentence_complexity,
            comma_density=syntactic.comma_density,
            semicolon_rate=syntactic.semicolon_rate,
            parenthetical_rate=syntactic.parenthetical_rate,
            
            # Metadata
            n_clauses=trajectory.n_waypoints,
            n_sentences=syntactic.n_sentences,
            word_count=n_words,
            extraction_time_ms=elapsed,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo cognitive extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognitive fingerprint extraction")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--specHO", type=str, help="Path to SpecHO installation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        # Demo
        text = """
        The implementation of distributed systems requires careful consideration 
        of several key trade-offs. First, there's the fundamental tension between 
        consistency and availability—the CAP theorem suggests we can't have both 
        during network partitions. However, in practice, many systems find 
        workable compromises. For example, eventual consistency allows high 
        availability while still converging to a consistent state. That said, 
        this approach isn't suitable for all applications; financial systems, 
        for instance, typically require stronger guarantees. It's worth noting 
        that the specific requirements depend heavily on the use case.
        """
    
    extractor = CognitiveExtractor(specHO_path=args.specHO)
    fingerprint = extractor.extract(text, sample_id="demo", source_model="unknown")
    
    if args.json:
        print(json.dumps(fingerprint.to_dict(), indent=2))
        return
    
    print("="*60)
    print("COGNITIVE FINGERPRINT")
    print("="*60)
    
    print(f"\nMetadata:")
    print(f"  Words: {fingerprint.word_count}")
    print(f"  Sentences: {fingerprint.n_sentences}")
    print(f"  Clauses: {fingerprint.n_clauses}")
    print(f"  Extraction time: {fingerprint.extraction_time_ms:.0f}ms")
    
    print(f"\n[Layer A] TRAJECTORY (High Trust) - 5 dims")
    vec_a = fingerprint.to_vector_layer_a()
    names_a = ["concept_jump_mean", "concept_jump_variance", "path_tortuosity", 
               "turning_angle_mean", "return_rate"]
    for name, val in zip(names_a, vec_a):
        print(f"  {name:22s}: {val:.4f}")
    
    print(f"\n[Layer B] SPECIO (Medium-High Trust) - 15 dims")
    print(f"  phonetic:    μ={fingerprint.phonetic_mean:.3f} σ={fingerprint.phonetic_std:.3f}")
    print(f"  structural:  μ={fingerprint.structural_mean:.3f} σ={fingerprint.structural_std:.3f}")
    print(f"  semantic:    μ={fingerprint.semantic_mean:.3f} σ={fingerprint.semantic_std:.3f}")
    print(f"  cooccurrence: {fingerprint.cooccurrence_rate:.3f}")
    print(f"  burstiness:   {fingerprint.burstiness:.3f}")
    
    print(f"\n[Layer C] EPISTEMIC (Medium Trust) - 6 dims")
    print(f"  hedge_density:    {fingerprint.hedge_density:.2f}")
    print(f"  hedge_clustering: {fingerprint.hedge_clustering:.2f}")
    print(f"  confidence_mean:  {fingerprint.confidence_mean:.2f}")
    print(f"  confidence_arc:   {fingerprint.confidence_arc:+.2f}")
    
    print(f"\n[Layer C] TRANSITIONS (Medium Trust) - 6 dims")
    print(f"  additive:     {fingerprint.additive_rate:.2f}")
    print(f"  contrastive:  {fingerprint.contrastive_rate:.2f}")
    print(f"  causal:       {fingerprint.causal_rate:.2f}")
    print(f"  temporal:     {fingerprint.temporal_rate:.2f}")
    print(f"  exemplifying: {fingerprint.exemplifying_rate:.2f}")
    print(f"  reformulating: {fingerprint.reformulating_rate:.2f}")
    
    print(f"\n[Layer D] SYNTACTIC (Lower Trust) - 7 dims")
    print(f"  clause_length: {fingerprint.clause_length_mean:.1f} ± {fingerprint.clause_length_std:.1f}")
    print(f"  rhythm_autocorr: {fingerprint.clause_rhythm_autocorr:+.2f}")
    print(f"  complexity: {fingerprint.sentence_complexity:.1f} clauses/sentence")
    print(f"  comma_density: {fingerprint.comma_density:.1f}")
    
    print(f"\n[FULL VECTOR] {len(fingerprint.to_vector_full())} dimensions")
    print(f"  {fingerprint.to_vector_full()[:10]}...")


if __name__ == "__main__":
    main()
