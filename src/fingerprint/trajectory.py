#!/usr/bin/env python3
"""
Trajectory Analyzer

Computes geometric features of semantic trajectories through embedding space.
Treats text as a path, not a bag of features.

The 5 Trajectory Features:
1. concept_jump_mean     - Average step size between clauses
2. concept_jump_variance - Rhythm regularity (low = metronomic, high = erratic)
3. path_tortuosity       - Wandering vs direct (1.0 = straight line)
4. turning_angle_mean    - Sharp vs gradual direction changes
5. return_rate           - How often the path revisits earlier territory

These are hard to spoof because they emerge from weights, not surface choices.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class TrajectoryFeatures:
    """Geometric features of a semantic trajectory."""
    
    # Core trajectory metrics (5 dims)
    concept_jump_mean: float = 0.0
    concept_jump_variance: float = 0.0
    path_tortuosity: float = 1.0
    turning_angle_mean: float = 0.0
    return_rate: float = 0.0
    
    # Diagnostic (not for classification, for debugging)
    n_waypoints: int = 0
    path_length: float = 0.0
    displacement: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Return the 5 core features as a vector."""
        return np.array([
            self.concept_jump_mean,
            self.concept_jump_variance,
            self.path_tortuosity,
            self.turning_angle_mean,
            self.return_rate,
        ])
    
    @property
    def feature_names(self) -> list[str]:
        return [
            "concept_jump_mean",
            "concept_jump_variance", 
            "path_tortuosity",
            "turning_angle_mean",
            "return_rate",
        ]


class TrajectoryAnalyzer:
    """
    Analyzes the geometric properties of semantic trajectories.
    
    Treats each clause as a waypoint in embedding space.
    The "fingerprint" is how the model navigates between waypoints.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence embedding model.
        
        Args:
            embedding_model: SentenceTransformer model name
        """
        self.model = None
        self.model_name = embedding_model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("[TrajectoryAnalyzer] sentence-transformers not available")
            self.model = None
    
    def segment_clauses(self, text: str) -> list[str]:
        """
        Segment text into clauses (waypoints).
        
        Uses punctuation and conjunctions as boundaries.
        More granular than sentences for better trajectory resolution.
        """
        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clauses = []
        for sentence in sentences:
            # Split on clause boundaries within sentences
            # Semicolons, colons, em-dashes, and coordinating conjunctions
            parts = re.split(
                r'(?<=[,;:])\s+|'           # After comma, semicolon, colon
                r'\s*—\s*|'                  # Em-dash
                r'\s+(?:but|and|or|yet|so|however|therefore|thus|although|while)\s+',
                sentence,
                flags=re.IGNORECASE
            )
            
            for part in parts:
                part = part.strip()
                # Minimum clause length: 3 words
                if len(part.split()) >= 3:
                    clauses.append(part)
        
        return clauses
    
    def embed_clauses(self, clauses: list[str]) -> np.ndarray:
        """
        Embed each clause into semantic space.
        
        Returns:
            Array of shape (n_clauses, embedding_dim)
        """
        if self.model is None:
            # Fallback: random but deterministic embeddings
            import hashlib
            embeddings = []
            for clause in clauses:
                seed = int(hashlib.md5(clause.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                embeddings.append(rng.randn(384))  # Match MiniLM dim
            return np.array(embeddings)
        
        return self.model.encode(clauses, convert_to_numpy=True)
    
    def compute_trajectory(self, embeddings: np.ndarray) -> TrajectoryFeatures:
        """
        Compute geometric features of the embedding trajectory.
        
        Args:
            embeddings: Array of shape (n_waypoints, embedding_dim)
            
        Returns:
            TrajectoryFeatures with the 5 core metrics
        """
        n = len(embeddings)
        
        if n < 2:
            return TrajectoryFeatures(n_waypoints=n)
        
        # === Step Vectors (directions between consecutive points) ===
        steps = np.diff(embeddings, axis=0)  # (n-1, dim)
        
        # === Step Lengths (concept jumps) ===
        step_lengths = np.linalg.norm(steps, axis=1)  # (n-1,)
        
        concept_jump_mean = float(np.mean(step_lengths))
        concept_jump_variance = float(np.var(step_lengths))
        
        # === Path Length & Displacement ===
        path_length = float(np.sum(step_lengths))
        displacement = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
        
        # Tortuosity: how winding is the path?
        # 1.0 = straight line, higher = more wandering
        if displacement > 1e-6:
            path_tortuosity = path_length / displacement
        else:
            # No net displacement means very high tortuosity (circular)
            path_tortuosity = float(path_length * 10)  # Scaled indicator
        
        # === Turning Angles ===
        # Angle between consecutive step vectors
        turning_angles = []
        for i in range(len(steps) - 1):
            v1, v2 = steps[i], steps[i + 1]
            
            # Normalize
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 > 1e-6 and n2 > 1e-6:
                # Cosine of angle
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)  # Radians
                turning_angles.append(angle)
        
        if turning_angles:
            turning_angle_mean = float(np.mean(turning_angles))
        else:
            turning_angle_mean = 0.0
        
        # === Return Rate (Orbit Behavior) ===
        # How often does the trajectory return to within threshold of an earlier point?
        return_threshold = concept_jump_mean * 0.5  # Half a typical step
        returns = 0
        comparisons = 0
        
        for i in range(2, n):
            for j in range(0, i - 1):  # Don't compare to immediate predecessor
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                if dist < return_threshold:
                    returns += 1
                    break  # Count at most one return per position
            comparisons += 1
        
        return_rate = returns / max(comparisons, 1)
        
        return TrajectoryFeatures(
            concept_jump_mean=concept_jump_mean,
            concept_jump_variance=concept_jump_variance,
            path_tortuosity=path_tortuosity,
            turning_angle_mean=turning_angle_mean,
            return_rate=return_rate,
            n_waypoints=n,
            path_length=path_length,
            displacement=displacement,
        )
    
    def analyze(self, text: str) -> TrajectoryFeatures:
        """
        Full trajectory analysis pipeline.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TrajectoryFeatures with geometric metrics
        """
        # Segment into clauses
        clauses = self.segment_clauses(text)
        
        if len(clauses) < 2:
            return TrajectoryFeatures(n_waypoints=len(clauses))
        
        # Embed clauses
        embeddings = self.embed_clauses(clauses)
        
        # Compute trajectory features
        return self.compute_trajectory(embeddings)


# =============================================================================
# EXTENDED FINGERPRINT (15 original + 5 trajectory = 20 dimensions)
# =============================================================================

@dataclass 
class ExtendedFingerprint:
    """
    20-dimensional fingerprint: 15 original SpecHO + 5 trajectory.
    
    Designed for A/B testing against the 15-dim baseline.
    """
    
    # Identity
    sample_id: str = ""
    source_model: str = ""
    
    # === ORIGINAL SPECIO (15 dims) ===
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
    
    # === TRAJECTORY (5 dims) ===
    concept_jump_mean: float = 0.0
    concept_jump_variance: float = 0.0
    path_tortuosity: float = 1.0
    turning_angle_mean: float = 0.0
    return_rate: float = 0.0
    
    # Metadata
    n_clauses: int = 0
    
    def to_vector_15(self) -> np.ndarray:
        """Original 15-dim vector for baseline comparison."""
        return np.array([
            self.phonetic_mean, self.phonetic_std, self.phonetic_max,
            self.structural_mean, self.structural_std, self.structural_max,
            self.semantic_mean, self.semantic_std, self.semantic_max,
            self.cooccurrence_rate, self.geometric_mean,
            self.overall_mean, self.overall_std, self.overall_max,
            self.burstiness,
        ])
    
    def to_vector_20(self) -> np.ndarray:
        """Extended 20-dim vector with trajectory."""
        return np.array([
            # Original 15
            self.phonetic_mean, self.phonetic_std, self.phonetic_max,
            self.structural_mean, self.structural_std, self.structural_max,
            self.semantic_mean, self.semantic_std, self.semantic_max,
            self.cooccurrence_rate, self.geometric_mean,
            self.overall_mean, self.overall_std, self.overall_max,
            self.burstiness,
            # Trajectory 5
            self.concept_jump_mean, self.concept_jump_variance,
            self.path_tortuosity, self.turning_angle_mean,
            self.return_rate,
        ])
    
    def to_vector_trajectory_only(self) -> np.ndarray:
        """Just the 5 trajectory features."""
        return np.array([
            self.concept_jump_mean, self.concept_jump_variance,
            self.path_tortuosity, self.turning_angle_mean,
            self.return_rate,
        ])
    
    @property
    def feature_names_15(self) -> list[str]:
        return [
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max",
            "burstiness",
        ]
    
    @property
    def feature_names_20(self) -> list[str]:
        return self.feature_names_15 + [
            "concept_jump_mean", "concept_jump_variance",
            "path_tortuosity", "turning_angle_mean",
            "return_rate",
        ]


# =============================================================================
# EXTENDED EXTRACTOR
# =============================================================================

class ExtendedExtractor:
    """
    Extracts 20-dimensional fingerprints (15 original + 5 trajectory).
    
    Usage:
        extractor = ExtendedExtractor()
        fp = extractor.extract(text, sample_id="sample_1", source_model="claude")
        
        # A/B test
        vec_15 = fp.to_vector_15()  # Baseline
        vec_20 = fp.to_vector_20()  # Extended
    """
    
    def __init__(self, specHO_path: str = None):
        """
        Initialize extractors.
        
        Args:
            specHO_path: Path to SpecHO installation for original 15 features
        """
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.specHO_detector = None
        
        self._init_specHO(specHO_path)
    
    def _init_specHO(self, path: str = None):
        """Initialize SpecHO detector for original features."""
        if path:
            import sys
            sys.path.insert(0, path)
        
        try:
            from specHO.detector import SpecHODetector
            self.specHO_detector = SpecHODetector(
                semantic_model_path="all-MiniLM-L6-v2"
            )
        except ImportError:
            self.specHO_detector = None
    
    def _extract_original_15(self, text: str) -> dict:
        """Extract original 15 SpecHO features."""
        if self.specHO_detector is None:
            # Mock extraction
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
        
        # Real SpecHO extraction
        result = self.specHO_detector.analyze(text)
        
        if not hasattr(result, 'pair_analyses') or not result.pair_analyses:
            return {k: 0.5 for k in [
                "phonetic_mean", "phonetic_std", "phonetic_max",
                "structural_mean", "structural_std", "structural_max",
                "semantic_mean", "semantic_std", "semantic_max",
                "cooccurrence_rate", "geometric_mean",
                "overall_mean", "overall_std", "overall_max",
                "burstiness",
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
    ) -> ExtendedFingerprint:
        """
        Extract 20-dimensional fingerprint.
        
        Args:
            text: Text to analyze
            sample_id: Sample identifier
            source_model: Model that generated the text
            
        Returns:
            ExtendedFingerprint with 15 original + 5 trajectory features
        """
        # Original 15 features
        original = self._extract_original_15(text)
        
        # Trajectory 5 features
        trajectory = self.trajectory_analyzer.analyze(text)
        
        return ExtendedFingerprint(
            sample_id=sample_id,
            source_model=source_model,
            
            # Original 15
            phonetic_mean=original["phonetic_mean"],
            phonetic_std=original["phonetic_std"],
            phonetic_max=original["phonetic_max"],
            structural_mean=original["structural_mean"],
            structural_std=original["structural_std"],
            structural_max=original["structural_max"],
            semantic_mean=original["semantic_mean"],
            semantic_std=original["semantic_std"],
            semantic_max=original["semantic_max"],
            cooccurrence_rate=original["cooccurrence_rate"],
            geometric_mean=original["geometric_mean"],
            overall_mean=original["overall_mean"],
            overall_std=original["overall_std"],
            overall_max=original["overall_max"],
            burstiness=original["burstiness"],
            
            # Trajectory 5
            concept_jump_mean=trajectory.concept_jump_mean,
            concept_jump_variance=trajectory.concept_jump_variance,
            path_tortuosity=trajectory.path_tortuosity,
            turning_angle_mean=trajectory.turning_angle_mean,
            return_rate=trajectory.return_rate,
            
            n_clauses=trajectory.n_waypoints,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the trajectory analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trajectory analysis demo")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    
    args = parser.parse_args()
    
    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        # Demo text
        text = """
        The implementation of distributed systems requires careful consideration of 
        consistency, availability, and partition tolerance. However, achieving all 
        three simultaneously is provably impossible; this is the CAP theorem. 
        Therefore, architects must choose which properties to prioritize. Some 
        systems favor consistency over availability, while others make the opposite 
        tradeoff. The choice depends on the specific use case and failure modes 
        that matter most. For financial systems, consistency is paramount. For 
        social media, availability often takes precedence.
        """
    
    print("="*60)
    print("TRAJECTORY ANALYSIS")
    print("="*60)
    
    analyzer = TrajectoryAnalyzer()
    
    # Show clause segmentation
    clauses = analyzer.segment_clauses(text)
    print(f"\nClauses ({len(clauses)} waypoints):")
    for i, clause in enumerate(clauses):
        print(f"  [{i}] {clause[:60]}...")
    
    # Compute trajectory
    features = analyzer.analyze(text)
    
    print(f"\nTrajectory Features:")
    print(f"  concept_jump_mean:     {features.concept_jump_mean:.4f}")
    print(f"  concept_jump_variance: {features.concept_jump_variance:.4f}")
    print(f"  path_tortuosity:       {features.path_tortuosity:.4f}")
    print(f"  turning_angle_mean:    {features.turning_angle_mean:.4f} rad ({np.degrees(features.turning_angle_mean):.1f}°)")
    print(f"  return_rate:           {features.return_rate:.4f}")
    
    print(f"\nDiagnostics:")
    print(f"  path_length:  {features.path_length:.4f}")
    print(f"  displacement: {features.displacement:.4f}")
    print(f"  n_waypoints:  {features.n_waypoints}")
    
    # Full extended extraction
    print("\n" + "="*60)
    print("EXTENDED FINGERPRINT (20-dim)")
    print("="*60)
    
    extractor = ExtendedExtractor()
    fp = extractor.extract(text, sample_id="demo", source_model="unknown")
    
    print(f"\n15-dim baseline: {fp.to_vector_15()[:5]}... (showing first 5)")
    print(f"20-dim extended: {fp.to_vector_20()[:5]}... (showing first 5)")
    print(f"5-dim trajectory: {fp.to_vector_trajectory_only()}")


if __name__ == "__main__":
    main()
