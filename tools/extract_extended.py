#!/usr/bin/env python3
"""
Extended Fingerprint Extraction (20-dim)

Extracts 20-dimensional fingerprints:
- 15 original SpecHO features (echo patterns)
- 5 trajectory features (semantic navigation geometry)

For A/B testing against the 15-dim baseline.

Usage:
    python extract_extended.py --corpus data/corpus/merged/corpus.json
    python extract_extended.py --corpus data/corpus/ --recursive
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class ExtendedFingerprint:
    """20-dimensional fingerprint: 15 original + 5 trajectory."""
    
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
    n_pairs: int = 0
    extraction_time_ms: float = 0.0


class ExtendedExtractor:
    """
    Extracts 20-dimensional fingerprints.
    
    Combines:
    - Original 15-dim SpecHO (echo patterns at clause boundaries)
    - New 5-dim trajectory (geometric properties of semantic path)
    """
    
    def __init__(self, specHO_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.specHO_detector = None
        self.sentence_model = None
        self.embedding_model_name = embedding_model
        
        self._init_specHO(specHO_path)
        self._init_embedding_model()
    
    def _init_specHO(self, path: str = None):
        """Initialize SpecHO detector."""
        if path:
            sys.path.insert(0, path)
        
        try:
            from specHO.detector import SpecHODetector
            self.specHO_detector = SpecHODetector(
                semantic_model_path="all-MiniLM-L6-v2"
            )
            print("[ExtendedExtractor] SpecHO detector: ✓")
        except ImportError:
            print("[ExtendedExtractor] SpecHO detector: ✗ (using mock)")
            self.specHO_detector = None
    
    def _init_embedding_model(self):
        """Initialize sentence embedding model for trajectory."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            print(f"[ExtendedExtractor] Sentence model ({self.embedding_model_name}): ✓")
        except ImportError:
            print("[ExtendedExtractor] Sentence model: ✗ (using mock)")
            self.sentence_model = None
    
    def _segment_clauses(self, text: str) -> list[str]:
        """Segment text into clauses (waypoints for trajectory)."""
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clauses = []
        for sentence in sentences:
            parts = re.split(
                r'(?<=[,;:])\s+|'
                r'\s*—\s*|'
                r'\s+(?:but|and|or|yet|so|however|therefore|thus|although|while)\s+',
                sentence,
                flags=re.IGNORECASE
            )
            
            for part in parts:
                part = part.strip()
                if len(part.split()) >= 3:
                    clauses.append(part)
        
        return clauses
    
    def _embed_clauses(self, clauses: list[str]) -> np.ndarray:
        """Embed clauses into semantic space."""
        if self.sentence_model is None:
            import hashlib
            embeddings = []
            for clause in clauses:
                seed = int(hashlib.md5(clause.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                embeddings.append(rng.randn(384))
            return np.array(embeddings) if embeddings else np.zeros((0, 384))
        
        if not clauses:
            return np.zeros((0, 384))
        
        return self.sentence_model.encode(clauses, convert_to_numpy=True)
    
    def _compute_trajectory_features(self, embeddings: np.ndarray) -> dict:
        """Compute 5 trajectory features from embedding sequence."""
        n = len(embeddings)
        
        if n < 2:
            return {
                "concept_jump_mean": 0.0,
                "concept_jump_variance": 0.0,
                "path_tortuosity": 1.0,
                "turning_angle_mean": 0.0,
                "return_rate": 0.0,
            }
        
        # Step vectors and lengths
        steps = np.diff(embeddings, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)
        
        concept_jump_mean = float(np.mean(step_lengths))
        concept_jump_variance = float(np.var(step_lengths))
        
        # Path length and displacement
        path_length = float(np.sum(step_lengths))
        displacement = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
        
        if displacement > 1e-6:
            path_tortuosity = path_length / displacement
        else:
            path_tortuosity = path_length * 10  # High tortuosity for circular paths
        
        # Turning angles
        turning_angles = []
        for i in range(len(steps) - 1):
            v1, v2 = steps[i], steps[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if n1 > 1e-6 and n2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                turning_angles.append(np.arccos(cos_angle))
        
        turning_angle_mean = float(np.mean(turning_angles)) if turning_angles else 0.0
        
        # Return rate
        return_threshold = concept_jump_mean * 0.5 if concept_jump_mean > 0 else 0.1
        returns = 0
        comparisons = 0
        
        for i in range(2, n):
            for j in range(0, i - 1):
                if np.linalg.norm(embeddings[i] - embeddings[j]) < return_threshold:
                    returns += 1
                    break
            comparisons += 1
        
        return_rate = returns / max(comparisons, 1)
        
        return {
            "concept_jump_mean": concept_jump_mean,
            "concept_jump_variance": concept_jump_variance,
            "path_tortuosity": float(path_tortuosity),
            "turning_angle_mean": turning_angle_mean,
            "return_rate": float(return_rate),
        }
    
    def _extract_original_15(self, text: str) -> dict:
        """Extract original 15 SpecHO features."""
        if self.specHO_detector is None:
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
                "n_pairs": int(rng.uniform(5, 15)),
            }
        
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
            "n_pairs": len(pairs),
        }
    
    def extract(self, text: str, sample_id: str = "", source_model: str = "") -> ExtendedFingerprint:
        """
        Extract 20-dimensional fingerprint.
        
        Args:
            text: Text to analyze
            sample_id: Sample identifier
            source_model: Model that generated the text
            
        Returns:
            ExtendedFingerprint with 15 original + 5 trajectory features
        """
        start = time.time()
        
        # Original 15 features
        original = self._extract_original_15(text)
        
        # Trajectory 5 features
        clauses = self._segment_clauses(text)
        embeddings = self._embed_clauses(clauses)
        trajectory = self._compute_trajectory_features(embeddings)
        
        elapsed = (time.time() - start) * 1000
        
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
            concept_jump_mean=trajectory["concept_jump_mean"],
            concept_jump_variance=trajectory["concept_jump_variance"],
            path_tortuosity=trajectory["path_tortuosity"],
            turning_angle_mean=trajectory["turning_angle_mean"],
            return_rate=trajectory["return_rate"],
            
            n_clauses=len(clauses),
            n_pairs=original.get("n_pairs", 0),
            extraction_time_ms=elapsed,
        )


def load_corpus(path: str, recursive: bool = False) -> list[dict]:
    """Load corpus from file or directory."""
    corpus = []
    path = Path(path)
    
    if path.is_file():
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        corpus.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    corpus = data
    elif path.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        for file in path.glob(pattern):
            if file.name.startswith("."):
                continue
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            corpus.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        corpus.extend(data)
    
    return corpus


def extract_fingerprints(
    corpus_path: str,
    output_path: str,
    specHO_path: str = None,
    recursive: bool = False,
) -> list[ExtendedFingerprint]:
    """Extract extended fingerprints from corpus."""
    
    print(f"\nLoading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path, recursive)
    print(f"Loaded {len(corpus)} samples")
    
    extractor = ExtendedExtractor(specHO_path)
    fingerprints = []
    
    print(f"\nExtracting 20-dim fingerprints...")
    for i, sample in enumerate(corpus):
        sample_id = sample.get("sample_id", f"sample_{i}")
        source_model = sample.get("model_name") or sample.get("model_id") or "unknown"
        text = sample.get("text", "")
        
        if not text:
            continue
        
        print(f"  [{i+1}/{len(corpus)}] {sample_id} ({source_model})...", end=" ", flush=True)
        
        fp = extractor.extract(text, sample_id, source_model)
        fingerprints.append(fp)
        
        print(f"✓ ({fp.n_clauses} clauses, {fp.extraction_time_ms:.0f}ms)")
    
    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        json.dump([asdict(fp) for fp in fingerprints], f, indent=2)
    
    print(f"\nSaved {len(fingerprints)} fingerprints to {output}")
    
    # Summary
    from collections import Counter
    model_counts = Counter(fp.source_model for fp in fingerprints)
    print("\nBy model:")
    for model, count in sorted(model_counts.items()):
        fps = [fp for fp in fingerprints if fp.source_model == model]
        mean_traj = np.mean([fp.concept_jump_mean for fp in fps])
        mean_tort = np.mean([fp.path_tortuosity for fp in fps])
        print(f"  {model}: {count} samples, jump={mean_traj:.3f}, tortuosity={mean_tort:.2f}")
    
    return fingerprints


def main():
    parser = argparse.ArgumentParser(
        description="Extract 20-dim extended fingerprints (15 original + 5 trajectory)",
        epilog="""
The 5 trajectory features measure how the model navigates semantic space:
- concept_jump_mean:     Average step size between clauses
- concept_jump_variance: Rhythm regularity
- path_tortuosity:       Wandering vs direct path
- turning_angle_mean:    Sharp vs gradual direction changes
- return_rate:           How often the path revisits territory
"""
    )
    parser.add_argument(
        "--corpus", type=str, required=True,
        help="Path to corpus file or directory"
    )
    parser.add_argument(
        "--output", type=str, default="data/fingerprints/extended.json",
        help="Output path (default: data/fingerprints/extended.json)"
    )
    parser.add_argument(
        "--specHO", type=str, default=None,
        help="Path to SpecHO installation"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search directories"
    )
    
    args = parser.parse_args()
    
    extract_fingerprints(
        corpus_path=args.corpus,
        output_path=args.output,
        specHO_path=args.specHO,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
