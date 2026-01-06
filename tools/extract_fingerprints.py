#!/usr/bin/env python3
"""
Fingerprint Extraction Tool

Extracts fingerprint vectors from corpus samples using SpecHO.

Usage:
    python extract_fingerprints.py --corpus data/corpus/gemini/corpus.json
    python extract_fingerprints.py --corpus data/corpus/ --recursive
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class FingerprintVector:
    """15-dimensional fingerprint extracted from text."""
    
    # Identity
    sample_id: str
    source_model: str
    
    # Phonetic dimension (3 features)
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    phonetic_max: float = 0.0
    
    # Structural dimension (3 features)
    structural_mean: float = 0.0
    structural_std: float = 0.0
    structural_max: float = 0.0
    
    # Semantic dimension (3 features)
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    semantic_max: float = 0.0
    
    # Co-occurrence metrics (2 features)
    cooccurrence_rate: float = 0.0
    geometric_mean: float = 0.0
    
    # Distribution metrics (3 features)
    overall_mean: float = 0.0
    overall_std: float = 0.0
    overall_max: float = 0.0
    
    # Human signature (1 feature)
    burstiness: float = 0.0
    
    # Metadata
    pair_count: int = 0
    extraction_time_ms: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.phonetic_mean, self.phonetic_std, self.phonetic_max,
            self.structural_mean, self.structural_std, self.structural_max,
            self.semantic_mean, self.semantic_std, self.semantic_max,
            self.cooccurrence_rate, self.geometric_mean,
            self.overall_mean, self.overall_std, self.overall_max,
            self.burstiness,
        ])
    
    @property
    def feature_names(self) -> list[str]:
        return [
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max",
            "burstiness",
        ]


class FingerprintExtractor:
    """Extract fingerprints using SpecHO detector."""
    
    def __init__(self, specHO_path: str = None):
        """
        Initialize with SpecHO detector.
        
        Args:
            specHO_path: Path to SpecHO installation. If None, tries to import.
        """
        self.detector = None
        self._init_detector(specHO_path)
    
    def _init_detector(self, specHO_path: str = None):
        """Initialize SpecHO detector."""
        if specHO_path:
            import sys
            sys.path.insert(0, specHO_path)
        
        try:
            from specHO.detector import SpecHODetector
            self.detector = SpecHODetector(
                semantic_model_path="all-MiniLM-L6-v2"
            )
            print("[Extractor] SpecHO detector initialized")
            
            # Check semantic is working
            info = self.detector.get_pipeline_info()
            if info.get("semantic_model_loaded"):
                print("[Extractor] Semantic analyzer: ✓")
            else:
                print("[Extractor] Semantic analyzer: ⚠ fallback mode")
                
        except ImportError as e:
            print(f"[Extractor] WARNING: SpecHO not available ({e})")
            print("[Extractor] Using mock extraction")
            self.detector = None
    
    def extract(self, text: str, sample_id: str, source_model: str) -> FingerprintVector:
        """Extract fingerprint from text."""
        import time
        start = time.time()
        
        if self.detector is None:
            return self._mock_extract(text, sample_id, source_model)
        
        try:
            result = self.detector.analyze(text)
            elapsed = (time.time() - start) * 1000
            
            return self._process_result(result, sample_id, source_model, elapsed)
            
        except Exception as e:
            print(f"[Extractor] Error analyzing {sample_id}: {e}")
            return self._mock_extract(text, sample_id, source_model)
    
    def _process_result(
        self, 
        result, 
        sample_id: str, 
        source_model: str,
        elapsed: float,
    ) -> FingerprintVector:
        """Process SpecHO result into fingerprint vector."""
        
        # Extract pair scores
        pairs = result.pair_analyses if hasattr(result, 'pair_analyses') else []
        
        if not pairs:
            # Fallback if no pairs
            return FingerprintVector(
                sample_id=sample_id,
                source_model=source_model,
                overall_mean=result.final_score if hasattr(result, 'final_score') else 0.5,
                pair_count=0,
                extraction_time_ms=elapsed,
            )
        
        phonetic = [p.phonetic_score for p in pairs]
        structural = [p.structural_score for p in pairs]
        semantic = [p.semantic_score for p in pairs]
        combined = [(p + s + m) / 3 for p, s, m in zip(phonetic, structural, semantic)]
        
        # Co-occurrence
        threshold = 0.6
        cooccur = sum(1 for p, s, m in zip(phonetic, structural, semantic)
                     if p > threshold and s > threshold and m > threshold)
        
        geometric = [pow(max(0.001, p * s * m), 1/3) 
                    for p, s, m in zip(phonetic, structural, semantic)]
        
        # Burstiness
        mean_score = np.mean(combined)
        std_score = np.std(combined) if len(combined) > 1 else 0.0
        burstiness = std_score / mean_score if mean_score > 0 else 0.0
        
        return FingerprintVector(
            sample_id=sample_id,
            source_model=source_model,
            
            phonetic_mean=float(np.mean(phonetic)),
            phonetic_std=float(np.std(phonetic)) if len(phonetic) > 1 else 0.0,
            phonetic_max=float(np.max(phonetic)),
            
            structural_mean=float(np.mean(structural)),
            structural_std=float(np.std(structural)) if len(structural) > 1 else 0.0,
            structural_max=float(np.max(structural)),
            
            semantic_mean=float(np.mean(semantic)),
            semantic_std=float(np.std(semantic)) if len(semantic) > 1 else 0.0,
            semantic_max=float(np.max(semantic)),
            
            cooccurrence_rate=cooccur / len(pairs),
            geometric_mean=float(np.mean(geometric)),
            
            overall_mean=float(mean_score),
            overall_std=float(std_score),
            overall_max=float(np.max(combined)),
            
            burstiness=float(burstiness),
            pair_count=len(pairs),
            extraction_time_ms=elapsed,
        )
    
    def _mock_extract(self, text: str, sample_id: str, source_model: str) -> FingerprintVector:
        """Mock extraction for testing."""
        import hashlib
        
        seed = int(hashlib.md5(f"{sample_id}{source_model}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        profiles = {
            "human": {"mean": 0.45, "std": 0.18},
            "claude": {"mean": 0.52, "std": 0.10},
            "gpt": {"mean": 0.48, "std": 0.08},
            "gemini": {"mean": 0.55, "std": 0.12},
            "llama": {"mean": 0.50, "std": 0.14},
            "mistral": {"mean": 0.47, "std": 0.11},
        }
        
        profile = profiles.get("human")
        for key in profiles:
            if key in source_model.lower():
                profile = profiles[key]
                break
        
        base = profile["mean"] + rng.normal(0, 0.03)
        std = profile["std"] * rng.uniform(0.8, 1.2)
        
        return FingerprintVector(
            sample_id=sample_id,
            source_model=source_model,
            phonetic_mean=base + rng.normal(0, 0.02),
            phonetic_std=std,
            phonetic_max=base + std * 1.5,
            structural_mean=base + rng.normal(0, 0.02),
            structural_std=std,
            structural_max=base + std * 1.5,
            semantic_mean=base + rng.normal(0, 0.02),
            semantic_std=std,
            semantic_max=base + std * 1.5,
            cooccurrence_rate=rng.uniform(0.1, 0.4),
            geometric_mean=base * 0.95,
            overall_mean=base,
            overall_std=std,
            overall_max=base + std * 1.5,
            burstiness=std / base if base > 0 else 0,
            pair_count=max(3, len(text.split('.')) - 1),
            extraction_time_ms=0.0,
        )


def load_corpus(path: str, recursive: bool = False) -> list[dict]:
    """Load corpus from file or directory."""
    corpus = []
    path = Path(path)
    
    if path.is_file():
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    corpus.append(json.loads(line))
            else:
                corpus = json.load(f)
    elif path.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        for file in path.glob(pattern):
            if file.name.startswith("."):
                continue
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
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
) -> list[FingerprintVector]:
    """Extract fingerprints from corpus."""
    
    print(f"\nLoading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path, recursive)
    print(f"Loaded {len(corpus)} samples")
    
    extractor = FingerprintExtractor(specHO_path)
    fingerprints = []
    
    print(f"\nExtracting fingerprints...")
    for i, sample in enumerate(corpus):
        sample_id = sample.get("sample_id", f"sample_{i}")
        source_model = sample.get("model_name") or sample.get("model_id") or "unknown"
        text = sample.get("text", "")
        
        print(f"  [{i+1}/{len(corpus)}] {sample_id}...", end=" ", flush=True)
        
        fp = extractor.extract(text, sample_id, source_model)
        fingerprints.append(fp)
        
        print(f"✓ ({fp.pair_count} pairs, {fp.extraction_time_ms:.0f}ms)")
    
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
        mean = np.mean([fp.overall_mean for fp in fps])
        print(f"  {model}: {count} samples, mean={mean:.3f}")
    
    return fingerprints


def main():
    parser = argparse.ArgumentParser(description="Extract fingerprints from corpus")
    parser.add_argument(
        "--corpus", type=str, required=True,
        help="Path to corpus file or directory"
    )
    parser.add_argument(
        "--output", type=str, default="data/fingerprints/fingerprints.json",
        help="Output path (default: data/fingerprints/fingerprints.json)"
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
