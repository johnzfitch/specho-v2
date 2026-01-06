#!/usr/bin/env python3
"""
Model Verifier API

Production interface for model identity verification.
Integrates with AURORA Trust Dimension.

Usage:
    # As library
    from verifier import ModelVerifier
    
    verifier = ModelVerifier("models/fingerprint_db.json")
    result = verifier.verify(text, claimed_model="claude-sonnet-4")
    
    # CLI
    python verifier.py --text "Some text to verify" --claimed claude-sonnet-4
    python verifier.py --file document.txt --identify
"""

import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum


class TrustLevel(Enum):
    """Trust assessment levels."""
    HIGH = "high"           # >85% confidence, verified
    MEDIUM = "medium"       # 70-85% confidence
    LOW = "low"             # 50-70% confidence
    REJECT = "reject"       # <50% confidence, cannot verify


@dataclass
class ModelMatch:
    """A potential model match."""
    model_id: str
    model_family: str
    confidence: float
    distance: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    """Result of model verification."""
    
    # Identification
    detected_model: str
    detected_family: str
    confidence: float
    
    # Alternatives
    alternatives: list[ModelMatch]
    
    # Claim verification (if claimed_model provided)
    claimed_model: Optional[str] = None
    claim_verified: Optional[bool] = None
    claim_confidence: Optional[float] = None
    
    # Trust assessment
    trust_level: TrustLevel = TrustLevel.LOW
    trust_factors: list[str] = None
    
    # Metadata
    fingerprint_version: str = "1.0"
    analysis_time_ms: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["trust_level"] = self.trust_level.value
        d["alternatives"] = [a.to_dict() if hasattr(a, 'to_dict') else a for a in self.alternatives]
        return d


class FingerprintDatabase:
    """Database of reference model fingerprints."""
    
    def __init__(self, path: str = None):
        self.models = {}
        self.version = "1.0"
        self.updated = None
        
        if path and Path(path).exists():
            self.load(path)
    
    def add_model(
        self,
        model_id: str,
        model_family: str,
        centroid: np.ndarray,
        covariance: np.ndarray = None,
        sample_count: int = 0,
    ):
        """Add or update a model profile."""
        self.models[model_id] = {
            "model_id": model_id,
            "model_family": model_family,
            "centroid": centroid.tolist(),
            "covariance": covariance.tolist() if covariance is not None else None,
            "sample_count": sample_count,
            "added": datetime.now().isoformat(),
        }
        self.updated = datetime.now().isoformat()
    
    def get_centroid(self, model_id: str) -> Optional[np.ndarray]:
        """Get centroid for a model."""
        if model_id in self.models:
            return np.array(self.models[model_id]["centroid"])
        return None
    
    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def save(self, path: str):
        """Save database to file."""
        data = {
            "version": self.version,
            "updated": self.updated,
            "models": self.models,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load database from file."""
        with open(path) as f:
            data = json.load(f)
        
        self.version = data.get("version", "1.0")
        self.updated = data.get("updated")
        self.models = data.get("models", {})
    
    @classmethod
    def from_fingerprints(cls, fingerprints: list[dict]) -> 'FingerprintDatabase':
        """Build database from extracted fingerprints."""
        from collections import defaultdict
        
        db = cls()
        
        # Group fingerprints by model
        by_model = defaultdict(list)
        for fp in fingerprints:
            model = fp.get("source_model", "unknown")
            by_model[model].append(fp)
        
        # Compute centroids
        for model, fps in by_model.items():
            vectors = []
            for fp in fps:
                vectors.append(np.array([
                    fp.get("phonetic_mean", 0.5),
                    fp.get("phonetic_std", 0.1),
                    fp.get("phonetic_max", 0.6),
                    fp.get("structural_mean", 0.5),
                    fp.get("structural_std", 0.1),
                    fp.get("structural_max", 0.6),
                    fp.get("semantic_mean", 0.5),
                    fp.get("semantic_std", 0.1),
                    fp.get("semantic_max", 0.6),
                    fp.get("cooccurrence_rate", 0.2),
                    fp.get("geometric_mean", 0.5),
                    fp.get("overall_mean", 0.5),
                    fp.get("overall_std", 0.1),
                    fp.get("overall_max", 0.6),
                    fp.get("burstiness", 0.2),
                ]))
            
            vectors = np.array(vectors)
            centroid = np.mean(vectors, axis=0)
            covariance = np.cov(vectors.T) if len(vectors) > 1 else None
            
            # Infer family from model name
            family = "unknown"
            if "claude" in model.lower():
                family = "anthropic"
            elif "gpt" in model.lower():
                family = "openai"
            elif "gemini" in model.lower():
                family = "google"
            elif "llama" in model.lower():
                family = "meta"
            elif "mistral" in model.lower():
                family = "mistral"
            elif "human" in model.lower():
                family = "human"
            
            db.add_model(
                model_id=model,
                model_family=family,
                centroid=centroid,
                covariance=covariance,
                sample_count=len(fps),
            )
        
        return db


class ModelVerifier:
    """
    Verify model identity claims against output fingerprints.
    
    Example:
        verifier = ModelVerifier("models/fingerprint_db.json")
        
        # Identify unknown text
        result = verifier.identify(text)
        print(f"Most likely: {result.detected_model} ({result.confidence:.0%})")
        
        # Verify a claim
        result = verifier.verify(text, claimed_model="claude-sonnet-4")
        if result.claim_verified:
            print("Claim verified!")
        else:
            print(f"Claim failed - detected {result.detected_model}")
    """
    
    CONFIDENCE_THRESHOLDS = {
        "high": 0.85,
        "medium": 0.70,
        "low": 0.50,
    }
    
    def __init__(
        self,
        database_path: str = None,
        specHO_path: str = None,
    ):
        """
        Initialize the verifier.
        
        Args:
            database_path: Path to fingerprint database JSON
            specHO_path: Path to SpecHO installation for extraction
        """
        self.database = FingerprintDatabase(database_path)
        self.extractor = None
        self.classifier = None
        
        self._init_extractor(specHO_path)
    
    def _init_extractor(self, specHO_path: str = None):
        """Initialize fingerprint extractor."""
        if specHO_path:
            import sys
            sys.path.insert(0, specHO_path)
        
        try:
            from specHO.detector import SpecHODetector
            self._detector = SpecHODetector(semantic_model_path="all-MiniLM-L6-v2")
            print("[Verifier] SpecHO detector initialized")
        except ImportError:
            print("[Verifier] SpecHO not available, using mock extraction")
            self._detector = None
    
    def _extract_fingerprint(self, text: str) -> np.ndarray:
        """Extract fingerprint vector from text."""
        if self._detector is None:
            # Mock extraction
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            return rng.uniform(0.3, 0.7, 15)
        
        result = self._detector.analyze(text)
        
        if not hasattr(result, 'pair_analyses') or not result.pair_analyses:
            return np.full(15, 0.5)
        
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
        burstiness = std_score / mean_score if mean_score > 0 else 0.0
        
        return np.array([
            np.mean(phonetic), np.std(phonetic) if len(phonetic) > 1 else 0, np.max(phonetic),
            np.mean(structural), np.std(structural) if len(structural) > 1 else 0, np.max(structural),
            np.mean(semantic), np.std(semantic) if len(semantic) > 1 else 0, np.max(semantic),
            cooccur / len(pairs), np.mean(geometric),
            mean_score, std_score, np.max(combined),
            burstiness,
        ])
    
    def _compute_distances(self, fingerprint: np.ndarray) -> list[tuple[str, str, float]]:
        """Compute distances to all model centroids."""
        distances = []
        
        for model_id, model_data in self.database.models.items():
            centroid = np.array(model_data["centroid"])
            dist = np.linalg.norm(fingerprint - centroid)
            family = model_data.get("model_family", "unknown")
            distances.append((model_id, family, dist))
        
        distances.sort(key=lambda x: x[2])
        return distances
    
    def _distance_to_confidence(self, distance: float, distances: list) -> float:
        """Convert distance to confidence score using softmax."""
        if not distances:
            return 0.0
        
        # Use negative distances for softmax (smaller distance = higher score)
        all_dists = np.array([d[2] for d in distances])
        
        # Softmax with temperature
        temperature = 0.5
        exp_scores = np.exp(-all_dists / temperature)
        softmax = exp_scores / np.sum(exp_scores)
        
        # Find index of this distance
        idx = np.argmin(np.abs(all_dists - distance))
        return float(softmax[idx])
    
    def identify(self, text: str, top_k: int = 5) -> VerificationResult:
        """
        Identify the most likely source model for text.
        
        Args:
            text: Text to analyze
            top_k: Number of alternatives to return
            
        Returns:
            VerificationResult with identification
        """
        import time
        start = time.time()
        
        # Extract fingerprint
        fingerprint = self._extract_fingerprint(text)
        
        # Compute distances to all models
        distances = self._compute_distances(fingerprint)
        
        if not distances:
            return VerificationResult(
                detected_model="unknown",
                detected_family="unknown",
                confidence=0.0,
                alternatives=[],
                trust_level=TrustLevel.REJECT,
                trust_factors=["No models in database"],
                analysis_time_ms=(time.time() - start) * 1000,
                timestamp=datetime.now().isoformat(),
            )
        
        # Best match
        best_model, best_family, best_dist = distances[0]
        best_conf = self._distance_to_confidence(best_dist, distances)
        
        # Alternatives
        alternatives = []
        for model_id, family, dist in distances[1:top_k+1]:
            conf = self._distance_to_confidence(dist, distances)
            alternatives.append(ModelMatch(
                model_id=model_id,
                model_family=family,
                confidence=conf,
                distance=dist,
            ))
        
        # Determine trust level
        if best_conf >= self.CONFIDENCE_THRESHOLDS["high"]:
            trust_level = TrustLevel.HIGH
        elif best_conf >= self.CONFIDENCE_THRESHOLDS["medium"]:
            trust_level = TrustLevel.MEDIUM
        elif best_conf >= self.CONFIDENCE_THRESHOLDS["low"]:
            trust_level = TrustLevel.LOW
        else:
            trust_level = TrustLevel.REJECT
        
        # Trust factors
        trust_factors = []
        if best_conf >= 0.85:
            trust_factors.append("High confidence match")
        if len(alternatives) > 0 and alternatives[0].confidence < best_conf * 0.5:
            trust_factors.append("Clear separation from alternatives")
        if best_family == "human":
            trust_factors.append("Detected as human-written")
        
        elapsed = (time.time() - start) * 1000
        
        return VerificationResult(
            detected_model=best_model,
            detected_family=best_family,
            confidence=best_conf,
            alternatives=alternatives,
            trust_level=trust_level,
            trust_factors=trust_factors,
            analysis_time_ms=elapsed,
            timestamp=datetime.now().isoformat(),
        )
    
    def verify(
        self,
        text: str,
        claimed_model: str,
    ) -> VerificationResult:
        """
        Verify that text matches a claimed model identity.
        
        Args:
            text: Text to analyze
            claimed_model: Model ID being claimed (e.g., "claude-sonnet-4")
            
        Returns:
            VerificationResult with claim verification
        """
        # Get identification
        result = self.identify(text)
        
        # Check claim
        result.claimed_model = claimed_model
        
        # Normalize for comparison
        claimed_lower = claimed_model.lower()
        detected_lower = result.detected_model.lower()
        
        # Check if claim matches (fuzzy)
        match = False
        if claimed_lower == detected_lower:
            match = True
        elif claimed_lower in detected_lower or detected_lower in claimed_lower:
            match = True
        
        # Get confidence for claimed model specifically
        claim_conf = 0.0
        for alt in result.alternatives:
            if claimed_lower in alt.model_id.lower():
                claim_conf = max(claim_conf, alt.confidence)
        if claimed_lower in result.detected_model.lower():
            claim_conf = result.confidence
        
        result.claim_verified = match or claim_conf >= self.CONFIDENCE_THRESHOLDS["medium"]
        result.claim_confidence = claim_conf if not match else result.confidence
        
        # Adjust trust factors
        if result.claim_verified:
            result.trust_factors.append("Claimed model verified")
        else:
            result.trust_factors.append(f"Claimed {claimed_model} but detected {result.detected_model}")
            if result.trust_level in (TrustLevel.HIGH, TrustLevel.MEDIUM):
                result.trust_level = TrustLevel.LOW
        
        return result


# =============================================================================
# AURORA Integration
# =============================================================================

@dataclass
class AURORATrustEvidence:
    """Evidence for AURORA trust calculation."""
    source: str
    type: str
    value: float
    weight: float
    details: dict = None


class AURORATrustAdapter:
    """
    Adapter for AURORA protocol trust dimension.
    
    Converts SpecHO fingerprint verification into AURORA trust evidence.
    """
    
    def __init__(self, verifier: ModelVerifier):
        self.verifier = verifier
    
    def verify_agent_identity(
        self,
        agent_output: str,
        claimed_model: str,
    ) -> dict:
        """
        Verify agent identity claim for AURORA trust dimension.
        
        Args:
            agent_output: Text output from the agent
            claimed_model: Model ID from agent card
            
        Returns:
            AURORA-compatible trust assessment
        """
        result = self.verifier.verify(agent_output, claimed_model)
        
        # Build evidence
        evidence = AURORATrustEvidence(
            source="specHO_fingerprint",
            type="statistical",
            value=result.confidence if result.claim_verified else 0.3,
            weight=0.6,  # Fingerprint is strong but not definitive
            details={
                "detected_model": result.detected_model,
                "claim_verified": result.claim_verified,
                "alternatives": [a.to_dict() for a in result.alternatives[:3]],
            }
        )
        
        # Compute trust score
        trust_score = 0.0
        if result.claim_verified:
            trust_score = min(1.0, result.confidence * 1.1)  # Slight boost for verified
        else:
            # Penalize but don't zero out - could be model drift
            trust_score = 0.3 * result.claim_confidence
        
        # Generate alerts
        alerts = []
        if not result.claim_verified:
            alerts.append({
                "level": "warning",
                "message": f"Agent claims {claimed_model} but fingerprint matches {result.detected_model}",
                "confidence": result.confidence,
            })
        
        return {
            "dimension": "identity",
            "trust_score": trust_score,
            "verified": result.claim_verified,
            "evidence": [asdict(evidence)],
            "alerts": alerts,
            "timestamp": result.timestamp,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model Identity Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Identify text
    python verifier.py --text "Some text to analyze" --identify
    
    # Verify a claim
    python verifier.py --text "Some text" --claimed claude-sonnet-4
    
    # From file
    python verifier.py --file document.txt --identify
    
    # Build database from fingerprints
    python verifier.py --build-db fingerprints.json --output models/db.json
""")
    
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--claimed", type=str, help="Claimed model ID")
    parser.add_argument("--identify", action="store_true", help="Identify source model")
    parser.add_argument("--database", type=str, default="data/models/fingerprint_db.json",
                       help="Path to fingerprint database")
    parser.add_argument("--specHO", type=str, help="Path to SpecHO installation")
    parser.add_argument("--build-db", type=str, metavar="FINGERPRINTS",
                       help="Build database from fingerprints file")
    parser.add_argument("--output", type=str, help="Output path for built database")
    
    args = parser.parse_args()
    
    # Build database mode
    if args.build_db:
        print(f"Building database from {args.build_db}...")
        with open(args.build_db) as f:
            fingerprints = json.load(f)
        
        db = FingerprintDatabase.from_fingerprints(fingerprints)
        
        output_path = args.output or "data/models/fingerprint_db.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        db.save(output_path)
        
        print(f"Database saved to {output_path}")
        print(f"Models: {', '.join(db.list_models())}")
        return
    
    # Get text
    text = None
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        print("Provide --text or --file")
        return
    
    # Initialize verifier
    verifier = ModelVerifier(
        database_path=args.database,
        specHO_path=args.specHO,
    )
    
    # Run verification
    if args.claimed:
        result = verifier.verify(text, args.claimed)
        
        print(f"\n{'='*60}")
        print("VERIFICATION RESULT")
        print(f"{'='*60}")
        print(f"\nClaimed: {args.claimed}")
        print(f"Detected: {result.detected_model}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"\nClaim Verified: {'YES ✓' if result.claim_verified else 'NO ✗'}")
        print(f"Trust Level: {result.trust_level.value.upper()}")
        
    else:
        result = verifier.identify(text)
        
        print(f"\n{'='*60}")
        print("IDENTIFICATION RESULT")
        print(f"{'='*60}")
        print(f"\nDetected Model: {result.detected_model}")
        print(f"Model Family: {result.detected_family}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Trust Level: {result.trust_level.value.upper()}")
        
        if result.alternatives:
            print("\nAlternatives:")
            for alt in result.alternatives:
                print(f"  {alt.model_id}: {alt.confidence:.1%}")
    
    print(f"\nAnalysis time: {result.analysis_time_ms:.1f}ms")


if __name__ == "__main__":
    main()
