#!/usr/bin/env python3
"""
SpecHO Integrated Fingerprinting System

Unified interface combining:
- SpecHOUnified: 77-dimensional fingerprint extraction
- FingerprintStore: Auto-accumulating reference database
- ModelVerifier: AURORA-compatible verification

Usage:
    from specHO_integrated import SpecHO
    
    # Initialize with auto-storage
    specho = SpecHO(data_dir="data/fingerprints")
    
    # Extract + store in one call
    result = specho.analyze("Some text", model_id="claude-sonnet-4")
    
    # Later: verify against accumulated profiles
    verification = specho.verify("Unknown text", claimed_model="claude-sonnet-4")
    
    # AURORA integration
    trust_evidence = specho.get_aurora_evidence("Agent output", claimed="claude-sonnet-4")
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class TrustLevel(Enum):
    """AURORA trust levels."""
    HIGH = "high"           # >85% confidence
    MEDIUM = "medium"       # 70-85% confidence
    LOW = "low"             # 50-70% confidence
    REJECT = "reject"       # <50% confidence


@dataclass
class AnalysisResult:
    """Result from analyze() call."""
    # Fingerprint
    fingerprint_id: str
    model_id: str
    vector_dims: int
    
    # Scores
    final_score: float
    z_score: float
    confidence: float
    
    # Key features
    path_tortuosity: float
    semantic_mean: float
    punctuation_pair_pct: float
    
    # Metadata
    text_length: int
    extraction_time_ms: float
    stored: bool
    timestamp: str


@dataclass
class VerificationResult:
    """Result from verify() call."""
    # Detection
    detected_model: str
    detected_family: str
    confidence: float
    
    # Claim verification
    claimed_model: Optional[str] = None
    claim_verified: Optional[bool] = None
    claim_confidence: Optional[float] = None
    
    # Trust
    trust_level: TrustLevel = TrustLevel.LOW
    trust_factors: List[str] = None
    
    # Alternatives
    alternatives: List[Dict] = None
    
    # Metadata
    analysis_time_ms: float = 0.0
    timestamp: str = ""


class SpecHO:
    """
    SpecHO Integrated Fingerprinting System.
    
    Main entry point for all fingerprinting operations:
    - Extract fingerprints
    - Auto-store to reference database
    - Verify claims against stored profiles
    - Generate AURORA trust evidence
    
    Example:
        # Initialize
        specho = SpecHO("data/fingerprints")
        
        # During normal operation: analyze and store
        for output in model_outputs:
            result = specho.analyze(output.text, model_id=output.model)
        
        # For verification: check claimed identity
        verification = specho.verify(
            agent_output,
            claimed_model="claude-sonnet-4"
        )
        
        if not verification.claim_verified:
            raise TrustViolation(f"Agent claims {verification.claimed_model} "
                               f"but matches {verification.detected_model}")
    """
    
    def __init__(
        self,
        data_dir: str = "data/fingerprints",
        auto_store: bool = True,
        lazy_load: bool = True,
    ):
        """
        Initialize SpecHO system.
        
        Args:
            data_dir: Directory for fingerprint storage
            auto_store: Automatically store fingerprints after extraction
            lazy_load: Delay loading models until first use
        """
        self.data_dir = Path(data_dir)
        self.auto_store = auto_store
        
        self._store = None
        self._extractor = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            self._init_components()
    
    def _init_components(self):
        """Initialize all components."""
        if self._store is None:
            from fingerprint_store import FingerprintStore
            self._store = FingerprintStore(
                data_dir=str(self.data_dir),
                auto_save=True,
                update_profiles=True,
            )
        
        if self._extractor is None:
            from specho_unified import SpecHOUnified
            self._extractor = SpecHOUnified(lazy_load=self._lazy_load)
    
    @property
    def store(self):
        """Get fingerprint store."""
        if self._store is None:
            self._init_components()
        return self._store
    
    @property
    def extractor(self):
        """Get unified extractor."""
        if self._extractor is None:
            self._init_components()
        return self._extractor
    
    # =========================================================================
    # Core API
    # =========================================================================
    
    def analyze(
        self,
        text: str,
        model_id: str = None,
        store: bool = None,
        prompt: str = "",
        category: str = "",
    ) -> AnalysisResult:
        """
        Analyze text and optionally store fingerprint.
        
        This is the main entry point for fingerprinting. When model_id is
        provided, the fingerprint is stored to the reference database.
        
        Args:
            text: Text to analyze
            model_id: Model identifier (required for storage)
            store: Override auto_store setting
            prompt: Original prompt (optional context)
            category: Text category (optional context)
            
        Returns:
            AnalysisResult with fingerprint and key features
        """
        should_store = store if store is not None else self.auto_store
        
        if should_store and model_id:
            # Extract and store
            fingerprint, stored = self.store.extract_and_store(
                text=text,
                model_id=model_id,
                prompt=prompt,
                category=category,
            )
            fingerprint_id = stored.fingerprint_id if stored else "not_stored"
        else:
            # Extract only
            fingerprint = self.extractor.extract(text)
            fingerprint_id = "not_stored"
        
        # Build result
        fp39 = fingerprint.fingerprint_39d
        micro = fingerprint.micro
        
        return AnalysisResult(
            fingerprint_id=fingerprint_id,
            model_id=model_id or "unknown",
            vector_dims=len(fingerprint.to_flat_vector()),
            final_score=fingerprint.final_score,
            z_score=fingerprint.z_score,
            confidence=fingerprint.confidence,
            path_tortuosity=fp39.path_tortuosity,
            semantic_mean=fp39.semantic_mean,
            punctuation_pair_pct=micro.pair_punctuation.percentage,
            text_length=fingerprint.text_length,
            extraction_time_ms=fingerprint.extraction_time_ms,
            stored=should_store and model_id is not None,
            timestamp=datetime.now().isoformat(),
        )
    
    def verify(
        self,
        text: str,
        claimed_model: str = None,
    ) -> VerificationResult:
        """
        Verify text against stored model profiles.
        
        Args:
            text: Text to verify
            claimed_model: Model identity claim to verify
            
        Returns:
            VerificationResult with detection and claim verification
        """
        result = self.store.verify(text, claimed_model=claimed_model)
        
        # Compute trust level
        conf = result["confidence"]
        if conf >= 0.85:
            trust_level = TrustLevel.HIGH
        elif conf >= 0.70:
            trust_level = TrustLevel.MEDIUM
        elif conf >= 0.50:
            trust_level = TrustLevel.LOW
        else:
            trust_level = TrustLevel.REJECT
        
        # Trust factors
        trust_factors = []
        if conf >= 0.85:
            trust_factors.append("High confidence match")
        if result.get("claim_verified"):
            trust_factors.append("Claimed model verified")
        elif claimed_model:
            trust_factors.append(f"Claimed {claimed_model} but detected {result['detected_model']}")
        
        profile = self.store.get_model_profile(result["detected_model"])
        if profile and profile.get("sample_count", 0) >= 50:
            trust_factors.append(f"Profile based on {profile['sample_count']} samples")
        elif profile:
            trust_factors.append(f"Limited profile ({profile.get('sample_count', 0)} samples)")
        
        return VerificationResult(
            detected_model=result["detected_model"],
            detected_family=result["detected_family"],
            confidence=result["confidence"],
            claimed_model=claimed_model,
            claim_verified=result.get("claim_verified"),
            claim_confidence=result.get("claim_confidence"),
            trust_level=trust_level,
            trust_factors=trust_factors,
            alternatives=result.get("alternatives", []),
            analysis_time_ms=result.get("extraction_time_ms", 0),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
        )
    
    def identify(self, text: str) -> VerificationResult:
        """
        Identify most likely source model for text.
        
        Shorthand for verify() without a claimed model.
        
        Args:
            text: Text to identify
            
        Returns:
            VerificationResult with detected model and confidence
        """
        return self.verify(text, claimed_model=None)
    
    # =========================================================================
    # AURORA Integration
    # =========================================================================
    
    def get_aurora_evidence(
        self,
        text: str,
        claimed_model: str,
    ) -> Dict:
        """
        Generate AURORA-compatible trust evidence.
        
        For integration with AURORA Trust Dimension.
        
        Args:
            text: Agent output text
            claimed_model: Model identity from agent card
            
        Returns:
            AURORA trust evidence dict
        """
        verification = self.verify(text, claimed_model)
        
        # Compute trust score
        if verification.claim_verified:
            trust_score = min(1.0, verification.confidence * 1.1)
        else:
            trust_score = 0.3 * (verification.claim_confidence or 0)
        
        # Build evidence
        evidence = {
            "source": "specHO_fingerprint",
            "type": "statistical",
            "value": verification.confidence if verification.claim_verified else 0.3,
            "weight": 0.6,
            "details": {
                "detected_model": verification.detected_model,
                "detected_family": verification.detected_family,
                "claim_verified": verification.claim_verified,
                "confidence": verification.confidence,
                "alternatives": verification.alternatives[:3] if verification.alternatives else [],
            }
        }
        
        # Generate alerts
        alerts = []
        if not verification.claim_verified:
            alerts.append({
                "level": "warning",
                "message": f"Agent claims {claimed_model} but fingerprint matches {verification.detected_model}",
                "confidence": verification.confidence,
            })
        
        return {
            "dimension": "identity",
            "trust_score": trust_score,
            "verified": verification.claim_verified,
            "evidence": [evidence],
            "alerts": alerts,
            "trust_level": verification.trust_level.value,
            "timestamp": verification.timestamp,
        }
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def batch_analyze(
        self,
        samples: List[Dict],
        progress_callback=None,
    ) -> Dict:
        """
        Process multiple samples.
        
        Args:
            samples: List of dicts with 'text' and 'model_id' keys
            progress_callback: Optional callback(current, total, model_id)
            
        Returns:
            Summary statistics
        """
        return self.store.batch_extract_and_store(samples, progress_callback)
    
    def load_corpus_and_analyze(
        self,
        corpus_path: str,
        progress_callback=None,
    ) -> Dict:
        """
        Load corpus file and analyze all samples.
        
        Handles various corpus formats (samples_500.json, etc.)
        
        Args:
            corpus_path: Path to corpus JSON file
            progress_callback: Optional progress callback
            
        Returns:
            Summary statistics
        """
        with open(corpus_path) as f:
            data = json.load(f)
        
        # Expand corpus to flat sample list
        if isinstance(data, dict) and "samples" in data:
            raw_samples = data["samples"]
        elif isinstance(data, list):
            raw_samples = data
        else:
            raise ValueError(f"Unknown corpus format in {corpus_path}")
        
        samples = []
        for sample in raw_samples:
            # Human text
            human_text = sample.get("human", sample.get("human_text", ""))
            if human_text:
                samples.append({
                    "text": human_text,
                    "model_id": "human",
                    "prompt": sample.get("prompt", ""),
                    "category": sample.get("category", ""),
                })
            
            # AI variants
            ai_variants = sample.get("ai_variants", sample.get("ai_generated", {}))
            for model_id, text in ai_variants.items():
                samples.append({
                    "text": text,
                    "model_id": model_id,
                    "prompt": sample.get("prompt", ""),
                    "category": sample.get("category", ""),
                })
        
        logging.info(f"Loaded {len(samples)} samples from {corpus_path}")
        return self.batch_analyze(samples, progress_callback)
    
    # =========================================================================
    # Status & Reporting
    # =========================================================================
    
    def status(self) -> Dict:
        """Get system status."""
        models = self.store.list_models()
        
        total_samples = 0
        model_stats = {}
        for model in models:
            stats = self.store.get_model_stats(model)
            total_samples += stats.get("sample_count", 0)
            model_stats[model] = {
                "samples": stats.get("sample_count", 0),
                "family": stats.get("model_family", "unknown"),
                "has_covariance": stats.get("has_covariance", False),
            }
        
        return {
            "data_dir": str(self.data_dir),
            "total_models": len(models),
            "total_samples": total_samples,
            "models": model_stats,
            "ready_for_verification": len(models) >= 2 and total_samples >= 10,
        }
    
    def report(self) -> str:
        """Generate text status report."""
        status = self.status()
        
        lines = [
            "=" * 60,
            "SpecHO Fingerprint System Status",
            "=" * 60,
            "",
            f"Data Directory: {status['data_dir']}",
            f"Total Models: {status['total_models']}",
            f"Total Samples: {status['total_samples']}",
            f"Ready for Verification: {'YES' if status['ready_for_verification'] else 'NO'}",
            "",
            "Model Profiles:",
        ]
        
        for model, stats in status['models'].items():
            cov = "✓" if stats['has_covariance'] else "✗"
            lines.append(f"  {model}: {stats['samples']} samples [{stats['family']}] (cov: {cov})")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_instance = None

def get_specho(data_dir: str = "data/fingerprints") -> SpecHO:
    """Get or create default SpecHO instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = SpecHO(data_dir=data_dir)
    return _default_instance


def analyze(text: str, model_id: str = None, **kwargs) -> AnalysisResult:
    """Quick analyze with default instance."""
    return get_specho().analyze(text, model_id, **kwargs)


def verify(text: str, claimed_model: str = None) -> VerificationResult:
    """Quick verify with default instance."""
    return get_specho().verify(text, claimed_model)


def identify(text: str) -> VerificationResult:
    """Quick identify with default instance."""
    return get_specho().identify(text)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for SpecHO integrated system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SpecHO Integrated Fingerprinting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze and store
    python specHO_integrated.py --analyze --text "Some text" --model claude-sonnet-4
    
    # Verify identity claim
    python specHO_integrated.py --verify --file output.txt --claimed gpt-4o
    
    # Identify source model
    python specHO_integrated.py --identify --text "Some text"
    
    # Load and process corpus
    python specHO_integrated.py --corpus samples_500.json
    
    # Show system status
    python specHO_integrated.py --status
    
    # AURORA evidence
    python specHO_integrated.py --aurora --text "Agent output" --claimed claude-sonnet-4
""")
    
    parser.add_argument("--analyze", action="store_true", help="Analyze text")
    parser.add_argument("--verify", action="store_true", help="Verify identity claim")
    parser.add_argument("--identify", action="store_true", help="Identify source model")
    parser.add_argument("--corpus", type=str, metavar="FILE", help="Load and process corpus")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--aurora", action="store_true", help="Generate AURORA evidence")
    
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--model", type=str, help="Model ID for analysis")
    parser.add_argument("--claimed", type=str, help="Claimed model for verification")
    parser.add_argument("--data-dir", type=str, default="data/fingerprints")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    specho = SpecHO(data_dir=args.data_dir)
    
    if args.status:
        print(specho.report())
        return
    
    if args.corpus:
        print(f"Loading corpus: {args.corpus}")
        
        def progress(current, total, model):
            if current % 50 == 0 or current == total:
                print(f"  Progress: {current}/{total} ({model})")
        
        results = specho.load_corpus_and_analyze(args.corpus, progress)
        
        print(f"\nResults:")
        print(f"  Success: {results.get('success', 0)}")
        print(f"  Skipped: {results.get('skipped', 0)}")
        print(f"  Failed: {results.get('failed', 0)}")
        return
    
    # Get text
    text = None
    if args.text:
        text = args.text
    elif args.file:
        text = open(args.file).read()
    
    if args.analyze:
        if not text:
            print("Error: Provide --text or --file")
            return
        
        result = specho.analyze(text, model_id=args.model)
        
        print(f"\nAnalysis Result:")
        print(f"  Fingerprint ID: {result.fingerprint_id}")
        print(f"  Model: {result.model_id}")
        print(f"  Vector dims: {result.vector_dims}")
        print(f"  Final score: {result.final_score:.3f}")
        print(f"  Path tortuosity: {result.path_tortuosity:.2f}")
        print(f"  Semantic mean: {result.semantic_mean:.3f}")
        print(f"  Punct pair %: {result.punctuation_pair_pct:.1f}%")
        print(f"  Stored: {result.stored}")
        return
    
    if args.verify or args.identify:
        if not text:
            print("Error: Provide --text or --file")
            return
        
        result = specho.verify(text, claimed_model=args.claimed)
        
        print(f"\nVerification Result:")
        print(f"  Detected: {result.detected_model}")
        print(f"  Family: {result.detected_family}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Trust Level: {result.trust_level.value.upper()}")
        
        if args.claimed:
            print(f"\n  Claimed: {args.claimed}")
            print(f"  Verified: {'YES ✓' if result.claim_verified else 'NO ✗'}")
        
        print(f"\n  Trust factors:")
        for factor in result.trust_factors or []:
            print(f"    - {factor}")
        return
    
    if args.aurora:
        if not text or not args.claimed:
            print("Error: Provide --text/--file and --claimed")
            return
        
        evidence = specho.get_aurora_evidence(text, args.claimed)
        print("\nAURORA Trust Evidence:")
        print(json.dumps(evidence, indent=2))
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
