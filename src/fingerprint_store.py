#!/usr/bin/env python3
"""
SpecHO Fingerprint Store

Auto-accumulating reference database that stores fingerprints per model.
Every extraction automatically updates the model profile.

Usage:
    from fingerprint_store import FingerprintStore
    
    store = FingerprintStore("data/fingerprints")
    
    # Extract and store in one call
    result = store.extract_and_store(text, model_id="claude-sonnet-4")
    
    # Get model profile
    profile = store.get_model_profile("claude-sonnet-4")
    print(f"Samples: {profile['sample_count']}")
    print(f"Centroid: {profile['centroid']}")
    
    # Verify against stored profiles
    verification = store.verify(text, claimed_model="claude-sonnet-4")
"""

import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

# Import 137D registry (primary) or fall back to old unified extractor
try:
    from .extractors.registry import get_registry, CombinedResult
    REGISTRY_AVAILABLE = True
except ImportError:
    try:
        from src.extractors.registry import get_registry, CombinedResult
        REGISTRY_AVAILABLE = True
    except ImportError:
        REGISTRY_AVAILABLE = False

# Legacy fallback
try:
    from specho_unified.specho_unified import SpecHOUnified, UnifiedFingerprint
except ImportError:
    SpecHOUnified = None
    UnifiedFingerprint = None


@dataclass
class ModelProfile:
    """Aggregated model fingerprint profile."""
    model_id: str
    model_family: str
    sample_count: int = 0
    
    # Centroid (mean fingerprint)
    centroid: List[float] = field(default_factory=list)
    
    # Covariance for Mahalanobis distance
    covariance: List[List[float]] = None
    
    # Feature-level stats for interpretability
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    first_seen: str = ""
    last_updated: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelProfile':
        return cls(**data)


@dataclass
class StoredFingerprint:
    """Individual stored fingerprint with metadata."""
    fingerprint_id: str
    model_id: str
    timestamp: str
    
    # The fingerprint vector (77D unified or 39D cognitive)
    vector: List[float]
    feature_names: List[str]
    
    # Full fingerprint data
    full_data: Dict = field(default_factory=dict)
    
    # Extraction metadata
    text_hash: str = ""  # SHA256 of input text (for dedup)
    text_length: int = 0
    extraction_time_ms: float = 0.0
    
    # Optional context
    prompt: str = ""
    category: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class FingerprintStore:
    """
    Auto-accumulating fingerprint database.
    
    Stores fingerprints per model and maintains running statistics
    for model profiles (centroid, covariance).
    
    Storage structure:
        data_dir/
        ├── profiles.json          # Aggregated model profiles
        ├── config.json            # Store configuration
        ├── fingerprints/
        │   ├── claude-sonnet-4/
        │   │   ├── fp_001.json
        │   │   ├── fp_002.json
        │   │   └── ...
        │   ├── gpt-4o/
        │   │   └── ...
        │   └── human/
        │       └── ...
        └── audit_log.jsonl        # Extraction history
    """
    
    MODEL_FAMILIES = {
        "claude": "anthropic",
        "gpt": "openai",
        "gemini": "google",
        "llama": "meta",
        "mistral": "mistralai",
        "qwen": "alibaba",
        "yi": "01ai",
        "human": "human",
    }
    
    def __init__(
        self,
        data_dir: str = "data/fingerprints",
        auto_save: bool = True,
        update_profiles: bool = True,
    ):
        """
        Initialize fingerprint store.
        
        Args:
            data_dir: Directory for fingerprint storage
            auto_save: Automatically save after each extraction
            update_profiles: Update model profiles after each extraction
        """
        self.data_dir = Path(data_dir)
        self.auto_save = auto_save
        self.update_profiles = update_profiles
        
        # Thread safety
        self._lock = threading.Lock()
        
        # In-memory caches
        self._profiles: Dict[str, ModelProfile] = {}
        self._fingerprints_by_model: Dict[str, List[StoredFingerprint]] = defaultdict(list)
        
        # Initialize extractor
        self._extractor = None
        
        # Initialize storage
        self._init_storage()
        self._load_existing()
    
    def _init_storage(self):
        """Create storage directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "fingerprints").mkdir(exist_ok=True)
        
        # Create config if not exists
        config_path = self.data_dir / "config.json"
        if not config_path.exists():
            feature_version = "registry_137d" if REGISTRY_AVAILABLE else "unified_77d"
            config = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "feature_version": feature_version,
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
    
    def _load_existing(self):
        """Load existing profiles and fingerprints."""
        # Load profiles
        profiles_path = self.data_dir / "profiles.json"
        if profiles_path.exists():
            with open(profiles_path) as f:
                data = json.load(f)
            for model_id, profile_data in data.items():
                self._profiles[model_id] = ModelProfile.from_dict(profile_data)
        
        # Load fingerprint metadata (not full data, for memory efficiency)
        fp_dir = self.data_dir / "fingerprints"
        for model_dir in fp_dir.iterdir():
            if model_dir.is_dir():
                model_id = model_dir.name
                count = len(list(model_dir.glob("*.json")))
                logging.debug(f"Found {count} fingerprints for {model_id}")
    
    def _get_extractor(self):
        """Lazy-load extractor (137D registry or legacy unified)."""
        if self._extractor is None:
            if REGISTRY_AVAILABLE:
                self._extractor = get_registry(quiet=True)
                self._use_registry = True
            elif SpecHOUnified is not None:
                self._extractor = SpecHOUnified()
                self._use_registry = False
            else:
                raise ImportError("No extractor available. Install extractors package.")
        return self._extractor
    
    def _infer_family(self, model_id: str) -> str:
        """Infer model family from model ID."""
        model_lower = model_id.lower()
        for key, family in self.MODEL_FAMILIES.items():
            if key in model_lower:
                return family
        return "unknown"
    
    def _hash_text(self, text: str) -> str:
        """Generate SHA256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _generate_id(self, model_id: str) -> str:
        """Generate unique fingerprint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{model_id}_{timestamp}_{rand}"
    
    # =========================================================================
    # Core API
    # =========================================================================
    
    def extract_and_store(
        self,
        text: str,
        model_id: str,
        prompt: str = "",
        category: str = "",
        skip_duplicate: bool = True,
    ) -> Tuple[UnifiedFingerprint, StoredFingerprint]:
        """
        Extract fingerprint and store it in the database.
        
        This is the main entry point. Every call:
        1. Extracts the full fingerprint
        2. Stores it in the model's directory
        3. Updates the model profile
        4. Logs to audit trail
        
        Args:
            text: Text to fingerprint
            model_id: Model identifier (e.g., "claude-sonnet-4", "gpt-4o", "human")
            prompt: Original prompt (optional, for context)
            category: Text category (optional, for analysis)
            skip_duplicate: Skip if identical text already fingerprinted
            
        Returns:
            Tuple of (UnifiedFingerprint, StoredFingerprint)
        """
        with self._lock:
            # Check for duplicate
            text_hash = self._hash_text(text)
            if skip_duplicate and self._is_duplicate(model_id, text_hash):
                logging.info(f"Skipping duplicate text for {model_id}")
                # Return existing fingerprint
                existing = self._get_by_hash(model_id, text_hash)
                if existing:
                    return None, existing
            
            # Extract fingerprint
            extractor = self._get_extractor()

            if getattr(self, '_use_registry', False):
                # 137D registry path
                result = extractor.extract_all(text)
                feature_names = sorted(result.features.keys())
                # Ensure standard Python floats for JSON serialization
                vector = [float(result.features[f]) for f in feature_names]
                full_data = {
                    "features": {k: float(v) for k, v in result.features.items()},
                    "by_group": {k: v.to_dict() for k, v in result.by_group.items()},
                    "extraction_time_ms": float(result.total_time_ms),
                    "n_features": result.n_features,
                }
                extraction_time_ms = result.total_time_ms
            else:
                # Legacy unified path
                fingerprint = extractor.extract(text)
                flat_vector = fingerprint.to_flat_vector()
                feature_names = list(flat_vector.keys())
                vector = list(flat_vector.values())
                full_data = fingerprint.to_dict()
                extraction_time_ms = fingerprint.extraction_time_ms
            
            stored = StoredFingerprint(
                fingerprint_id=self._generate_id(model_id),
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                vector=vector,
                feature_names=feature_names,
                full_data=full_data,
                text_hash=text_hash,
                text_length=len(text),
                extraction_time_ms=extraction_time_ms,
                prompt=prompt,
                category=category,
            )
            
            # Store fingerprint
            self._store_fingerprint(stored)
            
            # Update model profile
            if self.update_profiles:
                self._update_profile(model_id, vector, feature_names)
            
            # Audit log
            self._log_extraction(stored)
            
            # Auto-save profiles
            if self.auto_save:
                self._save_profiles()

            # Return result (None for registry path since there's no UnifiedFingerprint)
            return None, stored
    
    def _store_fingerprint(self, stored: StoredFingerprint):
        """Store fingerprint to disk."""
        model_dir = self.data_dir / "fingerprints" / stored.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        fp_path = model_dir / f"{stored.fingerprint_id}.json"
        with open(fp_path, "w") as f:
            json.dump(stored.to_dict(), f, indent=2)
        
        # Update in-memory cache
        self._fingerprints_by_model[stored.model_id].append(stored)
    
    def _update_profile(
        self,
        model_id: str,
        vector: List[float],
        feature_names: List[str],
    ):
        """Update model profile with new fingerprint."""
        vector = np.array(vector)
        
        if model_id not in self._profiles:
            # Create new profile
            self._profiles[model_id] = ModelProfile(
                model_id=model_id,
                model_family=self._infer_family(model_id),
                sample_count=0,
                centroid=[],
                first_seen=datetime.now().isoformat(),
            )
        
        profile = self._profiles[model_id]
        n = profile.sample_count
        
        if n == 0:
            # First sample
            profile.centroid = vector.tolist()
            profile.feature_means = dict(zip(feature_names, vector.tolist()))
        else:
            # Incremental mean update: new_mean = old_mean + (x - old_mean) / (n + 1)
            old_centroid = np.array(profile.centroid)
            new_centroid = old_centroid + (vector - old_centroid) / (n + 1)
            profile.centroid = new_centroid.tolist()
            
            # Update feature means
            for name, val in zip(feature_names, vector):
                old_val = profile.feature_means.get(name, val)
                profile.feature_means[name] = old_val + (val - old_val) / (n + 1)
        
        profile.sample_count = n + 1
        profile.last_updated = datetime.now().isoformat()
        
        # Update covariance every 10 samples (expensive)
        if profile.sample_count >= 10 and profile.sample_count % 10 == 0:
            self._update_covariance(model_id)
    
    def _update_covariance(self, model_id: str):
        """Recompute covariance matrix for model."""
        model_dir = self.data_dir / "fingerprints" / model_id
        if not model_dir.exists():
            return
        
        vectors = []
        for fp_path in model_dir.glob("*.json"):
            with open(fp_path) as f:
                data = json.load(f)
            vectors.append(data["vector"])
        
        if len(vectors) < 2:
            return
        
        vectors = np.array(vectors)
        try:
            cov = np.cov(vectors.T)
            self._profiles[model_id].covariance = cov.tolist()
        except Exception as e:
            logging.warning(f"Failed to compute covariance for {model_id}: {e}")
    
    def _is_duplicate(self, model_id: str, text_hash: str) -> bool:
        """Check if text already fingerprinted for this model."""
        model_dir = self.data_dir / "fingerprints" / model_id
        if not model_dir.exists():
            return False
        
        for fp_path in model_dir.glob("*.json"):
            with open(fp_path) as f:
                data = json.load(f)
            if data.get("text_hash") == text_hash:
                return True
        return False
    
    def _get_by_hash(self, model_id: str, text_hash: str) -> Optional[StoredFingerprint]:
        """Get stored fingerprint by text hash."""
        model_dir = self.data_dir / "fingerprints" / model_id
        if not model_dir.exists():
            return None
        
        for fp_path in model_dir.glob("*.json"):
            with open(fp_path) as f:
                data = json.load(f)
            if data.get("text_hash") == text_hash:
                return StoredFingerprint(**data)
        return None
    
    def _log_extraction(self, stored: StoredFingerprint):
        """Append to audit log."""
        log_path = self.data_dir / "audit_log.jsonl"
        entry = {
            "timestamp": stored.timestamp,
            "model_id": stored.model_id,
            "fingerprint_id": stored.fingerprint_id,
            "text_length": stored.text_length,
            "extraction_time_ms": stored.extraction_time_ms,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _save_profiles(self):
        """Save profiles to disk."""
        profiles_path = self.data_dir / "profiles.json"
        data = {
            model_id: profile.to_dict()
            for model_id, profile in self._profiles.items()
        }
        with open(profiles_path, "w") as f:
            json.dump(data, f, indent=2)
    
    # =========================================================================
    # Query API
    # =========================================================================
    
    def get_model_profile(self, model_id: str) -> Optional[Dict]:
        """Get profile for a model."""
        if model_id in self._profiles:
            return self._profiles[model_id].to_dict()
        return None
    
    def list_models(self) -> List[str]:
        """List all models with stored fingerprints."""
        return list(self._profiles.keys())
    
    def get_model_stats(self, model_id: str) -> Dict:
        """Get detailed statistics for a model."""
        profile = self._profiles.get(model_id)
        if not profile:
            return {"error": "Model not found"}
        
        return {
            "model_id": model_id,
            "model_family": profile.model_family,
            "sample_count": profile.sample_count,
            "first_seen": profile.first_seen,
            "last_updated": profile.last_updated,
            "has_covariance": profile.covariance is not None,
            "centroid_dims": len(profile.centroid),
            "top_features": dict(sorted(
                profile.feature_means.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]),
        }
    
    def get_all_fingerprints(self, model_id: str) -> List[Dict]:
        """Load all fingerprints for a model."""
        model_dir = self.data_dir / "fingerprints" / model_id
        if not model_dir.exists():
            return []
        
        fingerprints = []
        for fp_path in sorted(model_dir.glob("*.json")):
            with open(fp_path) as f:
                fingerprints.append(json.load(f))
        return fingerprints
    
    # =========================================================================
    # Verification API
    # =========================================================================
    
    def verify(
        self,
        text: str,
        claimed_model: str = None,
    ) -> Dict:
        """
        Verify text against stored model profiles.

        Args:
            text: Text to verify
            claimed_model: Optional model claim to verify

        Returns:
            Verification result with detected model and confidence
        """
        extractor = self._get_extractor()

        if getattr(self, '_use_registry', False):
            result = extractor.extract_all(text)
            feature_names = sorted(result.features.keys())
            vector = np.array([result.features[f] for f in feature_names])
            extraction_time_ms = result.total_time_ms
        else:
            fingerprint = extractor.extract(text)
            vector = np.array(list(fingerprint.to_flat_vector().values()))
            extraction_time_ms = fingerprint.extraction_time_ms
        
        # Compare against all profiles
        distances = {}
        for model_id, profile in self._profiles.items():
            if not profile.centroid:
                continue
            
            centroid = np.array(profile.centroid)
            
            # Euclidean distance
            dist = np.linalg.norm(vector - centroid)
            distances[model_id] = dist
        
        if not distances:
            return {
                "error": "No model profiles available",
                "detected_model": None,
                "confidence": 0.0,
            }
        
        # Sort by distance
        sorted_models = sorted(distances.items(), key=lambda x: x[1])
        
        # Convert distance to confidence (inverse, normalized)
        min_dist = sorted_models[0][1]
        max_dist = sorted_models[-1][1] if len(sorted_models) > 1 else min_dist + 1
        
        def dist_to_conf(d):
            if max_dist == min_dist:
                return 0.5
            normalized = (d - min_dist) / (max_dist - min_dist)
            return max(0, 1 - normalized)
        
        detected_model = sorted_models[0][0]
        detected_conf = dist_to_conf(sorted_models[0][1])
        
        alternatives = [
            {
                "model_id": model_id,
                "distance": float(dist),
                "confidence": dist_to_conf(dist),
            }
            for model_id, dist in sorted_models[:5]
        ]
        
        result = {
            "detected_model": detected_model,
            "detected_family": self._profiles[detected_model].model_family,
            "confidence": detected_conf,
            "alternatives": alternatives,
            "extraction_time_ms": extraction_time_ms,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Verify claim if provided
        if claimed_model:
            result["claimed_model"] = claimed_model
            claimed_lower = claimed_model.lower()
            detected_lower = detected_model.lower()
            
            # Fuzzy match
            match = claimed_lower == detected_lower or \
                    claimed_lower in detected_lower or \
                    detected_lower in claimed_lower
            
            # Get distance to claimed model
            claimed_dist = distances.get(claimed_model, float('inf'))
            claimed_conf = dist_to_conf(claimed_dist) if claimed_model in distances else 0.0
            
            result["claim_verified"] = match or claimed_conf > 0.7
            result["claim_confidence"] = claimed_conf
            
            if not result["claim_verified"]:
                result["alert"] = f"Claimed {claimed_model} but fingerprint matches {detected_model}"
        
        return result
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def batch_extract_and_store(
        self,
        samples: List[Dict],
        progress_callback=None,
    ) -> Dict:
        """
        Process multiple samples in batch.
        
        Args:
            samples: List of dicts with 'text' and 'model_id' keys
            progress_callback: Optional callback(current, total, model_id)
            
        Returns:
            Summary statistics
        """
        total = len(samples)
        results = {"success": 0, "skipped": 0, "failed": 0, "by_model": defaultdict(int)}
        
        for i, sample in enumerate(samples):
            text = sample.get("text", "")
            model_id = sample.get("model_id", sample.get("source_model", "unknown"))
            prompt = sample.get("prompt", "")
            category = sample.get("category", "")
            
            try:
                fp, stored = self.extract_and_store(
                    text=text,
                    model_id=model_id,
                    prompt=prompt,
                    category=category,
                )
                
                if fp is None:
                    results["skipped"] += 1
                else:
                    results["success"] += 1
                    results["by_model"][model_id] += 1
                    
            except Exception as e:
                logging.error(f"Failed to process sample {i}: {e}")
                results["failed"] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, model_id)
        
        return dict(results)
    
    def export_profiles_for_verifier(self, output_path: str):
        """Export profiles in format compatible with ModelVerifier."""
        export_data = {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_id, profile in self._profiles.items():
            export_data["models"][model_id] = {
                "model_id": model_id,
                "model_family": profile.model_family,
                "centroid": profile.centroid,
                "covariance": profile.covariance,
                "sample_count": profile.sample_count,
                "added": profile.first_seen,
            }
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"Exported {len(self._profiles)} profiles to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for fingerprint store operations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SpecHO Fingerprint Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract and store a fingerprint
    python fingerprint_store.py --extract --text "Some text" --model claude-sonnet-4
    
    # Extract from file
    python fingerprint_store.py --extract --file doc.txt --model gpt-4o
    
    # Verify text against stored profiles
    python fingerprint_store.py --verify --text "Some text" --claimed claude-sonnet-4
    
    # List stored models
    python fingerprint_store.py --list
    
    # Show model stats
    python fingerprint_store.py --stats claude-sonnet-4
    
    # Batch process corpus
    python fingerprint_store.py --batch corpus.json
    
    # Export for verifier
    python fingerprint_store.py --export fingerprint_db.json
""")
    
    parser.add_argument("--extract", action="store_true", help="Extract and store fingerprint")
    parser.add_argument("--verify", action="store_true", help="Verify text against profiles")
    parser.add_argument("--list", action="store_true", help="List stored models")
    parser.add_argument("--stats", type=str, metavar="MODEL", help="Show model statistics")
    parser.add_argument("--batch", type=str, metavar="CORPUS", help="Batch process corpus file")
    parser.add_argument("--export", type=str, metavar="OUTPUT", help="Export profiles for verifier")
    
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--model", type=str, help="Model ID for extraction")
    parser.add_argument("--claimed", type=str, help="Claimed model for verification")
    parser.add_argument("--data-dir", type=str, default="data/fingerprints", help="Storage directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    store = FingerprintStore(data_dir=args.data_dir)
    
    if args.list:
        models = store.list_models()
        print(f"\nStored Models ({len(models)}):")
        for model in models:
            stats = store.get_model_stats(model)
            print(f"  {model}: {stats['sample_count']} samples")
        return
    
    if args.stats:
        stats = store.get_model_stats(args.stats)
        print(f"\n{args.stats} Statistics:")
        print(json.dumps(stats, indent=2))
        return
    
    if args.export:
        store.export_profiles_for_verifier(args.export)
        print(f"Exported to {args.export}")
        return
    
    if args.batch:
        with open(args.batch) as f:
            data = json.load(f)
        
        # Handle different corpus formats
        if isinstance(data, dict):
            samples = data.get("samples", [])
        else:
            samples = data
        
        # Expand samples with ai_variants
        expanded = []
        for sample in samples:
            human_text = sample.get("human", sample.get("human_text", ""))
            if human_text:
                expanded.append({
                    "text": human_text,
                    "model_id": "human",
                    "prompt": sample.get("prompt", ""),
                    "category": sample.get("category", ""),
                })
            
            ai_variants = sample.get("ai_variants", sample.get("ai_generated", {}))
            for model, text in ai_variants.items():
                expanded.append({
                    "text": text,
                    "model_id": model,
                    "prompt": sample.get("prompt", ""),
                    "category": sample.get("category", ""),
                })
        
        print(f"Processing {len(expanded)} samples...")
        
        def progress(current, total, model):
            if current % 10 == 0:
                print(f"  {current}/{total} ({model})")
        
        results = store.batch_extract_and_store(expanded, progress_callback=progress)
        
        print(f"\nResults:")
        print(f"  Success: {results['success']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        print(f"\nBy model:")
        for model, count in results.get("by_model", {}).items():
            print(f"  {model}: {count}")
        return
    
    # Get text
    text = None
    if args.text:
        text = args.text
    elif args.file:
        text = open(args.file).read()
    
    if args.extract:
        if not text:
            print("Error: Provide --text or --file")
            return
        if not args.model:
            print("Error: Provide --model")
            return
        
        fp, stored = store.extract_and_store(text, args.model)
        
        print(f"\nFingerprint stored:")
        print(f"  ID: {stored.fingerprint_id}")
        print(f"  Model: {stored.model_id}")
        print(f"  Vector dims: {len(stored.vector)}")
        print(f"  Extraction time: {stored.extraction_time_ms:.1f}ms")
        
        stats = store.get_model_stats(args.model)
        print(f"\nModel profile updated:")
        print(f"  Total samples: {stats['sample_count']}")
        return
    
    if args.verify:
        if not text:
            print("Error: Provide --text or --file")
            return
        
        result = store.verify(text, claimed_model=args.claimed)
        
        print(f"\nVerification Result:")
        print(f"  Detected: {result['detected_model']}")
        print(f"  Family: {result['detected_family']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if args.claimed:
            print(f"\n  Claimed: {args.claimed}")
            print(f"  Verified: {'YES ✓' if result['claim_verified'] else 'NO ✗'}")
            if result.get('alert'):
                print(f"  Alert: {result['alert']}")
        
        print(f"\nAlternatives:")
        for alt in result['alternatives'][:3]:
            print(f"  {alt['model_id']}: {alt['confidence']:.1%}")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
