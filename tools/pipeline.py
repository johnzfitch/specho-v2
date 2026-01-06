#!/usr/bin/env python3
"""
SpecHO Fingerprint Pipeline Orchestrator

Runs the complete fingerprinting workflow:
1. Generate corpus (or load existing)
2. Extract fingerprints
3. Analyze separability
4. Train classifier
5. Build verification database
6. Run stability tests

Usage:
    python pipeline.py --phase all                    # Run everything
    python pipeline.py --phase validate              # Quick validation only
    python pipeline.py --phase train                 # Train classifier only
    python pipeline.py --config config.yaml          # Use config file
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import shutil


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    
    # Paths
    project_root: str = "."
    corpus_dir: str = "data/corpus"
    fingerprint_dir: str = "data/fingerprints"
    model_dir: str = "data/models"
    experiment_dir: str = "experiments"
    specHO_path: str = None
    
    # Corpus generation
    gemini_samples: int = 10
    local_samples: int = 10
    use_arena: bool = False
    arena_samples: int = 100
    
    # Generation settings
    prompt_styles: list = field(default_factory=lambda: ["neutral"])
    temperatures: list = field(default_factory=lambda: [0.7])
    
    # Models to use
    gemini_models: list = field(default_factory=lambda: [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ])
    local_models: list = field(default_factory=lambda: [
        "llama3.1:8b",
        "mistral:7b",
    ])
    
    # Analysis settings
    min_samples_per_model: int = 5
    classifier_type: str = "random_forest"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


class Pipeline:
    """Orchestrates the fingerprinting pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.root = Path(config.project_root)
        self.tools_dir = self.root / "tools"
        self.results = {}
        
    def log(self, msg: str, level: str = "info"):
        """Log a message."""
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "step": "▶️",
        }.get(level, "")
        print(f"{prefix} {msg}")
    
    def run_tool(self, tool: str, args: list = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a tool script."""
        cmd = [sys.executable, str(self.tools_dir / tool)]
        if args:
            cmd.extend(args)
        
        self.log(f"Running: {' '.join(cmd)}", "step")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )
        
        if result.returncode != 0 and check:
            self.log(f"Tool failed: {result.stderr}", "error")
        
        return result
    
    def phase_generate_corpus(self) -> dict:
        """Phase 1: Generate corpus from all sources."""
        self.log("PHASE: Generate Corpus", "step")
        results = {"sources": [], "total_samples": 0}
        
        corpus_dir = self.root / self.config.corpus_dir
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for Google API key
        if os.environ.get("GOOGLE_API_KEY"):
            self.log("Generating Gemini corpus...")
            result = self.run_tool("generate_gemini.py", [
                "--samples", str(self.config.gemini_samples),
                "--models", ",".join(self.config.gemini_models),
                "--styles", ",".join(self.config.prompt_styles),
                "--temperatures", ",".join(str(t) for t in self.config.temperatures),
                "--output", str(corpus_dir / "gemini"),
            ], check=False)
            
            if result.returncode == 0:
                results["sources"].append("gemini")
                self.log("Gemini corpus generated", "success")
            else:
                self.log("Gemini generation failed (check API key)", "warning")
        else:
            self.log("GOOGLE_API_KEY not set, skipping Gemini", "warning")
        
        # Check for Ollama
        ollama_check = subprocess.run(["which", "ollama"], capture_output=True)
        if ollama_check.returncode == 0:
            self.log("Generating local model corpus...")
            result = self.run_tool("generate_local.py", [
                "--samples", str(self.config.local_samples),
                "--models", ",".join(self.config.local_models),
                "--output", str(corpus_dir / "local"),
            ], check=False)
            
            if result.returncode == 0:
                results["sources"].append("local")
                self.log("Local corpus generated", "success")
            else:
                self.log("Local generation failed", "warning")
        else:
            self.log("Ollama not installed, skipping local models", "warning")
        
        # Arena data
        if self.config.use_arena:
            self.log("Loading Arena data...")
            result = self.run_tool("load_arena.py", [
                "--samples", str(self.config.arena_samples),
                "--output", str(corpus_dir / "arena"),
            ], check=False)
            
            if result.returncode == 0:
                results["sources"].append("arena")
                self.log("Arena data loaded", "success")
        
        # Merge all corpora
        if len(results["sources"]) > 0:
            self.log("Merging corpora...")
            result = self.run_tool("merge_corpus.py", [
                "--auto",
                "--output", str(corpus_dir / "merged"),
            ])
            
            # Count samples
            merged_file = corpus_dir / "merged" / "corpus.json"
            if merged_file.exists():
                with open(merged_file) as f:
                    samples = json.load(f)
                results["total_samples"] = len(samples)
                self.log(f"Total corpus: {len(samples)} samples", "success")
        
        return results
    
    def phase_extract_fingerprints(self) -> dict:
        """Phase 2: Extract fingerprints from corpus."""
        self.log("PHASE: Extract Fingerprints", "step")
        
        corpus_file = self.root / self.config.corpus_dir / "merged" / "corpus.json"
        if not corpus_file.exists():
            self.log("No merged corpus found", "error")
            return {"error": "No corpus"}
        
        fingerprint_dir = self.root / self.config.fingerprint_dir
        fingerprint_dir.mkdir(parents=True, exist_ok=True)
        
        args = [
            "--corpus", str(corpus_file),
            "--output", str(fingerprint_dir / "fingerprints.json"),
        ]
        
        if self.config.specHO_path:
            args.extend(["--specHO", self.config.specHO_path])
        
        result = self.run_tool("extract_fingerprints.py", args)
        
        if result.returncode != 0:
            return {"error": "Extraction failed"}
        
        # Count fingerprints
        fp_file = fingerprint_dir / "fingerprints.json"
        if fp_file.exists():
            with open(fp_file) as f:
                fingerprints = json.load(f)
            return {"count": len(fingerprints)}
        
        return {"error": "No fingerprints generated"}
    
    def phase_analyze(self) -> dict:
        """Phase 3: Analyze fingerprint separability."""
        self.log("PHASE: Analyze Separability", "step")
        
        fp_file = self.root / self.config.fingerprint_dir / "fingerprints.json"
        if not fp_file.exists():
            self.log("No fingerprints found", "error")
            return {"error": "No fingerprints"}
        
        experiment_dir = self.root / self.config.experiment_dir / "validation"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.run_tool("analyze_fingerprints.py", [
            "--fingerprints", str(fp_file),
            "--output", str(experiment_dir),
        ])
        
        # Load results
        report_file = experiment_dir / "report.json"
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            return {
                "accuracy": report.get("classifier", {}).get("cv_accuracy_mean"),
                "models": report.get("n_models"),
                "recommendation": report.get("recommendation"),
            }
        
        return {"error": "Analysis failed"}
    
    def phase_stability(self) -> dict:
        """Phase 4: Run stability tests."""
        self.log("PHASE: Stability Testing", "step")
        
        fp_file = self.root / self.config.fingerprint_dir / "fingerprints.json"
        if not fp_file.exists():
            return {"error": "No fingerprints"}
        
        stability_dir = self.root / self.config.experiment_dir / "stability"
        stability_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.run_tool("test_stability.py", [
            "--fingerprints", str(fp_file),
            "--output", str(stability_dir),
        ])
        
        # Load results
        results_file = stability_dir / "stability_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            # Summarize
            style_stable = sum(
                1 for v in results.get("style_stability", {}).values()
                if isinstance(v, dict) and v.get("stable")
            )
            domain_stable = sum(
                1 for v in results.get("domain_stability", {}).values()
                if isinstance(v, dict) and v.get("stable")
            )
            
            return {
                "style_stable_models": style_stable,
                "domain_stable_models": domain_stable,
            }
        
        return {"error": "Stability test failed"}
    
    def phase_build_database(self) -> dict:
        """Phase 5: Build verification database."""
        self.log("PHASE: Build Verification Database", "step")
        
        fp_file = self.root / self.config.fingerprint_dir / "fingerprints.json"
        if not fp_file.exists():
            return {"error": "No fingerprints"}
        
        model_dir = self.root / self.config.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Use verifier to build database
        sys.path.insert(0, str(self.root / "src" / "verifier"))
        try:
            from verifier import FingerprintDatabase
            
            with open(fp_file) as f:
                fingerprints = json.load(f)
            
            db = FingerprintDatabase.from_fingerprints(fingerprints)
            db.save(str(model_dir / "fingerprint_db.json"))
            
            return {
                "models": db.list_models(),
                "path": str(model_dir / "fingerprint_db.json"),
            }
        except ImportError as e:
            self.log(f"Could not import verifier: {e}", "error")
            return {"error": str(e)}
    
    def phase_validate(self) -> dict:
        """Quick validation phase - minimal corpus, basic analysis."""
        self.log("PHASE: Quick Validation", "step")
        
        # Override config for quick run
        self.config.gemini_samples = 5
        self.config.local_samples = 5
        self.config.use_arena = False
        self.config.prompt_styles = ["neutral"]
        
        results = {
            "corpus": self.phase_generate_corpus(),
            "fingerprints": self.phase_extract_fingerprints(),
            "analysis": self.phase_analyze(),
        }
        
        # Determine GO/NOGO
        accuracy = results.get("analysis", {}).get("accuracy", 0)
        if accuracy and accuracy > 0.7:
            results["decision"] = "GO - Fingerprinting viable"
        elif accuracy and accuracy > 0.5:
            results["decision"] = "INVESTIGATE - Some signal, needs work"
        else:
            results["decision"] = "NOGO - Insufficient discrimination"
        
        return results
    
    def run(self, phase: str = "all") -> dict:
        """Run the pipeline."""
        self.log(f"Starting SpecHO Fingerprint Pipeline", "info")
        self.log(f"Phase: {phase}", "info")
        self.log(f"Project root: {self.root}", "info")
        
        start_time = datetime.now()
        
        if phase == "validate":
            results = self.phase_validate()
        elif phase == "corpus":
            results = self.phase_generate_corpus()
        elif phase == "extract":
            results = self.phase_extract_fingerprints()
        elif phase == "analyze":
            results = self.phase_analyze()
        elif phase == "stability":
            results = self.phase_stability()
        elif phase == "database":
            results = self.phase_build_database()
        elif phase == "all":
            results = {
                "corpus": self.phase_generate_corpus(),
                "fingerprints": self.phase_extract_fingerprints(),
                "analysis": self.phase_analyze(),
                "stability": self.phase_stability(),
                "database": self.phase_build_database(),
            }
        else:
            self.log(f"Unknown phase: {phase}", "error")
            return {"error": f"Unknown phase: {phase}"}
        
        elapsed = (datetime.now() - start_time).total_seconds()
        results["elapsed_seconds"] = elapsed
        results["timestamp"] = datetime.now().isoformat()
        
        # Save results
        results_dir = self.root / self.config.experiment_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"pipeline_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"Pipeline complete in {elapsed:.1f}s", "success")
        self.log(f"Results saved to {results_file}", "info")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="SpecHO Fingerprint Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
    validate    Quick validation (minimal corpus, GO/NOGO decision)
    corpus      Generate corpus only
    extract     Extract fingerprints only
    analyze     Analyze separability only
    stability   Run stability tests only
    database    Build verification database only
    all         Run complete pipeline

Examples:
    # Quick validation
    python pipeline.py --phase validate
    
    # Full pipeline
    python pipeline.py --phase all
    
    # With custom config
    python pipeline.py --config my_config.yaml --phase all
    
    # Generate default config
    python pipeline.py --generate-config config.yaml
""")
    
    parser.add_argument(
        "--phase", type=str, default="validate",
        choices=["validate", "corpus", "extract", "analyze", "stability", "database", "all"],
        help="Pipeline phase to run"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--generate-config", type=str, metavar="PATH",
        help="Generate default config file and exit"
    )
    parser.add_argument(
        "--specHO", type=str,
        help="Path to SpecHO installation"
    )
    parser.add_argument(
        "--root", type=str, default=".",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Generate config mode
    if args.generate_config:
        config = PipelineConfig()
        config.to_yaml(args.generate_config)
        print(f"Config saved to {args.generate_config}")
        return
    
    # Load or create config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    # Override from args
    if args.specHO:
        config.specHO_path = args.specHO
    if args.root:
        config.project_root = args.root
    
    # Run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run(args.phase)
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
