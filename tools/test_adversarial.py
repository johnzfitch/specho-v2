#!/usr/bin/env python3
"""
Adversarial Fingerprint Testing

Tests whether models can be prompted to mimic other models' fingerprints.

Attack vectors tested:
1. Style mimicry: "Write like GPT-4 would write"
2. Pattern avoidance: "Avoid repetitive sentence structures"
3. Explicit anti-fingerprinting: "Make your writing unpredictable"
4. Cross-model prompting: Asking Claude to write like GPT

Usage:
    python test_adversarial.py --model gemini-1.5-flash --target claude
    python test_adversarial.py --all-attacks
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import time


# =============================================================================
# ADVERSARIAL PROMPTS
# =============================================================================

ATTACK_PROMPTS = {
    # Style mimicry attacks
    "mimic_gpt": {
        "category": "mimicry",
        "description": "Attempt to mimic GPT-4's writing style",
        "prompt": """Write like GPT-4 would write. Be direct, confident, and use clear paragraph structure.

Rewrite the following text:
{text}

Write in GPT-4's characteristic style:""",
    },
    
    "mimic_claude": {
        "category": "mimicry",
        "description": "Attempt to mimic Claude's writing style",
        "prompt": """Write like Claude would write. Use thoughtful hedging, consider multiple perspectives, and be nuanced.

Rewrite the following text:
{text}

Write in Claude's characteristic style:""",
    },
    
    "mimic_gemini": {
        "category": "mimicry",
        "description": "Attempt to mimic Gemini's writing style",
        "prompt": """Write like Google's Gemini would write. Be informative, balanced, and comprehensive.

Rewrite the following text:
{text}

Write in Gemini's characteristic style:""",
    },
    
    # Pattern avoidance attacks
    "avoid_repetition": {
        "category": "avoidance",
        "description": "Avoid repetitive patterns",
        "prompt": """Rewrite this text while carefully avoiding any repetitive sentence structures or word patterns. Make each sentence unique in its construction.

Text:
{text}

Rewritten (with maximum variety):""",
    },
    
    "randomize_structure": {
        "category": "avoidance",
        "description": "Randomize sentence structure",
        "prompt": """Rewrite this text with deliberately randomized sentence structures. Vary sentence length dramatically. Mix simple and complex sentences unpredictably.

Text:
{text}

Randomized version:""",
    },
    
    "unpredictable_transitions": {
        "category": "avoidance",
        "description": "Use unpredictable transitions",
        "prompt": """Rewrite this text using unexpected and varied transitions between sentences. Avoid common connective patterns.

Text:
{text}

Version with varied transitions:""",
    },
    
    # Anti-fingerprinting attacks
    "anti_detection": {
        "category": "anti_fingerprint",
        "description": "Explicit anti-detection prompt",
        "prompt": """Rewrite this text in a way that would be difficult for AI detection systems to classify. Aim for a natural, human-like writing style that doesn't follow typical AI patterns.

Text:
{text}

Human-like rewrite:""",
    },
    
    "maximize_entropy": {
        "category": "anti_fingerprint",
        "description": "Maximize writing entropy",
        "prompt": """Rewrite this text with maximum linguistic entropy. Use unusual word choices, varied syntax, and unpredictable punctuation while preserving meaning.

Text:
{text}

High-entropy version:""",
    },
    
    "style_mixing": {
        "category": "anti_fingerprint",
        "description": "Mix multiple writing styles",
        "prompt": """Rewrite this text by mixing multiple writing styles throughout - some sentences formal, some casual, some technical, some conversational. Create a stylistic patchwork.

Text:
{text}

Mixed-style version:""",
    },
    
    # Human mimicry attacks
    "mimic_human": {
        "category": "human_mimicry",
        "description": "Attempt to write like a human",
        "prompt": """Rewrite this text exactly as a human would write it. Include natural imperfections - slight redundancies, casual phrasing, thoughts that develop as you write. Don't over-optimize or over-structure.

Text:
{text}

Natural human-style rewrite:""",
    },
    
    "add_imperfections": {
        "category": "human_mimicry",
        "description": "Add human-like imperfections",
        "prompt": """Rewrite this text with deliberate human-like qualities: occasional wordiness, natural tangents, varied formality, and the kind of flow that comes from thinking while writing.

Text:
{text}

Imperfect human-style version:""",
    },
}

SEED_TEXTS = [
    {
        "id": "technical",
        "text": """The implementation of microservices architecture requires careful consideration of service boundaries and communication patterns. Unlike monolithic applications, microservices distribute functionality across independently deployable units. This approach offers benefits including improved scalability and technology flexibility, but introduces complexity in areas such as distributed transactions and service discovery."""
    },
    {
        "id": "creative",
        "text": """The old lighthouse keeper had seen many storms, but none quite like this one. The wind howled through the cracks in the stone walls, carrying with it the salt and fury of an angry sea. He climbed the spiral staircase one more time, his joints protesting with each step."""
    },
]


# =============================================================================
# ADVERSARIAL TESTER
# =============================================================================

class AdversarialTester:
    """Test fingerprint robustness against adversarial prompts."""
    
    def __init__(
        self,
        fingerprint_db_path: str = None,
        specHO_path: str = None,
    ):
        self.db = None
        self.extractor = None
        self._load_database(fingerprint_db_path)
        self._init_extractor(specHO_path)
    
    def _load_database(self, path: str):
        """Load fingerprint database for comparison."""
        if path and Path(path).exists():
            with open(path) as f:
                self.db = json.load(f)
    
    def _init_extractor(self, specHO_path: str = None):
        """Initialize fingerprint extractor."""
        if specHO_path:
            import sys
            sys.path.insert(0, specHO_path)
        
        try:
            from specHO.detector import SpecHODetector
            self._detector = SpecHODetector(semantic_model_path="all-MiniLM-L6-v2")
        except ImportError:
            self._detector = None
    
    def extract_fingerprint(self, text: str) -> np.ndarray:
        """Extract fingerprint from text."""
        if self._detector is None:
            # Mock
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
    
    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> dict:
        """Compare two fingerprints."""
        distance = np.linalg.norm(fp1 - fp2)
        correlation = np.corrcoef(fp1, fp2)[0, 1]
        cosine = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
        
        return {
            "euclidean_distance": float(distance),
            "correlation": float(correlation),
            "cosine_similarity": float(cosine),
        }
    
    def test_attack(
        self,
        attack_id: str,
        source_model: str,
        generate_fn,  # Function that generates text given a prompt
        baseline_fingerprint: np.ndarray = None,
    ) -> dict:
        """
        Test a single adversarial attack.
        
        Args:
            attack_id: ID of attack from ATTACK_PROMPTS
            source_model: Model being tested
            generate_fn: Function(prompt) -> text
            baseline_fingerprint: Fingerprint of normal output for comparison
        """
        attack = ATTACK_PROMPTS.get(attack_id)
        if not attack:
            return {"error": f"Unknown attack: {attack_id}"}
        
        results = {
            "attack_id": attack_id,
            "category": attack["category"],
            "description": attack["description"],
            "source_model": source_model,
            "samples": [],
        }
        
        for seed in SEED_TEXTS:
            prompt = attack["prompt"].format(text=seed["text"])
            
            try:
                adversarial_text = generate_fn(prompt)
                adversarial_fp = self.extract_fingerprint(adversarial_text)
                
                sample_result = {
                    "seed_id": seed["id"],
                    "text_length": len(adversarial_text),
                    "fingerprint": adversarial_fp.tolist(),
                }
                
                # Compare to baseline if provided
                if baseline_fingerprint is not None:
                    comparison = self.compare_fingerprints(adversarial_fp, baseline_fingerprint)
                    sample_result["vs_baseline"] = comparison
                
                results["samples"].append(sample_result)
                
            except Exception as e:
                results["samples"].append({
                    "seed_id": seed["id"],
                    "error": str(e),
                })
        
        # Aggregate
        valid_samples = [s for s in results["samples"] if "fingerprint" in s]
        if valid_samples:
            fps = np.array([s["fingerprint"] for s in valid_samples])
            results["mean_fingerprint"] = fps.mean(axis=0).tolist()
            results["std_fingerprint"] = fps.std(axis=0).tolist()
            
            if baseline_fingerprint is not None:
                results["mean_distance_from_baseline"] = np.mean([
                    s["vs_baseline"]["euclidean_distance"] for s in valid_samples
                ])
        
        return results
    
    def run_full_adversarial_suite(
        self,
        source_model: str,
        generate_fn,
        baseline_fingerprint: np.ndarray = None,
    ) -> dict:
        """Run all adversarial attacks."""
        results = {
            "source_model": source_model,
            "timestamp": datetime.now().isoformat(),
            "attacks": {},
        }
        
        for attack_id in ATTACK_PROMPTS:
            print(f"  Testing {attack_id}...", end=" ", flush=True)
            attack_result = self.test_attack(
                attack_id, source_model, generate_fn, baseline_fingerprint
            )
            results["attacks"][attack_id] = attack_result
            print("✓")
        
        # Summary
        results["summary"] = self._summarize_attacks(results["attacks"], baseline_fingerprint)
        
        return results
    
    def _summarize_attacks(self, attacks: dict, baseline_fp: np.ndarray) -> dict:
        """Summarize attack effectiveness."""
        summary = {
            "by_category": defaultdict(list),
            "most_effective": None,
            "least_effective": None,
        }
        
        for attack_id, result in attacks.items():
            if "mean_distance_from_baseline" in result:
                category = result["category"]
                distance = result["mean_distance_from_baseline"]
                summary["by_category"][category].append({
                    "attack_id": attack_id,
                    "distance_shift": distance,
                })
        
        # Find most/least effective
        all_attacks = []
        for cat_attacks in summary["by_category"].values():
            all_attacks.extend(cat_attacks)
        
        if all_attacks:
            all_attacks.sort(key=lambda x: x["distance_shift"], reverse=True)
            summary["most_effective"] = all_attacks[0] if all_attacks else None
            summary["least_effective"] = all_attacks[-1] if all_attacks else None
        
        return summary


def generate_with_ollama(model: str, prompt: str) -> str:
    """Generate text using Ollama."""
    import subprocess
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def generate_with_gemini(model: str, prompt: str) -> str:
    """Generate text using Gemini."""
    import google.generativeai as genai
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text


def generate_report(results: dict, output_path: str):
    """Generate HTML adversarial testing report."""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Adversarial Fingerprint Testing Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .danger {{ color: red; }}
        .metric {{ font-size: 1.5em; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Adversarial Fingerprint Testing</h1>
        <p>Model: {results.get('source_model', 'Unknown')}</p>
        <p>Generated: {results.get('timestamp', '')}</p>
        
        <div class="card">
            <h2>Summary</h2>
"""
    
    summary = results.get("summary", {})
    if summary.get("most_effective"):
        html += f"""
            <p><strong>Most Effective Attack:</strong> {summary['most_effective']['attack_id']} 
               (distance shift: {summary['most_effective']['distance_shift']:.3f})</p>
"""
    if summary.get("least_effective"):
        html += f"""
            <p><strong>Least Effective Attack:</strong> {summary['least_effective']['attack_id']}
               (distance shift: {summary['least_effective']['distance_shift']:.3f})</p>
"""
    
    html += """
        </div>
        
        <div class="card">
            <h2>Attack Results by Category</h2>
"""
    
    # Group by category
    by_category = defaultdict(list)
    for attack_id, attack_result in results.get("attacks", {}).items():
        by_category[attack_result.get("category", "unknown")].append((attack_id, attack_result))
    
    for category, attacks in sorted(by_category.items()):
        html += f"""
            <h3>{category.replace('_', ' ').title()}</h3>
            <table>
                <tr>
                    <th>Attack</th>
                    <th>Description</th>
                    <th>Distance Shift</th>
                    <th>Status</th>
                </tr>
"""
        for attack_id, result in attacks:
            distance = result.get("mean_distance_from_baseline", 0)
            if distance > 0.3:
                status = '<span class="danger">⚠️ High</span>'
            elif distance > 0.15:
                status = '<span class="warning">⚡ Medium</span>'
            else:
                status = '<span class="success">✓ Low</span>'
            
            html += f"""
                <tr>
                    <td>{attack_id}</td>
                    <td>{result.get('description', '')}</td>
                    <td>{distance:.3f}</td>
                    <td>{status}</td>
                </tr>
"""
        html += "</table>"
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Adversarial fingerprint testing")
    parser.add_argument("--model", type=str, help="Model to test (Ollama model name)")
    parser.add_argument("--gemini", type=str, help="Gemini model to test")
    parser.add_argument("--attack", type=str, help="Specific attack to test")
    parser.add_argument("--all-attacks", action="store_true", help="Run all attacks")
    parser.add_argument("--output", type=str, default="experiments/adversarial",
                       help="Output directory")
    parser.add_argument("--specHO", type=str, help="Path to SpecHO installation")
    parser.add_argument("--list-attacks", action="store_true", help="List available attacks")
    
    args = parser.parse_args()
    
    if args.list_attacks:
        print("\nAvailable adversarial attacks:\n")
        for attack_id, attack in ATTACK_PROMPTS.items():
            print(f"  {attack_id}")
            print(f"    Category: {attack['category']}")
            print(f"    {attack['description']}")
            print()
        return
    
    if not args.model and not args.gemini:
        print("Specify --model (Ollama) or --gemini (AI Studio)")
        return
    
    # Initialize tester
    tester = AdversarialTester(specHO_path=args.specHO)
    
    # Set up generator
    if args.model:
        model_name = args.model
        generate_fn = lambda p: generate_with_ollama(args.model, p)
    else:
        model_name = args.gemini
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        generate_fn = lambda p: generate_with_gemini(args.gemini, p)
    
    # Get baseline fingerprint
    print(f"\nGenerating baseline fingerprint for {model_name}...")
    baseline_text = generate_fn(f"Rewrite this: {SEED_TEXTS[0]['text']}")
    baseline_fp = tester.extract_fingerprint(baseline_text)
    print(f"Baseline fingerprint: {baseline_fp[:5]}... (showing first 5 dims)")
    
    # Run tests
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.attack:
        print(f"\nTesting attack: {args.attack}")
        result = tester.test_attack(args.attack, model_name, generate_fn, baseline_fp)
        results = {"attacks": {args.attack: result}}
    else:
        print(f"\nRunning full adversarial suite...")
        results = tester.run_full_adversarial_suite(model_name, generate_fn, baseline_fp)
    
    # Save results
    results_file = output_dir / f"adversarial_{model_name.replace(':', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate report
    report_path = generate_report(results, str(output_dir / "adversarial_report.html"))
    
    print(f"\nResults saved to {results_file}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    import os
    main()
