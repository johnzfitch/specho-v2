#!/usr/bin/env python3
"""
Fingerprint Stability Tester

Tests whether fingerprints remain stable across:
- Different prompt styles (minimal, neutral, improve, formal, casual)
- Different domains (academic, technical, creative, etc.)
- Different temperatures (0.5, 0.7, 0.9)

Success criteria:
- Intra-model correlation > 0.8 = fingerprints are stable
- Intra-model correlation < 0.5 = fingerprints are unstable

Usage:
    python test_stability.py --fingerprints data/fingerprints/fingerprints.json
    python test_stability.py --corpus data/corpus/merged/corpus.json --extract
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
from itertools import combinations


def load_fingerprints(path: str) -> list[dict]:
    """Load fingerprints from file."""
    with open(path) as f:
        return json.load(f)


def fingerprint_to_vector(fp: dict) -> np.ndarray:
    """Convert fingerprint to numpy array."""
    return np.array([
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
    ])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


class StabilityTester:
    """Test fingerprint stability across variations."""
    
    def __init__(self, fingerprints: list[dict]):
        self.fingerprints = fingerprints
        self._index_by_model = defaultdict(list)
        self._index_by_style = defaultdict(list)
        self._index_by_domain = defaultdict(list)
        self._index_by_seed = defaultdict(list)
        
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for quick lookup."""
        for i, fp in enumerate(self.fingerprints):
            model = fp.get("source_model", "unknown")
            style = fp.get("prompt_style", "unknown")
            domain = fp.get("domain", "unknown")
            seed = fp.get("seed_id", "unknown")
            
            self._index_by_model[model].append(i)
            self._index_by_style[style].append(i)
            self._index_by_domain[domain].append(i)
            self._index_by_seed[seed].append(i)
    
    def test_style_stability(self) -> dict:
        """Test if fingerprints are stable across prompt styles."""
        results = {}
        
        styles = list(self._index_by_style.keys())
        if len(styles) < 2:
            return {"error": "Need at least 2 prompt styles to test stability"}
        
        # For each model, compare fingerprints across styles
        for model, indices in self._index_by_model.items():
            if model == "human":
                continue
            
            # Group by style
            by_style = defaultdict(list)
            for i in indices:
                style = self.fingerprints[i].get("prompt_style", "unknown")
                by_style[style].append(fingerprint_to_vector(self.fingerprints[i]))
            
            if len(by_style) < 2:
                continue
            
            # Compute average fingerprint per style
            style_centroids = {}
            for style, vectors in by_style.items():
                style_centroids[style] = np.mean(vectors, axis=0)
            
            # Compare all pairs of styles
            correlations = []
            for s1, s2 in combinations(style_centroids.keys(), 2):
                corr = pearson_correlation(style_centroids[s1], style_centroids[s2])
                correlations.append({
                    "style_1": s1,
                    "style_2": s2,
                    "correlation": corr,
                })
            
            results[model] = {
                "styles_tested": list(style_centroids.keys()),
                "mean_correlation": np.mean([c["correlation"] for c in correlations]),
                "min_correlation": min(c["correlation"] for c in correlations),
                "pairwise": correlations,
                "stable": np.mean([c["correlation"] for c in correlations]) > 0.7,
            }
        
        return results
    
    def test_domain_stability(self) -> dict:
        """Test if fingerprints are stable across domains."""
        results = {}
        
        domains = list(self._index_by_domain.keys())
        if len(domains) < 2:
            return {"error": "Need at least 2 domains to test stability"}
        
        for model, indices in self._index_by_model.items():
            if model == "human":
                continue
            
            # Group by domain
            by_domain = defaultdict(list)
            for i in indices:
                domain = self.fingerprints[i].get("domain", "unknown")
                by_domain[domain].append(fingerprint_to_vector(self.fingerprints[i]))
            
            if len(by_domain) < 2:
                continue
            
            # Compute centroids
            domain_centroids = {}
            for domain, vectors in by_domain.items():
                domain_centroids[domain] = np.mean(vectors, axis=0)
            
            # Compare pairs
            correlations = []
            for d1, d2 in combinations(domain_centroids.keys(), 2):
                corr = pearson_correlation(domain_centroids[d1], domain_centroids[d2])
                correlations.append({
                    "domain_1": d1,
                    "domain_2": d2,
                    "correlation": corr,
                })
            
            results[model] = {
                "domains_tested": list(domain_centroids.keys()),
                "mean_correlation": np.mean([c["correlation"] for c in correlations]),
                "min_correlation": min(c["correlation"] for c in correlations),
                "pairwise": correlations,
                "stable": np.mean([c["correlation"] for c in correlations]) > 0.7,
            }
        
        return results
    
    def test_seed_stability(self) -> dict:
        """Test if fingerprints are stable across different seed texts."""
        results = {}
        
        for model, indices in self._index_by_model.items():
            if model == "human":
                continue
            
            # Group by seed
            by_seed = defaultdict(list)
            for i in indices:
                seed = self.fingerprints[i].get("seed_id", "unknown")
                by_seed[seed].append(fingerprint_to_vector(self.fingerprints[i]))
            
            if len(by_seed) < 2:
                continue
            
            # Get mean fingerprint per seed
            seed_centroids = {}
            for seed, vectors in by_seed.items():
                seed_centroids[seed] = np.mean(vectors, axis=0)
            
            # Compare all pairs
            correlations = []
            for s1, s2 in combinations(seed_centroids.keys(), 2):
                corr = pearson_correlation(seed_centroids[s1], seed_centroids[s2])
                correlations.append(corr)
            
            results[model] = {
                "seeds_tested": list(seed_centroids.keys()),
                "mean_correlation": float(np.mean(correlations)),
                "std_correlation": float(np.std(correlations)),
                "min_correlation": float(min(correlations)),
                "stable": np.mean(correlations) > 0.7,
            }
        
        return results
    
    def test_inter_model_separation(self) -> dict:
        """Test if different models are distinguishable."""
        results = {}
        
        # Compute centroid for each model
        model_centroids = {}
        for model, indices in self._index_by_model.items():
            vectors = [fingerprint_to_vector(self.fingerprints[i]) for i in indices]
            model_centroids[model] = np.mean(vectors, axis=0)
        
        if len(model_centroids) < 2:
            return {"error": "Need at least 2 models to test separation"}
        
        # Compare all model pairs
        comparisons = []
        for m1, m2 in combinations(model_centroids.keys(), 2):
            dist = np.linalg.norm(model_centroids[m1] - model_centroids[m2])
            corr = pearson_correlation(model_centroids[m1], model_centroids[m2])
            
            comparisons.append({
                "model_1": m1,
                "model_2": m2,
                "distance": float(dist),
                "correlation": float(corr),
            })
        
        # Sort by distance (most different first)
        comparisons.sort(key=lambda x: -x["distance"])
        
        results = {
            "models": list(model_centroids.keys()),
            "mean_distance": np.mean([c["distance"] for c in comparisons]),
            "mean_correlation": np.mean([c["correlation"] for c in comparisons]),
            "most_different": comparisons[:3],
            "most_similar": comparisons[-3:],
            "all_comparisons": comparisons,
        }
        
        return results
    
    def run_all_tests(self) -> dict:
        """Run all stability tests."""
        return {
            "style_stability": self.test_style_stability(),
            "domain_stability": self.test_domain_stability(),
            "seed_stability": self.test_seed_stability(),
            "model_separation": self.test_inter_model_separation(),
            "summary": self._generate_summary(),
        }
    
    def _generate_summary(self) -> dict:
        """Generate overall summary."""
        models = list(self._index_by_model.keys())
        
        return {
            "total_fingerprints": len(self.fingerprints),
            "models": models,
            "styles": list(self._index_by_style.keys()),
            "domains": list(self._index_by_domain.keys()),
            "seeds": list(self._index_by_seed.keys()),
        }


def generate_report(
    results: dict,
    output_path: str,
) -> str:
    """Generate HTML stability report."""
    
    def stability_badge(stable: bool) -> str:
        if stable:
            return '<span style="color: green; font-weight: bold;">✓ STABLE</span>'
        return '<span style="color: red; font-weight: bold;">✗ UNSTABLE</span>'
    
    def corr_color(corr: float) -> str:
        if corr > 0.8:
            return "green"
        elif corr > 0.6:
            return "orange"
        return "red"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fingerprint Stability Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; }}
        .metric {{ font-size: 1.5em; font-weight: bold; }}
        .label {{ color: #666; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fingerprint Stability Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="card">
            <h2>Summary</h2>
            <div class="grid">
                <div>
                    <div class="metric">{results['summary']['total_fingerprints']}</div>
                    <div class="label">Total Fingerprints</div>
                </div>
                <div>
                    <div class="metric">{len(results['summary']['models'])}</div>
                    <div class="label">Models</div>
                </div>
                <div>
                    <div class="metric">{len(results['summary']['styles'])}</div>
                    <div class="label">Prompt Styles</div>
                </div>
                <div>
                    <div class="metric">{len(results['summary']['domains'])}</div>
                    <div class="label">Domains</div>
                </div>
            </div>
        </div>
"""
    
    # Style stability
    if "error" not in results.get("style_stability", {}):
        html += """
        <div class="card">
            <h2>Prompt Style Stability</h2>
            <p>Do fingerprints remain consistent across different prompt styles?</p>
            <table>
                <tr><th>Model</th><th>Styles</th><th>Mean Correlation</th><th>Status</th></tr>
"""
        for model, data in results.get("style_stability", {}).items():
            if isinstance(data, dict):
                corr = data.get("mean_correlation", 0)
                stable = data.get("stable", False)
                html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{len(data.get('styles_tested', []))}</td>
                    <td style="color: {corr_color(corr)}">{corr:.3f}</td>
                    <td>{stability_badge(stable)}</td>
                </tr>
"""
        html += "</table></div>"
    
    # Domain stability
    if "error" not in results.get("domain_stability", {}):
        html += """
        <div class="card">
            <h2>Domain Stability</h2>
            <p>Do fingerprints remain consistent across different content domains?</p>
            <table>
                <tr><th>Model</th><th>Domains</th><th>Mean Correlation</th><th>Status</th></tr>
"""
        for model, data in results.get("domain_stability", {}).items():
            if isinstance(data, dict):
                corr = data.get("mean_correlation", 0)
                stable = data.get("stable", False)
                html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{len(data.get('domains_tested', []))}</td>
                    <td style="color: {corr_color(corr)}">{corr:.3f}</td>
                    <td>{stability_badge(stable)}</td>
                </tr>
"""
        html += "</table></div>"
    
    # Model separation
    sep = results.get("model_separation", {})
    if "error" not in sep:
        html += f"""
        <div class="card">
            <h2>Model Separation</h2>
            <p>How distinguishable are different models?</p>
            
            <h3>Most Different Pairs</h3>
            <table>
                <tr><th>Model 1</th><th>Model 2</th><th>Distance</th><th>Correlation</th></tr>
"""
        for comp in sep.get("most_different", []):
            html += f"""
                <tr>
                    <td>{comp['model_1']}</td>
                    <td>{comp['model_2']}</td>
                    <td>{comp['distance']:.3f}</td>
                    <td>{comp['correlation']:.3f}</td>
                </tr>
"""
        html += """
            </table>
            
            <h3>Most Similar Pairs (Potential Confusion)</h3>
            <table>
                <tr><th>Model 1</th><th>Model 2</th><th>Distance</th><th>Correlation</th></tr>
"""
        for comp in sep.get("most_similar", []):
            html += f"""
                <tr>
                    <td>{comp['model_1']}</td>
                    <td>{comp['model_2']}</td>
                    <td>{comp['distance']:.3f}</td>
                    <td style="color: {corr_color(1 - comp['correlation'])}">{comp['correlation']:.3f}</td>
                </tr>
"""
        html += "</table></div>"
    
    html += """
    </div>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test fingerprint stability")
    parser.add_argument(
        "--fingerprints", type=str, required=True,
        help="Path to fingerprints JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/stability",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("Loading fingerprints...")
    fingerprints = load_fingerprints(args.fingerprints)
    print(f"Loaded {len(fingerprints)} fingerprints")
    
    print("\nRunning stability tests...")
    tester = StabilityTester(fingerprints)
    results = tester.run_all_tests()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "stability_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    report_path = generate_report(results, str(output_dir / "stability_report.html"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("STABILITY TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\nFingerprints: {results['summary']['total_fingerprints']}")
    print(f"Models: {', '.join(results['summary']['models'])}")
    
    # Style stability summary
    style_results = results.get("style_stability", {})
    if "error" not in style_results:
        stable_count = sum(1 for v in style_results.values() if isinstance(v, dict) and v.get("stable"))
        print(f"\nStyle Stability: {stable_count}/{len(style_results)} models stable")
    
    # Domain stability summary
    domain_results = results.get("domain_stability", {})
    if "error" not in domain_results:
        stable_count = sum(1 for v in domain_results.values() if isinstance(v, dict) and v.get("stable"))
        print(f"Domain Stability: {stable_count}/{len(domain_results)} models stable")
    
    # Separation summary
    sep = results.get("model_separation", {})
    if "error" not in sep:
        print(f"\nModel Separation:")
        print(f"  Mean distance: {sep.get('mean_distance', 0):.3f}")
        print(f"  Mean correlation: {sep.get('mean_correlation', 0):.3f}")
    
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
