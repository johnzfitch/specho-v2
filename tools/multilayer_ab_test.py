#!/usr/bin/env python3
"""
Multi-Layer A/B Test

Tests all fingerprint layer combinations to find optimal configuration.

Configurations tested:
- Layer A only (5 dims)           - Trajectory
- Layer AB (20 dims)              - Trajectory + SpecHO
- Layer ABC (32 dims)             - + Epistemic + Transitions
- Full ABCD (39 dims)             - + Syntactic

Also tests:
- Each layer independently
- Various combinations

The goal: Find the smallest fingerprint that achieves target accuracy.

Usage:
    python multilayer_ab_test.py --fingerprints data/fingerprints/cognitive.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import sys


@dataclass
class LayerTestResult:
    """Result of testing a layer configuration."""
    
    config_name: str
    dimensions: int
    features: List[str]
    
    accuracy: float
    accuracy_std: float
    f1: float
    
    per_class_accuracy: Dict[str, float]
    
    training_time_ms: float


@dataclass
class MultiLayerTestResults:
    """Results of full multi-layer test."""
    
    n_samples: int
    n_models: int
    models: List[str]
    
    results: Dict[str, LayerTestResult]
    
    optimal_config: str
    optimal_accuracy: float
    
    recommendations: List[str]
    
    timestamp: str = ""


# Layer definitions
LAYER_A = [
    "concept_jump_mean", "concept_jump_variance",
    "path_tortuosity", "turning_angle_mean", "return_rate",
]

LAYER_B = [
    "phonetic_mean", "phonetic_std", "phonetic_max",
    "structural_mean", "structural_std", "structural_max",
    "semantic_mean", "semantic_std", "semantic_max",
    "cooccurrence_rate", "geometric_mean",
    "overall_mean", "overall_std", "overall_max", "burstiness",
]

LAYER_C_EPISTEMIC = [
    "hedge_density", "hedge_clustering", "hedge_position_bias",
    "confidence_mean", "confidence_variance", "confidence_arc",
]

LAYER_C_TRANSITIONS = [
    "additive_rate", "contrastive_rate", "causal_rate",
    "temporal_rate", "exemplifying_rate", "reformulating_rate",
]

LAYER_D = [
    "clause_length_mean", "clause_length_std", "clause_rhythm_autocorr",
    "sentence_complexity", "comma_density", "semicolon_rate",
    "parenthetical_rate",
]

# Test configurations
CONFIGURATIONS = {
    "A_trajectory": LAYER_A,
    "B_specHO": LAYER_B,
    "C_epistemic": LAYER_C_EPISTEMIC,
    "C_transitions": LAYER_C_TRANSITIONS,
    "D_syntactic": LAYER_D,
    "AB": LAYER_A + LAYER_B,
    "ABC": LAYER_A + LAYER_B + LAYER_C_EPISTEMIC + LAYER_C_TRANSITIONS,
    "ABCD_full": LAYER_A + LAYER_B + LAYER_C_EPISTEMIC + LAYER_C_TRANSITIONS + LAYER_D,
    "A_C": LAYER_A + LAYER_C_EPISTEMIC + LAYER_C_TRANSITIONS,
    "B_C": LAYER_B + LAYER_C_EPISTEMIC + LAYER_C_TRANSITIONS,
    "A_D": LAYER_A + LAYER_D,
}


def load_fingerprints(path: str) -> list[dict]:
    """Load fingerprints from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_features(fp: dict, feature_names: List[str]) -> np.ndarray:
    """Extract specified features from fingerprint."""
    return np.array([fp.get(f, 0.0) for f in feature_names])


def test_configuration(
    fingerprints: List[dict],
    features: List[str],
    config_name: str,
) -> LayerTestResult:
    """Test a single layer configuration."""
    import time
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import f1_score, accuracy_score
    except ImportError:
        return LayerTestResult(
            config_name=config_name,
            dimensions=len(features),
            features=features,
            accuracy=0,
            accuracy_std=0,
            f1=0,
            per_class_accuracy={},
            training_time_ms=0,
        )
    
    # Prepare data
    X = np.array([extract_features(fp, features) for fp in fingerprints])
    y = np.array([fp.get("source_model", "unknown") for fp in fingerprints])
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    n_classes = len(le.classes_)
    min_per_class = min(Counter(y_encoded).values())
    
    if min_per_class < 2:
        return LayerTestResult(
            config_name=config_name,
            dimensions=len(features),
            features=features,
            accuracy=0,
            accuracy_std=0,
            f1=0,
            per_class_accuracy={},
            training_time_ms=0,
        )
    
    cv_folds = min(5, min_per_class)
    
    # Train and evaluate
    start = time.time()
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
    )
    
    cv_scores = cross_val_score(clf, X, y_encoded, cv=cv_folds)
    y_pred = cross_val_predict(clf, X, y_encoded, cv=cv_folds)
    
    elapsed = (time.time() - start) * 1000
    
    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = (y_encoded == cls_idx)
        if mask.sum() > 0:
            per_class[cls_name] = float(accuracy_score(y_encoded[mask], y_pred[mask]))
    
    return LayerTestResult(
        config_name=config_name,
        dimensions=len(features),
        features=features,
        accuracy=float(np.mean(cv_scores)),
        accuracy_std=float(np.std(cv_scores)),
        f1=float(f1_score(y_encoded, y_pred, average='weighted')),
        per_class_accuracy=per_class,
        training_time_ms=elapsed,
    )


def run_multilayer_test(fingerprints: List[dict]) -> MultiLayerTestResults:
    """Run full multi-layer A/B test."""
    
    models = list(set(fp.get("source_model", "unknown") for fp in fingerprints))
    
    print(f"\nRunning multi-layer test on {len(fingerprints)} samples, {len(models)} models")
    print("="*60)
    
    results = {}
    
    for config_name, features in CONFIGURATIONS.items():
        print(f"  Testing {config_name} ({len(features)} dims)...", end=" ", flush=True)
        result = test_configuration(fingerprints, features, config_name)
        results[config_name] = result
        print(f"{result.accuracy:.1%}")
    
    # Find optimal
    optimal = max(results.items(), key=lambda x: x[1].accuracy)
    optimal_config = optimal[0]
    optimal_accuracy = optimal[1].accuracy
    
    # Generate recommendations
    recommendations = []
    
    # Best accuracy
    if optimal_accuracy >= 0.85:
        recommendations.append(f"✓ Excellent discrimination ({optimal_accuracy:.0%}) with {optimal_config}")
    elif optimal_accuracy >= 0.70:
        recommendations.append(f"○ Good discrimination ({optimal_accuracy:.0%}) with {optimal_config}")
    else:
        recommendations.append(f"✗ Weak discrimination ({optimal_accuracy:.0%}) - consider more samples")
    
    # Layer contributions
    a_acc = results.get("A_trajectory", LayerTestResult("", 0, [], 0, 0, 0, {}, 0)).accuracy
    b_acc = results.get("B_specHO", LayerTestResult("", 0, [], 0, 0, 0, {}, 0)).accuracy
    ab_acc = results.get("AB", LayerTestResult("", 0, [], 0, 0, 0, {}, 0)).accuracy
    
    if a_acc > 0.5:
        recommendations.append(f"→ Trajectory alone has signal ({a_acc:.0%}) - hard to spoof")
    if ab_acc > a_acc + 0.05:
        recommendations.append(f"→ SpecHO adds value (+{(ab_acc-a_acc):.0%})")
    
    # Efficiency check
    full_acc = results.get("ABCD_full", LayerTestResult("", 0, [], 0, 0, 0, {}, 0)).accuracy
    ab_acc = results.get("AB", LayerTestResult("", 0, [], 0, 0, 0, {}, 0)).accuracy
    
    if abs(full_acc - ab_acc) < 0.02:
        recommendations.append(f"→ Layers C+D add minimal value - consider 20-dim for efficiency")
    
    return MultiLayerTestResults(
        n_samples=len(fingerprints),
        n_models=len(models),
        models=models,
        results={k: asdict(v) for k, v in results.items()},
        optimal_config=optimal_config,
        optimal_accuracy=optimal_accuracy,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat(),
    )


def generate_report(results: MultiLayerTestResults, output_path: str) -> str:
    """Generate HTML report."""
    
    configs = sorted(results.results.items(), key=lambda x: -x[1]["accuracy"])
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Layer Fingerprint Test</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: system-ui; }}
        .glass {{ background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); }}
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-cyan-400 mb-2">🧪 Multi-Layer Fingerprint Test</h1>
        <p class="text-slate-400 mb-8">Finding the optimal layer combination</p>
        
        <div class="grid grid-cols-3 gap-4 mb-8">
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-cyan-400">{results.n_samples}</div>
                <div class="text-slate-400 text-sm">Samples</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-green-400">{results.optimal_accuracy:.0%}</div>
                <div class="text-slate-400 text-sm">Best Accuracy</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-2xl font-bold text-purple-400">{results.optimal_config}</div>
                <div class="text-slate-400 text-sm">Optimal Config</div>
            </div>
        </div>
        
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Recommendations</h2>
            <ul class="space-y-2">
"""
    
    for rec in results.recommendations:
        html += f'<li class="text-slate-300">{rec}</li>'
    
    html += """
            </ul>
        </div>
        
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Configuration Comparison</h2>
            <div id="accuracy-chart" style="height: 400px;"></div>
        </div>
        
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Accuracy vs Dimensions</h2>
            <div id="efficiency-chart" style="height: 300px;"></div>
        </div>
        
        <div class="glass rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-4">All Results</h2>
            <table class="w-full">
                <tr class="text-left text-slate-400 border-b border-slate-700">
                    <th class="p-2">Configuration</th>
                    <th class="p-2">Dims</th>
                    <th class="p-2">Accuracy</th>
                    <th class="p-2">F1</th>
                    <th class="p-2">Time</th>
                </tr>
"""
    
    for config_name, result in configs:
        highlight = "bg-green-900/30" if config_name == results.optimal_config else ""
        html += f"""
                <tr class="{highlight} border-b border-slate-800">
                    <td class="p-2 font-medium">{config_name}</td>
                    <td class="p-2">{result['dimensions']}</td>
                    <td class="p-2">{result['accuracy']:.1%}</td>
                    <td class="p-2">{result['f1']:.1%}</td>
                    <td class="p-2">{result['training_time_ms']:.0f}ms</td>
                </tr>
"""
    
    config_names = [c[0] for c in configs]
    accuracies = [c[1]["accuracy"] for c in configs]
    dimensions = [c[1]["dimensions"] for c in configs]
    
    html += f"""
            </table>
        </div>
    </div>
    
    <script>
        // Accuracy chart
        Plotly.newPlot('accuracy-chart', [{{
            x: {json.dumps(config_names)},
            y: {json.dumps(accuracies)},
            type: 'bar',
            marker: {{
                color: {json.dumps(accuracies)},
                colorscale: [[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#22c55e']],
            }}
        }}], {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            yaxis: {{ title: 'Accuracy', range: [0, 1], gridcolor: '#334155' }},
            xaxis: {{ tickangle: -45 }},
        }});
        
        // Efficiency chart
        Plotly.newPlot('efficiency-chart', [{{
            x: {json.dumps(dimensions)},
            y: {json.dumps(accuracies)},
            text: {json.dumps(config_names)},
            mode: 'markers+text',
            textposition: 'top center',
            marker: {{ size: 15, color: '#22d3ee' }},
        }}], {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ title: 'Dimensions', gridcolor: '#334155' }},
            yaxis: {{ title: 'Accuracy', gridcolor: '#334155' }},
        }});
    </script>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Multi-layer fingerprint A/B test")
    parser.add_argument("--fingerprints", required=True, help="Path to cognitive fingerprints")
    parser.add_argument("--output", default="experiments/multilayer", help="Output directory")
    
    args = parser.parse_args()
    
    print("Loading fingerprints...")
    fingerprints = load_fingerprints(args.fingerprints)
    print(f"Loaded {len(fingerprints)} fingerprints")
    
    # Check for cognitive features
    if "concept_jump_mean" not in fingerprints[0]:
        print("ERROR: Fingerprints don't have cognitive features.")
        print("Run extract_cognitive.py first.")
        return
    
    results = run_multilayer_test(fingerprints)
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "multilayer_results.json", "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    
    report_path = generate_report(results, str(output_dir / "multilayer_report.html"))
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-LAYER TEST RESULTS")
    print("="*60)
    
    print(f"\nOptimal: {results.optimal_config} ({results.optimal_accuracy:.1%})")
    
    print("\nAll configurations:")
    for config, result in sorted(results.results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = "★" if config == results.optimal_config else " "
        print(f"  {marker} {config:20s}: {result['accuracy']:.1%} ({result['dimensions']} dims)")
    
    print("\nRecommendations:")
    for rec in results.recommendations:
        print(f"  {rec}")
    
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
