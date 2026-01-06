#!/usr/bin/env python3
"""
A/B Test: 15-dim Baseline vs 20-dim Extended Fingerprints

The only question that matters: Does trajectory help?

This tool:
1. Extracts both 15-dim and 20-dim fingerprints from corpus
2. Trains classifiers on each
3. Compares accuracy, per-class performance, feature importance
4. Generates report with GO/NOGO recommendation

Usage:
    python ab_test.py --corpus data/corpus/merged/corpus.json
    python ab_test.py --fingerprints data/fingerprints/extended.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ABTestResult:
    """Results of A/B comparison."""
    
    # Sample info
    n_samples: int
    n_models: int
    models: list[str]
    
    # Baseline (15-dim)
    baseline_accuracy: float
    baseline_f1: float
    baseline_per_class: dict
    
    # Extended (20-dim)  
    extended_accuracy: float
    extended_f1: float
    extended_per_class: dict
    
    # Trajectory only (5-dim)
    trajectory_only_accuracy: float
    trajectory_only_f1: float
    
    # Comparison
    accuracy_delta: float
    f1_delta: float
    trajectory_helps: bool
    
    # Feature importance
    baseline_top_features: list
    extended_top_features: list
    trajectory_feature_ranks: dict
    
    # Recommendation
    recommendation: str
    
    timestamp: str = ""


def load_fingerprints(path: str) -> list[dict]:
    """Load fingerprints from JSON file."""
    with open(path) as f:
        return json.load(f)


def fingerprint_to_vectors(fp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 15-dim, 20-dim, and 5-dim trajectory vectors from fingerprint.
    
    Returns:
        (vec_15, vec_20, vec_trajectory)
    """
    # 15-dim baseline
    vec_15 = np.array([
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
    
    # 5-dim trajectory
    vec_trajectory = np.array([
        fp.get("concept_jump_mean", 0.5),
        fp.get("concept_jump_variance", 0.1),
        fp.get("path_tortuosity", 1.0),
        fp.get("turning_angle_mean", 0.5),
        fp.get("return_rate", 0.1),
    ])
    
    # 20-dim extended
    vec_20 = np.concatenate([vec_15, vec_trajectory])
    
    return vec_15, vec_20, vec_trajectory


FEATURE_NAMES_15 = [
    "phonetic_mean", "phonetic_std", "phonetic_max",
    "structural_mean", "structural_std", "structural_max",
    "semantic_mean", "semantic_std", "semantic_max",
    "cooccurrence_rate", "geometric_mean",
    "overall_mean", "overall_std", "overall_max",
    "burstiness",
]

FEATURE_NAMES_20 = FEATURE_NAMES_15 + [
    "concept_jump_mean", "concept_jump_variance",
    "path_tortuosity", "turning_angle_mean",
    "return_rate",
]

FEATURE_NAMES_TRAJECTORY = [
    "concept_jump_mean", "concept_jump_variance",
    "path_tortuosity", "turning_angle_mean",
    "return_rate",
]


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """
    Train classifier and evaluate with cross-validation.
    
    Returns:
        Dict with accuracy, f1, per_class metrics, feature importance
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import f1_score, accuracy_score
    except ImportError:
        return {"error": "sklearn not installed"}
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    n_classes = len(le.classes_)
    min_per_class = min(Counter(y_encoded).values())
    
    # Need at least 2 samples per class for stratified CV
    if min_per_class < 2:
        return {"error": f"Need at least 2 samples per class (min: {min_per_class})"}
    
    cv_folds = min(5, min_per_class)
    
    # Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
    )
    
    # Cross-validation scores
    cv_scores = cross_val_score(clf, X, y_encoded, cv=cv_folds)
    
    # Get predictions for per-class metrics
    y_pred = cross_val_predict(clf, X, y_encoded, cv=cv_folds)
    
    # Fit on all data for feature importance
    clf.fit(X, y_encoded)
    importance = dict(zip(feature_names, clf.feature_importances_))
    
    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = (y_encoded == cls_idx)
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_encoded[mask], y_pred[mask])
            per_class[cls_name] = float(cls_acc)
    
    return {
        "accuracy": float(np.mean(cv_scores)),
        "accuracy_std": float(np.std(cv_scores)),
        "f1": float(f1_score(y_encoded, y_pred, average='weighted')),
        "per_class": per_class,
        "feature_importance": importance,
        "top_features": sorted(importance.items(), key=lambda x: -x[1])[:5],
        "cv_folds": cv_folds,
        "n_classes": n_classes,
    }


def run_ab_test(fingerprints: list[dict]) -> ABTestResult:
    """
    Run A/B test comparing 15-dim vs 20-dim fingerprints.
    
    Args:
        fingerprints: List of fingerprint dicts with all 20 features
        
    Returns:
        ABTestResult with comparison metrics
    """
    # Prepare data
    X_15 = []
    X_20 = []
    X_traj = []
    y = []
    
    for fp in fingerprints:
        model = fp.get("source_model", "unknown")
        vec_15, vec_20, vec_traj = fingerprint_to_vectors(fp)
        
        X_15.append(vec_15)
        X_20.append(vec_20)
        X_traj.append(vec_traj)
        y.append(model)
    
    X_15 = np.array(X_15)
    X_20 = np.array(X_20)
    X_traj = np.array(X_traj)
    y = np.array(y)
    
    models = list(set(y))
    
    print(f"\nRunning A/B test on {len(fingerprints)} samples, {len(models)} models")
    
    # Evaluate each configuration
    print("\n[A] Baseline (15-dim)...")
    result_15 = train_and_evaluate(X_15, y, FEATURE_NAMES_15)
    
    print("[B] Extended (20-dim)...")
    result_20 = train_and_evaluate(X_20, y, FEATURE_NAMES_20)
    
    print("[C] Trajectory only (5-dim)...")
    result_traj = train_and_evaluate(X_traj, y, FEATURE_NAMES_TRAJECTORY)
    
    # Check for errors
    if "error" in result_15 or "error" in result_20:
        error = result_15.get("error") or result_20.get("error")
        return ABTestResult(
            n_samples=len(fingerprints),
            n_models=len(models),
            models=models,
            baseline_accuracy=0,
            baseline_f1=0,
            baseline_per_class={},
            extended_accuracy=0,
            extended_f1=0,
            extended_per_class={},
            trajectory_only_accuracy=0,
            trajectory_only_f1=0,
            accuracy_delta=0,
            f1_delta=0,
            trajectory_helps=False,
            baseline_top_features=[],
            extended_top_features=[],
            trajectory_feature_ranks={},
            recommendation=f"ERROR: {error}",
            timestamp=datetime.now().isoformat(),
        )
    
    # Compute deltas
    accuracy_delta = result_20["accuracy"] - result_15["accuracy"]
    f1_delta = result_20["f1"] - result_15["f1"]
    
    # Check if trajectory features appear in top importance
    traj_ranks = {}
    if "feature_importance" in result_20:
        sorted_features = sorted(
            result_20["feature_importance"].items(),
            key=lambda x: -x[1]
        )
        for rank, (feat, _) in enumerate(sorted_features):
            if feat in FEATURE_NAMES_TRAJECTORY:
                traj_ranks[feat] = rank + 1  # 1-indexed
    
    # Determine if trajectory helps
    trajectory_helps = (
        accuracy_delta > 0.02  # At least 2% improvement
        or any(rank <= 10 for rank in traj_ranks.values())  # Or a trajectory feature is top-10
    )
    
    # Generate recommendation
    if result_20["accuracy"] >= 0.85 and trajectory_helps:
        recommendation = "STRONG: Extended fingerprint outperforms baseline. Ship 20-dim."
    elif result_20["accuracy"] >= 0.70 and accuracy_delta > 0:
        recommendation = "GOOD: Trajectory adds value. Consider 20-dim for production."
    elif result_15["accuracy"] >= 0.70 and accuracy_delta <= 0:
        recommendation = "BASELINE SUFFICIENT: Trajectory doesn't help. Use 15-dim."
    elif result_traj["accuracy"] >= 0.50:
        recommendation = "TRAJECTORY SIGNAL: Trajectory alone has discriminative power. Investigate."
    else:
        recommendation = "WEAK: Neither configuration achieves good discrimination. Revisit features."
    
    return ABTestResult(
        n_samples=len(fingerprints),
        n_models=len(models),
        models=models,
        
        baseline_accuracy=result_15["accuracy"],
        baseline_f1=result_15["f1"],
        baseline_per_class=result_15.get("per_class", {}),
        
        extended_accuracy=result_20["accuracy"],
        extended_f1=result_20["f1"],
        extended_per_class=result_20.get("per_class", {}),
        
        trajectory_only_accuracy=result_traj.get("accuracy", 0),
        trajectory_only_f1=result_traj.get("f1", 0),
        
        accuracy_delta=accuracy_delta,
        f1_delta=f1_delta,
        trajectory_helps=trajectory_helps,
        
        baseline_top_features=result_15.get("top_features", []),
        extended_top_features=result_20.get("top_features", []),
        trajectory_feature_ranks=traj_ranks,
        
        recommendation=recommendation,
        timestamp=datetime.now().isoformat(),
    )


def generate_report(result: ABTestResult, output_path: str) -> str:
    """Generate HTML report for A/B test."""
    
    def bar(value: float, max_val: float = 1.0) -> str:
        pct = min(100, value / max_val * 100)
        color = "#4CAF50" if value >= 0.7 else "#FF9800" if value >= 0.5 else "#f44336"
        return f'<div style="background: {color}; width: {pct}%; height: 20px; border-radius: 4px;"></div>'
    
    delta_color = "#4CAF50" if result.accuracy_delta > 0 else "#f44336" if result.accuracy_delta < 0 else "#999"
    delta_sign = "+" if result.accuracy_delta > 0 else ""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>A/B Test: 15-dim vs 20-dim Fingerprints</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1, h2 {{ color: #333; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ font-size: 2.5em; font-weight: bold; }}
        .label {{ color: #666; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .delta {{ color: {delta_color}; font-size: 1.2em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        .bar-container {{ width: 200px; background: #eee; border-radius: 4px; }}
        .recommendation {{ font-size: 1.2em; padding: 20px; border-left: 4px solid {"#4CAF50" if "STRONG" in result.recommendation or "GOOD" in result.recommendation else "#FF9800"}; background: #f9f9f9; }}
        .winner {{ background: #e8f5e9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>A/B Test: Baseline (15-dim) vs Extended (20-dim)</h1>
        <p>Testing whether trajectory features improve model discrimination</p>
        <p>Generated: {result.timestamp}</p>
        
        <div class="card">
            <h2>Summary</h2>
            <div class="grid">
                <div>
                    <div class="metric">{result.n_samples}</div>
                    <div class="label">Samples</div>
                </div>
                <div>
                    <div class="metric">{result.n_models}</div>
                    <div class="label">Models</div>
                </div>
                <div>
                    <div class="metric delta">{delta_sign}{result.accuracy_delta:.1%}</div>
                    <div class="label">Accuracy Delta</div>
                </div>
                <div>
                    <div class="metric">{"✓" if result.trajectory_helps else "✗"}</div>
                    <div class="label">Trajectory Helps</div>
                </div>
            </div>
        </div>
        
        <div class="card recommendation">
            <strong>RECOMMENDATION:</strong> {result.recommendation}
        </div>
        
        <div class="card">
            <h2>Accuracy Comparison</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th></th>
                </tr>
                <tr class="{"winner" if result.baseline_accuracy > result.extended_accuracy else ""}">
                    <td><strong>[A] Baseline (15-dim)</strong></td>
                    <td>{result.baseline_accuracy:.1%}</td>
                    <td>{result.baseline_f1:.1%}</td>
                    <td class="bar-container">{bar(result.baseline_accuracy)}</td>
                </tr>
                <tr class="{"winner" if result.extended_accuracy > result.baseline_accuracy else ""}">
                    <td><strong>[B] Extended (20-dim)</strong></td>
                    <td>{result.extended_accuracy:.1%}</td>
                    <td>{result.extended_f1:.1%}</td>
                    <td class="bar-container">{bar(result.extended_accuracy)}</td>
                </tr>
                <tr>
                    <td><strong>[C] Trajectory Only (5-dim)</strong></td>
                    <td>{result.trajectory_only_accuracy:.1%}</td>
                    <td>{result.trajectory_only_f1:.1%}</td>
                    <td class="bar-container">{bar(result.trajectory_only_accuracy)}</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Feature Importance (Extended 20-dim)</h2>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Type</th></tr>
"""
    
    for rank, (feat, imp) in enumerate(result.extended_top_features, 1):
        feat_type = "🚀 TRAJECTORY" if feat in FEATURE_NAMES_TRAJECTORY else "📊 Original"
        html += f"""
                <tr>
                    <td>{rank}</td>
                    <td>{feat}</td>
                    <td>{imp:.3f}</td>
                    <td>{feat_type}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>Trajectory Feature Ranks (in 20-dim model)</h2>
            <table>
                <tr><th>Feature</th><th>Rank (out of 20)</th><th>Signal Strength</th></tr>
"""
    
    for feat in FEATURE_NAMES_TRAJECTORY:
        rank = result.trajectory_feature_ranks.get(feat, "N/A")
        strength = "🔥 Strong" if isinstance(rank, int) and rank <= 5 else "✓ Moderate" if isinstance(rank, int) and rank <= 10 else "○ Weak"
        html += f"""
                <tr>
                    <td>{feat}</td>
                    <td>{rank}</td>
                    <td>{strength}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>Per-Model Accuracy (Extended)</h2>
            <table>
                <tr><th>Model</th><th>Accuracy</th><th></th></tr>
"""
    
    for model, acc in sorted(result.extended_per_class.items(), key=lambda x: -x[1]):
        html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{acc:.1%}</td>
                    <td class="bar-container">{bar(acc)}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>Interpretation</h2>
            <ul>
                <li><strong>If Extended > Baseline:</strong> Trajectory captures real signal about model cognition</li>
                <li><strong>If Trajectory-Only > 50%:</strong> Path geometry alone discriminates models</li>
                <li><strong>If trajectory features rank high:</strong> They're not just noise</li>
                <li><strong>Watch for:</strong> Overfitting (20 features on small samples)</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="A/B test: 15-dim baseline vs 20-dim extended fingerprints",
        epilog="""
This test answers one question: Does trajectory geometry improve model discrimination?

If yes: Ship 20-dim fingerprints
If no: Stick with 15-dim baseline
"""
    )
    parser.add_argument(
        "--fingerprints", type=str, required=True,
        help="Path to fingerprints JSON with all 20 features"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/ab_test",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load fingerprints
    print(f"Loading fingerprints from {args.fingerprints}...")
    fingerprints = load_fingerprints(args.fingerprints)
    print(f"Loaded {len(fingerprints)} fingerprints")
    
    # Check for trajectory features
    sample = fingerprints[0]
    has_trajectory = "concept_jump_mean" in sample
    
    if not has_trajectory:
        print("\nWARNING: Fingerprints don't have trajectory features.")
        print("Run extract_extended.py first to get 20-dim fingerprints.")
        return
    
    # Run A/B test
    result = run_ab_test(fingerprints)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "ab_test_results.json"
    with open(results_file, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    report_path = generate_report(result, str(output_dir / "ab_test_report.html"))
    
    # Print summary
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)
    
    print(f"\nSamples: {result.n_samples}")
    print(f"Models: {', '.join(result.models)}")
    
    print(f"\n[A] Baseline (15-dim):    {result.baseline_accuracy:.1%} accuracy")
    print(f"[B] Extended (20-dim):    {result.extended_accuracy:.1%} accuracy")
    print(f"[C] Trajectory-only (5):  {result.trajectory_only_accuracy:.1%} accuracy")
    
    print(f"\nDelta: {'+' if result.accuracy_delta > 0 else ''}{result.accuracy_delta:.1%}")
    print(f"Trajectory helps: {'YES ✓' if result.trajectory_helps else 'NO ✗'}")
    
    print(f"\n{result.recommendation}")
    
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
