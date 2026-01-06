#!/usr/bin/env python3
"""
Fingerprint Analysis Tool

Analyzes extracted fingerprints for model discrimination.

Usage:
    python analyze_fingerprints.py --fingerprints data/fingerprints/fingerprints.json
    python analyze_fingerprints.py --fingerprints data/fingerprints/ --recursive
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional


def load_fingerprints(path: str, recursive: bool = False) -> list[dict]:
    """Load fingerprints from file or directory."""
    fingerprints = []
    path = Path(path)
    
    if path.is_file():
        with open(path) as f:
            fingerprints = json.load(f)
    elif path.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        for file in path.glob(pattern):
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    fingerprints.extend(data)
    
    return fingerprints


def fingerprint_to_vector(fp: dict) -> np.ndarray:
    """Convert fingerprint dict to numpy array."""
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


FEATURE_NAMES = [
    "phonetic_mean", "phonetic_std", "phonetic_max",
    "structural_mean", "structural_std", "structural_max",
    "semantic_mean", "semantic_std", "semantic_max",
    "cooccurrence_rate", "geometric_mean",
    "overall_mean", "overall_std", "overall_max",
    "burstiness",
]


def compute_separability(fingerprints: list[dict]) -> dict:
    """Compute separability metrics between models."""
    
    # Group by model
    by_model = defaultdict(list)
    for fp in fingerprints:
        model = fp.get("source_model", "unknown")
        by_model[model].append(fingerprint_to_vector(fp))
    
    models = list(by_model.keys())
    
    # Compute centroids
    centroids = {}
    for model, vectors in by_model.items():
        centroids[model] = np.mean(vectors, axis=0)
    
    # Pairwise distances
    distances = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            dist = np.linalg.norm(centroids[m1] - centroids[m2])
            distances[f"{m1}_vs_{m2}"] = float(dist)
    
    # Within-class variance
    variances = {}
    for model, vectors in by_model.items():
        if len(vectors) > 1:
            variances[model] = float(np.mean(np.var(vectors, axis=0)))
        else:
            variances[model] = 0.0
    
    return {
        "models": models,
        "n_samples_per_model": {m: len(v) for m, v in by_model.items()},
        "centroids": {m: c.tolist() for m, c in centroids.items()},
        "pairwise_distances": distances,
        "within_class_variance": variances,
    }


def train_classifier(fingerprints: list[dict]) -> dict:
    """Train classifier and evaluate."""
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return {"error": "sklearn not installed. Run: pip install scikit-learn"}
    
    # Prepare data
    X = np.array([fingerprint_to_vector(fp) for fp in fingerprints])
    y = np.array([fp.get("source_model", "unknown") for fp in fingerprints])
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    n_classes = len(le.classes_)
    n_samples = len(y)
    
    # Cross-validation
    cv_folds = min(5, min(np.bincount(y_encoded)))  # Ensure enough samples per fold
    if cv_folds < 2:
        return {"error": f"Not enough samples per class for cross-validation (min 2 needed)"}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_scores = cross_val_score(rf, X, y_encoded, cv=cv_folds)
    
    # Train final model for feature importance
    rf.fit(X, y_encoded)
    importance = dict(zip(FEATURE_NAMES, rf.feature_importances_))
    
    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "classes": le.classes_.tolist(),
        "cv_folds": cv_folds,
        "cv_accuracy_mean": float(np.mean(rf_scores)),
        "cv_accuracy_std": float(np.std(rf_scores)),
        "cv_scores": rf_scores.tolist(),
        "feature_importance": importance,
        "top_features": sorted(importance.items(), key=lambda x: -x[1])[:5],
        "success": np.mean(rf_scores) > 0.6,
    }


def generate_visualization(
    fingerprints: list[dict],
    output_path: str,
) -> dict:
    """Generate HTML visualization."""
    
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {"error": "sklearn not installed"}
    
    X = np.array([fingerprint_to_vector(fp) for fp in fingerprints])
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Prepare points for visualization
    colors = {
        "human": "#4CAF50",
        "claude": "#9C27B0",
        "gpt": "#2196F3",
        "gemini": "#FF9800",
        "grok": "#F44336",
        "llama": "#00BCD4",
        "mistral": "#795548",
        "phi": "#607D8B",
    }
    
    def get_color(model: str) -> str:
        for key, color in colors.items():
            if key in model.lower():
                return color
        return "#9E9E9E"
    
    points = []
    for i, fp in enumerate(fingerprints):
        model = fp.get("source_model", "unknown")
        points.append({
            "x": float(X_pca[i, 0]),
            "y": float(X_pca[i, 1]),
            "model": model,
            "id": fp.get("sample_id", f"sample_{i}"),
            "color": get_color(model),
        })
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SpecHO Fingerprint Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fafafa; }}
        h1 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        #plot {{ width: 100%; height: 600px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; }}
        .legend {{ margin-top: 20px; display: flex; flex-wrap: wrap; gap: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SpecHO Model Fingerprint Analysis</h1>
        <p>PCA projection of 15-dimensional fingerprint vectors. Clusters indicate distinguishable model signatures.</p>
        
        <div id="plot"></div>
        
        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background: #4CAF50;"></span> Human</div>
            <div class="legend-item"><span class="legend-color" style="background: #9C27B0;"></span> Claude</div>
            <div class="legend-item"><span class="legend-color" style="background: #2196F3;"></span> GPT</div>
            <div class="legend-item"><span class="legend-color" style="background: #FF9800;"></span> Gemini</div>
            <div class="legend-item"><span class="legend-color" style="background: #F44336;"></span> Grok</div>
            <div class="legend-item"><span class="legend-color" style="background: #00BCD4;"></span> Llama</div>
            <div class="legend-item"><span class="legend-color" style="background: #795548;"></span> Mistral</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(fingerprints)}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(set(p['model'] for p in points))}</div>
                <div class="stat-label">Model Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pca.explained_variance_ratio_[0]*100:.1f}%</div>
                <div class="stat-label">PC1 Variance</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pca.explained_variance_ratio_[1]*100:.1f}%</div>
                <div class="stat-label">PC2 Variance</div>
            </div>
        </div>
    </div>
    
    <script>
        const points = {json.dumps(points)};
        
        const byModel = {{}};
        points.forEach(p => {{
            if (!byModel[p.model]) byModel[p.model] = {{x: [], y: [], text: [], color: p.color}};
            byModel[p.model].x.push(p.x);
            byModel[p.model].y.push(p.y);
            byModel[p.model].text.push(p.id);
        }});
        
        const traces = Object.entries(byModel).map(([model, data]) => ({{
            x: data.x,
            y: data.y,
            text: data.text,
            mode: 'markers',
            type: 'scatter',
            name: model,
            marker: {{ color: data.color, size: 12, opacity: 0.7 }},
            hovertemplate: '<b>%{{text}}</b><br>Model: ' + model + '<extra></extra>'
        }}));
        
        Plotly.newPlot('plot', traces, {{
            title: 'Model Fingerprint Space (PCA)',
            xaxis: {{ title: 'PC1', zeroline: false }},
            yaxis: {{ title: 'PC2', zeroline: false }},
            hovermode: 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
        }});
    </script>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    
    return {
        "path": output_path,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def generate_report(
    fingerprints: list[dict],
    output_dir: str,
) -> dict:
    """Generate comprehensive analysis report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nComputing separability metrics...")
    separability = compute_separability(fingerprints)
    
    print("Training classifier...")
    classifier = train_classifier(fingerprints)
    
    print("Generating visualization...")
    viz = generate_visualization(fingerprints, str(output_path / "fingerprint_space.html"))
    
    # Generate recommendation
    if classifier.get("error"):
        recommendation = classifier["error"]
    else:
        acc = classifier["cv_accuracy_mean"]
        if acc > 0.85:
            recommendation = "STRONG: Fingerprinting shows excellent discrimination. Proceed to full corpus."
        elif acc > 0.70:
            recommendation = "GOOD: Fingerprinting viable. Consider expanding feature set or corpus."
        elif acc > 0.50:
            recommendation = "WEAK: Some discrimination, but needs improvement. Analyze top features."
        else:
            recommendation = "INSUFFICIENT: Features don't discriminate well. Revisit hypothesis."
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(fingerprints),
        "n_models": len(separability["models"]),
        "models": separability["models"],
        "separability": separability,
        "classifier": classifier,
        "visualization": viz,
        "recommendation": recommendation,
    }
    
    # Save report
    report_file = output_path / "report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nSamples: {len(fingerprints)}")
    print(f"Models: {len(separability['models'])}")
    print(f"  {', '.join(separability['models'])}")
    
    if not classifier.get("error"):
        print(f"\nClassifier Accuracy: {classifier['cv_accuracy_mean']:.1%} (±{classifier['cv_accuracy_std']:.1%})")
        print(f"Success: {'YES ✓' if classifier['success'] else 'NO ✗'}")
        
        print("\nTop Features:")
        for feat, imp in classifier["top_features"]:
            print(f"  {feat}: {imp:.3f}")
    
    print(f"\n{recommendation}")
    print(f"\nOutputs:")
    print(f"  Report: {report_file}")
    print(f"  Visualization: {output_path / 'fingerprint_space.html'}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze fingerprints")
    parser.add_argument(
        "--fingerprints", type=str, required=True,
        help="Path to fingerprints file or directory"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/validation",
        help="Output directory"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search directories"
    )
    
    args = parser.parse_args()
    
    fingerprints = load_fingerprints(args.fingerprints, args.recursive)
    
    if not fingerprints:
        print("No fingerprints found")
        return
    
    generate_report(fingerprints, args.output)


if __name__ == "__main__":
    main()
