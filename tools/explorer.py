#!/usr/bin/env python3
"""
Interactive Fingerprint Explorer

Generates an interactive web dashboard for exploring fingerprint data.

Features:
- 3D fingerprint space visualization
- Model comparison tool
- Feature importance breakdown
- Sample browser

Usage:
    python explorer.py --fingerprints data/fingerprints/fingerprints.json
    python explorer.py --corpus data/corpus/merged/corpus.json --extract
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


FEATURE_NAMES = [
    "phonetic_mean", "phonetic_std", "phonetic_max",
    "structural_mean", "structural_std", "structural_max",
    "semantic_mean", "semantic_std", "semantic_max",
    "cooccurrence_rate", "geometric_mean",
    "overall_mean", "overall_std", "overall_max",
    "burstiness",
]

MODEL_COLORS = {
    "human": "#4CAF50",
    "claude": "#9C27B0",
    "gpt": "#2196F3",
    "gemini": "#FF9800",
    "grok": "#F44336",
    "llama": "#00BCD4",
    "mistral": "#795548",
    "phi": "#607D8B",
    "qwen": "#E91E63",
    "gemma": "#3F51B5",
}


def get_color(model: str) -> str:
    """Get color for a model."""
    model_lower = model.lower()
    for key, color in MODEL_COLORS.items():
        if key in model_lower:
            return color
    return "#9E9E9E"


def generate_explorer(fingerprints: list[dict], output_path: str) -> str:
    """Generate interactive HTML explorer."""
    
    # Group by model
    by_model = defaultdict(list)
    for fp in fingerprints:
        model = fp.get("source_model", "unknown")
        by_model[model].append(fp)
    
    # Compute statistics
    model_stats = {}
    for model, fps in by_model.items():
        vectors = np.array([
            [fp.get(f, 0.5) for f in FEATURE_NAMES]
            for fp in fps
        ])
        model_stats[model] = {
            "count": len(fps),
            "mean": vectors.mean(axis=0).tolist(),
            "std": vectors.std(axis=0).tolist() if len(vectors) > 1 else [0] * 15,
            "color": get_color(model),
        }
    
    # PCA for visualization
    try:
        from sklearn.decomposition import PCA
        
        all_vectors = np.array([
            [fp.get(f, 0.5) for f in FEATURE_NAMES]
            for fp in fingerprints
        ])
        
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(all_vectors)
        
        pca_2d = PCA(n_components=2)
        coords_2d = pca_2d.fit_transform(all_vectors)
        
        explained_var = pca.explained_variance_ratio_.tolist()
    except ImportError:
        # Fallback: random projection
        all_vectors = np.array([
            [fp.get(f, 0.5) for f in FEATURE_NAMES]
            for fp in fingerprints
        ])
        coords_3d = all_vectors[:, :3]
        coords_2d = all_vectors[:, :2]
        explained_var = [0.33, 0.33, 0.33]
    
    # Build data for visualization
    points_3d = []
    points_2d = []
    samples = []
    
    for i, fp in enumerate(fingerprints):
        model = fp.get("source_model", "unknown")
        sample_id = fp.get("sample_id", f"sample_{i}")
        
        points_3d.append({
            "x": float(coords_3d[i, 0]),
            "y": float(coords_3d[i, 1]),
            "z": float(coords_3d[i, 2]),
            "model": model,
            "id": sample_id,
            "color": get_color(model),
        })
        
        points_2d.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "model": model,
            "id": sample_id,
            "color": get_color(model),
        })
        
        samples.append({
            "id": sample_id,
            "model": model,
            "features": {f: fp.get(f, 0.5) for f in FEATURE_NAMES},
        })
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SpecHO Fingerprint Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }}
        .glass {{ background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border: 1px solid rgba(148, 163, 184, 0.1); }}
        .plot-container {{ height: 500px; }}
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-cyan-400 mb-2">🔬 SpecHO Fingerprint Explorer</h1>
            <p class="text-slate-400">Interactive visualization of model fingerprint space</p>
        </div>
        
        <!-- Stats Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-cyan-400">{len(fingerprints)}</div>
                <div class="text-slate-400 text-sm">Total Samples</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-purple-400">{len(by_model)}</div>
                <div class="text-slate-400 text-sm">Models</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-green-400">{explained_var[0]*100:.1f}%</div>
                <div class="text-slate-400 text-sm">PC1 Variance</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-3xl font-bold text-yellow-400">{explained_var[1]*100:.1f}%</div>
                <div class="text-slate-400 text-sm">PC2 Variance</div>
            </div>
        </div>
        
        <!-- Main Visualization -->
        <div class="grid md:grid-cols-2 gap-6 mb-8">
            <!-- 3D Plot -->
            <div class="glass rounded-lg p-4">
                <h2 class="text-xl font-semibold mb-4">3D Fingerprint Space</h2>
                <div id="plot3d" class="plot-container"></div>
            </div>
            
            <!-- 2D Plot -->
            <div class="glass rounded-lg p-4">
                <h2 class="text-xl font-semibold mb-4">2D Projection (PCA)</h2>
                <div id="plot2d" class="plot-container"></div>
            </div>
        </div>
        
        <!-- Model Comparison -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Model Fingerprint Profiles</h2>
            <div id="radar" style="height: 400px;"></div>
        </div>
        
        <!-- Feature Distribution -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Feature Distributions by Model</h2>
            <div id="boxplot" style="height: 400px;"></div>
        </div>
        
        <!-- Model Legend -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Model Legend</h2>
            <div class="flex flex-wrap gap-4">
"""
    
    for model, stats in model_stats.items():
        html += f"""
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 rounded-full" style="background: {stats['color']}"></div>
                    <span>{model}</span>
                    <span class="text-slate-500">({stats['count']})</span>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <!-- Sample Browser -->
        <div class="glass rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-4">Sample Browser</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-left text-slate-400 border-b border-slate-700">
                            <th class="p-2">ID</th>
                            <th class="p-2">Model</th>
                            <th class="p-2">Overall Mean</th>
                            <th class="p-2">Burstiness</th>
                            <th class="p-2">Co-occurrence</th>
                        </tr>
                    </thead>
                    <tbody id="sample-table">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Data
        const points3d = POINTS_3D_DATA;
        const points2d = POINTS_2D_DATA;
        const modelStats = MODEL_STATS_DATA;
        const samples = SAMPLES_DATA;
        const featureNames = FEATURE_NAMES_DATA;
        
        // 3D Scatter Plot
        const traces3d = [];
        const modelGroups = {{}};
        
        points3d.forEach(p => {{
            if (!modelGroups[p.model]) {{
                modelGroups[p.model] = {{x: [], y: [], z: [], text: [], color: p.color}};
            }}
            modelGroups[p.model].x.push(p.x);
            modelGroups[p.model].y.push(p.y);
            modelGroups[p.model].z.push(p.z);
            modelGroups[p.model].text.push(p.id);
        }});
        
        Object.entries(modelGroups).forEach(([model, data]) => {{
            traces3d.push({{
                x: data.x,
                y: data.y,
                z: data.z,
                text: data.text,
                type: 'scatter3d',
                mode: 'markers',
                name: model,
                marker: {{ color: data.color, size: 6, opacity: 0.8 }},
            }});
        }});
        
        Plotly.newPlot('plot3d', traces3d, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            scene: {{
                xaxis: {{ title: 'PC1', gridcolor: '#334155' }},
                yaxis: {{ title: 'PC2', gridcolor: '#334155' }},
                zaxis: {{ title: 'PC3', gridcolor: '#334155' }},
                bgcolor: 'rgba(0,0,0,0)',
            }},
            margin: {{ l: 0, r: 0, t: 0, b: 0 }},
        }});
        
        // 2D Scatter Plot
        const traces2d = [];
        const modelGroups2d = {{}};
        
        points2d.forEach(p => {{
            if (!modelGroups2d[p.model]) {{
                modelGroups2d[p.model] = {{x: [], y: [], text: [], color: p.color}};
            }}
            modelGroups2d[p.model].x.push(p.x);
            modelGroups2d[p.model].y.push(p.y);
            modelGroups2d[p.model].text.push(p.id);
        }});
        
        Object.entries(modelGroups2d).forEach(([model, data]) => {{
            traces2d.push({{
                x: data.x,
                y: data.y,
                text: data.text,
                type: 'scatter',
                mode: 'markers',
                name: model,
                marker: {{ color: data.color, size: 10, opacity: 0.7 }},
            }});
        }});
        
        Plotly.newPlot('plot2d', traces2d, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ title: 'PC1', gridcolor: '#334155', zeroline: false }},
            yaxis: {{ title: 'PC2', gridcolor: '#334155', zeroline: false }},
            margin: {{ l: 50, r: 20, t: 20, b: 50 }},
        }});
        
        // Radar Chart
        const radarTraces = Object.entries(modelStats).map(([model, stats]) => ({{
            type: 'scatterpolar',
            r: stats.mean.slice(0, 9),  // First 9 features for readability
            theta: featureNames.slice(0, 9),
            fill: 'toself',
            name: model,
            line: {{ color: stats.color }},
            opacity: 0.6,
        }}));
        
        Plotly.newPlot('radar', radarTraces, {{
            polar: {{
                bgcolor: 'rgba(0,0,0,0)',
                radialaxis: {{ visible: true, range: [0, 1], gridcolor: '#334155' }},
                angularaxis: {{ gridcolor: '#334155' }},
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            margin: {{ l: 80, r: 80, t: 40, b: 40 }},
        }});
        
        // Box Plot for feature distributions
        const boxTraces = [];
        ['overall_mean', 'burstiness', 'cooccurrence_rate'].forEach(feature => {{
            Object.entries(modelStats).forEach(([model, stats]) => {{
                const idx = featureNames.indexOf(feature);
                // Get all values for this model
                const values = samples
                    .filter(s => s.model === model)
                    .map(s => s.features[feature]);
                
                boxTraces.push({{
                    y: values,
                    name: `${{model}} - ${{feature}}`,
                    type: 'box',
                    marker: {{ color: stats.color }},
                }});
            }});
        }});
        
        Plotly.newPlot('boxplot', boxTraces.slice(0, 12), {{  // Limit for readability
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            yaxis: {{ gridcolor: '#334155' }},
            margin: {{ l: 50, r: 20, t: 20, b: 100 }},
            boxmode: 'group',
        }});
        
        // Sample Table
        const tbody = document.getElementById('sample-table');
        samples.slice(0, 50).forEach(s => {{
            const row = document.createElement('tr');
            row.className = 'border-b border-slate-800 hover:bg-slate-800/50';
            row.innerHTML = `
                <td class="p-2">${{s.id}}</td>
                <td class="p-2">${{s.model}}</td>
                <td class="p-2">${{s.features.overall_mean?.toFixed(3) || 'N/A'}}</td>
                <td class="p-2">${{s.features.burstiness?.toFixed(3) || 'N/A'}}</td>
                <td class="p-2">${{s.features.cooccurrence_rate?.toFixed(3) || 'N/A'}}</td>
            `;
            tbody.appendChild(row);
        }});
    </script>
</body>
</html>
"""
    
    # Replace data placeholders
    html = html.replace("POINTS_3D_DATA", json.dumps(points_3d))
    html = html.replace("POINTS_2D_DATA", json.dumps(points_2d))
    html = html.replace("MODEL_STATS_DATA", json.dumps(model_stats))
    html = html.replace("SAMPLES_DATA", json.dumps(samples))
    html = html.replace("FEATURE_NAMES_DATA", json.dumps(FEATURE_NAMES))
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Interactive fingerprint explorer")
    parser.add_argument("--fingerprints", type=str, required=True,
                       help="Path to fingerprints JSON")
    parser.add_argument("--output", type=str, default="experiments/explorer.html",
                       help="Output HTML path")
    
    args = parser.parse_args()
    
    print("Loading fingerprints...")
    with open(args.fingerprints) as f:
        fingerprints = json.load(f)
    
    print(f"Loaded {len(fingerprints)} fingerprints")
    
    print("Generating explorer...")
    path = generate_explorer(fingerprints, args.output)
    
    print(f"Explorer saved to {path}")


if __name__ == "__main__":
    main()
