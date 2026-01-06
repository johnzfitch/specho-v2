#!/usr/bin/env python3
"""
Trajectory Visualization

Visualizes how different models navigate semantic space.
Shows the "cognitive gait" - the rhythm and geometry of thought.

Creates:
1. 2D/3D trajectory plots showing paths through embedding space
2. Step size distributions by model
3. Turning angle distributions by model
4. Comparative "gait signatures"

Usage:
    python visualize_trajectory.py --corpus data/corpus/merged/corpus.json
    python visualize_trajectory.py --compare claude gpt-4o gemini
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


MODEL_COLORS = {
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
    for key, color in MODEL_COLORS.items():
        if key in model.lower():
            return color
    return "#9E9E9E"


class TrajectoryVisualizer:
    """Visualize semantic trajectories."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.sentence_model = None
        self._init_model(embedding_model)
    
    def _init_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(model_name)
        except ImportError:
            print("[Visualizer] sentence-transformers not available, using mock")
            self.sentence_model = None
    
    def segment_clauses(self, text: str) -> list[str]:
        """Segment text into clauses."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clauses = []
        for sentence in sentences:
            parts = re.split(
                r'(?<=[,;:])\s+|'
                r'\s*—\s*|'
                r'\s+(?:but|and|or|yet|so|however|therefore|thus|although|while)\s+',
                sentence,
                flags=re.IGNORECASE
            )
            
            for part in parts:
                part = part.strip()
                if len(part.split()) >= 3:
                    clauses.append(part)
        
        return clauses
    
    def embed_clauses(self, clauses: list[str]) -> np.ndarray:
        """Embed clauses."""
        if self.sentence_model is None:
            import hashlib
            embeddings = []
            for clause in clauses:
                seed = int(hashlib.md5(clause.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                embeddings.append(rng.randn(384))
            return np.array(embeddings) if embeddings else np.zeros((0, 384))
        
        if not clauses:
            return np.zeros((0, 384))
        
        return self.sentence_model.encode(clauses, convert_to_numpy=True)
    
    def compute_trajectory_stats(self, embeddings: np.ndarray) -> dict:
        """Compute trajectory statistics."""
        if len(embeddings) < 2:
            return {
                "step_lengths": [],
                "turning_angles": [],
                "tortuosity": 1.0,
            }
        
        steps = np.diff(embeddings, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1).tolist()
        
        turning_angles = []
        for i in range(len(steps) - 1):
            v1, v2 = steps[i], steps[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                turning_angles.append(np.arccos(cos_angle))
        
        path_length = sum(step_lengths)
        displacement = np.linalg.norm(embeddings[-1] - embeddings[0])
        tortuosity = path_length / max(displacement, 1e-6)
        
        return {
            "step_lengths": step_lengths,
            "turning_angles": turning_angles,
            "tortuosity": tortuosity,
        }
    
    def process_corpus(self, corpus: list[dict]) -> dict:
        """Process corpus and collect trajectory stats by model."""
        stats_by_model = defaultdict(lambda: {
            "step_lengths": [],
            "turning_angles": [],
            "tortuosities": [],
            "trajectories_2d": [],
        })
        
        # For PCA projection
        all_embeddings = []
        embedding_info = []  # (model, sample_idx, clause_idx)
        
        for sample in corpus:
            model = sample.get("model_name") or sample.get("model_id") or "unknown"
            text = sample.get("text", "")
            
            if not text:
                continue
            
            clauses = self.segment_clauses(text)
            if len(clauses) < 2:
                continue
            
            embeddings = self.embed_clauses(clauses)
            stats = self.compute_trajectory_stats(embeddings)
            
            stats_by_model[model]["step_lengths"].extend(stats["step_lengths"])
            stats_by_model[model]["turning_angles"].extend(stats["turning_angles"])
            stats_by_model[model]["tortuosities"].append(stats["tortuosity"])
            
            # Collect for PCA
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                embedding_info.append((model, len(stats_by_model[model]["trajectories_2d"]), i))
        
        # PCA projection for visualization
        if all_embeddings:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                projected = pca.fit_transform(np.array(all_embeddings))
                
                # Reconstruct trajectories in 2D
                traj_2d_by_model = defaultdict(list)
                current_traj = []
                current_model = None
                current_sample = None
                
                for i, (model, sample_idx, clause_idx) in enumerate(embedding_info):
                    if current_model != model or current_sample != sample_idx:
                        if current_traj:
                            traj_2d_by_model[current_model].append(current_traj)
                        current_traj = []
                        current_model = model
                        current_sample = sample_idx
                    
                    current_traj.append(projected[i].tolist())
                
                if current_traj:
                    traj_2d_by_model[current_model].append(current_traj)
                
                for model, trajs in traj_2d_by_model.items():
                    stats_by_model[model]["trajectories_2d"] = trajs
                    
            except ImportError:
                pass
        
        return dict(stats_by_model)


def generate_visualization(stats: dict, output_path: str) -> str:
    """Generate HTML visualization."""
    
    models = list(stats.keys())
    
    # Compute summary stats for each model
    model_summaries = {}
    for model, data in stats.items():
        step_lengths = data.get("step_lengths", [])
        turning_angles = data.get("turning_angles", [])
        tortuosities = data.get("tortuosities", [])
        
        model_summaries[model] = {
            "color": get_color(model),
            "step_mean": np.mean(step_lengths) if step_lengths else 0,
            "step_std": np.std(step_lengths) if len(step_lengths) > 1 else 0,
            "angle_mean": np.mean(turning_angles) if turning_angles else 0,
            "angle_std": np.std(turning_angles) if len(turning_angles) > 1 else 0,
            "tortuosity_mean": np.mean(tortuosities) if tortuosities else 1,
            "n_trajectories": len(tortuosities),
        }
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cognitive Gait Analysis - Trajectory Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: system-ui; }}
        .glass {{ background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); }}
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold text-cyan-400 mb-2">🧠 Cognitive Gait Analysis</h1>
        <p class="text-slate-400 mb-8">How do different models navigate semantic space?</p>
        
        <!-- Model Signatures -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
"""
    
    for model, summary in model_summaries.items():
        html += f"""
            <div class="glass rounded-lg p-4">
                <div class="flex items-center gap-2 mb-2">
                    <div class="w-3 h-3 rounded-full" style="background: {summary['color']}"></div>
                    <span class="font-semibold">{model}</span>
                </div>
                <div class="text-sm text-slate-400">
                    <div>Step size: {summary['step_mean']:.3f} ± {summary['step_std']:.3f}</div>
                    <div>Turn angle: {np.degrees(summary['angle_mean']):.1f}° ± {np.degrees(summary['angle_std']):.1f}°</div>
                    <div>Tortuosity: {summary['tortuosity_mean']:.2f}</div>
                    <div class="text-xs text-slate-500">{summary['n_trajectories']} trajectories</div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <!-- Trajectory Plot -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Semantic Trajectories (PCA Projection)</h2>
            <p class="text-slate-400 text-sm mb-4">Each line is one text's path through meaning space. Different "gaits" = different models.</p>
            <div id="trajectory-plot" style="height: 500px;"></div>
        </div>
        
        <!-- Step Size Distribution -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Step Size Distribution</h2>
            <p class="text-slate-400 text-sm mb-4">How far does each model "jump" between concepts?</p>
            <div id="step-plot" style="height: 400px;"></div>
        </div>
        
        <!-- Turning Angle Distribution -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Turning Angle Distribution</h2>
            <p class="text-slate-400 text-sm mb-4">Sharp turns (high angles) vs smooth flow (low angles)</p>
            <div id="angle-plot" style="height: 400px;"></div>
        </div>
        
        <!-- Interpretation -->
        <div class="glass rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-4">Interpretation Guide</h2>
            <ul class="text-slate-300 space-y-2">
                <li><strong>Step Size:</strong> Large = confident conceptual leaps. Small = careful, incremental moves.</li>
                <li><strong>Turning Angle:</strong> High variance = erratic. Low variance = predictable flow.</li>
                <li><strong>Tortuosity:</strong> High = wandering path. Low = direct, efficient argument.</li>
                <li><strong>The Fingerprint:</strong> Each model has a distinctive combination of these properties - its "cognitive gait."</li>
            </ul>
        </div>
    </div>
    
    <script>
        const stats = STATS_DATA;
        const summaries = SUMMARIES_DATA;
        
        // Trajectory plot
        const trajTraces = [];
        Object.entries(stats).forEach(([model, data]) => {{
            if (data.trajectories_2d) {{
                data.trajectories_2d.slice(0, 5).forEach((traj, i) => {{  // Limit to 5 per model
                    if (traj.length > 1) {{
                        trajTraces.push({{
                            x: traj.map(p => p[0]),
                            y: traj.map(p => p[1]),
                            mode: 'lines+markers',
                            name: i === 0 ? model : '',
                            showlegend: i === 0,
                            line: {{ color: summaries[model]?.color || '#999', width: 2 }},
                            marker: {{ size: 6 }},
                            opacity: 0.7,
                        }});
                    }}
                }});
            }}
        }});
        
        Plotly.newPlot('trajectory-plot', trajTraces, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ title: 'PC1', gridcolor: '#334155', zeroline: false }},
            yaxis: {{ title: 'PC2', gridcolor: '#334155', zeroline: false }},
            legend: {{ orientation: 'h', y: -0.1 }},
        }});
        
        // Step size box plot
        const stepTraces = Object.entries(stats).map(([model, data]) => ({{
            y: data.step_lengths || [],
            name: model,
            type: 'box',
            marker: {{ color: summaries[model]?.color || '#999' }},
        }}));
        
        Plotly.newPlot('step-plot', stepTraces, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            yaxis: {{ title: 'Step Size (embedding distance)', gridcolor: '#334155' }},
        }});
        
        // Turning angle histogram
        const angleTraces = Object.entries(stats).map(([model, data]) => ({{
            x: (data.turning_angles || []).map(a => a * 180 / Math.PI),  // Convert to degrees
            name: model,
            type: 'histogram',
            opacity: 0.6,
            marker: {{ color: summaries[model]?.color || '#999' }},
            nbinsx: 30,
        }}));
        
        Plotly.newPlot('angle-plot', angleTraces, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ title: 'Turning Angle (degrees)', gridcolor: '#334155' }},
            yaxis: {{ title: 'Count', gridcolor: '#334155' }},
            barmode: 'overlay',
        }});
    </script>
</body>
</html>
"""
    
    # Insert data
    html = html.replace("STATS_DATA", json.dumps({
        k: {
            "step_lengths": v.get("step_lengths", [])[:1000],  # Limit size
            "turning_angles": v.get("turning_angles", [])[:1000],
            "tortuosities": v.get("tortuosities", []),
            "trajectories_2d": v.get("trajectories_2d", [])[:5],  # Limit trajectories
        }
        for k, v in stats.items()
    }))
    html = html.replace("SUMMARIES_DATA", json.dumps(model_summaries))
    
    Path(output_path).write_text(html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize semantic trajectories")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus")
    parser.add_argument("--output", type=str, default="experiments/trajectory_viz.html",
                       help="Output HTML path")
    
    args = parser.parse_args()
    
    print("Loading corpus...")
    with open(args.corpus) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} samples")
    
    print("\nComputing trajectory statistics...")
    viz = TrajectoryVisualizer()
    stats = viz.process_corpus(corpus)
    
    print("\nGenerating visualization...")
    path = generate_visualization(stats, args.output)
    
    print(f"\nVisualization saved to {path}")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL GAIT SIGNATURES")
    print("="*50)
    for model, data in stats.items():
        step_lengths = data.get("step_lengths", [])
        turning_angles = data.get("turning_angles", [])
        tortuosities = data.get("tortuosities", [])
        
        if step_lengths:
            print(f"\n{model}:")
            print(f"  Step size:   {np.mean(step_lengths):.3f} ± {np.std(step_lengths):.3f}")
            print(f"  Turn angle:  {np.degrees(np.mean(turning_angles)):.1f}° ± {np.degrees(np.std(turning_angles)):.1f}°")
            print(f"  Tortuosity:  {np.mean(tortuosities):.2f}")


if __name__ == "__main__":
    main()
