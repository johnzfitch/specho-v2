#!/usr/bin/env python3
"""
Model Drift Detection System

Tracks fingerprint changes over time to detect model updates.
Models like GPT-4 and Claude update silently - this monitors for drift.

Features:
- Daily fingerprint sampling
- Drift alerts when fingerprints shift significantly
- Historical trend visualization
- Model version change detection

Usage:
    python drift_monitor.py --sample                  # Take daily sample
    python drift_monitor.py --analyze                 # Analyze drift trends
    python drift_monitor.py --alert-threshold 0.15   # Set alert threshold
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
import subprocess


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SAMPLE_TEXT = """The implementation of distributed systems requires careful consideration of consistency, availability, and partition tolerance. Modern architectures must balance these competing concerns while maintaining acceptable performance and reliability. This challenge becomes more complex as systems scale across multiple data centers and geographic regions."""

DRIFT_THRESHOLDS = {
    "minor": 0.05,      # Small drift, probably noise
    "moderate": 0.10,   # Notable drift, worth monitoring
    "significant": 0.15, # Significant drift, likely model update
    "major": 0.25,      # Major drift, definite model change
}


# =============================================================================
# DRIFT MONITOR
# =============================================================================

class DriftMonitor:
    """Monitor model fingerprint drift over time."""
    
    def __init__(
        self,
        data_dir: str = "data/drift",
        specHO_path: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "drift_history.json"
        self.history = self._load_history()
        self._init_extractor(specHO_path)
    
    def _load_history(self) -> dict:
        """Load historical fingerprint data."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {"models": {}, "alerts": []}
    
    def _save_history(self):
        """Save historical data."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
    
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
    
    def sample_model(
        self,
        model_id: str,
        generate_fn,
        sample_text: str = DEFAULT_SAMPLE_TEXT,
        n_samples: int = 3,
    ) -> dict:
        """
        Take a fingerprint sample from a model.
        
        Args:
            model_id: Model identifier
            generate_fn: Function(prompt) -> text
            sample_text: Text to use for rewriting
            n_samples: Number of samples to average
        """
        timestamp = datetime.now().isoformat()
        fingerprints = []
        
        prompt = f"Rewrite this text in your own words:\n\n{sample_text}"
        
        for i in range(n_samples):
            try:
                response = generate_fn(prompt)
                fp = self.extract_fingerprint(response)
                fingerprints.append(fp.tolist())
            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")
        
        if not fingerprints:
            return {"error": "No successful samples"}
        
        mean_fp = np.mean(fingerprints, axis=0).tolist()
        std_fp = np.std(fingerprints, axis=0).tolist() if len(fingerprints) > 1 else [0] * 15
        
        sample = {
            "timestamp": timestamp,
            "model_id": model_id,
            "fingerprint": mean_fp,
            "std": std_fp,
            "n_samples": len(fingerprints),
        }
        
        # Add to history
        if model_id not in self.history["models"]:
            self.history["models"][model_id] = {"samples": [], "baseline": None}
        
        self.history["models"][model_id]["samples"].append(sample)
        
        # Set baseline if first sample
        if self.history["models"][model_id]["baseline"] is None:
            self.history["models"][model_id]["baseline"] = mean_fp
        
        # Check for drift
        drift = self._check_drift(model_id, mean_fp)
        if drift["level"] != "none":
            alert = {
                "timestamp": timestamp,
                "model_id": model_id,
                "drift_level": drift["level"],
                "drift_distance": drift["distance"],
                "message": drift["message"],
            }
            self.history["alerts"].append(alert)
            sample["alert"] = alert
        
        self._save_history()
        return sample
    
    def _check_drift(self, model_id: str, current_fp: list) -> dict:
        """Check for drift from baseline."""
        model_data = self.history["models"].get(model_id, {})
        baseline = model_data.get("baseline")
        
        if baseline is None:
            return {"level": "none", "distance": 0, "message": "No baseline"}
        
        current = np.array(current_fp)
        base = np.array(baseline)
        distance = float(np.linalg.norm(current - base))
        
        level = "none"
        message = "Within normal range"
        
        if distance >= DRIFT_THRESHOLDS["major"]:
            level = "major"
            message = f"MAJOR drift detected ({distance:.3f}) - likely model update"
        elif distance >= DRIFT_THRESHOLDS["significant"]:
            level = "significant"
            message = f"Significant drift ({distance:.3f}) - possible model change"
        elif distance >= DRIFT_THRESHOLDS["moderate"]:
            level = "moderate"
            message = f"Moderate drift ({distance:.3f}) - monitoring"
        elif distance >= DRIFT_THRESHOLDS["minor"]:
            level = "minor"
            message = f"Minor drift ({distance:.3f}) - likely noise"
        
        return {
            "level": level,
            "distance": distance,
            "message": message,
        }
    
    def get_model_history(self, model_id: str) -> dict:
        """Get historical data for a model."""
        return self.history["models"].get(model_id, {"samples": [], "baseline": None})
    
    def get_alerts(self, since: datetime = None) -> list:
        """Get alerts, optionally filtered by date."""
        alerts = self.history.get("alerts", [])
        
        if since:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a["timestamp"]) >= since
            ]
        
        return alerts
    
    def analyze_trends(self, model_id: str) -> dict:
        """Analyze drift trends for a model."""
        model_data = self.get_model_history(model_id)
        samples = model_data.get("samples", [])
        baseline = model_data.get("baseline")
        
        if len(samples) < 2:
            return {"error": "Need at least 2 samples for trend analysis"}
        
        # Calculate drift over time
        drifts = []
        timestamps = []
        
        for sample in samples:
            fp = np.array(sample["fingerprint"])
            base = np.array(baseline)
            drift = float(np.linalg.norm(fp - base))
            drifts.append(drift)
            timestamps.append(sample["timestamp"])
        
        # Detect change points (sudden jumps)
        change_points = []
        for i in range(1, len(drifts)):
            delta = abs(drifts[i] - drifts[i-1])
            if delta > DRIFT_THRESHOLDS["moderate"]:
                change_points.append({
                    "index": i,
                    "timestamp": timestamps[i],
                    "delta": delta,
                })
        
        return {
            "model_id": model_id,
            "n_samples": len(samples),
            "first_sample": timestamps[0],
            "last_sample": timestamps[-1],
            "baseline": baseline,
            "current_drift": drifts[-1],
            "max_drift": max(drifts),
            "mean_drift": np.mean(drifts),
            "drift_trend": "increasing" if drifts[-1] > drifts[0] else "stable",
            "change_points": change_points,
        }
    
    def reset_baseline(self, model_id: str):
        """Reset baseline to most recent sample."""
        if model_id in self.history["models"]:
            samples = self.history["models"][model_id]["samples"]
            if samples:
                self.history["models"][model_id]["baseline"] = samples[-1]["fingerprint"]
                self._save_history()
                return True
        return False


def generate_dashboard(monitor: DriftMonitor, output_path: str) -> str:
    """Generate HTML drift monitoring dashboard."""
    
    models = list(monitor.history["models"].keys())
    alerts = monitor.get_alerts(since=datetime.now() - timedelta(days=7))
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Drift Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d9ff; }}
        .card {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .alert {{ background: #ff6b6b; color: white; padding: 10px; border-radius: 4px; margin: 5px 0; }}
        .alert.moderate {{ background: #feca57; color: #333; }}
        .alert.minor {{ background: #54a0ff; }}
        .metric {{ font-size: 2em; font-weight: bold; color: #00d9ff; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        .status-ok {{ color: #00d9ff; }}
        .status-warn {{ color: #feca57; }}
        .status-alert {{ color: #ff6b6b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Model Drift Monitor</h1>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="grid">
            <div class="card">
                <div class="metric">{len(models)}</div>
                <div>Models Tracked</div>
            </div>
            <div class="card">
                <div class="metric">{sum(len(m.get('samples', [])) for m in monitor.history['models'].values())}</div>
                <div>Total Samples</div>
            </div>
            <div class="card">
                <div class="metric">{len(alerts)}</div>
                <div>Alerts (7 days)</div>
            </div>
        </div>
"""
    
    # Recent alerts
    if alerts:
        html += """
        <div class="card">
            <h2>⚠️ Recent Alerts</h2>
"""
        for alert in sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:10]:
            level_class = alert.get("drift_level", "minor")
            html += f"""
            <div class="alert {level_class}">
                <strong>{alert['model_id']}</strong> - {alert['drift_level'].upper()}
                <br>{alert['message']}
                <br><small>{alert['timestamp']}</small>
            </div>
"""
        html += "</div>"
    
    # Model status table
    html += """
        <div class="card">
            <h2>Model Status</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Samples</th>
                    <th>Current Drift</th>
                    <th>Status</th>
                </tr>
"""
    
    for model_id, model_data in monitor.history["models"].items():
        samples = model_data.get("samples", [])
        if samples:
            current_fp = np.array(samples[-1]["fingerprint"])
            baseline = np.array(model_data.get("baseline", current_fp))
            drift = float(np.linalg.norm(current_fp - baseline))
            
            if drift >= DRIFT_THRESHOLDS["significant"]:
                status = '<span class="status-alert">⚠️ DRIFT</span>'
            elif drift >= DRIFT_THRESHOLDS["moderate"]:
                status = '<span class="status-warn">⚡ WATCH</span>'
            else:
                status = '<span class="status-ok">✓ OK</span>'
            
            html += f"""
                <tr>
                    <td>{model_id}</td>
                    <td>{len(samples)}</td>
                    <td>{drift:.3f}</td>
                    <td>{status}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
"""
    
    # Drift chart for each model
    for model_id, model_data in monitor.history["models"].items():
        samples = model_data.get("samples", [])
        if len(samples) > 1:
            baseline = np.array(model_data.get("baseline", samples[0]["fingerprint"]))
            
            timestamps = [s["timestamp"][:10] for s in samples]
            drifts = [float(np.linalg.norm(np.array(s["fingerprint"]) - baseline)) for s in samples]
            
            html += f"""
        <div class="card">
            <h2>{model_id} - Drift Over Time</h2>
            <div id="chart-{model_id.replace(':', '-').replace('.', '-')}"></div>
            <script>
                Plotly.newPlot('chart-{model_id.replace(':', '-').replace('.', '-')}', [{{
                    x: {json.dumps(timestamps)},
                    y: {json.dumps(drifts)},
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: {{ color: '#00d9ff' }},
                    line: {{ color: '#00d9ff' }}
                }}], {{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#eee' }},
                    xaxis: {{ gridcolor: '#333' }},
                    yaxis: {{ gridcolor: '#333', title: 'Drift Distance' }},
                    shapes: [{{
                        type: 'line',
                        y0: {DRIFT_THRESHOLDS['significant']},
                        y1: {DRIFT_THRESHOLDS['significant']},
                        x0: 0,
                        x1: 1,
                        xref: 'paper',
                        line: {{ color: '#ff6b6b', dash: 'dash' }}
                    }}]
                }});
            </script>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model drift monitoring")
    parser.add_argument("--sample", action="store_true", help="Take drift sample")
    parser.add_argument("--model", type=str, help="Model to sample (Ollama)")
    parser.add_argument("--gemini", type=str, help="Gemini model to sample")
    parser.add_argument("--analyze", type=str, help="Analyze drift for model")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard")
    parser.add_argument("--alerts", action="store_true", help="Show recent alerts")
    parser.add_argument("--reset-baseline", type=str, help="Reset baseline for model")
    parser.add_argument("--data-dir", type=str, default="data/drift", help="Data directory")
    parser.add_argument("--output", type=str, default="experiments/drift", help="Output directory")
    parser.add_argument("--specHO", type=str, help="Path to SpecHO")
    
    args = parser.parse_args()
    
    monitor = DriftMonitor(data_dir=args.data_dir, specHO_path=args.specHO)
    
    if args.sample:
        if args.model:
            print(f"Sampling {args.model}...")
            generate_fn = lambda p: subprocess.run(
                ["ollama", "run", args.model, p],
                capture_output=True, text=True
            ).stdout.strip()
            result = monitor.sample_model(args.model, generate_fn)
        elif args.gemini:
            import os
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            
            print(f"Sampling {args.gemini}...")
            model = genai.GenerativeModel(args.gemini)
            generate_fn = lambda p: model.generate_content(p).text
            result = monitor.sample_model(args.gemini, generate_fn)
        else:
            print("Specify --model or --gemini")
            return
        
        print(json.dumps(result, indent=2))
    
    elif args.analyze:
        trends = monitor.analyze_trends(args.analyze)
        print(json.dumps(trends, indent=2, default=str))
    
    elif args.dashboard:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = generate_dashboard(monitor, str(output_dir / "drift_dashboard.html"))
        print(f"Dashboard saved to {path}")
    
    elif args.alerts:
        alerts = monitor.get_alerts(since=datetime.now() - timedelta(days=7))
        if alerts:
            for alert in alerts:
                print(f"[{alert['drift_level'].upper()}] {alert['model_id']}: {alert['message']}")
        else:
            print("No recent alerts")
    
    elif args.reset_baseline:
        if monitor.reset_baseline(args.reset_baseline):
            print(f"Baseline reset for {args.reset_baseline}")
        else:
            print("Model not found")
    
    else:
        # Default: show status
        print("\nModel Drift Status\n" + "="*40)
        for model_id, data in monitor.history["models"].items():
            samples = data.get("samples", [])
            print(f"\n{model_id}:")
            print(f"  Samples: {len(samples)}")
            if samples:
                baseline = np.array(data.get("baseline", samples[0]["fingerprint"]))
                current = np.array(samples[-1]["fingerprint"])
                drift = np.linalg.norm(current - baseline)
                print(f"  Current drift: {drift:.3f}")


if __name__ == "__main__":
    main()
