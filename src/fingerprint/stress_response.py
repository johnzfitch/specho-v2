#!/usr/bin/env python3
"""
Cognitive Stress Response Framework

Tests how fingerprints DEFORM under different cognitive loads.

The insight: The fingerprint signature isn't just a static vector.
It's how that vector CHANGES under stress that reveals model identity.

Like cardiac stress testing - you don't just measure the heart at rest,
you measure it on a treadmill.

Cognitive Modes (Operations):
1. Neutral    - Baseline rewrite
2. Critique   - Find flaws (forces contrastive, hedging)
3. Creative   - Reimagine (forces divergence, entropy)
4. Synthesize - Abstract (forces conceptual jumps)
5. Extract    - Compress (forces pruning)
6. Explain    - Teach (forces explicit bridging)

The DEFORMATION SIGNATURE is model-specific:
- Claude compresses under critique (more hedging)
- GPT expands under creative (more confidence)
- Gemini stays stable (trained for consistency)
- Grok destabilizes (personality amplifies)

Usage:
    from stress_response import StressResponseAnalyzer
    
    analyzer = StressResponseAnalyzer()
    response = analyzer.measure_response(model_generate_fn, seed_text)
    
    # Deformation signature is the fingerprint of how the model responds to stress
    deformation_signature = response.compute_deformation_signature()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cognitive import CognitiveExtractor, CognitiveFingerprint


# =============================================================================
# COGNITIVE MODE DEFINITIONS
# =============================================================================

COGNITIVE_MODES = {
    "neutral": {
        "name": "Neutral Rewrite",
        "prompt": "Rewrite the following text in your own words, preserving the meaning:\n\n{text}",
        "description": "Baseline - minimal cognitive load",
    },
    "critique": {
        "name": "Critical Analysis", 
        "prompt": "Critically analyze the following text. Identify flaws, weaknesses, limitations, and potential counterarguments:\n\n{text}",
        "description": "Forces contrastive transitions, hedging, epistemic caution",
    },
    "creative": {
        "name": "Creative Reimagining",
        "prompt": "Creatively reimagine the following text. Present the same ideas in a completely different, more engaging way:\n\n{text}",
        "description": "Forces divergent thinking, stylistic variation, semantic jumps",
    },
    "synthesize": {
        "name": "Abstract Synthesis",
        "prompt": "Synthesize the following text into broader implications. Connect it to larger themes and draw out the deeper significance:\n\n{text}",
        "description": "Forces conceptual abstraction, large semantic jumps",
    },
    "extract": {
        "name": "Essential Extraction",
        "prompt": "Extract only the essential claims from the following text. Remove all unnecessary detail and keep only what matters:\n\n{text}",
        "description": "Forces compression, pruning, topic coherence",
    },
    "explain": {
        "name": "Clear Explanation",
        "prompt": "Explain the following text as if to a smart non-expert. Make every concept clear and explicit:\n\n{text}",
        "description": "Forces explicit bridging, additive transitions, clarity",
    },
}

# Envelope constraint for fair comparison
ENVELOPE_CONSTRAINT = """
Write exactly 6-8 sentences. No bullet points. No numbered lists. No headers.
Keep each sentence between 12 and 25 words.
"""


@dataclass
class ModeFingerprint:
    """Fingerprint extracted in a specific cognitive mode."""
    
    mode: str
    fingerprint: CognitiveFingerprint
    prompt_used: str
    response_text: str = ""


@dataclass
class StressResponse:
    """Complete stress response profile for a model."""
    
    # Baseline
    baseline: ModeFingerprint
    
    # Stressed fingerprints
    stressed: Dict[str, ModeFingerprint] = field(default_factory=dict)
    
    # Metadata
    model_id: str = ""
    seed_text: str = ""
    
    def compute_deformations(self) -> Dict[str, np.ndarray]:
        """
        Compute deformation vectors from baseline.
        
        Returns:
            Dict mapping mode -> deformation vector (stressed - baseline)
        """
        baseline_vec = self.baseline.fingerprint.to_vector_full()
        
        deformations = {}
        for mode, mode_fp in self.stressed.items():
            stressed_vec = mode_fp.fingerprint.to_vector_full()
            deformations[mode] = stressed_vec - baseline_vec
        
        return deformations
    
    def compute_deformation_magnitudes(self) -> Dict[str, float]:
        """Compute magnitude of deformation per mode."""
        deformations = self.compute_deformations()
        return {mode: float(np.linalg.norm(vec)) for mode, vec in deformations.items()}
    
    def compute_deformation_signature(self) -> np.ndarray:
        """
        Compute the DEFORMATION SIGNATURE.
        
        This is a meta-fingerprint that captures how the model responds to stress.
        It's the concatenation of all deformation vectors, normalized.
        """
        deformations = self.compute_deformations()
        
        # Fixed order
        modes = ["critique", "creative", "synthesize", "extract", "explain"]
        
        signature_parts = []
        for mode in modes:
            if mode in deformations:
                # Normalize deformation
                vec = deformations[mode]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                signature_parts.append(vec)
            else:
                # Pad with zeros if mode not present
                signature_parts.append(np.zeros(39))
        
        return np.concatenate(signature_parts)
    
    def compute_layer_deformations(self) -> Dict[str, Dict[str, float]]:
        """
        Compute deformations per trust layer.
        
        Returns:
            Dict mapping mode -> {layer_a, layer_b, layer_c, layer_d}
        """
        baseline_a = self.baseline.fingerprint.to_vector_layer_a()
        baseline_ab = self.baseline.fingerprint.to_vector_layer_ab()
        baseline_abc = self.baseline.fingerprint.to_vector_layer_abc()
        baseline_full = self.baseline.fingerprint.to_vector_full()
        
        layer_deformations = {}
        
        for mode, mode_fp in self.stressed.items():
            stressed_a = mode_fp.fingerprint.to_vector_layer_a()
            stressed_ab = mode_fp.fingerprint.to_vector_layer_ab()
            stressed_abc = mode_fp.fingerprint.to_vector_layer_abc()
            stressed_full = mode_fp.fingerprint.to_vector_full()
            
            layer_deformations[mode] = {
                "layer_a_trajectory": float(np.linalg.norm(stressed_a - baseline_a)),
                "layer_b_specHO": float(np.linalg.norm(stressed_ab[5:20] - baseline_ab[5:20])),
                "layer_c_epistemic": float(np.linalg.norm(stressed_abc[20:26] - baseline_abc[20:26])),
                "layer_c_transitions": float(np.linalg.norm(stressed_abc[26:32] - baseline_abc[26:32])),
                "layer_d_syntactic": float(np.linalg.norm(stressed_full[32:39] - baseline_full[32:39])),
            }
        
        return layer_deformations
    
    def get_most_sensitive_mode(self) -> str:
        """Return the mode that causes the most deformation."""
        magnitudes = self.compute_deformation_magnitudes()
        return max(magnitudes, key=magnitudes.get) if magnitudes else "unknown"
    
    def get_most_stable_mode(self) -> str:
        """Return the mode that causes the least deformation."""
        magnitudes = self.compute_deformation_magnitudes()
        return min(magnitudes, key=magnitudes.get) if magnitudes else "unknown"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "seed_text": self.seed_text[:200],
            "baseline": {
                "mode": self.baseline.mode,
                "fingerprint": self.baseline.fingerprint.to_vector_full().tolist(),
            },
            "stressed": {
                mode: {
                    "fingerprint": mfp.fingerprint.to_vector_full().tolist(),
                }
                for mode, mfp in self.stressed.items()
            },
            "deformation_magnitudes": self.compute_deformation_magnitudes(),
            "most_sensitive_mode": self.get_most_sensitive_mode(),
            "most_stable_mode": self.get_most_stable_mode(),
        }


class StressResponseAnalyzer:
    """
    Measures how model fingerprints deform under cognitive stress.
    
    The deformation signature is a meta-fingerprint that captures
    the model's characteristic response to different cognitive loads.
    """
    
    def __init__(self, specHO_path: str = None):
        """Initialize the analyzer."""
        self.extractor = CognitiveExtractor(specHO_path)
        self.modes = COGNITIVE_MODES
    
    def measure_response(
        self,
        generate_fn: Callable[[str], str],
        seed_text: str,
        model_id: str = "",
        modes: List[str] = None,
        use_envelope: bool = True,
    ) -> StressResponse:
        """
        Measure stress response for a model.
        
        Args:
            generate_fn: Function that takes prompt and returns generated text
            seed_text: Text to use as seed for all prompts
            model_id: Identifier for the model
            modes: List of modes to test (default: all)
            use_envelope: Whether to apply envelope constraints
            
        Returns:
            StressResponse with baseline and stressed fingerprints
        """
        modes = modes or ["neutral", "critique", "creative", "synthesize", "extract", "explain"]
        
        print(f"\nMeasuring stress response for {model_id or 'model'}...")
        
        # Extract baseline
        print(f"  [baseline] neutral...", end=" ", flush=True)
        baseline_prompt = self._build_prompt("neutral", seed_text, use_envelope)
        baseline_text = generate_fn(baseline_prompt)
        baseline_fp = self.extractor.extract(baseline_text, "baseline", model_id)
        baseline = ModeFingerprint(
            mode="neutral",
            fingerprint=baseline_fp,
            prompt_used=baseline_prompt,
            response_text=baseline_text,
        )
        print("✓")
        
        # Extract stressed fingerprints
        stressed = {}
        for mode in modes:
            if mode == "neutral":
                continue
            
            print(f"  [stressed] {mode}...", end=" ", flush=True)
            prompt = self._build_prompt(mode, seed_text, use_envelope)
            response = generate_fn(prompt)
            fp = self.extractor.extract(response, f"stressed_{mode}", model_id)
            stressed[mode] = ModeFingerprint(
                mode=mode,
                fingerprint=fp,
                prompt_used=prompt,
                response_text=response,
            )
            print("✓")
        
        return StressResponse(
            baseline=baseline,
            stressed=stressed,
            model_id=model_id,
            seed_text=seed_text,
        )
    
    def _build_prompt(self, mode: str, seed_text: str, use_envelope: bool) -> str:
        """Build prompt for a cognitive mode."""
        mode_config = self.modes.get(mode, self.modes["neutral"])
        prompt = mode_config["prompt"].format(text=seed_text)
        
        if use_envelope:
            prompt = ENVELOPE_CONSTRAINT + "\n\n" + prompt
        
        return prompt
    
    def compare_responses(
        self,
        responses: List[StressResponse],
    ) -> dict:
        """
        Compare stress responses across multiple models.
        
        Args:
            responses: List of StressResponse from different models
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "models": [r.model_id for r in responses],
            "deformation_profiles": {},
            "distinctive_modes": {},
        }
        
        # Collect deformation magnitudes
        for response in responses:
            magnitudes = response.compute_deformation_magnitudes()
            comparison["deformation_profiles"][response.model_id] = magnitudes
        
        # Find distinctive modes (where models differ most)
        modes = ["critique", "creative", "synthesize", "extract", "explain"]
        for mode in modes:
            values = [
                comparison["deformation_profiles"].get(r.model_id, {}).get(mode, 0)
                for r in responses
            ]
            if len(values) > 1:
                comparison["distinctive_modes"][mode] = {
                    "variance": float(np.var(values)),
                    "range": float(max(values) - min(values)),
                    "values": {r.model_id: v for r, v in zip(responses, values)},
                }
        
        # Rank modes by distinctiveness
        comparison["modes_by_distinctiveness"] = sorted(
            comparison["distinctive_modes"].keys(),
            key=lambda m: comparison["distinctive_modes"][m]["variance"],
            reverse=True
        )
        
        return comparison


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_stress_report(response: StressResponse, output_path: str) -> str:
    """Generate HTML stress response report."""
    
    deformations = response.compute_deformation_magnitudes()
    layer_deformations = response.compute_layer_deformations()
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stress Response Analysis - {response.model_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: system-ui; }}
        .glass {{ background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); }}
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-cyan-400 mb-2">⚡ Cognitive Stress Response</h1>
        <p class="text-slate-400 mb-2">Model: {response.model_id or "Unknown"}</p>
        <p class="text-slate-500 text-sm mb-8">How does the fingerprint deform under cognitive load?</p>
        
        <!-- Summary -->
        <div class="grid grid-cols-3 gap-4 mb-8">
            <div class="glass rounded-lg p-4">
                <div class="text-2xl font-bold text-green-400">{response.get_most_stable_mode()}</div>
                <div class="text-slate-400 text-sm">Most Stable Mode</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-2xl font-bold text-red-400">{response.get_most_sensitive_mode()}</div>
                <div class="text-slate-400 text-sm">Most Sensitive Mode</div>
            </div>
            <div class="glass rounded-lg p-4">
                <div class="text-2xl font-bold text-cyan-400">{len(deformations)}</div>
                <div class="text-slate-400 text-sm">Modes Tested</div>
            </div>
        </div>
        
        <!-- Deformation Magnitudes -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Deformation Magnitudes by Mode</h2>
            <div id="magnitude-chart" style="height: 300px;"></div>
        </div>
        
        <!-- Layer Analysis -->
        <div class="glass rounded-lg p-4 mb-8">
            <h2 class="text-xl font-semibold mb-4">Deformation by Trust Layer</h2>
            <div id="layer-chart" style="height: 400px;"></div>
        </div>
        
        <!-- Interpretation -->
        <div class="glass rounded-lg p-4">
            <h2 class="text-xl font-semibold mb-4">Interpretation</h2>
            <ul class="text-slate-300 space-y-2">
                <li><strong>High deformation in Critique:</strong> Model hedges and becomes cautious under pressure</li>
                <li><strong>High deformation in Creative:</strong> Model's style shifts significantly when unconstrained</li>
                <li><strong>Low deformation overall:</strong> Model has stable fingerprint (possibly well-regularized)</li>
                <li><strong>Layer A (Trajectory) sensitive:</strong> Semantic navigation changes - hard to fake</li>
                <li><strong>Layer D (Syntactic) sensitive:</strong> Surface style changes - easier to mask</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Magnitude chart
        const modes = {json.dumps(list(deformations.keys()))};
        const magnitudes = {json.dumps(list(deformations.values()))};
        
        Plotly.newPlot('magnitude-chart', [{{
            x: modes,
            y: magnitudes,
            type: 'bar',
            marker: {{
                color: magnitudes.map(m => m > 0.5 ? '#ef4444' : m > 0.3 ? '#f59e0b' : '#22c55e')
            }}
        }}], {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            yaxis: {{ title: 'Deformation Magnitude', gridcolor: '#334155' }},
            xaxis: {{ gridcolor: '#334155' }},
        }});
        
        // Layer chart
        const layerData = {json.dumps(layer_deformations)};
        const layers = ['layer_a_trajectory', 'layer_b_specHO', 'layer_c_epistemic', 'layer_c_transitions', 'layer_d_syntactic'];
        const layerNames = ['A: Trajectory', 'B: SpecHO', 'C: Epistemic', 'C: Transitions', 'D: Syntactic'];
        
        const layerTraces = Object.entries(layerData).map(([mode, values]) => ({{
            x: layerNames,
            y: layers.map(l => values[l]),
            name: mode,
            type: 'bar',
        }}));
        
        Plotly.newPlot('layer-chart', layerTraces, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            yaxis: {{ title: 'Deformation', gridcolor: '#334155' }},
            barmode: 'group',
        }});
    </script>
</body>
</html>
"""
    
    Path(output_path).write_text(html)
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo stress response analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognitive stress response analysis")
    parser.add_argument("--seed-text", type=str, help="Seed text for analysis")
    parser.add_argument("--seed-file", type=str, help="File containing seed text")
    parser.add_argument("--model", type=str, help="Model to test (ollama name)")
    parser.add_argument("--output", type=str, default="experiments/stress",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.seed_file:
        seed_text = open(args.seed_file).read()
    elif args.seed_text:
        seed_text = args.seed_text
    else:
        # Default seed
        seed_text = """
        The development of artificial intelligence has progressed rapidly over the 
        past decade. Machine learning models have achieved remarkable results in 
        natural language processing, computer vision, and game playing. However, 
        concerns remain about safety, alignment, and the potential for misuse. 
        Researchers are working to address these challenges while continuing to 
        push the boundaries of what AI systems can accomplish.
        """
    
    analyzer = StressResponseAnalyzer()
    
    if args.model:
        # Use specified model via ollama
        import subprocess
        
        def generate_fn(prompt: str) -> str:
            result = subprocess.run(
                ["ollama", "run", args.model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.stdout.strip()
        
        response = analyzer.measure_response(
            generate_fn,
            seed_text,
            model_id=args.model,
        )
    else:
        # Demo with mock generator
        print("\n[Demo mode - using mock generator]")
        
        def mock_generate(prompt: str) -> str:
            # Generate deterministic but varied responses
            import hashlib
            seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
            
            responses = [
                "The rapid advancement of AI technology presents both opportunities and challenges. Systems now demonstrate capabilities once thought impossible. However, significant risks remain unaddressed. The path forward requires careful consideration of safety measures. Researchers continue to develop new frameworks for responsible AI development. The future remains uncertain but full of potential.",
                "One must critically examine the claims made about AI progress. While achievements in narrow domains are real, generalizations often overstate capabilities. The hype cycle obscures genuine limitations and risks. Furthermore, alignment research remains underfunded and understudied. We should approach bold predictions with appropriate skepticism. The challenges ahead are more complex than many acknowledge.",
                "Imagine a world transformed by intelligent machines working alongside humans. The possibilities stretch beyond current imagination into realms of wonder. Perhaps creativity itself will be redefined through this partnership. New forms of art, science, and expression await discovery. The adventure of consciousness exploration has only begun. What dreams may come from silicon minds?",
            ]
            
            return responses[seed % len(responses)]
        
        response = analyzer.measure_response(
            mock_generate,
            seed_text,
            model_id="mock_model",
        )
    
    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_dir / "stress_response.json", "w") as f:
        json.dump(response.to_dict(), f, indent=2)
    
    # Generate report
    report_path = generate_stress_report(response, str(output_dir / "stress_report.html"))
    
    # Print summary
    print("\n" + "="*60)
    print("STRESS RESPONSE ANALYSIS")
    print("="*60)
    
    print(f"\nModel: {response.model_id}")
    print(f"Most stable mode: {response.get_most_stable_mode()}")
    print(f"Most sensitive mode: {response.get_most_sensitive_mode()}")
    
    print("\nDeformation magnitudes:")
    for mode, magnitude in response.compute_deformation_magnitudes().items():
        bar = "█" * int(magnitude * 20)
        print(f"  {mode:12s}: {magnitude:.3f} {bar}")
    
    print(f"\nDeformation signature shape: {response.compute_deformation_signature().shape}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
