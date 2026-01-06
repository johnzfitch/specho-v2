#!/usr/bin/env python3
"""
Local Model Corpus Generator via Ollama (FREE, UNLIMITED)

Uses your RTX 4090 to generate samples from open-weight models.

Setup:
    1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    2. Pull models: ollama pull llama3.1:8b
    3. Run: python generate_local.py

Usage:
    python generate_local.py                    # Default models
    python generate_local.py --samples 50       # 50 samples per model
    python generate_local.py --models llama3.1:8b,mistral:7b
"""

import subprocess
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED_TEXTS = [
    {
        "id": "academic_01",
        "domain": "academic",
        "text": """The relationship between economic inequality and social mobility has been extensively studied in recent decades. Research consistently shows that countries with higher levels of income inequality tend to have lower rates of intergenerational mobility. This phenomenon, sometimes called the Great Gatsby Curve, suggests that the circumstances of one's birth have increasingly predictive power over lifetime outcomes. However, the causal mechanisms remain contested among scholars."""
    },
    {
        "id": "technical_01",
        "domain": "technical",
        "text": """The implementation of microservices architecture requires careful consideration of service boundaries and communication patterns. Unlike monolithic applications, microservices distribute functionality across independently deployable units. This approach offers benefits including improved scalability and technology flexibility, but introduces complexity in areas such as distributed transactions and service discovery. Teams must weigh these tradeoffs against their specific operational capabilities and business requirements."""
    },
    {
        "id": "creative_01",
        "domain": "creative",
        "text": """The old lighthouse keeper had seen many storms, but none quite like this one. The wind howled through the cracks in the stone walls, carrying with it the salt and fury of an angry sea. He climbed the spiral staircase one more time, his joints protesting with each step. At the top, the great lens turned steadily, sending its beam into the chaos. Somewhere out there, ships depended on that light. He would not let them down."""
    },
    {
        "id": "conversational_01",
        "domain": "conversational",
        "text": """So I finally tried that new coffee place downtown, and honestly? It was pretty good but not worth the hype. The latte art was impressive, I'll give them that. But the barista seemed kinda stressed and the music was way too loud for a Sunday morning. My friend disagrees completely though - she says it's the best coffee in the city. I guess I just don't get it."""
    },
    {
        "id": "journalistic_01",
        "domain": "journalistic",
        "text": """City officials announced Thursday that the proposed transit expansion project will move forward despite budget concerns raised by the oversight committee. The $2.3 billion initiative aims to extend light rail service to underserved neighborhoods in the eastern district. Critics argue the timeline is unrealistic, while supporters point to economic development opportunities. Construction is expected to begin in early 2027."""
    },
]

DEFAULT_MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "gemma2:9b",
    "phi3:medium",
]

AVAILABLE_MODELS = [
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2:3b",
    "mistral:7b",
    "mixtral:8x7b",
    "gemma2:9b",
    "gemma2:27b",
    "phi3:mini",
    "phi3:medium",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "codellama:7b",
    "deepseek-coder:6.7b",
]

PROMPT_TEMPLATE = """Rewrite the following text in your own words while preserving the core meaning and information. Maintain a similar length and tone.

Text to rewrite:
{text}

Rewritten version:"""


@dataclass
class Sample:
    sample_id: str
    seed_id: str
    domain: str
    model_id: str
    model_name: str
    prompt_style: str
    temperature: float
    text: str
    word_count: int
    timestamp: str
    generation_time_ms: float


# =============================================================================
# GENERATION
# =============================================================================

def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_available_models() -> list[str]:
    """List models available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        pass
    return []


def pull_model(model: str) -> bool:
    """Pull a model if not already available."""
    print(f"  Pulling {model}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for large models
        )
        if result.returncode == 0:
            print("✓")
            return True
        else:
            print(f"✗ ({result.stderr.strip()})")
            return False
    except subprocess.TimeoutExpired:
        print("✗ (timeout)")
        return False


def generate_rewrite(
    model: str,
    text: str,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """Generate a rewrite using Ollama."""
    
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    start = time.time()
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=120,  # 2 min timeout
    )
    elapsed = (time.time() - start) * 1000
    
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    
    return result.stdout.strip(), elapsed


def generate_corpus(
    models: list[str] = None,
    samples_per_model: int = 10,
    output_dir: str = "data/corpus/local",
    auto_pull: bool = True,
) -> list[Sample]:
    """Generate corpus from local models."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if models is None:
        models = DEFAULT_MODELS
    
    # Check which models are available
    available = list_available_models()
    print(f"\nAvailable models: {available}")
    
    models_to_use = []
    for model in models:
        if model in available:
            models_to_use.append(model)
        elif auto_pull:
            print(f"\nModel {model} not found, pulling...")
            if pull_model(model):
                models_to_use.append(model)
        else:
            print(f"  Skipping {model} (not available)")
    
    if not models_to_use:
        print("ERROR: No models available")
        return []
    
    corpus = []
    
    # Add human originals
    for seed in SEED_TEXTS:
        corpus.append(Sample(
            sample_id=f"{seed['id']}_human",
            seed_id=seed["id"],
            domain=seed["domain"],
            model_id="human",
            model_name="human",
            prompt_style="original",
            temperature=0.0,
            text=seed["text"],
            word_count=len(seed["text"].split()),
            timestamp=datetime.now().isoformat(),
            generation_time_ms=0.0,
        ))
    
    runs_per_seed = max(1, samples_per_model // len(SEED_TEXTS))
    total = len(models_to_use) * len(SEED_TEXTS) * runs_per_seed
    current = 0
    
    print(f"\n{'='*60}")
    print(f"Generating {total} samples from {len(models_to_use)} models")
    print(f"{'='*60}\n")
    
    for model in models_to_use:
        print(f"\n[{model}]")
        
        for run in range(runs_per_seed):
            for seed in SEED_TEXTS:
                current += 1
                sample_id = f"{seed['id']}_{model.replace(':', '-')}_r{run}"
                
                print(f"  [{current}/{total}] {sample_id}...", end=" ", flush=True)
                
                try:
                    rewrite, gen_time = generate_rewrite(model, seed["text"])
                    
                    sample = Sample(
                        sample_id=sample_id,
                        seed_id=seed["id"],
                        domain=seed["domain"],
                        model_id=model,
                        model_name=model.split(":")[0],
                        prompt_style="neutral",
                        temperature=0.7,
                        text=rewrite,
                        word_count=len(rewrite.split()),
                        timestamp=datetime.now().isoformat(),
                        generation_time_ms=gen_time,
                    )
                    corpus.append(sample)
                    print(f"✓ ({gen_time/1000:.1f}s, {sample.word_count} words)")
                    
                except Exception as e:
                    print(f"✗ ({e})")
    
    # Save corpus
    corpus_file = output_path / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump([asdict(s) for s in corpus], f, indent=2)
    
    jsonl_file = output_path / "corpus.jsonl"
    with open(jsonl_file, "w") as f:
        for sample in corpus:
            f.write(json.dumps(asdict(sample)) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(corpus)}")
    print(f"Output: {corpus_file}")
    
    from collections import Counter
    model_counts = Counter(s.model_name for s in corpus)
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
    
    return corpus


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate fingerprint corpus using local Ollama models (FREE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
    {', '.join(AVAILABLE_MODELS)}

Examples:
    # Default (4 models, 10 samples each)
    python generate_local.py
    
    # Specific models
    python generate_local.py --models llama3.1:8b,mistral:7b
    
    # More samples
    python generate_local.py --samples 50
""")
    
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated models (default: {','.join(DEFAULT_MODELS)})"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Target samples per model (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="data/corpus/local",
        help="Output directory (default: data/corpus/local)"
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Don't auto-pull missing models"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # Check Ollama
    if not check_ollama():
        print("ERROR: Ollama not installed or not running")
        print("Install: curl -fsSL https://ollama.com/install.sh | sh")
        return
    
    if args.list:
        print("Installed models:")
        for model in list_available_models():
            print(f"  {model}")
        print("\nAvailable to pull:")
        for model in AVAILABLE_MODELS:
            print(f"  {model}")
        return
    
    models = args.models.split(",") if args.models else None
    
    generate_corpus(
        models=models,
        samples_per_model=args.samples,
        output_dir=args.output,
        auto_pull=not args.no_pull,
    )


if __name__ == "__main__":
    main()
