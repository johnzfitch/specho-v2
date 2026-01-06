#!/usr/bin/env python3
"""
Gemini Corpus Generator via AI Studio (FREE)

This is your primary high-volume generation channel.
AI Studio provides free API access to Gemini models.

Setup:
    1. Get API key: https://aistudio.google.com/app/apikey
    2. Set environment variable: export GOOGLE_API_KEY="your-key"
    3. Run: python generate_gemini.py

Usage:
    python generate_gemini.py                      # Default: 10 samples/model
    python generate_gemini.py --samples 50         # 50 samples per model
    python generate_gemini.py --styles all         # All rewrite styles
    python generate_gemini.py --models gemini-2.0-flash-exp,gemini-1.5-pro
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not installed")
    print("Run: pip install google-generativeai")
    sys.exit(1)


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

MODELS = {
    "gemini-2.0-flash": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
}

PROMPT_STYLES = {
    "neutral": """Rewrite the following text in your own words while preserving the core meaning and information. Maintain a similar length and tone.

Text to rewrite:
{text}

Rewritten version:""",

    "minimal": """Rewrite the following text with minimal changes while preserving all meaning exactly.

Text to rewrite:
{text}

Rewritten version:""",

    "improve": """Rewrite the following text, improving its clarity and flow while preserving the core meaning.

Text to rewrite:
{text}

Improved version:""",

    "formal": """Rewrite the following text in formal academic style while preserving the meaning.

Text to rewrite:
{text}

Formal version:""",

    "casual": """Rewrite the following text in a casual, conversational tone while preserving the meaning.

Text to rewrite:
{text}

Casual version:""",
}


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

def generate_rewrite(
    model_id: str,
    text: str,
    style: str = "neutral",
    temperature: float = 0.7,
) -> tuple[str, float]:
    """Generate a rewrite using specified Gemini model."""
    
    model = genai.GenerativeModel(model_id)
    prompt = PROMPT_STYLES[style].format(text=text)
    
    start = time.time()
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )
    )
    elapsed = (time.time() - start) * 1000
    
    return response.text, elapsed


def generate_corpus(
    samples_per_model: int = 10,
    models: list[str] = None,
    styles: list[str] = None,
    temperatures: list[float] = None,
    output_dir: str = "data/corpus/gemini",
    rate_limit: float = 1.0,
) -> list[Sample]:
    """Generate corpus from Gemini models."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if models is None:
        models = list(MODELS.keys())
    if styles is None:
        styles = ["neutral"]
    if temperatures is None:
        temperatures = [0.7]
    
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
    
    # Calculate how many runs per seed
    runs_per_seed = max(1, samples_per_model // len(SEED_TEXTS))
    
    total_generations = len(models) * len(styles) * len(temperatures) * len(SEED_TEXTS) * runs_per_seed
    current = 0
    
    print(f"\n{'='*60}")
    print(f"Generating {total_generations} samples")
    print(f"Models: {models}")
    print(f"Styles: {styles}")
    print(f"Temperatures: {temperatures}")
    print(f"Runs per seed: {runs_per_seed}")
    print(f"{'='*60}\n")
    
    for model_name in models:
        model_id = MODELS[model_name]
        print(f"\n[{model_name}]")
        
        for style in styles:
            for temp in temperatures:
                for run in range(runs_per_seed):
                    for seed in SEED_TEXTS:
                        current += 1
                        sample_id = f"{seed['id']}_{model_name}_{style}_t{temp}_r{run}"
                        
                        print(f"  [{current}/{total_generations}] {sample_id}...", end=" ", flush=True)
                        
                        try:
                            rewrite, gen_time = generate_rewrite(
                                model_id=model_id,
                                text=seed["text"],
                                style=style,
                                temperature=temp,
                            )
                            
                            sample = Sample(
                                sample_id=sample_id,
                                seed_id=seed["id"],
                                domain=seed["domain"],
                                model_id=model_id,
                                model_name=model_name,
                                prompt_style=style,
                                temperature=temp,
                                text=rewrite,
                                word_count=len(rewrite.split()),
                                timestamp=datetime.now().isoformat(),
                                generation_time_ms=gen_time,
                            )
                            corpus.append(sample)
                            print(f"✓ ({gen_time:.0f}ms, {sample.word_count} words)")
                            
                        except Exception as e:
                            print(f"✗ ({e})")
                        
                        time.sleep(rate_limit)
    
    # Save corpus
    corpus_file = output_path / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump([asdict(s) for s in corpus], f, indent=2)
    
    # Save JSONL for streaming
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
    
    # Count by model
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
        description="Generate fingerprint corpus using Google AI Studio (FREE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation (10 samples per model, neutral style only)
    python generate_gemini.py --samples 10
    
    # Full generation (50 samples, all styles)
    python generate_gemini.py --samples 50 --styles all
    
    # Specific models only
    python generate_gemini.py --models gemini-2.0-flash,gemini-1.5-pro
    
    # Multiple temperatures
    python generate_gemini.py --temperatures 0.5,0.7,0.9
""")
    
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Target samples per model (default: 10)"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help=f"Comma-separated models (default: all). Options: {','.join(MODELS.keys())}"
    )
    parser.add_argument(
        "--styles", type=str, default="neutral",
        help=f"Comma-separated styles or 'all' (default: neutral). Options: {','.join(PROMPT_STYLES.keys())}"
    )
    parser.add_argument(
        "--temperatures", type=str, default="0.7",
        help="Comma-separated temperatures (default: 0.7)"
    )
    parser.add_argument(
        "--output", type=str, default="data/corpus/gemini",
        help="Output directory (default: data/corpus/gemini)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Seconds between requests (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    models = args.models.split(",") if args.models else None
    
    if args.styles == "all":
        styles = list(PROMPT_STYLES.keys())
    else:
        styles = args.styles.split(",")
    
    temperatures = [float(t) for t in args.temperatures.split(",")]
    
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Get your key at: https://aistudio.google.com/app/apikey")
        print("Then run: export GOOGLE_API_KEY='your-key'")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    # Test connection
    print("Testing API connection...", end=" ")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Say 'OK'")
        print(f"✓ ({response.text.strip()})")
    except Exception as e:
        print(f"✗ ({e})")
        sys.exit(1)
    
    # Generate
    corpus = generate_corpus(
        samples_per_model=args.samples,
        models=models,
        styles=styles,
        temperatures=temperatures,
        output_dir=args.output,
        rate_limit=args.rate_limit,
    )
    
    return corpus


if __name__ == "__main__":
    main()
