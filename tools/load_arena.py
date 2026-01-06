#!/usr/bin/env python3
"""
LMSYS Arena Data Loader

Loads pre-labeled model outputs from the Chatbot Arena dataset.
This is FREE access to millions of samples with verified model labels.

Dataset: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations

Setup:
    pip install datasets

Usage:
    python load_arena.py                          # Load 100 samples per model
    python load_arena.py --samples 500            # More samples
    python load_arena.py --models gpt-4,claude    # Specific models only
    python load_arena.py --list-models            # List available models
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional


def load_arena_dataset(cache_dir: str = None):
    """Load the LMSYS Arena dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed")
        print("Run: pip install datasets")
        return None
    
    print("Loading LMSYS Chatbot Arena dataset...")
    print("(This may take a few minutes on first run)")
    
    try:
        ds = load_dataset(
            "lmsys/chatbot_arena_conversations",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative dataset...")
        
        try:
            # Try the older version
            ds = load_dataset(
                "lmsys/lmsys-chat-1m",
                cache_dir=cache_dir,
            )
            return ds
        except Exception as e2:
            print(f"Error loading alternative: {e2}")
            return None


def list_available_models(ds) -> dict[str, int]:
    """List all models in the dataset with counts."""
    model_counts = Counter()
    
    print("Scanning dataset for models...")
    
    # Sample first 10k to get model distribution quickly
    for i, example in enumerate(ds["train"]):
        if i >= 10000:
            break
        
        if "model_a" in example:
            model_counts[example["model_a"]] += 1
        if "model_b" in example:
            model_counts[example["model_b"]] += 1
        if "model" in example:
            model_counts[example["model"]] += 1
    
    return dict(model_counts.most_common())


def normalize_model_name(model: str) -> tuple[str, str, str]:
    """Normalize model name to (family, name, version)."""
    model = model.lower().strip()
    
    patterns = [
        # Claude
        (r"claude-3[.-]opus", "anthropic", "claude-opus", "3"),
        (r"claude-3[.-]sonnet", "anthropic", "claude-sonnet", "3"),
        (r"claude-3[.-]haiku", "anthropic", "claude-haiku", "3"),
        (r"claude-2", "anthropic", "claude", "2"),
        (r"claude-instant", "anthropic", "claude-instant", "1"),
        (r"claude", "anthropic", "claude", "unknown"),
        
        # GPT
        (r"gpt-4o", "openai", "gpt-4o", ""),
        (r"gpt-4-turbo", "openai", "gpt-4-turbo", ""),
        (r"gpt-4-\d+", "openai", "gpt-4", "dated"),
        (r"gpt-4", "openai", "gpt-4", ""),
        (r"gpt-3\.5-turbo", "openai", "gpt-3.5-turbo", ""),
        (r"chatgpt", "openai", "chatgpt", ""),
        
        # Gemini
        (r"gemini-pro", "google", "gemini-pro", ""),
        (r"gemini-1\.5", "google", "gemini-1.5", ""),
        (r"gemini", "google", "gemini", ""),
        (r"bard", "google", "bard", ""),
        
        # Llama
        (r"llama-3[.-]70b", "meta", "llama3-70b", ""),
        (r"llama-3[.-]8b", "meta", "llama3-8b", ""),
        (r"llama-3", "meta", "llama3", ""),
        (r"llama-2-70b", "meta", "llama2-70b", ""),
        (r"llama-2-13b", "meta", "llama2-13b", ""),
        (r"llama-2-7b", "meta", "llama2-7b", ""),
        (r"llama-2", "meta", "llama2", ""),
        (r"llama", "meta", "llama", ""),
        
        # Mistral
        (r"mixtral-8x7b", "mistral", "mixtral-8x7b", ""),
        (r"mixtral", "mistral", "mixtral", ""),
        (r"mistral-7b", "mistral", "mistral-7b", ""),
        (r"mistral-large", "mistral", "mistral-large", ""),
        (r"mistral-medium", "mistral", "mistral-medium", ""),
        (r"mistral", "mistral", "mistral", ""),
        
        # Others
        (r"vicuna", "lmsys", "vicuna", ""),
        (r"wizardlm", "microsoft", "wizardlm", ""),
        (r"zephyr", "huggingface", "zephyr", ""),
        (r"yi-", "01ai", "yi", ""),
        (r"qwen", "alibaba", "qwen", ""),
        (r"gemma", "google", "gemma", ""),
        (r"phi-", "microsoft", "phi", ""),
        (r"command", "cohere", "command", ""),
    ]
    
    for pattern, family, name, version in patterns:
        if re.search(pattern, model):
            return family, name, version
    
    return "unknown", model, ""


def extract_samples(
    ds,
    target_models: list[str] = None,
    samples_per_model: int = 100,
    min_response_length: int = 50,
    max_response_length: int = 2000,
) -> list[dict]:
    """Extract samples from the dataset."""
    
    samples_by_model = defaultdict(list)
    target_set = set(m.lower() for m in target_models) if target_models else None
    
    print(f"\nExtracting samples (target: {samples_per_model} per model)...")
    
    processed = 0
    for example in ds["train"]:
        processed += 1
        
        if processed % 10000 == 0:
            total_collected = sum(len(v) for v in samples_by_model.values())
            print(f"  Processed {processed:,} rows, collected {total_collected} samples...")
        
        # Extract model and response
        # Dataset format varies, try different fields
        model = None
        response = None
        prompt = None
        
        # Format 1: model_a/model_b with conversation
        if "model_a" in example and "conversation_a" in example:
            for m, conv in [("model_a", "conversation_a"), ("model_b", "conversation_b")]:
                model = example.get(m)
                if model and example.get(conv):
                    # Get assistant response from conversation
                    for turn in example[conv]:
                        if turn.get("role") == "assistant":
                            response = turn.get("content", "")
                            break
                        elif turn.get("role") == "user":
                            prompt = turn.get("content", "")
                    
                    if response:
                        break
        
        # Format 2: single model with conversation
        elif "model" in example and "conversation" in example:
            model = example.get("model")
            for turn in example.get("conversation", []):
                if turn.get("role") == "assistant":
                    response = turn.get("content", "")
                elif turn.get("role") == "user":
                    prompt = turn.get("content", "")
        
        if not model or not response:
            continue
        
        # Filter by length
        if len(response) < min_response_length or len(response) > max_response_length:
            continue
        
        # Normalize model name
        family, name, version = normalize_model_name(model)
        
        # Filter by target models
        if target_set:
            if not any(t in name.lower() or t in model.lower() for t in target_set):
                continue
        
        # Check if we have enough for this model
        if len(samples_by_model[name]) >= samples_per_model:
            continue
        
        # Create sample
        sample = {
            "sample_id": f"arena_{processed}_{name}",
            "seed_id": f"arena_{processed}",
            "domain": "arena",
            "model_id": model,
            "model_family": family,
            "model_name": name,
            "prompt_style": "arena",
            "temperature": 0.0,  # Unknown
            "text": response,
            "word_count": len(response.split()),
            "char_count": len(response),
            "prompt": prompt[:500] if prompt else "",
            "source": "lmsys/chatbot_arena_conversations",
            "timestamp": datetime.now().isoformat(),
        }
        
        samples_by_model[name].append(sample)
        
        # Check if we have enough total
        if target_set:
            have_all = all(
                len(samples_by_model.get(m, [])) >= samples_per_model 
                for m in target_set
            )
            if have_all:
                break
        elif len(samples_by_model) >= 10 and all(
            len(v) >= samples_per_model for v in samples_by_model.values()
        ):
            break
    
    # Flatten
    all_samples = []
    for samples in samples_by_model.values():
        all_samples.extend(samples)
    
    return all_samples, dict(samples_by_model)


def save_arena_corpus(
    samples: list[dict],
    output_dir: str = "data/corpus/arena",
):
    """Save extracted samples."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Save main corpus
    corpus_file = output / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    jsonl_file = output / "corpus.jsonl"
    with open(jsonl_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    # Save stats
    stats = {
        "total_samples": len(samples),
        "by_model": dict(Counter(s["model_name"] for s in samples)),
        "by_family": dict(Counter(s["model_family"] for s in samples)),
        "source": "lmsys/chatbot_arena_conversations",
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return corpus_file


def main():
    parser = argparse.ArgumentParser(
        description="Load samples from LMSYS Chatbot Arena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load 100 samples per model (default)
    python load_arena.py
    
    # Load more samples
    python load_arena.py --samples 500
    
    # Specific models only
    python load_arena.py --models gpt-4,claude-3,gemini,llama-3
    
    # List available models
    python load_arena.py --list-models
""")
    
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Samples per model (default: 100)"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of target models"
    )
    parser.add_argument(
        "--output", type=str, default="data/corpus/arena",
        help="Output directory"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Cache directory for dataset"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    ds = load_arena_dataset(args.cache_dir)
    if ds is None:
        return
    
    print(f"Dataset loaded: {len(ds['train'])} conversations")
    
    # List models
    if args.list_models:
        models = list_available_models(ds)
        print("\nAvailable models (top 30):")
        for model, count in list(models.items())[:30]:
            print(f"  {model}: {count}")
        return
    
    # Parse target models
    target_models = None
    if args.models:
        target_models = [m.strip() for m in args.models.split(",")]
        print(f"Target models: {target_models}")
    
    # Extract samples
    samples, by_model = extract_samples(
        ds,
        target_models=target_models,
        samples_per_model=args.samples,
    )
    
    if not samples:
        print("No samples extracted")
        return
    
    # Save
    corpus_file = save_arena_corpus(samples, args.output)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal samples: {len(samples)}")
    
    print("\nBy model:")
    for model, model_samples in sorted(by_model.items(), key=lambda x: -len(x[1])):
        print(f"  {model}: {len(model_samples)}")
    
    print(f"\nOutput: {corpus_file}")


if __name__ == "__main__":
    main()
