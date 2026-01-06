#!/usr/bin/env python3
"""
Corpus Merger Tool

Combines corpora from multiple sources (Gemini, local, manual, Arena)
into a unified dataset with consistent schema.

Usage:
    python merge_corpus.py --sources data/corpus/gemini data/corpus/local data/corpus/manual
    python merge_corpus.py --auto  # Auto-discover all corpus directories
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional


UNIFIED_SCHEMA = {
    "sample_id": str,
    "seed_id": str,
    "domain": str,
    "model_id": str,
    "model_family": str,
    "model_name": str,
    "prompt_style": str,
    "temperature": float,
    "text": str,
    "word_count": int,
    "char_count": int,
    "sentence_count": int,
    "source_file": str,
    "timestamp": str,
}


def normalize_model_name(model_id: str) -> tuple[str, str]:
    """Normalize model ID to (family, name)."""
    model_id = model_id.lower()
    
    mappings = {
        # Anthropic
        "claude": ("anthropic", "claude"),
        "claude-sonnet": ("anthropic", "claude-sonnet"),
        "claude-opus": ("anthropic", "claude-opus"),
        "claude-haiku": ("anthropic", "claude-haiku"),
        
        # OpenAI
        "gpt-4o": ("openai", "gpt-4o"),
        "gpt-4": ("openai", "gpt-4"),
        "gpt-3.5": ("openai", "gpt-3.5"),
        "chatgpt": ("openai", "gpt-4o"),
        
        # Google
        "gemini": ("google", "gemini"),
        "gemini-2.0": ("google", "gemini-2.0"),
        "gemini-1.5": ("google", "gemini-1.5"),
        "gemini-pro": ("google", "gemini-pro"),
        "gemini-flash": ("google", "gemini-flash"),
        
        # xAI
        "grok": ("xai", "grok"),
        
        # Meta
        "llama": ("meta", "llama"),
        "llama3": ("meta", "llama3"),
        "llama2": ("meta", "llama2"),
        
        # Mistral
        "mistral": ("mistral", "mistral"),
        "mixtral": ("mistral", "mixtral"),
        
        # Google (open)
        "gemma": ("google-open", "gemma"),
        
        # Microsoft
        "phi": ("microsoft", "phi"),
        
        # Alibaba
        "qwen": ("alibaba", "qwen"),
        
        # Human
        "human": ("human", "human"),
    }
    
    for key, (family, name) in mappings.items():
        if key in model_id:
            # Try to get more specific name
            if "sonnet" in model_id:
                return family, "claude-sonnet"
            if "opus" in model_id:
                return family, "claude-opus"
            if "2.0" in model_id:
                return family, f"{name}-2.0"
            if "1.5" in model_id:
                return family, f"{name}-1.5"
            if "8b" in model_id:
                return family, f"{name}-8b"
            if "70b" in model_id:
                return family, f"{name}-70b"
            return family, name
    
    return ("unknown", model_id)


def count_sentences(text: str) -> int:
    """Rough sentence count."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def normalize_sample(sample: dict, source_file: str) -> dict:
    """Normalize a sample to unified schema."""
    
    # Get model info
    model_id = sample.get("model_id") or sample.get("model") or "unknown"
    family, name = normalize_model_name(model_id)
    
    text = sample.get("text", "")
    
    return {
        "sample_id": sample.get("sample_id", f"sample_{hash(text) % 100000}"),
        "seed_id": sample.get("seed_id", "unknown"),
        "domain": sample.get("domain", "unknown"),
        "model_id": model_id,
        "model_family": sample.get("model_family") or family,
        "model_name": sample.get("model_name") or name,
        "prompt_style": sample.get("prompt_style", "neutral"),
        "temperature": float(sample.get("temperature", 0.7)),
        "text": text,
        "word_count": sample.get("word_count") or len(text.split()),
        "char_count": len(text),
        "sentence_count": count_sentences(text),
        "source_file": source_file,
        "timestamp": sample.get("timestamp", datetime.now().isoformat()),
    }


def load_corpus_file(path: Path) -> list[dict]:
    """Load samples from a corpus file."""
    samples = []
    
    try:
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and "samples" in data:
                    samples = data["samples"]
    except Exception as e:
        print(f"  ⚠ Error loading {path}: {e}")
    
    return samples


def discover_corpus_dirs(base_path: str = "data/corpus") -> list[Path]:
    """Auto-discover corpus directories."""
    base = Path(base_path)
    if not base.exists():
        return []
    
    dirs = []
    for item in base.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            dirs.append(item)
    
    return dirs


def merge_corpora(
    sources: list[str],
    output_path: str = "data/corpus/merged",
    deduplicate: bool = True,
) -> dict:
    """Merge multiple corpora into unified dataset."""
    
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    source_stats = defaultdict(int)
    
    print(f"\nMerging corpora from {len(sources)} sources...")
    
    for source in sources:
        source_path = Path(source)
        
        if source_path.is_file():
            files = [source_path]
        elif source_path.is_dir():
            files = list(source_path.glob("*.json")) + list(source_path.glob("*.jsonl"))
        else:
            print(f"  ⚠ Source not found: {source}")
            continue
        
        for file in files:
            print(f"  Loading {file}...", end=" ")
            samples = load_corpus_file(file)
            
            normalized = []
            for s in samples:
                normalized.append(normalize_sample(s, str(file)))
            
            all_samples.extend(normalized)
            source_stats[str(source_path)] += len(normalized)
            print(f"✓ ({len(normalized)} samples)")
    
    # Deduplicate by text hash
    if deduplicate:
        seen = set()
        unique_samples = []
        duplicates = 0
        
        for sample in all_samples:
            text_hash = hash(sample["text"])
            if text_hash not in seen:
                seen.add(text_hash)
                unique_samples.append(sample)
            else:
                duplicates += 1
        
        print(f"\nRemoved {duplicates} duplicates")
        all_samples = unique_samples
    
    # Sort by model, then seed
    all_samples.sort(key=lambda x: (x["model_name"], x["seed_id"], x["sample_id"]))
    
    # Save merged corpus
    corpus_file = output / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    jsonl_file = output / "corpus.jsonl"
    with open(jsonl_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Generate statistics
    stats = {
        "total_samples": len(all_samples),
        "sources": dict(source_stats),
        "by_model": dict(Counter(s["model_name"] for s in all_samples)),
        "by_family": dict(Counter(s["model_family"] for s in all_samples)),
        "by_domain": dict(Counter(s["domain"] for s in all_samples)),
        "by_style": dict(Counter(s["prompt_style"] for s in all_samples)),
        "timestamp": datetime.now().isoformat(),
    }
    
    stats_file = output / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal samples: {len(all_samples)}")
    
    print("\nBy model:")
    for model, count in sorted(stats["by_model"].items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")
    
    print("\nBy family:")
    for family, count in sorted(stats["by_family"].items(), key=lambda x: -x[1]):
        print(f"  {family}: {count}")
    
    print(f"\nOutput: {corpus_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge corpora from multiple sources")
    parser.add_argument(
        "--sources", nargs="+",
        help="Paths to corpus files or directories"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-discover corpus directories in data/corpus/"
    )
    parser.add_argument(
        "--output", default="data/corpus/merged",
        help="Output directory"
    )
    parser.add_argument(
        "--no-dedupe", action="store_true",
        help="Don't remove duplicate samples"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        sources = [str(d) for d in discover_corpus_dirs()]
        if not sources:
            print("No corpus directories found in data/corpus/")
            return
        print(f"Auto-discovered: {sources}")
    elif args.sources:
        sources = args.sources
    else:
        print("Specify --sources or use --auto")
        return
    
    merge_corpora(
        sources=sources,
        output_path=args.output,
        deduplicate=not args.no_dedupe,
    )


if __name__ == "__main__":
    main()
