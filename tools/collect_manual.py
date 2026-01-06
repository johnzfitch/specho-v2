#!/usr/bin/env python3
"""
Manual Collection Helper for Web Interfaces

Guides you through collecting samples from:
- claude.ai (Claude)
- chat.openai.com (ChatGPT)
- x.com (Grok)

Usage:
    python collect_manual.py                    # Interactive guided mode
    python collect_manual.py --paste claude     # Paste samples for Claude
    python collect_manual.py --status           # Show collection progress
"""

import json
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

PROVIDERS = {
    "claude": {
        "name": "Claude",
        "url": "https://claude.ai",
        "model_id": "claude-sonnet-4",
        "model_name": "claude",
    },
    "chatgpt": {
        "name": "ChatGPT",
        "url": "https://chat.openai.com",
        "model_id": "gpt-4o",
        "model_name": "gpt-4o",
    },
    "grok": {
        "name": "Grok",
        "url": "https://x.com/i/grok",
        "model_id": "grok-2",
        "model_name": "grok",
    },
}

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
    source: str


# =============================================================================
# COLLECTION
# =============================================================================

def load_corpus(output_dir: str) -> list[Sample]:
    """Load existing corpus."""
    corpus_file = Path(output_dir) / "corpus.json"
    if corpus_file.exists():
        with open(corpus_file) as f:
            data = json.load(f)
            return [Sample(**s) for s in data]
    return []


def save_corpus(corpus: list[Sample], output_dir: str):
    """Save corpus to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpus_file = output_path / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump([asdict(s) for s in corpus], f, indent=2)
    
    jsonl_file = output_path / "corpus.jsonl"
    with open(jsonl_file, "w") as f:
        for sample in corpus:
            f.write(json.dumps(asdict(sample)) + "\n")


def show_status(output_dir: str):
    """Show collection progress."""
    corpus = load_corpus(output_dir)
    
    print("\n" + "="*60)
    print("COLLECTION STATUS")
    print("="*60)
    
    print(f"\nTotal samples: {len(corpus)}")
    
    # By provider
    from collections import defaultdict
    by_provider = defaultdict(list)
    for s in corpus:
        by_provider[s.model_name].append(s)
    
    for provider, samples in sorted(by_provider.items()):
        seeds_covered = set(s.seed_id for s in samples)
        print(f"\n{provider}:")
        print(f"  Samples: {len(samples)}")
        print(f"  Seeds covered: {len(seeds_covered)}/5")
        
        for seed in SEED_TEXTS:
            status = "✓" if seed["id"] in seeds_covered else "⬜"
            print(f"    {status} {seed['id']}")
    
    # What's missing
    print("\n" + "-"*60)
    print("MISSING:")
    for provider_id, provider in PROVIDERS.items():
        samples = by_provider.get(provider["model_name"], [])
        seeds_covered = set(s.seed_id for s in samples)
        missing = [s for s in SEED_TEXTS if s["id"] not in seeds_covered]
        
        if missing:
            print(f"\n{provider['name']} ({provider['url']}):")
            for seed in missing:
                print(f"  • {seed['id']}")


def interactive_guide(output_dir: str):
    """Interactive guided collection."""
    corpus = load_corpus(output_dir)
    
    print("\n" + "="*60)
    print("MANUAL COLLECTION GUIDE")
    print("="*60)
    print("\nI'll guide you through collecting samples from web interfaces.")
    print("For each seed text, I'll show you the prompt to paste.")
    print("Then paste the model's response back here.\n")
    
    # Find what's missing
    from collections import defaultdict
    by_provider = defaultdict(set)
    for s in corpus:
        by_provider[s.model_name].add(s.seed_id)
    
    for provider_id, provider in PROVIDERS.items():
        seeds_covered = by_provider.get(provider["model_name"], set())
        missing = [s for s in SEED_TEXTS if s["id"] not in seeds_covered]
        
        if not missing:
            print(f"✓ {provider['name']}: All seeds collected")
            continue
        
        print(f"\n{'='*60}")
        print(f"{provider['name']} - {len(missing)} seeds remaining")
        print(f"URL: {provider['url']}")
        print("="*60)
        
        input("\nPress Enter when ready to start...")
        
        for seed in missing:
            print(f"\n--- {seed['id']} ({seed['domain']}) ---")
            print("\n📋 COPY THIS PROMPT:")
            print("-"*40)
            prompt = PROMPT_TEMPLATE.format(text=seed["text"])
            print(prompt)
            print("-"*40)
            
            print(f"\n1. Go to {provider['url']}")
            print("2. Paste the prompt above")
            print("3. Copy the response")
            print("4. Paste it below (end with a blank line + 'DONE'):")
            print()
            
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == "DONE":
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            if not lines:
                print("⚠ Skipped")
                continue
            
            rewrite = "\n".join(lines)
            
            sample = Sample(
                sample_id=f"{seed['id']}_{provider['model_name']}",
                seed_id=seed["id"],
                domain=seed["domain"],
                model_id=provider["model_id"],
                model_name=provider["model_name"],
                prompt_style="neutral",
                temperature=0.7,
                text=rewrite,
                word_count=len(rewrite.split()),
                timestamp=datetime.now().isoformat(),
                source=provider["url"],
            )
            corpus.append(sample)
            save_corpus(corpus, output_dir)
            
            print(f"✓ Saved ({sample.word_count} words)")
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    show_status(output_dir)


def paste_mode(provider_id: str, output_dir: str):
    """Paste samples for a specific provider."""
    if provider_id not in PROVIDERS:
        print(f"Unknown provider: {provider_id}")
        print(f"Options: {', '.join(PROVIDERS.keys())}")
        return
    
    provider = PROVIDERS[provider_id]
    corpus = load_corpus(output_dir)
    
    # Find missing seeds
    existing = set(s.seed_id for s in corpus if s.model_name == provider["model_name"])
    missing = [s for s in SEED_TEXTS if s["id"] not in existing]
    
    if not missing:
        print(f"✓ All seeds already collected for {provider['name']}")
        return
    
    print(f"\nPaste mode for {provider['name']}")
    print(f"Missing: {len(missing)} seeds")
    print("\nFor each seed, paste the response then type 'DONE' on a new line.\n")
    
    for seed in missing:
        print(f"\n--- {seed['id']} ---")
        print(f"Prompt used:\n{PROMPT_TEMPLATE.format(text=seed['text'][:100]}...")
        print("\nPaste response (then 'DONE'):")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "DONE":
                    break
                lines.append(line)
            except EOFError:
                break
        
        if not lines:
            print("⚠ Skipped")
            continue
        
        rewrite = "\n".join(lines)
        
        sample = Sample(
            sample_id=f"{seed['id']}_{provider['model_name']}",
            seed_id=seed["id"],
            domain=seed["domain"],
            model_id=provider["model_id"],
            model_name=provider["model_name"],
            prompt_style="neutral",
            temperature=0.7,
            text=rewrite,
            word_count=len(rewrite.split()),
            timestamp=datetime.now().isoformat(),
            source=provider["url"],
        )
        corpus.append(sample)
        save_corpus(corpus, output_dir)
        
        print(f"✓ Saved ({sample.word_count} words)")
    
    show_status(output_dir)


def print_prompts():
    """Print all prompts for manual collection."""
    print("\n" + "="*60)
    print("PROMPTS FOR MANUAL COLLECTION")
    print("="*60)
    print("\nCopy each prompt, paste into the web interface, copy response.\n")
    
    for i, seed in enumerate(SEED_TEXTS, 1):
        print(f"\n{'='*60}")
        print(f"SEED {i}/5: {seed['id']} ({seed['domain']})")
        print("="*60)
        print("\n" + PROMPT_TEMPLATE.format(text=seed["text"]))
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manual collection helper for web interfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive guided collection
    python collect_manual.py
    
    # Check progress
    python collect_manual.py --status
    
    # Print all prompts
    python collect_manual.py --prompts
    
    # Paste samples for specific provider
    python collect_manual.py --paste claude
""")
    
    parser.add_argument(
        "--status", action="store_true",
        help="Show collection progress"
    )
    parser.add_argument(
        "--prompts", action="store_true",
        help="Print all prompts for manual collection"
    )
    parser.add_argument(
        "--paste", type=str, metavar="PROVIDER",
        help=f"Paste samples for provider ({', '.join(PROVIDERS.keys())})"
    )
    parser.add_argument(
        "--output", type=str, default="data/corpus/manual",
        help="Output directory (default: data/corpus/manual)"
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status(args.output)
    elif args.prompts:
        print_prompts()
    elif args.paste:
        paste_mode(args.paste, args.output)
    else:
        interactive_guide(args.output)


if __name__ == "__main__":
    main()
