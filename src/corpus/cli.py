#!/usr/bin/env python3
"""
PACT Corpus CLI

Command-line interface for collecting human text and generating AI rewrites.

Usage:
    # Collect all domains
    python -m corpus.cli collect --output ./data/corpus
    
    # Collect specific domain
    python -m corpus.cli collect --domain academic --output ./data/corpus
    
    # Generate AI rewrites
    python -m corpus.cli generate --corpus ./data/corpus/academic.jsonl --output ./data/rewrites
    
    # Build train/val/test splits
    python -m corpus.cli split --corpus ./data/corpus --output ./data/splits
    
    # Run SpecHO analysis on corpus
    python -m corpus.cli analyze --corpus ./data/corpus/academic.jsonl
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from corpus.schema import Domain, DOMAIN_CONFIGS
from corpus.pipeline import CollectionPipeline, SplitBuilder
from corpus.generator import RewriteGenerator
from corpus.registry import DomainRegistry

# Import collectors to register them
import corpus.collectors

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def cmd_collect(args):
    """Collect human-written text corpus."""
    
    pipeline = CollectionPipeline(
        output_dir=args.output,
        cache_dir=args.cache,
    )
    
    if args.domain:
        try:
            domain = Domain(args.domain)
            pipeline.collect_domain(domain, limit=args.limit)
        except ValueError:
            logger.error(f"Unknown domain: {args.domain}")
            logger.info(f"Available domains: {[d.value for d in Domain]}")
            return 1
    else:
        # Collect all
        domains = [Domain(d) for d in args.domains] if args.domains else None
        pipeline.collect_all(domains=domains, limit_per_domain=args.limit)
    
    # Export to parquet
    if args.parquet:
        pipeline.export_parquet()
    
    return 0


def cmd_generate(args):
    """Generate AI rewrites of corpus samples."""
    
    generator = RewriteGenerator(
        output_dir=args.output,
        rate_limit_delay=args.rate_limit,
    )
    
    asyncio.run(generator.process_corpus(
        corpus_path=args.corpus,
        runs_per_config=args.runs,
        limit=args.limit,
        resume_from=args.resume,
    ))
    
    return 0


def cmd_split(args):
    """Create train/val/test splits."""
    
    builder = SplitBuilder(
        corpus_dir=args.corpus,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    splits = builder.build_splits(stratify_by=args.stratify)
    
    for name, path in splits.items():
        logger.info(f"{name}: {path}")
    
    return 0


def cmd_list(args):
    """List available collectors and domains."""
    
    print("\n=== Available Domains ===")
    for domain in Domain:
        config = DOMAIN_CONFIGS.get(domain)
        if config:
            print(f"\n{domain.value}:")
            print(f"  Subdomains: {[s.value for s in config.subdomains]}")
            print(f"  Target samples: {config.target_samples}")
    
    print("\n=== Registered Collectors ===")
    for key in DomainRegistry.list_collectors():
        domain, subdomain, source = key
        print(f"  {domain.value}/{subdomain.value}: {source}")
    
    return 0


def cmd_stats(args):
    """Show corpus statistics."""
    
    pipeline = CollectionPipeline(output_dir=args.corpus)
    
    print("\n=== Corpus Statistics ===")
    
    total = 0
    for domain in Domain:
        count = pipeline.get_sample_count(domain)
        if count > 0:
            print(f"  {domain.value}: {count:,} samples")
            total += count
    
    print(f"\n  Total: {total:,} samples")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="PACT Corpus Collection and Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect human-written text")
    collect_parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    collect_parser.add_argument("-c", "--cache", type=Path, help="Cache directory")
    collect_parser.add_argument("-d", "--domain", type=str, help="Single domain to collect")
    collect_parser.add_argument("--domains", nargs="+", help="List of domains to collect")
    collect_parser.add_argument("-l", "--limit", type=int, help="Limit samples per domain")
    collect_parser.add_argument("--parquet", action="store_true", help="Export to parquet")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate AI rewrites")
    gen_parser.add_argument("corpus", type=Path, help="Corpus JSONL file")
    gen_parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    gen_parser.add_argument("-r", "--runs", type=int, default=3, help="Runs per config")
    gen_parser.add_argument("-l", "--limit", type=int, help="Max samples to process")
    gen_parser.add_argument("--resume", type=str, help="Resume from sample ID")
    gen_parser.add_argument("--rate-limit", type=float, default=0.5, help="Delay between API calls")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/val/test splits")
    split_parser.add_argument("corpus", type=Path, help="Corpus directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--stratify", choices=["domain", "subdomain", "source"], default="domain")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available collectors")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show corpus statistics")
    stats_parser.add_argument("corpus", type=Path, help="Corpus directory")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "collect": cmd_collect,
        "generate": cmd_generate,
        "split": cmd_split,
        "list": cmd_list,
        "stats": cmd_stats,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
