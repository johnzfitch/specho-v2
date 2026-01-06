"""
PACT Corpus Collection Pipeline

Orchestrates collection across all registered collectors,
handles deduplication, validation, and storage.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime
from dataclasses import asdict
import hashlib

# For parquet output
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from .schema import CorpusSample, Domain, Subdomain, DOMAIN_CONFIGS
from .registry import DomainRegistry, BaseCollector

logger = logging.getLogger(__name__)


class CollectionPipeline:
    """
    Main pipeline for collecting corpus data across all domains.
    
    Usage:
        pipeline = CollectionPipeline(output_dir="./corpus")
        
        # Collect everything
        pipeline.collect_all()
        
        # Or collect specific domains
        pipeline.collect_domain(Domain.ACADEMIC)
        
        # Export to different formats
        pipeline.export_parquet()
        pipeline.export_jsonl()
    """
    
    def __init__(
        self,
        output_dir: Path | str,
        cache_dir: Optional[Path | str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".pact_corpus" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track global deduplication
        self._global_hashes: set[str] = set()
        
        # Stats
        self.stats = {
            'total_collected': 0,
            'by_domain': {},
            'by_source': {},
            'duplicates_skipped': 0,
            'validation_failed': 0,
        }
    
    def collect_all(
        self,
        domains: Optional[list[Domain]] = None,
        limit_per_domain: Optional[int] = None,
    ) -> None:
        """
        Collect samples from all (or specified) domains.
        
        Args:
            domains: List of domains to collect (None = all)
            limit_per_domain: Override target samples per domain
        """
        domains = domains or list(Domain)
        
        for domain in domains:
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting domain: {domain.value}")
            logger.info(f"{'='*60}")
            
            try:
                self.collect_domain(domain, limit=limit_per_domain)
            except Exception as e:
                logger.error(f"Error collecting {domain.value}: {e}")
                continue
        
        # Save stats
        self._save_stats()
        
        logger.info(f"\n{'='*60}")
        logger.info("Collection complete!")
        logger.info(f"Total samples: {self.stats['total_collected']}")
        logger.info(f"{'='*60}")
    
    def collect_domain(
        self,
        domain: Domain,
        limit: Optional[int] = None,
    ) -> None:
        """Collect samples from a single domain using all registered collectors."""
        
        config = DOMAIN_CONFIGS.get(domain)
        if not config:
            logger.warning(f"No config for domain: {domain}")
            return
        
        target = limit or config.target_samples
        collected = 0
        
        # Get all collectors for this domain
        collectors = DomainRegistry.get_all_for_domain(domain)
        
        if not collectors:
            logger.warning(f"No collectors registered for domain: {domain}")
            return
        
        # Distribute target across collectors
        per_collector = target // len(collectors)
        
        # Output file for this domain
        domain_file = self.output_dir / f"{domain.value}.jsonl"
        
        with open(domain_file, 'a', encoding='utf-8') as f:
            for collector_class in collectors:
                if collected >= target:
                    break
                
                try:
                    collector = collector_class(
                        config=config,
                        cache_dir=self.cache_dir / collector_class.SOURCE_NAME,
                    )
                    
                    collector_limit = min(per_collector, target - collected)
                    
                    logger.info(f"Running collector: {collector_class.SOURCE_NAME} (target: {collector_limit})")
                    
                    for sample in collector.collect(limit=collector_limit):
                        # Global deduplication
                        if sample.content_hash in self._global_hashes:
                            self.stats['duplicates_skipped'] += 1
                            continue
                        
                        self._global_hashes.add(sample.content_hash)
                        
                        # Write sample
                        f.write(sample.to_json() + '\n')
                        
                        # Update stats
                        collected += 1
                        self.stats['total_collected'] += 1
                        self.stats['by_domain'][domain.value] = self.stats['by_domain'].get(domain.value, 0) + 1
                        self.stats['by_source'][collector_class.SOURCE_NAME] = (
                            self.stats['by_source'].get(collector_class.SOURCE_NAME, 0) + 1
                        )
                        
                except Exception as e:
                    logger.error(f"Error in collector {collector_class.SOURCE_NAME}: {e}")
                    continue
        
        logger.info(f"Domain {domain.value} complete: {collected} samples")
    
    def export_parquet(self, output_path: Optional[Path] = None) -> None:
        """Export all collected data to a single Parquet file."""
        
        if not HAS_PYARROW:
            logger.error("pyarrow not installed. Run: pip install pyarrow")
            return
        
        output_path = output_path or self.output_dir / "corpus.parquet"
        
        logger.info(f"Exporting to Parquet: {output_path}")
        
        # Collect all samples from domain files
        samples = []
        
        for jsonl_file in self.output_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        samples.append(self._flatten_sample(data))
                    except:
                        continue
        
        if not samples:
            logger.warning("No samples to export")
            return
        
        # Convert to PyArrow Table
        table = pa.Table.from_pylist(samples)
        
        # Write Parquet
        pq.write_table(table, output_path, compression='snappy')
        
        logger.info(f"Exported {len(samples)} samples to {output_path}")
    
    def _flatten_sample(self, data: dict) -> dict:
        """Flatten nested sample dict for Parquet."""
        flat = {
            'sample_id': data.get('sample_id'),
            'domain': data.get('domain'),
            'subdomain': data.get('subdomain'),
            'text': data.get('text'),
            'char_count': data.get('char_count'),
            'word_count': data.get('word_count'),
            'content_hash': data.get('content_hash'),
            'is_original': data.get('is_original', True),
        }
        
        # Flatten source metadata
        source = data.get('source', {})
        flat.update({
            'source_corpus': source.get('corpus_name'),
            'source_id': source.get('original_id'),
            'source_url': source.get('url'),
            'content_date': source.get('content_date'),
            'verified_pre_llm': source.get('verified_pre_llm', False),
        })
        
        # Generation metadata (for AI variants)
        flat.update({
            'parent_id': data.get('parent_id'),
            'model': data.get('model'),
            'model_family': data.get('model_family'),
            'prompt_id': data.get('prompt_id'),
            'prompt_style': data.get('prompt_style'),
            'generation_temp': data.get('generation_temp'),
            'run_number': data.get('run_number'),
        })
        
        # SpecHO analysis
        flat.update({
            'specho_score': data.get('specho_score'),
            'specho_confidence': data.get('specho_confidence'),
            'echo_phonetic': data.get('echo_phonetic'),
            'echo_structural': data.get('echo_structural'),
            'echo_semantic': data.get('echo_semantic'),
            'clause_pair_count': data.get('clause_pair_count'),
        })
        
        return flat
    
    def _save_stats(self) -> None:
        """Save collection statistics."""
        stats_file = self.output_dir / "collection_stats.json"
        
        self.stats['collection_completed'] = datetime.utcnow().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Stats saved to {stats_file}")
    
    def load_samples(
        self,
        domain: Optional[Domain] = None,
    ) -> Iterator[CorpusSample]:
        """Load collected samples as CorpusSample objects."""
        
        if domain:
            files = [self.output_dir / f"{domain.value}.jsonl"]
        else:
            files = list(self.output_dir.glob("*.jsonl"))
        
        for jsonl_file in files:
            if not jsonl_file.exists():
                continue
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        yield CorpusSample.from_dict(data)
                    except Exception as e:
                        logger.debug(f"Error loading sample: {e}")
                        continue
    
    def get_sample_count(self, domain: Optional[Domain] = None) -> int:
        """Get count of collected samples."""
        count = 0
        
        if domain:
            files = [self.output_dir / f"{domain.value}.jsonl"]
        else:
            files = list(self.output_dir.glob("*.jsonl"))
        
        for jsonl_file in files:
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    count += sum(1 for _ in f)
        
        return count


class SplitBuilder:
    """Creates train/val/test splits from collected corpus."""
    
    def __init__(
        self,
        corpus_dir: Path | str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    def build_splits(self, stratify_by: str = "domain") -> dict[str, Path]:
        """
        Create stratified train/val/test splits.
        
        Args:
            stratify_by: Field to stratify on ("domain", "subdomain", "source")
        
        Returns:
            Dict mapping split name to file path
        """
        import random
        random.seed(self.seed)
        
        # Load all samples grouped by stratification key
        groups: dict[str, list[dict]] = {}
        
        pipeline = CollectionPipeline(self.corpus_dir)
        
        for sample in pipeline.load_samples():
            key = getattr(sample, stratify_by, "unknown")
            if isinstance(key, Domain) or isinstance(key, Subdomain):
                key = key.value
            
            if key not in groups:
                groups[key] = []
            groups[key].append(sample.to_dict())
        
        # Split each group
        train_samples = []
        val_samples = []
        test_samples = []
        
        for key, samples in groups.items():
            random.shuffle(samples)
            
            n = len(samples)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train_samples.extend(samples[:train_end])
            val_samples.extend(samples[train_end:val_end])
            test_samples.extend(samples[val_end:])
        
        # Shuffle each split
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        # Write splits
        splits_dir = self.corpus_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        output_paths = {}
        
        for name, samples in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]:
            path = splits_dir / f"{name}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + '\n')
            
            output_paths[name] = path
            logger.info(f"{name}: {len(samples)} samples -> {path}")
        
        return output_paths
