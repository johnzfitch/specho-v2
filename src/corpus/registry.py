"""
PACT Corpus Collector Registry

Provides a unified interface for all data collectors.
Each domain/source combination has a registered collector.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Type
from pathlib import Path
import logging
from datetime import date

from .schema import CorpusSample, Domain, Subdomain, SourceMetadata, DomainConfig

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Base class for all corpus collectors.
    
    Subclasses implement the actual data fetching logic.
    The framework handles validation, deduplication, and storage.
    """
    
    # Override in subclasses
    DOMAIN: Domain
    SUBDOMAIN: Subdomain
    SOURCE_NAME: str
    SOURCE_VERSION: str = "1.0"
    LICENSE: str = "unknown"
    
    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: float = 1.0,  # seconds between requests
    ):
        self.config = config or self._default_config()
        self.cache_dir = cache_dir or Path.home() / ".pact_corpus" / "cache" / self.SOURCE_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self._seen_hashes = set()
    
    def _default_config(self) -> DomainConfig:
        """Get default config for this collector's domain."""
        from .schema import DOMAIN_CONFIGS
        return DOMAIN_CONFIGS.get(self.DOMAIN, DomainConfig(
            domain=self.DOMAIN,
            subdomains=[self.SUBDOMAIN],
            target_samples=1000,
        ))
    
    @abstractmethod
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """
        Yield corpus samples from this source.
        
        Implementations should:
        1. Fetch raw data (from API, files, database)
        2. Filter by date (pre-Nov 2022 if required)
        3. Clean and normalize text
        4. Yield CorpusSample objects
        
        Args:
            limit: Maximum samples to yield (None = use config target)
        
        Yields:
            CorpusSample objects
        """
        pass
    
    def validate_sample(self, sample: CorpusSample) -> bool:
        """Check if sample meets quality criteria."""
        config = self.config
        
        # Length checks
        if len(sample.text) < config.min_length:
            return False
        if len(sample.text) > config.max_length:
            return False
        if sample.word_count < config.min_words:
            return False
        
        # Deduplication
        if sample.content_hash in self._seen_hashes:
            return False
        
        # Pre-LLM requirement
        if config.require_pre_llm and not sample.source.verified_pre_llm:
            # Check if date is before Nov 2022
            if sample.source.content_date:
                cutoff = date(2022, 11, 1)
                if sample.source.content_date >= cutoff:
                    return False
            else:
                # No date, can't verify
                return False
        
        # Repetition check (simple n-gram analysis)
        if self._repetition_ratio(sample.text) > config.max_repetition_ratio:
            return False
        
        return True
    
    def _repetition_ratio(self, text: str, n: int = 3) -> float:
        """Calculate ratio of repeated n-grams."""
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        
        unique = len(set(ngrams))
        return 1 - (unique / len(ngrams))
    
    def create_sample(
        self,
        text: str,
        original_id: str,
        content_date: Optional[date] = None,
        url: Optional[str] = None,
        author_id_hash: Optional[str] = None,
        **extra_metadata
    ) -> CorpusSample:
        """Factory method for creating properly formatted samples."""
        
        # Determine if pre-LLM
        verified_pre_llm = False
        if content_date and content_date < date(2022, 11, 1):
            verified_pre_llm = True
        
        source = SourceMetadata(
            corpus_name=self.SOURCE_NAME,
            corpus_version=self.SOURCE_VERSION,
            original_id=original_id,
            url=url,
            license=self.LICENSE,
            content_date=content_date,
            content_date_precision="day" if content_date else "unknown",
            verified_pre_llm=verified_pre_llm,
            author_id_hash=author_id_hash,
        )
        
        sample_id = CorpusSample.create_id(
            self.DOMAIN, self.SUBDOMAIN, self.SOURCE_NAME, original_id
        )
        
        return CorpusSample(
            sample_id=sample_id,
            domain=self.DOMAIN,
            subdomain=self.SUBDOMAIN,
            source=source,
            text=text,
        )


class DomainRegistry:
    """Registry of all available collectors."""
    
    _collectors: dict[tuple[Domain, Subdomain, str], Type[BaseCollector]] = {}
    
    @classmethod
    def register(cls, collector_class: Type[BaseCollector]) -> Type[BaseCollector]:
        """Decorator to register a collector."""
        key = (collector_class.DOMAIN, collector_class.SUBDOMAIN, collector_class.SOURCE_NAME)
        cls._collectors[key] = collector_class
        logger.info(f"Registered collector: {key}")
        return collector_class
    
    @classmethod
    def get(
        cls,
        domain: Domain,
        subdomain: Subdomain,
        source: Optional[str] = None
    ) -> Optional[Type[BaseCollector]]:
        """Get a collector by domain/subdomain/source."""
        if source:
            return cls._collectors.get((domain, subdomain, source))
        
        # Return first matching domain/subdomain
        for key, collector in cls._collectors.items():
            if key[0] == domain and key[1] == subdomain:
                return collector
        return None
    
    @classmethod
    def list_collectors(cls, domain: Optional[Domain] = None) -> list[tuple[Domain, Subdomain, str]]:
        """List all registered collectors, optionally filtered by domain."""
        if domain:
            return [k for k in cls._collectors.keys() if k[0] == domain]
        return list(cls._collectors.keys())
    
    @classmethod
    def get_all_for_domain(cls, domain: Domain) -> list[Type[BaseCollector]]:
        """Get all collectors for a domain."""
        return [
            collector for key, collector in cls._collectors.items()
            if key[0] == domain
        ]


def get_collector(
    domain: Domain,
    subdomain: Subdomain,
    source: Optional[str] = None,
    **kwargs
) -> Optional[BaseCollector]:
    """Convenience function to get an instantiated collector."""
    collector_class = DomainRegistry.get(domain, subdomain, source)
    if collector_class:
        return collector_class(**kwargs)
    return None
