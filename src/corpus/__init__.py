"""
PACT Corpus Collection Framework

Harvests human-written text across multiple domains for training
the SpecHO watermark detector and building domain-specific baselines.
"""

from .schema import CorpusSample, DomainConfig, SourceMetadata
from .registry import DomainRegistry, get_collector
from .pipeline import CollectionPipeline

__all__ = [
    'CorpusSample',
    'DomainConfig', 
    'SourceMetadata',
    'DomainRegistry',
    'get_collector',
    'CollectionPipeline',
]
