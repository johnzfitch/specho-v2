"""
PACT Corpus Collectors

Import all collectors to trigger registration with DomainRegistry.
"""

# Import all collector modules to register them
from . import academic
from . import abstracts
from . import conversational
from . import legal
from . import pacer
from . import huggingface

# Make individual collectors available
from .academic import ASAPCollector, Persuade2Collector
from .abstracts import ArxivCollector, SemanticScholarCollector
from .conversational import PushshiftRedditCollector, ForumCollector, StackOverflowCollector
from .legal import CourtListenerCollector, CongressCollector, CRSReportCollector
from .pacer import PACERCollector, PACEROpinionsCollector
from .huggingface import HuggingFaceNYTCollector, HuggingFaceGenericCollector

__all__ = [
    # Academic
    'ASAPCollector',
    'Persuade2Collector',

    # Abstracts
    'ArxivCollector',
    'SemanticScholarCollector',

    # Conversational
    'PushshiftRedditCollector',
    'ForumCollector',
    'StackOverflowCollector',

    # Legal
    'CourtListenerCollector',
    'CongressCollector',
    'CRSReportCollector',

    # PACER
    'PACERCollector',
    'PACEROpinionsCollector',

    # HuggingFace
    'HuggingFaceNYTCollector',
    'HuggingFaceGenericCollector',
]
