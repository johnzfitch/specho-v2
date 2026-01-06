"""
PACT Corpus Schema

Defines the canonical data structures for corpus samples.
All collectors must produce CorpusSample objects.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional, Literal
from enum import Enum
import hashlib
import json


class Domain(str, Enum):
    """Primary content domains for cross-domain transfer studies."""
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    JOURNALISTIC = "journalistic"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"
    SOCIAL = "social"
    LEGAL = "legal"


class Subdomain(str, Enum):
    """Granular content categories within domains."""
    # Academic
    ESSAYS = "essays"
    ABSTRACTS = "abstracts"
    THESIS = "thesis"
    PEER_REVIEWS = "peer_reviews"
    
    # Conversational
    DISCORD = "discord"
    REDDIT = "reddit"
    FORUMS = "forums"
    IRC = "irc"
    
    # Journalistic
    NEWS = "news"
    EDITORIALS = "editorials"
    LONGFORM = "longform"
    
    # Technical
    DOCUMENTATION = "documentation"
    TUTORIALS = "tutorials"
    CODE_COMMENTS = "code_comments"
    BUG_REPORTS = "bug_reports"
    
    # Creative
    FICTION = "fiction"
    POETRY = "poetry"
    SCREENPLAYS = "screenplays"
    
    # Business
    EMAILS = "emails"
    REPORTS = "reports"
    PROPOSALS = "proposals"
    
    # Social
    TWEETS = "tweets"
    REVIEWS = "reviews"
    
    # Legal
    COURT_FILINGS = "court_filings"
    LEGISLATION = "legislation"
    POLICY = "policy"


@dataclass
class SourceMetadata:
    """Provenance information for a corpus sample."""
    corpus_name: str                    # e.g., "asap", "pushshift_reddit", "courtlistener"
    corpus_version: str                 # Version or date of corpus snapshot
    original_id: str                    # ID within source corpus
    url: Optional[str] = None           # Direct link if available
    license: Optional[str] = None       # License of source data
    collection_date: date = field(default_factory=date.today)
    
    # Temporal provenance (CRITICAL for pre-LLM verification)
    content_date: Optional[date] = None         # When content was written
    content_date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    verified_pre_llm: bool = False              # Confirmed before Nov 2022
    
    # Author provenance (anonymized)
    author_id_hash: Optional[str] = None        # Hashed author ID
    author_type: Literal["individual", "organization", "unknown"] = "unknown"


@dataclass
class CorpusSample:
    """
    Canonical sample in the PACT corpus.
    
    All collectors produce these. All generators consume these.
    SpecHO processes these. Baselines are built from these.
    """
    
    # === Identity ===
    sample_id: str                      # Globally unique: "{domain}_{subdomain}_{source}_{original_id}"
    
    # === Classification ===
    domain: Domain
    subdomain: Subdomain
    
    # === Provenance ===
    source: SourceMetadata
    
    # === Content ===
    text: str
    
    # === Computed Fields (populated after creation) ===
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    content_hash: str = ""              # SHA256 of normalized text
    
    # === Generation Fields (only for AI-generated variants) ===
    is_original: bool = True
    parent_id: Optional[str] = None     # sample_id of human original
    model: Optional[str] = None         # "gpt-4o-2024-08-06"
    model_family: Optional[str] = None  # "openai", "anthropic", etc.
    prompt_id: Optional[int] = None
    prompt_text: Optional[str] = None
    prompt_style: Optional[str] = None
    generation_temp: Optional[float] = None
    generation_timestamp: Optional[datetime] = None
    run_number: Optional[int] = None
    
    # === SpecHO Analysis (populated by detector) ===
    specho_score: Optional[float] = None
    specho_confidence: Optional[float] = None
    echo_phonetic: Optional[float] = None
    echo_structural: Optional[float] = None
    echo_semantic: Optional[float] = None
    clause_pair_count: Optional[int] = None
    
    def __post_init__(self):
        """Compute derived fields."""
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.sentence_count:
            # Rough estimate
            self.sentence_count = self.text.count('.') + self.text.count('!') + self.text.count('?')
        if not self.content_hash:
            normalized = ' '.join(self.text.lower().split())
            self.content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    @classmethod
    def create_id(cls, domain: Domain, subdomain: Subdomain, source: str, original_id: str) -> str:
        """Generate canonical sample ID."""
        return f"{domain.value}_{subdomain.value}_{source}_{original_id}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['domain'] = self.domain.value
        d['subdomain'] = self.subdomain.value
        d['source'] = asdict(self.source)
        return d
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str, indent=2)

    def to_jsonl(self) -> str:
        """Serialize to compact JSONL format (no indentation)."""
        return json.dumps(self.to_dict(), default=str)

    def get_formatted_text(self) -> str:
        """
        Get text formatted as users actually present it to AI models.

        Returns text with realistic formatting based on domain:
        - News: Markdown with headlines
        - Essays: Plain formatting with optional title
        - Code: Syntax highlighting with code fences
        - Legal: Document headers
        - Social: Plaintext
        """
        from .formatters import format_text

        # Build metadata dict from source
        metadata = {
            'title': getattr(self.source, 'title', None),
            'url': self.source.url,
        }

        return format_text(self.text, self.domain, self.subdomain, metadata)

    @classmethod
    def from_dict(cls, d: dict) -> 'CorpusSample':
        """Reconstruct from dictionary."""
        d['domain'] = Domain(d['domain'])
        d['subdomain'] = Subdomain(d['subdomain'])
        d['source'] = SourceMetadata(**d['source'])
        return cls(**d)


@dataclass
class DomainConfig:
    """Configuration for a collection domain."""
    domain: Domain
    subdomains: list[Subdomain]
    target_samples: int
    min_length: int = 100               # Minimum chars
    max_length: int = 10000             # Maximum chars
    require_pre_llm: bool = True        # Must be verifiably pre-Nov 2022
    language: str = "en"
    
    # Quality filters
    min_words: int = 20
    max_repetition_ratio: float = 0.3   # Max ratio of repeated n-grams
    require_complete_sentences: bool = True


# Default domain configurations
DOMAIN_CONFIGS = {
    Domain.ACADEMIC: DomainConfig(
        domain=Domain.ACADEMIC,
        subdomains=[Subdomain.ESSAYS, Subdomain.ABSTRACTS, Subdomain.PEER_REVIEWS],
        target_samples=2000,
        min_length=200,
        max_length=15000,
    ),
    Domain.CONVERSATIONAL: DomainConfig(
        domain=Domain.CONVERSATIONAL,
        subdomains=[Subdomain.DISCORD, Subdomain.REDDIT, Subdomain.FORUMS],
        target_samples=5000,
        min_length=50,
        max_length=5000,
        require_complete_sentences=False,  # Casual text may be fragments
    ),
    Domain.JOURNALISTIC: DomainConfig(
        domain=Domain.JOURNALISTIC,
        subdomains=[Subdomain.NEWS, Subdomain.EDITORIALS, Subdomain.LONGFORM],
        target_samples=1500,
        min_length=300,
        max_length=20000,
    ),
    Domain.TECHNICAL: DomainConfig(
        domain=Domain.TECHNICAL,
        subdomains=[Subdomain.DOCUMENTATION, Subdomain.TUTORIALS, Subdomain.BUG_REPORTS],
        target_samples=1500,
        min_length=100,
        max_length=10000,
    ),
    Domain.CREATIVE: DomainConfig(
        domain=Domain.CREATIVE,
        subdomains=[Subdomain.FICTION, Subdomain.SCREENPLAYS],  # Skip poetry - too short
        target_samples=1000,
        min_length=500,
        max_length=30000,
    ),
    Domain.BUSINESS: DomainConfig(
        domain=Domain.BUSINESS,
        subdomains=[Subdomain.EMAILS, Subdomain.REPORTS],
        target_samples=1000,
        min_length=100,
        max_length=15000,
    ),
    Domain.SOCIAL: DomainConfig(
        domain=Domain.SOCIAL,
        subdomains=[Subdomain.TWEETS, Subdomain.REVIEWS],
        target_samples=3000,
        min_length=30,
        max_length=2000,
        require_complete_sentences=False,
    ),
    Domain.LEGAL: DomainConfig(
        domain=Domain.LEGAL,
        subdomains=[Subdomain.COURT_FILINGS, Subdomain.LEGISLATION],
        target_samples=1000,
        min_length=500,
        max_length=50000,
    ),
}
