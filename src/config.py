"""Configuration management for SpecHO watermark detection system.

This module implements a three-tier configuration system that controls the
behavior of all five pipeline components. Each tier represents a different
stage of system maturity:

- Tier 1 (simple): MVP implementation with basic algorithms
- Tier 2 (robust): Production-ready with proven enhancements
- Tier 3 (research): Experimental optimization with advanced features

The configuration system allows component-level customization while maintaining
consistent profiles across the entire pipeline. Use load_config() to load a
profile with optional field-level overrides.

Tier: 1 (MVP)
Task: 1.2
Dependencies: Task 1.1 (models.py)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ClauseDetectionConfig:
    """Configuration for clause boundary detection and identification.

    Controls how the Clause Identifier component segments text into clauses
    and determines which clause pairs should be analyzed for echoes.

    Attributes:
        min_length: Minimum tokens required for a valid clause
        max_length: Maximum tokens allowed in a clause (fragments beyond merge)
        punctuation: Punctuation marks that indicate clause boundaries
        dependency_labels: Dependency parse labels indicating clause structure
        strict_mode: Whether to enforce strict clause validation rules
        cross_sentence_pairing: Whether to pair clauses across sentence boundaries (Tier 3)
    """
    min_length: int = 3
    max_length: int = 50
    punctuation: List[str] = field(default_factory=lambda: [";", "—", ":"])
    dependency_labels: List[str] = field(default_factory=lambda: ["ROOT", "conj", "advcl", "ccomp"])
    strict_mode: bool = False
    cross_sentence_pairing: bool = False


@dataclass
class PairRulesConfig:
    """Configuration for clause pairing rules.

    Defines which linguistic relationships constitute thematic clause pairs
    that should be analyzed for watermark echoes.

    Attributes:
        conjunctions: Coordinating conjunctions for Rule B
        transitions: Transitional phrases for Rule C
        min_pair_confidence: Minimum confidence threshold for accepting pairs (Tier 2+)
        use_confidence_weighting: Whether to weight pairs by confidence (Tier 2+)
    """
    conjunctions: List[str] = field(default_factory=lambda: ["but", "and", "or"])
    transitions: List[str] = field(default_factory=lambda: ["However,", "Therefore,", "Thus,"])
    min_pair_confidence: float = 0.0
    use_confidence_weighting: bool = False


@dataclass
class ZoneExtractionConfig:
    """Configuration for zone extraction from clause pairs.

    Controls how terminal and initial zones are extracted from clause pairs
    for echo analysis.

    Attributes:
        window_size: Number of content words to extract from each zone
        min_zone_length: Minimum words required in a zone
        exclude_discourse_markers: Whether to filter out transition words (Tier 2+)
        discourse_markers: List of discourse markers to exclude (Tier 2+)
        adaptive_window: Use adaptive window sizing based on clause length (Tier 3)
    """
    window_size: int = 3
    min_zone_length: int = 1
    exclude_discourse_markers: bool = False
    discourse_markers: List[str] = field(default_factory=lambda: ["however", "therefore", "thus", "moreover"])
    adaptive_window: bool = False


@dataclass
class PhoneticAnalysisConfig:
    """Configuration for phonetic echo analysis.

    Controls the algorithm used to measure phonetic similarity between zones.

    Attributes:
        algorithm: Similarity algorithm ('levenshtein', 'rime', or 'hungarian')
        top_k_matches: Number of best matches to average (Tier 2+)
        length_penalty: Penalty factor for length differences (Tier 2+)
        use_stress_patterns: Include stress pattern matching (Tier 3)
        cache_results: Enable LRU caching of phonetic computations (Tier 3)
    """
    algorithm: str = "levenshtein"
    top_k_matches: int = 1
    length_penalty: float = 0.0
    use_stress_patterns: bool = False
    cache_results: bool = False


@dataclass
class StructuralAnalysisConfig:
    """Configuration for structural echo analysis.

    Controls which structural features are compared and their relative weights.

    Attributes:
        pos_pattern_enabled: Enable POS pattern comparison
        pos_pattern_weight: Weight for POS pattern similarity
        syllable_count_enabled: Enable syllable count comparison
        syllable_count_weight: Weight for syllable similarity
        word_properties_enabled: Enable word property comparison (Tier 2+)
        word_properties_weight: Weight for word properties (Tier 2+)
    """
    pos_pattern_enabled: bool = True
    pos_pattern_weight: float = 0.5
    syllable_count_enabled: bool = True
    syllable_count_weight: float = 0.5
    word_properties_enabled: bool = False
    word_properties_weight: float = 0.0


@dataclass
class SemanticAnalysisConfig:
    """Configuration for semantic echo analysis.

    Controls the embedding model and semantic similarity computation.

    Attributes:
        model: Embedding model ('static' for Word2Vec/GloVe, or transformer name)
        use_antonym_detection: Detect and handle antonym relationships (Tier 2+)
        batch_size: Batch size for embedding computation (Tier 2+)
        device: Computation device ('cpu' or 'cuda')
        cache_embeddings: Enable embedding caching (Tier 2+)
    """
    model: str = "static"
    use_antonym_detection: bool = False
    batch_size: int = 1
    device: str = "cpu"
    cache_embeddings: bool = False


@dataclass
class ScoringConfig:
    """Configuration for weighted scoring and aggregation.

    Controls how individual echo scores are combined into pair scores and
    how pair scores are aggregated into document scores.

    Attributes:
        phonetic_weight: Weight for phonetic similarity
        structural_weight: Weight for structural similarity
        semantic_weight: Weight for semantic similarity
        missing_data_strategy: How to handle missing scores ('zero' or 'renorm')
        use_pair_confidence: Weight pairs by confidence (Tier 2+)
        aggregation_strategy: Document aggregation method ('mean', 'median', 'trimmed_mean', 'winsorized_mean')
        trim_percent: Percentage to trim for trimmed_mean (Tier 2+)
        outlier_removal: Enable outlier detection and removal (Tier 2+)
    """
    phonetic_weight: float = 0.33
    structural_weight: float = 0.33
    semantic_weight: float = 0.34  # Ensures sum = 1.0
    missing_data_strategy: str = "zero"
    use_pair_confidence: bool = False
    aggregation_strategy: str = "mean"
    trim_percent: float = 0.0
    outlier_removal: bool = False


@dataclass
class ValidationConfig:
    """Configuration for statistical validation.

    Controls baseline corpus processing and statistical significance testing.

    Attributes:
        baseline_path: Path to baseline statistics file
        corpus_sources: List of corpus directories for baseline building (Tier 2+)
        incremental_updates: Enable incremental baseline updates (Tier 2+)
        distribution_fitting: Use non-parametric distribution fitting (Tier 3)
    """
    baseline_path: str = "data/baseline/baseline_stats.pkl"
    corpus_sources: List[str] = field(default_factory=lambda: ["data/corpus/"])
    incremental_updates: bool = False
    distribution_fitting: bool = False


@dataclass
class SpecHOConfig:
    """Master configuration for the entire SpecHO detection pipeline.

    Aggregates component-level configurations into a single object that
    controls the behavior of all five pipeline components. Use one of the
    predefined PROFILES or create a custom configuration.

    Attributes:
        profile_name: Name of this configuration profile
        tier: Implementation tier (1=MVP, 2=Production, 3=Research)
        clause_detection: Configuration for clause identification
        pair_rules: Configuration for clause pairing rules
        zone_extraction: Configuration for zone extraction
        phonetic_analysis: Configuration for phonetic analyzer
        structural_analysis: Configuration for structural analyzer
        semantic_analysis: Configuration for semantic analyzer
        scoring: Configuration for scoring and aggregation
        validation: Configuration for statistical validation
    """
    profile_name: str = "simple"
    tier: int = 1
    clause_detection: ClauseDetectionConfig = field(default_factory=ClauseDetectionConfig)
    pair_rules: PairRulesConfig = field(default_factory=PairRulesConfig)
    zone_extraction: ZoneExtractionConfig = field(default_factory=ZoneExtractionConfig)
    phonetic_analysis: PhoneticAnalysisConfig = field(default_factory=PhoneticAnalysisConfig)
    structural_analysis: StructuralAnalysisConfig = field(default_factory=StructuralAnalysisConfig)
    semantic_analysis: SemanticAnalysisConfig = field(default_factory=SemanticAnalysisConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ============================================================================
# PREDEFINED CONFIGURATION PROFILES
# ============================================================================

# Tier 1: Simple Profile (MVP Implementation)
SIMPLE_PROFILE = SpecHOConfig(
    profile_name="simple",
    tier=1,
    clause_detection=ClauseDetectionConfig(
        min_length=3,
        max_length=50,
        punctuation=[";", "—", ":"],
        strict_mode=False,
    ),
    pair_rules=PairRulesConfig(
        conjunctions=["but", "and", "or"],
        transitions=["However,", "Therefore,", "Thus,"],
    ),
    zone_extraction=ZoneExtractionConfig(
        window_size=3,
        min_zone_length=1,
    ),
    phonetic_analysis=PhoneticAnalysisConfig(
        algorithm="levenshtein",
    ),
    structural_analysis=StructuralAnalysisConfig(
        pos_pattern_weight=0.5,
        syllable_count_weight=0.5,
    ),
    semantic_analysis=SemanticAnalysisConfig(
        model="static",
        device="cpu",
    ),
    scoring=ScoringConfig(
        phonetic_weight=0.33,
        structural_weight=0.33,
        semantic_weight=0.34,
        missing_data_strategy="zero",
        aggregation_strategy="mean",
    ),
    validation=ValidationConfig(
        baseline_path="data/baseline/baseline_stats.pkl",
    ),
)

# Tier 2: Robust Profile (Production-Ready)
ROBUST_PROFILE = SpecHOConfig(
    profile_name="robust",
    tier=2,
    clause_detection=ClauseDetectionConfig(
        min_length=3,
        max_length=50,
        punctuation=[";", "—", ":"],
        strict_mode=False,
    ),
    pair_rules=PairRulesConfig(
        conjunctions=["but", "yet", "however", "and", "or"],
        transitions=["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,"],
        min_pair_confidence=0.3,
        use_confidence_weighting=True,
    ),
    zone_extraction=ZoneExtractionConfig(
        window_size=3,
        min_zone_length=1,
        exclude_discourse_markers=True,
        discourse_markers=["however", "therefore", "thus", "moreover"],
    ),
    phonetic_analysis=PhoneticAnalysisConfig(
        algorithm="rime",
        top_k_matches=2,
        length_penalty=0.1,
    ),
    structural_analysis=StructuralAnalysisConfig(
        pos_pattern_enabled=True,
        pos_pattern_weight=0.4,
        syllable_count_enabled=True,
        syllable_count_weight=0.3,
        word_properties_enabled=True,
        word_properties_weight=0.3,
    ),
    semantic_analysis=SemanticAnalysisConfig(
        model="all-MiniLM-L6-v2",
        batch_size=32,
        device="cpu",
        cache_embeddings=True,
    ),
    scoring=ScoringConfig(
        phonetic_weight=0.4,
        structural_weight=0.3,
        semantic_weight=0.3,
        missing_data_strategy="renorm",
        use_pair_confidence=True,
        aggregation_strategy="trimmed_mean",
        trim_percent=0.1,
        outlier_removal=True,
    ),
    validation=ValidationConfig(
        baseline_path="data/baseline/baseline_stats.pkl",
        corpus_sources=["data/corpus/"],
        incremental_updates=False,
    ),
)

# Tier 3: Research Profile (Experimental Optimization)
RESEARCH_PROFILE = SpecHOConfig(
    profile_name="research",
    tier=3,
    clause_detection=ClauseDetectionConfig(
        min_length=2,
        max_length=100,
        punctuation=[";", "—", ":"],
        strict_mode=False,
        cross_sentence_pairing=True,
    ),
    pair_rules=PairRulesConfig(
        conjunctions=["but", "yet", "however", "and", "or", "nor", "so"],
        transitions=["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,", "Nonetheless,", "Furthermore,"],
        min_pair_confidence=0.2,
        use_confidence_weighting=True,
    ),
    zone_extraction=ZoneExtractionConfig(
        window_size=5,
        min_zone_length=1,
        exclude_discourse_markers=True,
        discourse_markers=["however", "therefore", "thus", "moreover"],
        adaptive_window=True,
    ),
    phonetic_analysis=PhoneticAnalysisConfig(
        algorithm="hungarian",
        use_stress_patterns=True,
        cache_results=True,
    ),
    structural_analysis=StructuralAnalysisConfig(
        pos_pattern_enabled=True,
        pos_pattern_weight=0.4,
        syllable_count_enabled=True,
        syllable_count_weight=0.3,
        word_properties_enabled=True,
        word_properties_weight=0.3,
    ),
    semantic_analysis=SemanticAnalysisConfig(
        model="sentence-transformers/all-mpnet-base-v2",
        use_antonym_detection=True,
        batch_size=64,
        device="cuda",
        cache_embeddings=True,
    ),
    scoring=ScoringConfig(
        phonetic_weight=0.4,
        structural_weight=0.3,
        semantic_weight=0.3,
        missing_data_strategy="renorm",
        use_pair_confidence=True,
        aggregation_strategy="winsorized_mean",
        trim_percent=0.05,
        outlier_removal=True,
    ),
    validation=ValidationConfig(
        baseline_path="data/baseline/baseline_stats.pkl",
        corpus_sources=["data/corpus/"],
        incremental_updates=False,
        distribution_fitting=True,
    ),
)

# Profile registry for easy lookup
PROFILES: Dict[str, SpecHOConfig] = {
    "simple": SIMPLE_PROFILE,
    "robust": ROBUST_PROFILE,
    "research": RESEARCH_PROFILE,
}


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(profile_name: str = "simple", overrides: Optional[Dict[str, Any]] = None) -> SpecHOConfig:
    """Load a configuration profile with optional field-level overrides.

    This function provides flexible configuration loading with three patterns:

    1. Load a predefined profile as-is:
       config = load_config("simple")

    2. Load a profile with component overrides:
       config = load_config("simple", {"scoring.phonetic_weight": 0.5})

    3. Load a profile with nested overrides:
       config = load_config("robust", {
           "phonetic_analysis.algorithm": "hungarian",
           "scoring.aggregation_strategy": "median"
       })

    Override keys use dot notation to specify nested fields. For example:
    - "scoring.phonetic_weight" sets config.scoring.phonetic_weight
    - "zone_extraction.window_size" sets config.zone_extraction.window_size

    Args:
        profile_name: Name of profile to load ('simple', 'robust', or 'research')
        overrides: Optional dictionary of field overrides using dot notation

    Returns:
        SpecHOConfig instance with requested profile and overrides applied

    Raises:
        ValueError: If profile_name is not recognized
        KeyError: If an override key references a non-existent field

    Examples:
        >>> config = load_config("simple")
        >>> config.profile_name
        'simple'

        >>> config = load_config("robust", {"scoring.phonetic_weight": 0.5})
        >>> config.scoring.phonetic_weight
        0.5
    """
    if profile_name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")

    # Get base configuration (make a copy to avoid modifying the original)
    config_dict = asdict(PROFILES[profile_name])

    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            # Parse dot notation (e.g., "scoring.phonetic_weight")
            parts = key.split(".")

            if len(parts) == 1:
                # Top-level override (e.g., "tier")
                if parts[0] not in config_dict:
                    raise KeyError(f"Unknown config field: {key}")
                config_dict[parts[0]] = value

            elif len(parts) == 2:
                # Component-level override (e.g., "scoring.phonetic_weight")
                component, field = parts
                if component not in config_dict:
                    raise KeyError(f"Unknown config component: {component}")
                if not isinstance(config_dict[component], dict):
                    raise KeyError(f"Cannot override field in non-dict component: {component}")
                if field not in config_dict[component]:
                    raise KeyError(f"Unknown field '{field}' in component '{component}'")
                config_dict[component][field] = value

            else:
                raise KeyError(f"Invalid override key format: {key}. Use 'field' or 'component.field'")

    # Reconstruct config object from dictionary
    return _dict_to_config(config_dict)


def _dict_to_config(config_dict: Dict[str, Any]) -> SpecHOConfig:
    """Internal helper to reconstruct SpecHOConfig from dictionary.

    This function handles the conversion of nested dictionaries back into
    the appropriate dataclass instances.

    Args:
        config_dict: Dictionary representation of configuration

    Returns:
        Fully reconstructed SpecHOConfig instance
    """
    return SpecHOConfig(
        profile_name=config_dict["profile_name"],
        tier=config_dict["tier"],
        clause_detection=ClauseDetectionConfig(**config_dict["clause_detection"]),
        pair_rules=PairRulesConfig(**config_dict["pair_rules"]),
        zone_extraction=ZoneExtractionConfig(**config_dict["zone_extraction"]),
        phonetic_analysis=PhoneticAnalysisConfig(**config_dict["phonetic_analysis"]),
        structural_analysis=StructuralAnalysisConfig(**config_dict["structural_analysis"]),
        semantic_analysis=SemanticAnalysisConfig(**config_dict["semantic_analysis"]),
        scoring=ScoringConfig(**config_dict["scoring"]),
        validation=ValidationConfig(**config_dict["validation"]),
    )
