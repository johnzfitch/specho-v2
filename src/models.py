"""Core data models for SpecHO watermark detection pipeline.

This module defines the fundamental data structures used throughout the
SpecHO detection system. These dataclasses represent the transformation
of text through each stage of the five-component pipeline:

1. Token: Individual word with linguistic annotations (preprocessor output)
2. Clause: Boundary-detected clause segment (clause identifier output)
3. ClausePair: Two related clauses with extracted zones for comparison
4. EchoScore: Similarity scores from the three echo analyzers
5. DocumentAnalysis: Complete detection results for a document

Tier: 1 (MVP)
Task: 1.1
Dependencies: None (foundation task)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    """Single token with linguistic annotations from preprocessing.

    Represents a word or punctuation mark enriched with linguistic features
    needed for watermark detection. The phonetic field enables phonetic echo
    analysis, POS tags enable structural analysis, and content-word status
    determines which tokens are extracted for zone comparison.

    Attributes:
        text: The raw word or punctuation string
        pos_tag: Part-of-speech tag (e.g., 'NOUN', 'VERB', 'ADJ')
        phonetic: ARPAbet phonetic transcription (e.g., 'HH EH L OW')
        is_content_word: True if noun/verb/adjective (not function word)
        syllable_count: Number of syllables in the word
    """
    text: str
    pos_tag: str
    phonetic: str
    is_content_word: bool
    syllable_count: int


@dataclass
class Clause:
    """Clause boundary with tokens extracted from dependency parse.

    Represents a syntactic unit containing a subject and predicate, identified
    through dependency parsing. Clauses are the fundamental units of analysis
    for the Echo Rule watermark, as echoing patterns manifest at clause
    boundaries rather than arbitrary word positions.

    Attributes:
        tokens: List of Token objects comprising this clause
        start_idx: Starting token index in the document
        end_idx: Ending token index in the document (exclusive)
        clause_type: Classification (e.g., 'main', 'subordinate', 'coordinate')
        head_idx: Token index of the clause anchor/head (ROOT, conj, advcl, ccomp verb)
    """
    tokens: List[Token]
    start_idx: int
    end_idx: int
    clause_type: str
    head_idx: int


@dataclass
class ClausePair:
    """Pair of clauses identified as thematically related for echo analysis.

    Represents two clauses that should be analyzed for watermark echoes based
    on their linguistic relationship (punctuation, conjunction, or transition).
    The zones contain the specific content words extracted for comparison:
    terminal words from clause A and initial words from clause B.

    Attributes:
        clause_a: First clause (provides terminal zone)
        clause_b: Second clause (provides initial zone)
        zone_a_tokens: Terminal content words from clause_a (typically last 3)
        zone_b_tokens: Initial content words from clause_b (typically first 3)
        pair_type: Rule that identified this pair ('punctuation', 'conjunction', 'transition')
    """
    clause_a: Clause
    clause_b: Clause
    zone_a_tokens: List[Token]
    zone_b_tokens: List[Token]
    pair_type: str


@dataclass
class EchoScore:
    """Similarity scores from three echo analyzers for a clause pair.

    Captures the multi-dimensional echoing pattern that constitutes the
    watermark signal. Each score measures a different type of similarity
    between zone A and zone B. The combined_score represents the weighted
    aggregation of the three dimensions.

    All scores are normalized to the range [0.0, 1.0] where:
    - 0.0 indicates no similarity
    - 1.0 indicates perfect similarity

    Attributes:
        phonetic_score: Phonetic similarity (sound correspondence)
        structural_score: Structural similarity (POS patterns, syllable counts)
        semantic_score: Semantic similarity (meaning relationships)
        combined_score: Weighted combination of the three scores
    """
    phonetic_score: float  # 0.0-1.0
    structural_score: float  # 0.0-1.0
    semantic_score: float  # 0.0-1.0
    combined_score: float  # 0.0-1.0


@dataclass
class DocumentAnalysis:
    """Complete watermark detection results for a document.

    Contains the full analysis output including all intermediate results
    and the final statistical verdict. This is the top-level object returned
    by the SpecHODetector after processing a text through the entire pipeline.

    The z_score and confidence provide the interpretable verdict:
    - z_score: Standard deviations above human baseline mean
    - confidence: Probability that this score is inconsistent with human writing

    Attributes:
        text: Original input text that was analyzed
        tokens: All tokens from preprocessing (for debugging/display)
        clause_pairs: All thematic clause pairs identified in the document
        echo_scores: Similarity scores for each clause pair
        final_score: Aggregated document-level echo score (0.0-1.0)
        z_score: Statistical significance relative to human baseline
        confidence: Confidence level that watermark is present (0.0-1.0)
    """
    text: str
    clause_pairs: List[ClausePair]
    echo_scores: List[EchoScore]
    final_score: float
    z_score: float
    confidence: float
    tokens: List[Token] = None  # Optional, added for UI display

    @property
    def document_score(self) -> float:
        """Alias for final_score for backward compatibility."""
        return self.final_score
