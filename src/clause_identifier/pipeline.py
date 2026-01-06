"""
Task 3.4: ClauseIdentifier Pipeline

Orchestrates clause detection, pairing, and zone extraction.
Tier 1 implementation - simple orchestration pattern.

Component: Clause Identifier
Tier: 1 (MVP)
"""

from typing import List, Tuple
from specHO.models import Token, ClausePair
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.clause_identifier.pair_rules import PairRulesEngine
from specHO.clause_identifier.zone_extractor import ZoneExtractor
import spacy.tokens


class ClauseIdentifier:
    """
    Orchestrator for the complete clause identification pipeline.

    Combines three components:
    1. ClauseBoundaryDetector - identifies clause boundaries
    2. PairRulesEngine - applies thematic pairing rules
    3. ZoneExtractor - extracts terminal and initial zones

    Tier 1: Simple orchestration with no error recovery or optimization.
    """

    def __init__(self):
        """
        Initialize the clause identifier pipeline.

        Creates instances of all three sub-components.
        """
        self.boundary_detector = ClauseBoundaryDetector()
        self.pair_engine = PairRulesEngine()
        self.zone_extractor = ZoneExtractor()

    def identify_pairs(
        self,
        tokens: List[Token],
        doc: spacy.tokens.Doc
    ) -> List[ClausePair]:
        """
        Identify and extract thematic clause pairs from tokenized text.

        This is the main API method that orchestrates all three components.

        Args:
            tokens: List of Token objects from LinguisticPreprocessor
            doc: spaCy Doc object with dependency parse

        Returns:
            List[ClausePair]: Clause pairs with zones populated

        Pipeline:
            1. Detect clause boundaries from dependency parse
            2. Apply pairing rules (A, B, C) to find thematic pairs
            3. Extract terminal/initial zones for each pair

        Example:
            >>> from specHO.preprocessor.pipeline import LinguisticPreprocessor
            >>> preprocessor = LinguisticPreprocessor()
            >>> identifier = ClauseIdentifier()
            >>>
            >>> text = "The cat sat quietly; the dog ran quickly."
            >>> tokens, doc = preprocessor.process(text)
            >>> pairs = identifier.identify_pairs(tokens, doc)
            >>>
            >>> # pairs[0] will have:
            >>> # - clause_a: "The cat sat quietly"
            >>> # - clause_b: "the dog ran quickly"
            >>> # - zone_a_tokens: [Token("cat"), Token("sat")]
            >>> # - zone_b_tokens: [Token("dog"), Token("ran")]
            >>> # - pair_type: "punctuation"
        """
        # Step 1: Detect clause boundaries
        clauses = self.boundary_detector.identify_clauses(doc, tokens)

        # Step 2: Apply pairing rules
        pairs = self.pair_engine.apply_all_rules(clauses, tokens, doc)

        # Step 3: Extract zones for each pair
        enriched_pairs = []
        for pair in pairs:
            zone_a, zone_b = self.zone_extractor.extract_zones(pair)

            # Create new ClausePair with zones populated
            enriched_pair = ClausePair(
                clause_a=pair.clause_a,
                clause_b=pair.clause_b,
                zone_a_tokens=zone_a,
                zone_b_tokens=zone_b,
                pair_type=pair.pair_type
            )
            enriched_pairs.append(enriched_pair)

        return enriched_pairs


def quick_identify_pairs(
    tokens: List[Token],
    doc: spacy.tokens.Doc
) -> List[ClausePair]:
    """
    Convenience function for one-off clause pair identification.

    Creates a ClauseIdentifier instance and processes the input.

    Args:
        tokens: List of Token objects from LinguisticPreprocessor
        doc: spaCy Doc object with dependency parse

    Returns:
        List[ClausePair]: Clause pairs with zones populated

    Example:
        >>> from specHO.preprocessor.pipeline import LinguisticPreprocessor
        >>> from specHO.clause_identifier.pipeline import quick_identify_pairs
        >>>
        >>> preprocessor = LinguisticPreprocessor()
        >>> text = "The cat sat; the dog ran."
        >>> tokens, doc = preprocessor.process(text)
        >>> pairs = quick_identify_pairs(tokens, doc)
    """
    identifier = ClauseIdentifier()
    return identifier.identify_pairs(tokens, doc)
