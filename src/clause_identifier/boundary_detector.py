"""
Clause Boundary Detector

Identifies clause boundaries in text using dependency parsing and linguistic heuristics.
This is Task 3.1 of the SpecHO watermark detection system.

The BoundaryDetector uses spaCy's dependency parse to identify:
- Main clauses (ROOT verbs)
- Coordinated clauses (conj relations)
- Subordinate clauses (advcl, ccomp relations)

Tier 1 Implementation (Simple Heuristics):
- Uses basic dependency labels: ROOT, conj, advcl, ccomp
- Simple punctuation rules: period, semicolon, em dash
- No edge case handling (quotes, parentheses, fragments)
- No sophisticated span merging
"""

import logging
from typing import List, Tuple, Set
from spacy.tokens import Doc as SpacyDoc, Token as SpacyToken

# Import our data model
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import Clause, Token


class ClauseBoundaryDetector:
    """Detects clause boundaries using dependency parsing and linguistic rules.

    The ClauseBoundaryDetector identifies syntactic clause boundaries by analyzing
    spaCy's dependency parse tree. It recognizes:

    - **Main clauses**: Clauses anchored by ROOT verbs (independent clauses)
    - **Coordinate clauses**: Clauses joined by coordinating conjunctions (and, but, or)
    - **Subordinate clauses**: Clauses that depend on main clauses (when, because, although)

    Tier 1 Algorithm:
    1. Find all ROOT verbs (main clause anchors)
    2. Find coordinated clauses (conj dependency relation)
    3. Find subordinate clauses (advcl, ccomp relations)
    4. For each clause anchor, determine span boundaries
    5. Classify clause type based on dependency relation

    The clause spans are determined by:
    - Start: Leftmost child of the clause anchor
    - End: Rightmost child of the clause anchor
    - Includes all tokens in the syntactic subtree

    Tier 1 Limitations (deferred to Tier 2):
    - No fragment merging (clauses < 3 tokens)
    - No quote/parenthesis trimming
    - No zero-length span handling
    - No overlapping span resolution

    Attributes:
        None (stateless detector)

    Examples:
        >>> detector = ClauseBoundaryDetector()
        >>> from preprocessor.dependency_parser import DependencyParser
        >>>
        >>> parser = DependencyParser()
        >>> doc = parser.parse("The cat sat, and the dog ran.")
        >>> clauses = detector.identify_clauses(doc)
        >>>
        >>> # Two clauses detected
        >>> len(clauses)
        2
        >>> clauses[0].clause_type
        'main'
        >>> clauses[1].clause_type
        'coordinate'
    """

    def __init__(self):
        """Initialize the ClauseBoundaryDetector.

        Tier 1 implementation requires no configuration.
        """
        logging.info("Initializing ClauseBoundaryDetector (Tier 1)")

    def identify_clauses(self, doc: SpacyDoc, tokens: List[Token]) -> List[Clause]:
        """Identify clause boundaries and create Clause objects.

        This is the main entry point for clause boundary detection. It analyzes
        the dependency parse tree to find clause anchors (ROOT, conj, advcl, ccomp)
        and determines the span of each clause.

        Algorithm:
        1. Find all ROOT verbs → main clauses
        2. Find all conj relations → coordinated clauses
        3. Find all advcl/ccomp relations → subordinate clauses
        4. For each clause anchor, determine span (leftmost to rightmost child)
        5. Extract Token objects for each span
        6. Create Clause dataclass with tokens, indices, and type

        Args:
            doc: spaCy Doc with dependency parse
            tokens: List[Token] from LinguisticPreprocessor (for Clause.tokens field)

        Returns:
            List of Clause objects, sorted by start_idx

        Raises:
            ValueError: If doc is empty or tokens list doesn't match doc length

        Examples:
            >>> detector = ClauseBoundaryDetector()
            >>> doc = parser.parse("The cat sat on the mat.")
            >>> tokens = preprocessor.process("The cat sat on the mat.")[0]
            >>> clauses = detector.identify_clauses(doc, tokens)
            >>>
            >>> # One main clause
            >>> len(clauses)
            1
            >>> clauses[0].clause_type
            'main'
            >>> len(clauses[0].tokens)
            6  # "The cat sat on the mat"
        """
        if not doc or len(list(doc)) == 0:
            logging.warning("Empty doc provided to ClauseBoundaryDetector")
            return []

        if len(tokens) != len(list(doc)):
            logging.warning(
                f"Token count mismatch: {len(tokens)} tokens vs {len(list(doc))} spaCy tokens. "
                "Proceeding with spaCy token count."
            )

        logging.debug(f"Detecting clause boundaries in document with {len(list(doc))} tokens")

        # Collect all clause anchors with their types
        clause_anchors: List[Tuple[SpacyToken, str]] = []

        # 1. Find ROOT verbs (main clauses)
        for token in doc:
            if token.dep_ == "ROOT":
                clause_anchors.append((token, "main"))
                logging.debug(f"Found main clause anchor: {token.text} (pos={token.i})")

        # 2. Find coordinated clauses (conj)
        for token in doc:
            if token.dep_ == "conj":
                clause_anchors.append((token, "coordinate"))
                logging.debug(f"Found coordinate clause anchor: {token.text} (pos={token.i})")

        # 3. Find subordinate clauses (advcl, ccomp)
        for token in doc:
            if token.dep_ in {"advcl", "ccomp"}:
                clause_anchors.append((token, "subordinate"))
                logging.debug(f"Found subordinate clause anchor: {token.text} (pos={token.i}, dep={token.dep_})")

        # Build Clause objects for each anchor
        clauses = []
        for anchor_token, clause_type in clause_anchors:
            clause = self._build_clause_from_anchor(anchor_token, clause_type, doc, tokens)
            clauses.append(clause)

        # Sort clauses by start position
        clauses.sort(key=lambda c: c.start_idx)

        # Tier 1 Fix: Make spans non-overlapping
        # This prevents downstream issues in zone extraction and pairing
        clauses = self._make_spans_non_overlapping(clauses, doc, tokens)

        logging.info(f"Identified {len(clauses)} clauses ({sum(1 for c in clauses if c.clause_type == 'main')} main, "
                    f"{sum(1 for c in clauses if c.clause_type == 'coordinate')} coordinate, "
                    f"{sum(1 for c in clauses if c.clause_type == 'subordinate')} subordinate)")

        return clauses

    def _build_clause_from_anchor(
        self,
        anchor: SpacyToken,
        clause_type: str,
        doc: SpacyDoc,
        tokens: List[Token]
    ) -> Clause:
        """Build a Clause object from a clause anchor token.

        Determines the span of the clause by finding all children (descendants)
        of the anchor token in the dependency tree. The clause includes:
        - The anchor token itself
        - All tokens that depend on the anchor (direct and indirect children)

        Tier 1: Simple span extraction (leftmost to rightmost child)
        Tier 2: Could add punctuation trimming, fragment filtering

        Args:
            anchor: Clause anchor token (ROOT, conj, advcl, or ccomp)
            clause_type: Type of clause ("main", "coordinate", "subordinate")
            doc: spaCy Doc for token lookup
            tokens: List[Token] for Clause.tokens field

        Returns:
            Clause object with tokens, indices, and type
        """
        # Find all descendants of anchor (subtree)
        subtree_indices = self._get_subtree_indices(anchor)

        # Determine span boundaries
        start_idx = min(subtree_indices)
        end_idx = max(subtree_indices)

        # Extract Token objects for this span
        # Handle potential token count mismatch
        clause_tokens = []
        for i in range(start_idx, end_idx + 1):
            if i < len(tokens):
                clause_tokens.append(tokens[i])
            else:
                # Fallback: create minimal Token from spaCy token
                spacy_token = doc[i]
                clause_tokens.append(Token(
                    text=spacy_token.text,
                    pos_tag=spacy_token.pos_,
                    phonetic="",
                    is_content_word=False,
                    syllable_count=0
                ))
                logging.debug(f"Created fallback Token for index {i}: {spacy_token.text}")

        logging.debug(f"Built {clause_type} clause: span=({start_idx}, {end_idx}), "
                     f"tokens={len(clause_tokens)}, text='{' '.join(t.text for t in clause_tokens[:5])}...'")

        return Clause(
            tokens=clause_tokens,
            start_idx=start_idx,
            end_idx=end_idx,
            clause_type=clause_type,
            head_idx=anchor.i
        )

    def _get_subtree_indices(self, token: SpacyToken) -> Set[int]:
        """Get all token indices in the subtree rooted at token.

        Recursively finds all descendants of a token in the dependency tree.
        This includes the token itself and all tokens that directly or
        indirectly depend on it.

        Args:
            token: Root of subtree to extract

        Returns:
            Set of token indices in the subtree

        Examples:
            >>> # For "The cat sat on the mat"
            >>> # "sat" is ROOT with children: "cat", "on"
            >>> # "cat" has child: "The"
            >>> # "on" has child: "mat" which has child "the"
            >>> indices = self._get_subtree_indices(sat_token)
            >>> # Returns: {0, 1, 2, 3, 4, 5} (all tokens)
        """
        indices = {token.i}  # Include the token itself

        # Recursively add all children
        for child in token.children:
            indices.update(self._get_subtree_indices(child))

        return indices

    def _make_spans_non_overlapping(
        self,
        clauses: List[Clause],
        doc: SpacyDoc,
        tokens: List[Token]
    ) -> List[Clause]:
        """Make clause spans non-overlapping by trimming at separators.

        Tier 1 Minimal Fix: Handle overlapping spans from dependency subtrees
        by trimming them at strong punctuation marks and ensuring clean boundaries.

        Algorithm:
        1. For each adjacent pair of clauses (sorted by start_idx)
        2. If they overlap, find separator punctuation in overlap region
        3. Split at separator: clause_a ends before it, clause_b starts after it
        4. If no separator, split at midpoint of overlap

        This ensures:
        - No token belongs to multiple clauses
        - Zone extraction (Task 3.3) gets clean boundaries
        - Pairing logic doesn't need to handle overlaps

        Args:
            clauses: List of potentially overlapping Clause objects
            doc: spaCy Doc for reference
            tokens: List[Token] for reconstructing clause tokens

        Returns:
            List of Clause objects with non-overlapping spans
        """
        if len(clauses) <= 1:
            return clauses

        # Strong separator punctuation
        SEPARATORS = {";", ":", "—", ",", ".", "?", "!"}

        result = []
        i = 0

        while i < len(clauses):
            current_clause = clauses[i]

            # Check if this clause overlaps with the next one
            if i + 1 < len(clauses):
                next_clause = clauses[i + 1]

                # Check for overlap
                if current_clause.end_idx >= next_clause.start_idx:
                    # Overlap detected
                    overlap_start = next_clause.start_idx
                    overlap_end = min(current_clause.end_idx, next_clause.end_idx)

                    logging.debug(f"Overlap detected between clause {i} and {i+1}: "
                                 f"overlap region [{overlap_start}, {overlap_end}]")

                    # Find separator in overlap region
                    # Prefer strong separators (; : —) over weak ones (, .)
                    STRONG_SEPARATORS = {";", ":", "—"}
                    separator_idx = None

                    # First pass: look for strong separators
                    for idx in range(overlap_start, overlap_end + 1):
                        if idx < len(tokens) and tokens[idx].text in STRONG_SEPARATORS:
                            separator_idx = idx
                            break

                    # Second pass: if no strong separator, look for any separator
                    if separator_idx is None:
                        for idx in range(overlap_start, overlap_end + 1):
                            if idx < len(tokens) and tokens[idx].text in SEPARATORS:
                                separator_idx = idx
                                break

                    if separator_idx is not None:
                        # Split at separator: clause_a gets tokens up to (not including) separator
                        # clause_b gets tokens after separator
                        new_current_end = separator_idx - 1
                        new_next_start = separator_idx + 1

                        logging.debug(f"Splitting at separator '{tokens[separator_idx].text}' "
                                     f"(idx {separator_idx})")
                    else:
                        # No separator found, split at midpoint
                        midpoint = (overlap_start + overlap_end) // 2
                        new_current_end = midpoint
                        new_next_start = midpoint + 1

                        logging.debug(f"No separator found, splitting at midpoint {midpoint}")

                    # Rebuild current clause with new end
                    if new_current_end >= current_clause.start_idx:
                        new_current_tokens = tokens[current_clause.start_idx:new_current_end + 1]
                        current_clause = Clause(
                            tokens=new_current_tokens,
                            start_idx=current_clause.start_idx,
                            end_idx=new_current_end,
                            clause_type=current_clause.clause_type,
                            head_idx=current_clause.head_idx
                        )

                    # Rebuild next clause with new start
                    if new_next_start <= next_clause.end_idx:
                        new_next_tokens = tokens[new_next_start:next_clause.end_idx + 1]
                        next_clause = Clause(
                            tokens=new_next_tokens,
                            start_idx=new_next_start,
                            end_idx=next_clause.end_idx,
                            clause_type=next_clause.clause_type,
                            head_idx=next_clause.head_idx
                        )
                        # Update the next clause in the list
                        clauses[i + 1] = next_clause

            result.append(current_clause)
            i += 1

        logging.debug(f"Span normalization complete: {len(result)} non-overlapping clauses")
        return result


# Convenience function for quick clause detection
def quick_identify_clauses(doc: SpacyDoc, tokens: List[Token]) -> List[Clause]:
    """Convenience function for one-off clause identification.

    Args:
        doc: spaCy Doc with dependency parse
        tokens: List[Token] from preprocessor

    Returns:
        List of Clause objects

    Examples:
        >>> clauses = quick_identify_clauses(doc, tokens)
    """
    detector = ClauseBoundaryDetector()
    return detector.identify_clauses(doc, tokens)
