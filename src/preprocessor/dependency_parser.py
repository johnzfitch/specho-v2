"""Dependency Parser for SpecHO watermark detection system.

This module provides dependency parsing using spaCy, extracting syntactic
relationships between words. The dependency tree is crucial for identifying
clause boundaries, which are the fundamental units for watermark detection.

Tier: 1 (MVP)
Task: 2.3
Dependencies: Task 1.1 (models.py), spacy
"""

import logging
from typing import List, Tuple
import spacy
from spacy.tokens import Doc as SpacyDoc, Token as SpacyToken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DependencyParser:
    """Dependency parser that extracts syntactic structures from text.

    Uses spaCy's dependency parser to build syntactic trees showing how
    words relate to each other grammatically. These dependency relations
    (ROOT, conj, advcl, ccomp, etc.) are used by the Clause Identifier
    to determine where clauses begin and end.

    The dependency tree reveals sentence structure:
    - ROOT: The main verb of a sentence
    - conj: Coordinated clauses (connected by and, but, or)
    - advcl: Adverbial clauses (subordinate clauses modifying verbs)
    - ccomp: Clausal complements (e.g., "I think [that he left]")

    Tier 1 Implementation:
    - Basic spaCy dependency parsing
    - Returns full spaCy Doc object for downstream use
    - Simple clause boundary detection helper
    - No complex edge case handling

    Attributes:
        nlp: spaCy language model with dependency parser enabled
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the DependencyParser with a spaCy model.

        Args:
            model_name: Name of spaCy model to load (default: 'en_core_web_sm')

        Raises:
            OSError: If spaCy model is not installed

        Examples:
            >>> parser = DependencyParser()
            >>> parser = DependencyParser("en_core_web_md")  # Use medium model
        """
        try:
            # Load spaCy with parser enabled (disable NER for speed)
            self.nlp = spacy.load(model_name, disable=["ner"])
            logging.info(f"Loaded spaCy model for dependency parsing: {model_name}")
        except OSError as e:
            logging.error(
                f"Failed to load spaCy model '{model_name}'. "
                f"Install it with: python -m spacy download {model_name}"
            )
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from e

    def parse(self, text: str) -> SpacyDoc:
        """Parse text and return spaCy Doc with dependency tree.

        Processes text through spaCy's NLP pipeline, including tokenization,
        POS tagging, and dependency parsing. Returns the full spaCy Doc object
        which contains the dependency tree structure.

        The Doc object provides access to:
        - Tokens and their attributes (text, POS, lemma, etc.)
        - Dependency relations (token.dep_, token.head)
        - Sentence boundaries
        - Syntactic tree structure

        Args:
            text: Raw text string to parse

        Returns:
            spacy.tokens.Doc object with full linguistic analysis

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("The cat sat on the mat.")
            >>> doc.text
            'The cat sat on the mat.'
            >>> len(doc)
            7
            >>> [token.dep_ for token in doc]
            ['det', 'nsubj', 'ROOT', 'prep', 'det', 'pobj', 'punct']
        """
        if not text or not text.strip():
            logging.warning("Received empty text for dependency parsing")
            return self.nlp("")

        doc = self.nlp(text)
        logging.debug(f"Parsed text with {len(doc)} tokens, {len(list(doc.sents))} sentences")
        return doc

    def get_clause_boundaries(self, doc: SpacyDoc) -> List[Tuple[int, int]]:
        """Extract basic clause boundaries from a dependency parse.

        Identifies potential clause boundaries using simple heuristics based on
        dependency labels. This is a helper method for the ClauseIdentifier
        component (Task 3.1), which will implement more sophisticated clause
        detection.

        Tier 1 heuristics:
        - ROOT verbs indicate main clauses
        - conj (conjunction) indicates coordinated clauses
        - advcl (adverbial clause) indicates subordinate clauses
        - ccomp (clausal complement) indicates embedded clauses

        Returns clause boundaries as (start_idx, end_idx) tuples where indices
        are token positions in the document.

        Args:
            doc: spaCy Doc object with dependency parse

        Returns:
            List of (start_idx, end_idx) tuples marking potential clause boundaries

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("The cat sat, and the dog ran.")
            >>> boundaries = parser.get_clause_boundaries(doc)
            >>> len(boundaries) >= 2  # At least two clauses
            True

        Notes:
            This is a simplified implementation. The full ClauseIdentifier
            (Task 3.1) will use more sophisticated algorithms and handle
            edge cases properly.
        """
        if not doc:
            logging.warning("Received empty doc for clause boundary detection")
            return []

        boundaries = []

        # Clause-indicating dependency labels (Tier 1 simple set)
        clause_labels = {"ROOT", "conj", "advcl", "ccomp"}

        for sent in doc.sents:
            # Find tokens with clause-indicating dependencies
            clause_heads = []

            for token in sent:
                if token.dep_ in clause_labels:
                    clause_heads.append(token)

            # If no clause heads found, treat entire sentence as one clause
            if not clause_heads:
                boundaries.append((sent.start, sent.end))
                continue

            # Create boundaries around each clause head
            # This is a simple approximation - Task 3.1 will refine this
            for i, head in enumerate(clause_heads):
                if i == 0:
                    # First clause starts at sentence beginning
                    start = sent.start
                else:
                    # Subsequent clauses start after previous head
                    start = clause_heads[i - 1].i + 1

                if i == len(clause_heads) - 1:
                    # Last clause ends at sentence end
                    end = sent.end
                else:
                    # Clause ends at next head
                    end = clause_heads[i + 1].i

                # Only add non-empty boundaries
                if end > start:
                    boundaries.append((start, end))

        logging.debug(f"Detected {len(boundaries)} potential clause boundaries")
        return boundaries

    def get_dependency_tree(self, doc: SpacyDoc) -> str:
        """Get a string representation of the dependency tree.

        Utility method for debugging and visualization. Returns a formatted
        string showing the dependency structure of the parsed text.

        Args:
            doc: spaCy Doc object with dependency parse

        Returns:
            String representation of dependency tree

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("The cat sat.")
            >>> tree = parser.get_dependency_tree(doc)
            >>> "ROOT" in tree
            True
        """
        if not doc:
            return ""

        lines = []
        for sent in doc.sents:
            lines.append(f"Sentence: {sent.text}")
            for token in sent:
                head_text = token.head.text if token.head != token else "ROOT"
                lines.append(
                    f"  {token.text:12s} | POS: {token.pos_:6s} | "
                    f"DEP: {token.dep_:10s} | HEAD: {head_text}"
                )
            lines.append("")

        return "\n".join(lines)

    def find_root_verbs(self, doc: SpacyDoc) -> List[SpacyToken]:
        """Find ROOT verbs in the document.

        ROOT verbs are the main verbs of sentences and serve as anchors
        for main clauses. This is a helper for clause identification.

        Args:
            doc: spaCy Doc object with dependency parse

        Returns:
            List of spaCy tokens that are ROOT verbs

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("The cat sat. The dog ran.")
            >>> roots = parser.find_root_verbs(doc)
            >>> len(roots)
            2
            >>> [token.text for token in roots]
            ['sat', 'ran']
        """
        roots = []
        for token in doc:
            if token.dep_ == "ROOT":
                roots.append(token)

        return roots

    def find_coordinated_clauses(self, doc: SpacyDoc) -> List[Tuple[SpacyToken, SpacyToken]]:
        """Find pairs of coordinated clauses (connected by conjunctions).

        Identifies clauses connected by coordinating conjunctions (and, but, or).
        Returns pairs of (head1, head2) where head2 is connected to head1 via
        a 'conj' dependency.

        Args:
            doc: spaCy Doc object with dependency parse

        Returns:
            List of (head1, head2) token pairs for coordinated clauses

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("The cat sat, and the dog ran.")
            >>> pairs = parser.find_coordinated_clauses(doc)
            >>> len(pairs) >= 1
            True
        """
        pairs = []

        for token in doc:
            if token.dep_ == "conj":
                # token is coordinated with token.head
                pairs.append((token.head, token))

        return pairs

    def find_subordinate_clauses(self, doc: SpacyDoc) -> List[Tuple[SpacyToken, SpacyToken]]:
        """Find subordinate clauses attached to main clauses.

        Identifies subordinate clauses (advcl, ccomp, etc.) and returns
        pairs of (main_verb, subordinate_verb).

        Args:
            doc: spaCy Doc object with dependency parse

        Returns:
            List of (main_verb, subordinate_verb) token pairs

        Examples:
            >>> parser = DependencyParser()
            >>> doc = parser.parse("She said that he left.")
            >>> pairs = parser.find_subordinate_clauses(doc)
            >>> len(pairs) >= 1
            True
        """
        subordinate_labels = {"advcl", "ccomp", "xcomp", "acl"}
        pairs = []

        for token in doc:
            if token.dep_ in subordinate_labels:
                # token is a subordinate clause head attached to token.head
                pairs.append((token.head, token))

        return pairs


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_parse(text: str, model_name: str = "en_core_web_sm") -> SpacyDoc:
    """Convenience function for one-off parsing without creating DependencyParser instance.

    Args:
        text: Text to parse
        model_name: spaCy model to use

    Returns:
        spaCy Doc object with dependency parse

    Examples:
        >>> doc = quick_parse("The cat sat.")
        >>> len(doc)
        4
    """
    parser = DependencyParser(model_name)
    return parser.parse(text)
