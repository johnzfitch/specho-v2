"""
Pair Rules Engine

Identifies thematic clause pairs using three rule types for echo analysis.
This is Task 3.2 of the SpecHO watermark detection system.

The PairRulesEngine applies three distinct pairing rules to identify "thematically related"
clause pairs that should be analyzed for echoes:

- **Rule A (Punctuation)**: Pairs separated by semicolon or em dash
- **Rule B (Conjunction)**: Pairs connected by coordinating conjunctions
- **Rule C (Transition)**: Pairs where second clause begins with transitional phrase

Tier 1 Implementation (Simple Rules):
- Rule A: Only semicolon (;) and em dash (—)
- Rule B: Only "but", "and", "or"
- Rule C: Only "However,", "Therefore,", "Thus," (case-sensitive, with comma)
- Simple deduplication by clause indices
- No confidence scoring or weighting
"""

import logging
import re
from typing import List, Set, Tuple, Dict, Any
from spacy.tokens import Doc as SpacyDoc

# Import our data models
from specHO.models import Clause, ClausePair, Token


class PairRulesEngine:
    """Identifies thematic clause pairs using punctuation, conjunction, and transition rules.

    The PairRulesEngine implements the three pairing rules defined in the Echo Rule
    watermark specification. These rules identify clause pairs that are likely to
    contain phonetic, structural, or semantic echoes.

    **Rule A - Punctuation Pairing**:
    Clauses separated by semicolon or em dash are thematically related. These
    punctuation marks signal a close relationship between clauses.
    - Trigger: `;` or `—` between clauses
    - Example: "The cat sat; the dog ran." → pair("cat sat", "dog ran")

    **Rule B - Conjunction Pairing**:
    Clauses joined by coordinating conjunctions show contrast or addition.
    - Trigger: "but", "and", "or" between clauses
    - Example: "The cat sat, and the dog ran." → pair("cat sat", "dog ran")

    **Rule C - Transition Pairing**:
    Clauses where the second starts with a transition word signal logical connection.
    - Trigger: "However,", "Therefore,", "Thus," at start of clause
    - Example: "The cat sat. However, the dog ran." → pair("cat sat", "dog ran")

    Tier 1 Limitations (deferred to Tier 2):
    - No confidence scoring
    - No rule weighting (all pairs treated equally)
    - No minimum clause length filtering
    - Basic deduplication only

    Attributes:
        None (stateless engine in Tier 1)

    Examples:
        >>> engine = PairRulesEngine()
        >>> # Assume we have clauses from BoundaryDetector
        >>> pairs = engine.apply_all_rules(clauses)
        >>> # Returns List[ClausePair]
    """

    # Tier 1 rule triggers (simple, no configuration)
    RULE_A_PUNCTUATION = {";", ":", "—", "–", "--"}  # Strong punctuation: semicolon, colon, em/en dash, double hyphen
    RULE_B_CONJUNCTIONS = {"but", "and", "or"}  # Case-insensitive
    RULE_C_TRANSITIONS = {"However,", "Therefore,", "Thus,"}  # Case-sensitive with comma

    def __init__(self):
        """Initialize the PairRulesEngine.

        Tier 1 implementation requires no configuration.
        """
        logging.info("Initializing PairRulesEngine (Tier 1)")

    def apply_all_rules(self, clauses: List[Clause], all_tokens: List[Token] = None, doc: SpacyDoc = None) -> List[ClausePair]:
        """Apply all three pairing rules and return deduplicated pairs.

        This is the main entry point. It applies Rule A, Rule B, and Rule C
        in sequence, then deduplicates the results with priority-based deduplication.

        Args:
            clauses: List of Clause objects from ClauseBoundaryDetector
            all_tokens: Complete list of Token objects from preprocessor (optional, for checking between-clause tokens)
            doc: spaCy Doc object (required for Rule A head-order pairing)

        Returns:
            Deduplicated list of ClausePair objects

        Examples:
            >>> engine = PairRulesEngine()
            >>> clauses = detector.identify_clauses(doc, tokens)
            >>> pairs = engine.apply_all_rules(clauses, tokens, doc)
            >>> len(pairs)  # Number of unique pairs
        """
        if not clauses or len(clauses) < 2:
            logging.warning("Insufficient clauses for pairing (need at least 2)")
            return []

        logging.debug(f"Applying pairing rules to {len(clauses)} clauses")

        # Apply each rule
        rule_a_pairs = self.apply_rule_a(clauses, all_tokens, doc)
        rule_b_pairs = self.apply_rule_b(clauses, all_tokens)
        rule_c_pairs = self.apply_rule_c(clauses)

        # Priority-based deduplication (Rule A > Rule B > Rule C)
        deduplicated_pairs = self._deduplicate_pairs_with_priority(rule_a_pairs, rule_b_pairs, rule_c_pairs)

        logging.info(f"Generated {len(deduplicated_pairs)} unique clause pairs "
                    f"(Rule A: {len(rule_a_pairs)}, Rule B: {len(rule_b_pairs)}, Rule C: {len(rule_c_pairs)})")

        return deduplicated_pairs

    def apply_rule_a(self, clauses: List[Clause], all_tokens: List[Token] = None, doc: SpacyDoc = None) -> List[ClausePair]:
        """Apply Rule A: Pair clauses separated by strong punctuation using head-order logic.

        Rule A identifies clause pairs separated by strong punctuation marks
        that indicate thematic relationship. This implementation uses clause head positions
        instead of token spans to handle dependency parse quirks.

        Tier 1 triggers: `;` `:` `—` `–` `--` (strong punctuation)

        Algorithm:
        1. Sort clauses by their head token position
        2. For each adjacent pair of clause heads
        3. Scan tokens between heads for strong punctuation
        4. Check dependency fallback (punct attached to either head)
        5. If strong punct found, create ClausePair

        Args:
            clauses: List of Clause objects
            all_tokens: Complete list of Token objects (for text scanning)
            doc: spaCy Doc object (for dependency fallback)

        Returns:
            List of ClausePair objects from Rule A

        Examples:
            >>> # "The cat sat; the dog ran."
            >>> # Two clauses separated by semicolon
            >>> pairs = engine.apply_rule_a(clauses, tokens, doc)
            >>> len(pairs)
            1
        """
        if doc is None:
            logging.warning("Rule A requires doc parameter for head-order pairing. Skipping.")
            return []

        pairs = []

        # Get clauses with head metadata, sorted by head position
        anchors_in_order = self._get_anchors_in_order(clauses, doc)

        # Generate adjacent pairs based on head order
        adjacent_pairs = self._get_adjacent_anchor_pairs(anchors_in_order)

        # Check each pair for strong punctuation between heads
        for pair_info in adjacent_pairs:
            clause_a = pair_info["clause_a"]
            clause_b = pair_info["clause_b"]
            head_a_idx = pair_info["head_a_idx"]
            head_b_idx = pair_info["head_b_idx"]

            if self._has_strong_punct_between(doc, head_a_idx, head_b_idx, all_tokens):
                # Ensure clause_a comes before clause_b in document order
                if clause_a.start_idx > clause_b.start_idx:
                    clause_a, clause_b = clause_b, clause_a

                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="punctuation"
                )
                pairs.append(pair)
                logging.debug(f"Rule A match: clause heads at {head_a_idx} <--> {head_b_idx}")

        logging.debug(f"Rule A generated {len(pairs)} pairs")
        return pairs

    def apply_rule_b(self, clauses: List[Clause], all_tokens: List[Token] = None) -> List[ClausePair]:
        """Apply Rule B: Pair clauses connected by coordinating conjunction.

        Rule B identifies clause pairs joined by coordinating conjunctions
        (and, but, or) that signal addition, contrast, or choice.

        Tier 1 triggers: "but", "and", "or" (case-insensitive)

        Algorithm:
        1. For each adjacent clause pair (i, i+1)
        2. Check if conjunction word appears between them
        3. If yes, create ClausePair

        Args:
            clauses: List of Clause objects sorted by position
            all_tokens: Complete list of Token objects (optional, for checking between-clause tokens)

        Returns:
            List of ClausePair objects from Rule B

        Examples:
            >>> # "The cat sat, and the dog ran."
            >>> # Two clauses connected by "and"
            >>> pairs = engine.apply_rule_b(clauses, tokens)
            >>> len(pairs)
            1
        """
        pairs = []

        for i in range(len(clauses) - 1):
            clause_a = clauses[i]
            clause_b = clauses[i + 1]

            # Check for conjunction between clauses
            if self._has_conjunction_trigger(clause_a, clause_b, all_tokens):
                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="conjunction"
                )
                pairs.append(pair)
                logging.debug(f"Rule B match: clause {i} <--> clause {i+1}")

        logging.debug(f"Rule B generated {len(pairs)} pairs")
        return pairs

    def apply_rule_c(self, clauses: List[Clause]) -> List[ClausePair]:
        """Apply Rule C: Pair clauses where second begins with transition word.

        Rule C identifies clause pairs where the second clause begins with
        a transitional phrase that signals logical relationship.

        Tier 1 triggers: "However,", "Therefore,", "Thus," (exact match with comma)

        Algorithm:
        1. For each clause starting at position 1
        2. Check if it begins with a transition trigger
        3. If yes, pair with previous clause

        Args:
            clauses: List of Clause objects sorted by position

        Returns:
            List of ClausePair objects from Rule C

        Examples:
            >>> # "The cat sat. However, the dog ran."
            >>> # Second clause starts with "However,"
            >>> pairs = engine.apply_rule_c(clauses)
            >>> len(pairs)
            1
        """
        pairs = []

        for i in range(1, len(clauses)):
            clause_a = clauses[i - 1]
            clause_b = clauses[i]

            # Check if clause_b starts with transition word
            if self._has_transition_trigger(clause_b):
                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="transition"
                )
                pairs.append(pair)
                logging.debug(f"Rule C match: clause {i-1} <--> clause {i}")

        logging.debug(f"Rule C generated {len(pairs)} pairs")
        return pairs

    def _get_anchors_in_order(self, clauses: List[Clause], doc: SpacyDoc) -> List[Dict[str, Any]]:
        """Sort clauses by their head token position.

        Args:
            clauses: List of Clause objects
            doc: spaCy Doc object

        Returns:
            List of dicts with clause metadata, sorted by head position
            Format: [{"clause": Clause, "head_idx": int, "head_token": SpacyToken}, ...]
        """
        anchors = []
        for clause in clauses:
            anchors.append({
                "clause": clause,
                "head_idx": clause.head_idx,
                "head_token": doc[clause.head_idx]
            })

        # Sort by head position
        anchors.sort(key=lambda x: x["head_idx"])
        return anchors

    def _get_adjacent_anchor_pairs(self, anchors_in_order: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate adjacent pairs of clause heads based on sorted head order.

        Args:
            anchors_in_order: List of anchor dicts sorted by head position

        Returns:
            List of dicts representing adjacent pairs
            Format: [{"clause_a": Clause, "clause_b": Clause, "head_a_idx": int, "head_b_idx": int}, ...]
        """
        pairs = []
        for i in range(len(anchors_in_order) - 1):
            a = anchors_in_order[i]
            b = anchors_in_order[i + 1]
            pairs.append({
                "clause_a": a["clause"],
                "clause_b": b["clause"],
                "head_a_idx": a["head_idx"],
                "head_b_idx": b["head_idx"]
            })
        return pairs

    def _has_strong_punct_between(self, doc: SpacyDoc, head_a_idx: int, head_b_idx: int, all_tokens: List[Token] = None) -> bool:
        """Check if strong punctuation exists between two clause heads.

        Scans the token window between head positions for strong punctuation.
        Includes dependency fallback for punct tokens attached to either head.

        Args:
            doc: spaCy Doc object
            head_a_idx: Token index of first clause head
            head_b_idx: Token index of second clause head
            all_tokens: Complete list of Token objects (for text scanning)

        Returns:
            True if strong punctuation found between heads
        """
        # Scan token window between heads
        for idx in range(head_a_idx + 1, head_b_idx):
            if idx < len(doc):
                token_text = doc[idx].text
                if token_text in self.RULE_A_PUNCTUATION:
                    logging.debug(f"Strong punct '{token_text}' found at token {idx} between heads {head_a_idx} and {head_b_idx}")
                    return True

        # Dependency fallback: check for punct children attached to either head
        head_a = doc[head_a_idx]
        head_b = doc[head_b_idx]

        for child in head_a.children:
            if child.dep_ == "punct" and child.text in self.RULE_A_PUNCTUATION:
                logging.debug(f"Strong punct '{child.text}' attached to head_a at {head_a_idx}")
                return True

        for child in head_b.children:
            if child.dep_ == "punct" and child.text in self.RULE_A_PUNCTUATION:
                logging.debug(f"Strong punct '{child.text}' attached to head_b at {head_b_idx}")
                return True

        return False

    def _has_conjunction_trigger(self, clause_a: Clause, clause_b: Clause, all_tokens: List[Token] = None) -> bool:
        """Check if coordinating conjunction appears between two clauses.

        Looks for "but", "and", "or" (case-insensitive) in:
        1. Tokens within both clauses
        2. Tokens between clauses

        Args:
            clause_a: First clause
            clause_b: Second clause
            all_tokens: Complete list of Token objects (for checking between-clause tokens)

        Returns:
            True if conjunction trigger found
        """
        # Check tokens in both clauses
        for token in clause_a.tokens + clause_b.tokens:
            if token.text.lower() in self.RULE_B_CONJUNCTIONS:
                return True

        # Check tokens between clauses (if any)
        if all_tokens and clause_b.start_idx > clause_a.end_idx + 1:
            for idx in range(clause_a.end_idx + 1, clause_b.start_idx):
                if idx < len(all_tokens) and all_tokens[idx].text.lower() in self.RULE_B_CONJUNCTIONS:
                    return True

        return False

    def _has_transition_trigger(self, clause: Clause) -> bool:
        """Check if clause begins with a transition word.

        Looks for "However,", "Therefore,", "Thus," (case-sensitive with comma)
        at the start of the clause.

        Args:
            clause: Clause to check

        Returns:
            True if clause starts with transition trigger
        """
        if not clause.tokens or len(clause.tokens) < 2:
            return False

        # Check first token + second token for "However," pattern
        # First token should be transition word, second should be comma
        first_token = clause.tokens[0].text

        # Option 1: "However," as single token
        if first_token in self.RULE_C_TRANSITIONS:
            return True

        # Option 2: "However" + "," as two tokens
        if len(clause.tokens) >= 2:
            first_word = first_token
            second_token = clause.tokens[1].text
            combined = f"{first_word}{second_token}"
            if combined in self.RULE_C_TRANSITIONS:
                return True

        return False

    def _deduplicate_pairs_with_priority(
        self,
        rule_a_pairs: List[ClausePair],
        rule_b_pairs: List[ClausePair],
        rule_c_pairs: List[ClausePair]
    ) -> List[ClausePair]:
        """Remove duplicate clause pairs with priority: Rule A > Rule B > Rule C.

        If the same clause pair is identified by multiple rules, keep only
        the pair from the highest-priority rule.

        Priority order:
        1. Rule A (punctuation) - strongest signal
        2. Rule B (conjunction) - medium signal
        3. Rule C (transition) - weakest signal

        Args:
            rule_a_pairs: Pairs from Rule A
            rule_b_pairs: Pairs from Rule B
            rule_c_pairs: Pairs from Rule C

        Returns:
            Deduplicated list of ClausePair objects

        Examples:
            >>> # If Rule A and Rule B both identify same pair
            >>> # Only the Rule A version is kept
        """
        seen = {}  # key -> (pair, priority)
        result = []

        # Priority values (lower = higher priority)
        PRIORITY = {"punctuation": 1, "conjunction": 2, "transition": 3}

        # Process all pairs
        all_pairs = [
            (pair, PRIORITY["punctuation"]) for pair in rule_a_pairs
        ] + [
            (pair, PRIORITY["conjunction"]) for pair in rule_b_pairs
        ] + [
            (pair, PRIORITY["transition"]) for pair in rule_c_pairs
        ]

        for pair, priority in all_pairs:
            # Create key from clause indices
            key = (
                pair.clause_a.start_idx,
                pair.clause_a.end_idx,
                pair.clause_b.start_idx,
                pair.clause_b.end_idx
            )

            if key not in seen:
                # New pair
                seen[key] = (pair, priority)
            else:
                # Duplicate found - keep higher priority (lower number)
                existing_pair, existing_priority = seen[key]
                if priority < existing_priority:
                    seen[key] = (pair, priority)
                    logging.debug(f"Replaced duplicate pair {key} with higher-priority rule ({pair.pair_type})")
                else:
                    logging.debug(f"Duplicate pair {key} kept existing higher-priority rule ({existing_pair.pair_type})")

        # Extract pairs from seen dict
        result = [pair for pair, priority in seen.values()]
        return result

    def _deduplicate_pairs(self, pairs: List[ClausePair]) -> List[ClausePair]:
        """Remove duplicate clause pairs based on clause indices.

        Tier 1 deduplication: Simple set-based deduplication using
        (start_idx_a, end_idx_a, start_idx_b, end_idx_b) as key.

        Note: This method is kept for backward compatibility but is superseded
        by _deduplicate_pairs_with_priority in the main pipeline.

        Args:
            pairs: List of ClausePair objects (may contain duplicates)

        Returns:
            Deduplicated list of ClausePair objects

        Examples:
            >>> # If Rule A and Rule B both identify same pair
            >>> # Only one copy is returned
        """
        seen = set()
        deduplicated = []

        for pair in pairs:
            # Create key from clause indices
            key = (
                pair.clause_a.start_idx,
                pair.clause_a.end_idx,
                pair.clause_b.start_idx,
                pair.clause_b.end_idx
            )

            if key not in seen:
                seen.add(key)
                deduplicated.append(pair)
            else:
                logging.debug(f"Duplicate pair removed: {key}")

        return deduplicated


# Convenience function for quick pairing
def quick_apply_rules(clauses: List[Clause], all_tokens: List[Token] = None, doc: SpacyDoc = None) -> List[ClausePair]:
    """Convenience function for one-off clause pairing.

    Args:
        clauses: List of Clause objects
        all_tokens: Complete list of Token objects (optional)
        doc: spaCy Doc object (required for Rule A)

    Returns:
        List of ClausePair objects

    Examples:
        >>> pairs = quick_apply_rules(clauses, tokens, doc)
    """
    engine = PairRulesEngine()
    return engine.apply_all_rules(clauses, all_tokens, doc)
