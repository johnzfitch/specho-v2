"""
Zone Extractor

Extracts terminal and initial zones from clause pairs for echo analysis.
This is Task 3.3 of the SpecHO watermark detection system.

The ZoneExtractor identifies the specific words within each clause that will be
compared for phonetic, structural, and semantic echoes:

- **Terminal Zone**: Last N content words from clause_a
- **Initial Zone**: First N content words from clause_b

Tier 1 Implementation (Simple Extraction):
- Fixed window size (N=3)
- Content words only (nouns, verbs, adjectives)
- Returns all available words if clause has < N content words
- No lemmatization or preprocessing
"""

from typing import List, Tuple
from specHO.models import Clause, ClausePair, Token


class ZoneExtractor:
    """Extract terminal and initial zones from clause pairs.

    The ZoneExtractor implements the zone extraction algorithm for the Echo Rule
    watermark detection. It identifies the specific content words that will be
    analyzed for phonetic, structural, and semantic similarity.

    **Terminal Zone**: The last N content words from the first clause (clause_a).
    These represent the "ending" of the thematic unit.

    **Initial Zone**: The first N content words from the second clause (clause_b).
    These represent the "beginning" of the next thematic unit.

    The Echo Rule watermark manifests in the similarity between these zones.

    Tier 1 Limitations (deferred to Tier 2):
    - No dynamic window sizing
    - No discourse marker exclusion
    - No lemmatization
    - No quote/parenthesis trimming

    Examples:
        >>> extractor = ZoneExtractor()
        >>> # clause_a: "The quick brown fox jumped"
        >>> # clause_b: "the lazy dog slept"
        >>> zone_a, zone_b = extractor.extract_zones(pair)
        >>> # zone_a: ["quick", "brown", "fox"] (last 3 content words)
        >>> # zone_b: ["lazy", "dog", "slept"] (first 3 content words)
    """

    def extract_zones(self, clause_pair: ClausePair) -> Tuple[List[Token], List[Token]]:
        """Extract zones from a single clause pair.

        Extracts the terminal zone from clause_a (last 3 content words) and
        the initial zone from clause_b (first 3 content words). These zones
        will be compared by the echo analyzers in Component 3.

        Args:
            clause_pair: ClausePair to analyze

        Returns:
            Tuple of (zone_a_tokens, zone_b_tokens) where each is a list of
            Token objects. Lists may contain fewer than 3 tokens if the clause
            has fewer than 3 content words.

        Examples:
            >>> extractor = ZoneExtractor()
            >>> zone_a, zone_b = extractor.extract_zones(pair)
            >>> len(zone_a)  # 3 or fewer
            >>> len(zone_b)  # 3 or fewer
        """
        zone_a = self.get_terminal_content_words(clause_pair.clause_a, n=3)
        zone_b = self.get_initial_content_words(clause_pair.clause_b, n=3)
        return zone_a, zone_b

    def get_terminal_content_words(self, clause: Clause, n: int = 3) -> List[Token]:
        """Get last N content words from clause.

        Extracts the terminal zone by selecting the last N content words from
        the clause. Content words are identified by the is_content_word field
        populated by the POSTagger (nouns, verbs, adjectives).

        Args:
            clause: Clause to extract from
            n: Number of content words to extract (default: 3)

        Returns:
            List of Token objects representing the last N content words.
            Returns fewer than N tokens if clause has fewer than N content words.
            Returns empty list if clause has no content words.

        Examples:
            >>> # clause.tokens = [The(det), cat(noun), sat(verb), quietly(adv)]
            >>> # Content words: [cat, sat]
            >>> tokens = extractor.get_terminal_content_words(clause, n=3)
            >>> len(tokens)
            2  # Only 2 content words available
            >>> [t.text for t in tokens]
            ['cat', 'sat']
        """
        content_words = [t for t in clause.tokens if t.is_content_word]
        return content_words[-n:]

    def get_initial_content_words(self, clause: Clause, n: int = 3) -> List[Token]:
        """Get first N content words from clause.

        Extracts the initial zone by selecting the first N content words from
        the clause. Content words are identified by the is_content_word field
        populated by the POSTagger (nouns, verbs, adjectives).

        Args:
            clause: Clause to extract from
            n: Number of content words to extract (default: 3)

        Returns:
            List of Token objects representing the first N content words.
            Returns fewer than N tokens if clause has fewer than N content words.
            Returns empty list if clause has no content words.

        Examples:
            >>> # clause.tokens = [However(adv), the(det), dog(noun), ran(verb), quickly(adv)]
            >>> # Content words: [dog, ran]
            >>> tokens = extractor.get_initial_content_words(clause, n=3)
            >>> len(tokens)
            2  # Only 2 content words available
            >>> [t.text for t in tokens]
            ['dog', 'ran']
        """
        content_words = [t for t in clause.tokens if t.is_content_word]
        return content_words[:n]


# Convenience function for quick zone extraction
def quick_extract_zones(clause_pair: ClausePair) -> Tuple[List[Token], List[Token]]:
    """Convenience function for one-off zone extraction.

    Args:
        clause_pair: ClausePair to extract zones from

    Returns:
        Tuple of (zone_a_tokens, zone_b_tokens)

    Examples:
        >>> zone_a, zone_b = quick_extract_zones(pair)
    """
    extractor = ZoneExtractor()
    return extractor.extract_zones(clause_pair)
