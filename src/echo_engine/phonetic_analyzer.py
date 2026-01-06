"""
Task 4.1: PhoneticEchoAnalyzer

Analyzes phonetic similarity between clause zones using Levenshtein distance
on ARPAbet phonetic representations.

Tier 1 implementation - simple Levenshtein-based phonetic comparison.

Component: Echo Engine
Tier: 1 (MVP)
"""

from typing import List
from specHO.models import Token
import Levenshtein


class PhoneticEchoAnalyzer:
    """
    Analyzes phonetic echoes between clause zones.

    Uses Levenshtein distance on ARPAbet phonetic transcriptions to measure
    phonetic similarity. Performs pairwise comparison between zone tokens
    and selects the best match.

    Tier 1: Simple algorithm with no optimization or caching.
    """

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Calculate phonetic similarity between two zones.

        Args:
            zone_a: List of Token objects from clause_a terminal zone
            zone_b: List of Token objects from clause_b initial zone

        Returns:
            float: Phonetic similarity score in [0,1] range
                  0 = no similarity, 1 = identical phonetics
                  0 if either zone is empty

        Algorithm:
            1. For each token in zone_a, compare with all tokens in zone_b
            2. Find the best (minimum distance) match for each token
            3. Average the normalized similarity scores
            4. Normalize distance to [0,1]: similarity = 1 - (distance / max_length)

        Example:
            >>> analyzer = PhoneticEchoAnalyzer()
            >>> # Tokens with phonetic field populated by PhoneticTranscriber
            >>> zone_a = [Token("cat", phonetic="K AE1 T", ...)]
            >>> zone_b = [Token("bat", phonetic="B AE1 T", ...)]
            >>> similarity = analyzer.analyze(zone_a, zone_b)
            >>> # Returns ~0.67 (2 of 3 phonemes match)
        """
        # Edge case: return 0 if either zone empty
        if not zone_a or not zone_b:
            return 0.0

        # Collect all pairwise similarity scores
        similarities = []

        for token_a in zone_a:
            # Skip tokens without phonetic transcription
            if token_a.phonetic is None:
                continue

            # Find best match for this token in zone_b
            best_similarity = 0.0

            for token_b in zone_b:
                # Skip tokens without phonetic transcription
                if token_b.phonetic is None:
                    continue

                # Calculate phonetic distance
                similarity = self.calculate_phonetic_similarity(
                    token_a.phonetic,
                    token_b.phonetic
                )

                # Track best match
                if similarity > best_similarity:
                    best_similarity = similarity

            # Add best match for this token
            if best_similarity > 0.0:
                similarities.append(best_similarity)

        # Return average similarity, or 0 if no valid comparisons
        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)

    def calculate_phonetic_similarity(self, phoneme_a: str, phoneme_b: str) -> float:
        """
        Calculate similarity between two phonetic transcriptions.

        Args:
            phoneme_a: ARPAbet phonetic string (e.g., "K AE1 T")
            phoneme_b: ARPAbet phonetic string (e.g., "B AE1 T")

        Returns:
            float: Normalized similarity in [0,1] range
                  1 - (levenshtein_distance / max_length)

        Normalization:
            - max_length = max(len(phoneme_a), len(phoneme_b))
            - Ensures similarity is in [0,1] range
            - 1.0 means identical phonetics
            - 0.0 means completely different

        Example:
            >>> analyzer = PhoneticEchoAnalyzer()
            >>> sim = analyzer.calculate_phonetic_similarity("K AE1 T", "B AE1 T")
            >>> # Levenshtein distance = 1 (K→B), max_length = 7
            >>> # similarity = 1 - (1/7) ≈ 0.857
        """
        # Handle empty strings
        if not phoneme_a or not phoneme_b:
            return 0.0

        # Calculate Levenshtein distance
        distance = Levenshtein.distance(phoneme_a, phoneme_b)

        # Normalize to [0,1] similarity
        max_length = max(len(phoneme_a), len(phoneme_b))

        if max_length == 0:
            return 1.0  # Both empty = identical

        similarity = 1.0 - (distance / max_length)

        # Clamp to [0,1] range (should already be, but be safe)
        return max(0.0, min(1.0, similarity))


def quick_phonetic_analysis(zone_a: List[Token], zone_b: List[Token]) -> float:
    """
    Convenience function for one-off phonetic analysis.

    Creates a PhoneticEchoAnalyzer instance and analyzes the zones.

    Args:
        zone_a: List of Token objects from clause_a terminal zone
        zone_b: List of Token objects from clause_b initial zone

    Returns:
        float: Phonetic similarity score in [0,1] range

    Example:
        >>> from specHO.echo_engine.phonetic_analyzer import quick_phonetic_analysis
        >>> similarity = quick_phonetic_analysis(zone_a, zone_b)
    """
    analyzer = PhoneticEchoAnalyzer()
    return analyzer.analyze(zone_a, zone_b)
