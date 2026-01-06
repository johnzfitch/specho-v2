"""
Structural Echo Analyzer for SpecHO Watermark Detector.

Analyzes structural similarity between clause zones based on POS patterns and syllable counts.
Part of Component 3: Echo Engine (Task 4.2).

Tier 1: Simple exact matching of POS patterns and syllable count comparison.
"""

from typing import List
from specHO.models import Token


class StructuralEchoAnalyzer:
    """
    Analyzes structural echoes between clause zones.

    Tier 1 algorithm:
    - POS pattern comparison: exact match of POS tag sequences
    - Syllable count similarity: normalized absolute difference
    - Combined score: pattern_sim * 0.5 + syllable_sim * 0.5
    """

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Calculate structural similarity between two zones.

        Args:
            zone_a: First zone (terminal content words from clause A)
            zone_b: Second zone (initial content words from clause B)

        Returns:
            Float in [0, 1] representing structural similarity.
            Returns 0.0 if either zone is empty.

        Algorithm:
            1. Extract POS patterns from both zones
            2. Calculate POS pattern similarity (exact match)
            3. Extract syllable counts from both zones
            4. Calculate syllable count similarity
            5. Combine: 0.5 * pattern_sim + 0.5 * syllable_sim
        """
        if not zone_a or not zone_b:
            return 0.0

        # Calculate individual similarity scores
        pos_similarity = self.compare_pos_patterns(zone_a, zone_b)
        syllable_similarity = self.compare_syllable_counts(zone_a, zone_b)

        # Combine with equal weighting (Tier 1)
        combined_score = (pos_similarity * 0.5) + (syllable_similarity * 0.5)

        # Ensure output is in [0, 1] range
        return max(0.0, min(1.0, combined_score))

    def compare_pos_patterns(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Compare POS tag patterns between two zones.

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            Float in [0, 1]. Returns 1.0 for exact match, 0.0 for no match.

        Tier 1 algorithm:
            - Extract POS tag sequences
            - Exact match comparison (all-or-nothing)
            - Returns 1.0 if patterns match exactly, 0.0 otherwise
        """
        if not zone_a or not zone_b:
            return 0.0

        # Extract POS tag sequences
        pos_pattern_a = [token.pos_tag for token in zone_a if token.pos_tag is not None]
        pos_pattern_b = [token.pos_tag for token in zone_b if token.pos_tag is not None]

        # Handle case where no valid POS tags
        if not pos_pattern_a or not pos_pattern_b:
            return 0.0

        # Tier 1: Exact match only
        # Convert to tuples for comparison
        if tuple(pos_pattern_a) == tuple(pos_pattern_b):
            return 1.0
        else:
            return 0.0

    def compare_syllable_counts(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Compare syllable count similarity between two zones.

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            Float in [0, 1] representing syllable count similarity.

        Tier 1 algorithm:
            - Sum total syllables in each zone
            - Calculate normalized similarity: 1 - (abs_diff / max_count)
            - Returns 0.0 if either zone has no syllable data
        """
        if not zone_a or not zone_b:
            return 0.0

        # Extract syllable counts
        syllables_a = [token.syllable_count for token in zone_a
                      if token.syllable_count is not None]
        syllables_b = [token.syllable_count for token in zone_b
                      if token.syllable_count is not None]

        # Handle case where no valid syllable counts
        if not syllables_a or not syllables_b:
            return 0.0

        # Calculate total syllables in each zone
        total_a = sum(syllables_a)
        total_b = sum(syllables_b)

        # Edge case: both zones have 0 syllables (shouldn't happen, but handle gracefully)
        if total_a == 0 and total_b == 0:
            return 1.0

        # Calculate similarity: 1 - (absolute_difference / max_count)
        max_count = max(total_a, total_b)
        if max_count == 0:
            return 0.0

        abs_diff = abs(total_a - total_b)
        similarity = 1.0 - (abs_diff / max_count)

        # Ensure output is in [0, 1] range
        return max(0.0, min(1.0, similarity))


def quick_structural_analysis(zone_a: List[Token], zone_b: List[Token]) -> float:
    """
    Convenience function for quick structural analysis.

    Args:
        zone_a: First zone
        zone_b: Second zone

    Returns:
        Structural similarity score in [0, 1]
    """
    analyzer = StructuralEchoAnalyzer()
    return analyzer.analyze(zone_a, zone_b)
