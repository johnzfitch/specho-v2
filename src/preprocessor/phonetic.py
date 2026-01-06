"""Phonetic Transcriber for SpecHO watermark detection system.

This module provides phonetic transcription using the CMU Pronouncing Dictionary,
converting words to ARPAbet representation. Phonetic transcription is essential
for detecting phonetic echoes between clause boundaries in watermarked text.

ARPAbet is a phonetic transcription code used by the CMU Pronouncing Dictionary:
- Consonants: B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH
- Vowels: AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
- Stress: 0 (no stress), 1 (primary), 2 (secondary) appended to vowels

Example: "hello" -> "HH AH0 L OW1"

Tier: 1 (MVP)
Task: 2.4
Dependencies: Task 1.1 (models.py), pronouncing library
"""

import logging
from typing import List, Optional, Dict
import pronouncing

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Token

# ============================================================================
# CONTRACTION MAPPINGS
# ============================================================================

# Mapping of contraction fragments to ARPAbet phonetics
# These are common fragments created by spaCy tokenization that don't exist
# in the CMU Pronouncing Dictionary
#
# Note: These represent the REDUCED forms heard in contractions, not the full words
# For example, "have" is "HH AE1 V" fully pronounced, but "'ve" is just "V"
CONTRACTION_PHONETICS: Dict[str, str] = {
    # Negative contractions
    "n't": "AH0 N T",  # not (as in "don't", "can't", "won't") - reduced from "N AA1 T"

    # Pronoun/verb contractions (reduced forms)
    "'m": "AH0 M",     # am (as in "I'm") - already in CMU dict
    "'re": "ER0",      # are (as in "you're", "they're") - reduced from "AA1 R"
    "'ve": "V",        # have (as in "I've", "they've") - reduced from "HH AE1 V"
    "'ll": "AH0 L",    # will (as in "I'll", "you'll") - reduced from "W IH1 L"
    "'d": "D",         # would/had (as in "I'd", "you'd") - just the final consonant

    # Possessive/is contractions (already in CMU dict, but included for completeness)
    "'s": "Z",         # is/possessive (as in "it's", "John's") - often realized as [z]

    # Less common
    "'t": "T",         # it (as in "'twas")
    "n'": "AH0 N",     # and (as in "rock 'n' roll")
}


class PhoneticTranscriber:
    """Phonetic transcriber that converts words to ARPAbet representation.

    Uses the CMU Pronouncing Dictionary (via the pronouncing library) to
    convert English words into phonetic representations. This phonetic form
    is used later by the PhoneticEchoAnalyzer (Task 4.1) to detect sound
    similarities between clause boundaries.

    The transcriber handles:
    - Dictionary lookups for known words
    - OOV (out-of-vocabulary) word fallbacks
    - Syllable counting
    - Stress pattern extraction

    Tier 1 Implementation:
    - Simple dictionary lookup using pronouncing library
    - Basic fallback for OOV words (return uppercase text)
    - Syllable counting via pronouncing.syllable_count()
    - No complex G2P (grapheme-to-phoneme) models

    Attributes:
        None (uses pronouncing library directly)
    """

    def __init__(self):
        """Initialize the PhoneticTranscriber.

        No setup required for Tier 1 - uses pronouncing library directly.
        Contraction mappings are defined as a class constant.

        Examples:
            >>> transcriber = PhoneticTranscriber()
        """
        logging.info("Initialized PhoneticTranscriber (Tier 1 - CMU Dict + contractions)")

    def transcribe(self, word: str) -> str:
        """Convert a word to ARPAbet phonetic representation.

        Uses the CMU Pronouncing Dictionary to look up the phonetic form of
        a word. If the word has multiple pronunciations, returns the first one.

        BUGFIX (2025-10-19): Added contraction handling
        - Checks CONTRACTION_PHONETICS mapping before CMU dict lookup
        - Handles spaCy tokenization artifacts: "n't", "'re", "'ve", "'ll", etc.
        - Prevents invalid phonetics like "N'T" for contraction fragments

        Tier 1 Lookup Strategy:
        1. Check contraction mapping (for spaCy artifacts)
        2. Check CMU Pronouncing Dictionary
        3. Fallback: return uppercase text for true OOV words

        Args:
            word: English word to transcribe (any casing)

        Returns:
            ARPAbet string (e.g., "HH AH0 L OW1") or uppercase word if OOV

        Examples:
            >>> transcriber = PhoneticTranscriber()
            >>> transcriber.transcribe("hello")
            'HH AH0 L OW1'
            >>> transcriber.transcribe("cat")
            'K AE1 T'
            >>> transcriber.transcribe("n't")  # Contraction fragment
            'AH0 N T'
            >>> transcriber.transcribe("'re")  # Another fragment
            'AA1 R'
            >>> transcriber.transcribe("xyz123")  # True OOV word
            'XYZ123'

        Notes:
            The pronouncing library returns a list of possible pronunciations.
            We use the first one for Tier 1 simplicity. Tier 2 may add
            context-aware pronunciation selection.
        """
        if not word or not word.strip():
            logging.debug("Empty word provided for transcription")
            return ""

        # Clean the word (remove punctuation, convert to lowercase for lookup)
        cleaned_word = word.strip().lower()

        # Strip trailing punctuation (but preserve apostrophes for now)
        for punct in ".,!?;:\"":
            cleaned_word = cleaned_word.rstrip(punct)

        if not cleaned_word:
            return ""

        # BUGFIX: Check contraction mappings (after punctuation strip, preserving apostrophes)
        # This handles spaCy tokenization artifacts like "n't", "'re", etc.
        if cleaned_word in CONTRACTION_PHONETICS:
            phonetic = CONTRACTION_PHONETICS[cleaned_word]
            logging.debug(f"Contraction '{word}' -> '{phonetic}' (from mapping)")
            return phonetic

        # Look up pronunciation in CMU dict
        phones_list = pronouncing.phones_for_word(cleaned_word)

        if phones_list:
            # Return first pronunciation (Tier 1 simplicity)
            phonetic = phones_list[0]
            logging.debug(f"Transcribed '{word}' -> '{phonetic}'")
            return phonetic
        else:
            # OOV word: fallback to uppercase (Tier 1 simple strategy)
            fallback = word.upper()
            logging.debug(f"OOV word '{word}' -> fallback '{fallback}'")
            return fallback

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU Pronouncing Dictionary.

        Uses pronouncing.syllable_count() which counts vowel phonemes with
        stress markers in the ARPAbet representation.

        Args:
            word: English word to count syllables for

        Returns:
            Number of syllables (integer >= 0)

        Examples:
            >>> transcriber = PhoneticTranscriber()
            >>> transcriber.count_syllables("hello")
            2
            >>> transcriber.count_syllables("cat")
            1
            >>> transcriber.count_syllables("beautiful")
            3
            >>> transcriber.count_syllables("xyz")  # OOV
            1

        Notes:
            For OOV words, we estimate syllables by counting vowel clusters.
            This is a Tier 1 approximation. Tier 2 may use more sophisticated
            syllable counting for OOV words.
        """
        if not word or not word.strip():
            return 0

        cleaned_word = word.strip().lower()

        # Remove punctuation
        for punct in ".,!?;:\"'":
            cleaned_word = cleaned_word.replace(punct, "")

        if not cleaned_word:
            return 0

        # Try pronouncing library
        try:
            # Get syllable count using pronouncing library
            phones_list = pronouncing.phones_for_word(cleaned_word)

            if phones_list:
                # Use pronouncing's syllable counting on first pronunciation
                # Note: pronouncing.syllable_count() counts stress markers
                phonetic = phones_list[0]
                syllables = pronouncing.syllable_count(phonetic)
                logging.debug(f"Syllables for '{word}': {syllables}")
                return syllables
            else:
                # OOV word: estimate syllables by counting vowel clusters
                return self._estimate_syllables(cleaned_word)

        except Exception as e:
            logging.warning(f"Error counting syllables for '{word}': {e}")
            return self._estimate_syllables(cleaned_word)

    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllables for OOV words by counting vowel clusters.

        This is a simple heuristic fallback for words not in the CMU dictionary.
        Tier 1 implementation uses basic vowel counting. Tier 2 may improve this.

        Args:
            word: Lowercase word without punctuation

        Returns:
            Estimated syllable count (minimum 1 for non-empty words)

        Examples:
            >>> transcriber = PhoneticTranscriber()
            >>> transcriber._estimate_syllables("xyz")
            1
            >>> transcriber._estimate_syllables("aardvark")  # If OOV
            2
        """
        if not word:
            return 0

        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        # Ensure at least 1 syllable for non-empty words
        return max(1, syllable_count)

    def get_stressed_syllables(self, phonetic: str) -> List[str]:
        """Extract stressed syllables from an ARPAbet string.

        Identifies syllables with primary stress (marked with '1' in ARPAbet)
        and secondary stress (marked with '2'). This is useful for rime-based
        phonetic analysis in Tier 2.

        Args:
            phonetic: ARPAbet string (e.g., "HH AH0 L OW1")

        Returns:
            List of phonemes that carry stress markers

        Examples:
            >>> transcriber = PhoneticTranscriber()
            >>> transcriber.get_stressed_syllables("HH AH0 L OW1")
            ['OW1']
            >>> transcriber.get_stressed_syllables("B IY1 Y UW2 T IH0 F AH0 L")
            ['IY1', 'UW2']
            >>> transcriber.get_stressed_syllables("K AE1 T")
            ['AE1']

        Notes:
            Tier 1 implementation simply finds phonemes ending in '1' or '2'.
            Tier 2 may add more sophisticated stress pattern analysis.
        """
        if not phonetic:
            return []

        stressed = []
        phonemes = phonetic.split()

        for phoneme in phonemes:
            # Check if phoneme ends with stress marker (1 or 2)
            if phoneme and phoneme[-1] in ['1', '2']:
                stressed.append(phoneme)

        return stressed

    def transcribe_tokens(self, tokens: List[Token]) -> List[Token]:
        """Enrich Token objects with phonetic and syllable_count fields.

        This is the main method for pipeline integration. Takes tokens with
        text populated and returns tokens with phonetic and syllable_count
        fields filled in.

        Args:
            tokens: List of Token objects with text field populated

        Returns:
            List of Token objects with phonetic and syllable_count fields set

        Examples:
            >>> from models import Token
            >>> transcriber = PhoneticTranscriber()
            >>> tokens = [Token("hello", "NOUN", "", False, 0)]
            >>> enriched = transcriber.transcribe_tokens(tokens)
            >>> enriched[0].phonetic
            'HH AH0 L OW1'
            >>> enriched[0].syllable_count
            2

        Notes:
            This method preserves all other Token fields (text, pos_tag,
            is_content_word) while enriching phonetic and syllable_count.
            Follows the placeholder pattern used by other preprocessor components.
        """
        if not tokens:
            logging.debug("Empty token list provided for transcription")
            return []

        enriched_tokens = []

        for token in tokens:
            # Transcribe the word
            phonetic = self.transcribe(token.text)
            syllables = self.count_syllables(token.text)

            # Create enriched token preserving all other fields
            enriched_token = Token(
                text=token.text,
                pos_tag=token.pos_tag,
                phonetic=phonetic,
                is_content_word=token.is_content_word,
                syllable_count=syllables
            )

            enriched_tokens.append(enriched_token)

        logging.debug(f"Transcribed {len(enriched_tokens)} tokens")
        return enriched_tokens


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_transcribe(word: str) -> str:
    """Convenience function for one-off transcription without creating PhoneticTranscriber instance.

    Args:
        word: Word to transcribe

    Returns:
        ARPAbet string or uppercase word if OOV

    Examples:
        >>> quick_transcribe("hello")
        'HH AH0 L OW1'
        >>> quick_transcribe("world")
        'W ER1 L D'
    """
    transcriber = PhoneticTranscriber()
    return transcriber.transcribe(word)


def get_rhyming_words(word: str, max_results: int = 10) -> List[str]:
    """Get words that rhyme with the given word.

    Uses pronouncing.rhymes() to find rhyming words in the CMU dictionary.
    This is a utility function that may be useful for testing and validation.

    Args:
        word: Word to find rhymes for
        max_results: Maximum number of rhymes to return

    Returns:
        List of rhyming words

    Examples:
        >>> rhymes = get_rhyming_words("cat", max_results=5)
        >>> "bat" in rhymes or "hat" in rhymes
        True
    """
    rhymes = pronouncing.rhymes(word.lower())
    return rhymes[:max_results]
