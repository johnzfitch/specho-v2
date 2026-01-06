"""Tokenizer for SpecHO watermark detection system.

This module provides text tokenization using spaCy, converting raw text strings
into lists of Token objects. The tokenizer handles contractions, hyphenated words,
and other edge cases automatically through spaCy's robust tokenization.

Tier: 1 (MVP)
Task: 2.1
Dependencies: Task 1.1 (models.py), spacy
"""

import logging
from typing import List
import spacy
from spacy.tokens import Doc as SpacyDoc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Token


class Tokenizer:
    """Tokenizer that converts raw text into Token objects using spaCy.

    This is the entry point for the linguistic preprocessing pipeline. It uses
    spaCy's tokenizer to segment text into words and punctuation marks,
    returning Token objects with the text field populated. Other Token fields
    (POS tags, phonetic transcriptions, etc.) are populated by downstream
    preprocessor components.

    Tier 1 Implementation:
    - Basic spaCy tokenization
    - Handles contractions (e.g., "don't" → "do", "n't")
    - Handles hyphenated words per spaCy's rules
    - Returns Token objects with text field only
    - Other fields set to placeholder values

    Attributes:
        nlp: spaCy language model for tokenization
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the Tokenizer with a spaCy model.

        Args:
            model_name: Name of spaCy model to load (default: 'en_core_web_sm')

        Raises:
            OSError: If spaCy model is not installed

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer = Tokenizer("en_core_web_md")  # Use medium model
        """
        try:
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
            logging.info(f"Loaded spaCy model: {model_name}")
        except OSError as e:
            logging.error(
                f"Failed to load spaCy model '{model_name}'. "
                f"Install it with: python -m spacy download {model_name}"
            )
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from e

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize text into a list of Token objects.

        Uses spaCy's tokenizer to segment text into individual tokens (words,
        punctuation, etc.). Returns Token objects with the text field populated.
        Other fields are set to placeholder values that will be filled by
        subsequent preprocessor components.

        Tier 1 behavior:
        - text: Set from spaCy token.text
        - pos_tag: Empty string (populated by POSTagger)
        - phonetic: Empty string (populated by PhoneticTranscriber)
        - is_content_word: False (populated by POSTagger)
        - syllable_count: 0 (populated by PhoneticTranscriber)

        Args:
            text: Raw text string to tokenize

        Returns:
            List of Token objects, one per token in the text

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.tokenize("Hello, world!")
            >>> len(tokens)
            4
            >>> [t.text for t in tokens]
            ['Hello', ',', 'world', '!']

            >>> tokens = tokenizer.tokenize("Don't worry about it.")
            >>> [t.text for t in tokens]
            ['Do', "n't", 'worry', 'about', 'it', '.']
        """
        if not text or not text.strip():
            logging.warning("Received empty text for tokenization")
            return []

        # Process text with spaCy
        doc = self.nlp(text)

        # Convert spaCy tokens to our Token objects
        tokens = []
        for spacy_token in doc:
            token = Token(
                text=spacy_token.text,
                pos_tag="",  # Filled by POSTagger (Task 2.2)
                phonetic="",  # Filled by PhoneticTranscriber (Task 2.4)
                is_content_word=False,  # Filled by POSTagger (Task 2.2)
                syllable_count=0,  # Filled by PhoneticTranscriber (Task 2.4)
            )
            tokens.append(token)

        logging.debug(f"Tokenized text into {len(tokens)} tokens")
        return tokens

    def tokenize_with_doc(self, text: str) -> tuple[List[Token], SpacyDoc]:
        """Tokenize text and return both Tokens and the spaCy Doc object.

        This variant returns both the Token list and the original spaCy Doc,
        which is needed by downstream components (particularly the Dependency
        Parser) that operate on the spaCy Doc structure.

        Args:
            text: Raw text string to tokenize

        Returns:
            Tuple of (List[Token], spacy.tokens.Doc)

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokens, doc = tokenizer.tokenize_with_doc("Hello world")
            >>> len(tokens) == len(doc)
            True
        """
        if not text or not text.strip():
            logging.warning("Received empty text for tokenization")
            return [], self.nlp("")

        # Process text with spaCy
        doc = self.nlp(text)

        # Convert spaCy tokens to our Token objects
        tokens = []
        for spacy_token in doc:
            token = Token(
                text=spacy_token.text,
                pos_tag="",
                phonetic="",
                is_content_word=False,
                syllable_count=0,
            )
            tokens.append(token)

        logging.debug(f"Tokenized text into {len(tokens)} tokens (with Doc)")
        return tokens, doc


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_tokenize(text: str, model_name: str = "en_core_web_sm") -> List[Token]:
    """Convenience function for one-off tokenization without creating Tokenizer instance.

    Args:
        text: Text to tokenize
        model_name: spaCy model to use

    Returns:
        List of Token objects

    Examples:
        >>> tokens = quick_tokenize("Hello world")
        >>> len(tokens)
        2
    """
    tokenizer = Tokenizer(model_name)
    return tokenizer.tokenize(text)
