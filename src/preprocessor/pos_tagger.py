"""POS Tagger for SpecHO watermark detection system.

This module provides part-of-speech tagging using spaCy, enriching Token objects
with POS tags and identifying content words (nouns, verbs, adjectives) that are
important for echo analysis and zone extraction.

Tier: 1 (MVP)
Task: 2.2
Dependencies: Task 1.1 (models.py), Task 2.1 (tokenizer.py), spacy
"""

import logging
from typing import List
import spacy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Token


class POSTagger:
    """POS tagger that enriches Token objects with part-of-speech tags.

    Uses spaCy's POS tagger to annotate tokens with their grammatical categories
    (NOUN, VERB, ADJ, etc.) and identifies content words that carry semantic meaning.
    Content words are crucial for zone extraction and echo analysis, as they are
    the words compared for phonetic, structural, and semantic similarity.

    Tier 1 Implementation:
    - Basic spaCy POS tagging
    - Simple content word identification (nouns, verbs, adjectives)
    - Binary is_content_word flag
    - Uses universal POS tags (UPOS)

    Content Word Definition (Tier 1):
    - NOUN: Common and proper nouns
    - VERB: All verb forms (except auxiliary "be", "have", "do")
    - ADJ: Adjectives
    - ADV: Adverbs (included as potential content carriers)

    Function Words (excluded):
    - DET: Determiners (the, a, an)
    - ADP: Prepositions (in, on, at)
    - CONJ/CCONJ/SCONJ: Conjunctions (and, but, because)
    - PRON: Pronouns (he, she, it)
    - AUX: Auxiliary verbs (is, has, do)
    - PART: Particles (to, not)
    - PUNCT: Punctuation

    Attributes:
        nlp: spaCy language model with POS tagger enabled
        content_pos_tags: Set of POS tags considered content words
    """

    # Universal POS tags that represent content words
    CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the POSTagger with a spaCy model.

        Args:
            model_name: Name of spaCy model to load (default: 'en_core_web_sm')

        Raises:
            OSError: If spaCy model is not installed
        """
        try:
            # Load spaCy with tagger enabled (disable parser and NER for speed)
            self.nlp = spacy.load(model_name, disable=["parser", "ner"])
            self.content_pos_tags = self.CONTENT_POS_TAGS
            logging.info(f"Loaded spaCy model for POS tagging: {model_name}")
        except OSError as e:
            logging.error(
                f"Failed to load spaCy model '{model_name}'. "
                f"Install it with: python -m spacy download {model_name}"
            )
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from e

    def tag(self, tokens: List[Token], spacy_doc=None) -> List[Token]:
        """Enrich tokens with POS tags and content word identification.

        Takes a list of Token objects (typically from Tokenizer) and enriches
        them with:
        - pos_tag: Universal POS tag (NOUN, VERB, ADJ, etc.)
        - is_content_word: Boolean indicating if token is a content word

        Note: This method processes text through spaCy, so the tokens list should
        match the original text structure. The token texts are reconstructed into
        a string for spaCy processing.

        Args:
            tokens: List of Token objects to enrich
            spacy_doc: Optional pre-existing spaCy Doc to use for POS tagging.
                      If provided, skips creating a new doc (ensures alignment).

        Returns:
            List of Token objects with pos_tag and is_content_word fields populated

        Examples:
            >>> from tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
            >>> pos_tagger = POSTagger()
            >>>
            >>> tokens = tokenizer.tokenize("The cat sat on the mat.")
            >>> enriched_tokens = pos_tagger.tag(tokens)
            >>>
            >>> enriched_tokens[1].text  # "cat"
            'cat'
            >>> enriched_tokens[1].pos_tag
            'NOUN'
            >>> enriched_tokens[1].is_content_word
            True
        """
        if not tokens:
            logging.warning("Received empty token list for POS tagging")
            return []

        # Use provided spaCy doc if available, otherwise create new one
        if spacy_doc is not None:
            doc = spacy_doc
            logging.debug(f"Using provided spaCy doc for POS tagging ({len(doc)} tokens)")
        else:
            # Reconstruct text from tokens for spaCy processing
            # This ensures alignment between our tokens and spaCy's tokens
            text = " ".join(t.text for t in tokens)

            # Process with spaCy
            doc = self.nlp(text)

        # Verify alignment (spaCy might tokenize differently)
        if len(doc) != len(tokens):
            logging.warning(
                f"Token count mismatch: input={len(tokens)}, spaCy={len(doc)}. "
                f"Using text-based matching."
            )
            # Fall back to text-based matching if counts don't match
            return self._tag_with_text_matching(tokens, doc)

        # Enrich our Token objects with spaCy's POS tags
        enriched_tokens = []
        for token, spacy_token in zip(tokens, doc):
            enriched_token = Token(
                text=token.text,
                pos_tag=spacy_token.pos_,  # Universal POS tag
                phonetic=token.phonetic,  # Preserve existing value
                is_content_word=self.is_content_word_from_pos(spacy_token.pos_),
                syllable_count=token.syllable_count,  # Preserve existing value
            )
            enriched_tokens.append(enriched_token)

        logging.debug(f"Tagged {len(enriched_tokens)} tokens with POS")
        return enriched_tokens

    def _tag_with_text_matching(self, tokens: List[Token], doc) -> List[Token]:
        """Fallback method for POS tagging when token alignment fails.

        Uses text-based matching to align tokens with spaCy doc when token
        counts don't match. Tries to match tokens by text content with
        position as a tiebreaker.

        Args:
            tokens: List of Token objects
            doc: spaCy Doc to extract POS tags from

        Returns:
            List of enriched Token objects
        """
        # Create list of spaCy tokens for matching
        spacy_tokens_list = list(doc)

        enriched_tokens = []
        spacy_idx = 0  # Track position in spaCy doc

        for token in tokens:
            # Try to find matching spaCy token starting from current position
            matched = False

            # Search forward in spaCy doc for matching text
            for offset in range(min(3, len(spacy_tokens_list) - spacy_idx)):
                if spacy_idx + offset < len(spacy_tokens_list):
                    spacy_token = spacy_tokens_list[spacy_idx + offset]

                    # Match if text is the same (case-insensitive)
                    if spacy_token.text.lower() == token.text.lower():
                        pos_tag = spacy_token.pos_
                        is_content = self.is_content_word_from_pos(pos_tag)
                        spacy_idx += offset + 1  # Move past matched token
                        matched = True
                        break

            if not matched:
                # No match found - use empty defaults
                pos_tag = ""
                is_content = False
                logging.debug(f"No POS match for token: {token.text}")
                spacy_idx += 1  # Still advance position

            enriched_token = Token(
                text=token.text,
                pos_tag=pos_tag,
                phonetic=token.phonetic,
                is_content_word=is_content,
                syllable_count=token.syllable_count,
            )
            enriched_tokens.append(enriched_token)

        return enriched_tokens

    def is_content_word(self, token: Token) -> bool:
        """Determine if a token is a content word.

        Checks if a token carries semantic content based on its POS tag.
        This method requires the token to already have a pos_tag set.

        Args:
            token: Token object with pos_tag populated

        Returns:
            True if token is a content word, False otherwise

        Examples:
            >>> token = Token("cat", "NOUN", "", False, 0)
            >>> pos_tagger = POSTagger()
            >>> pos_tagger.is_content_word(token)
            True

            >>> token = Token("the", "DET", "", False, 0)
            >>> pos_tagger.is_content_word(token)
            False
        """
        if not token.pos_tag:
            logging.warning(f"Token '{token.text}' has no POS tag")
            return False

        return self.is_content_word_from_pos(token.pos_tag)

    def is_content_word_from_pos(self, pos_tag: str) -> bool:
        """Determine if a POS tag represents a content word.

        Tier 1 implementation uses a simple set membership check against
        universal POS tags.

        Args:
            pos_tag: Universal POS tag (e.g., "NOUN", "VERB")

        Returns:
            True if POS tag indicates content word, False otherwise

        Examples:
            >>> pos_tagger = POSTagger()
            >>> pos_tagger.is_content_word_from_pos("NOUN")
            True
            >>> pos_tagger.is_content_word_from_pos("DET")
            False
        """
        return pos_tag in self.content_pos_tags

    def get_content_words(self, tokens: List[Token]) -> List[Token]:
        """Filter a list of tokens to only content words.

        Convenience method for extracting content words from a token list.
        Useful for zone extraction in the Clause Identifier.

        Args:
            tokens: List of Token objects with pos_tag populated

        Returns:
            List containing only tokens where is_content_word is True

        Examples:
            >>> tokens = [
            ...     Token("the", "DET", "", False, 0),
            ...     Token("cat", "NOUN", "", True, 0),
            ...     Token("sat", "VERB", "", True, 0),
            ... ]
            >>> pos_tagger = POSTagger()
            >>> content = pos_tagger.get_content_words(tokens)
            >>> [t.text for t in content]
            ['cat', 'sat']
        """
        return [t for t in tokens if t.is_content_word]

    def tag_text_directly(self, text: str) -> List[Token]:
        """Tag text directly without pre-tokenization.

        Convenience method that combines tokenization and POS tagging.
        Useful for quick processing, but less flexible than using separate
        Tokenizer and POSTagger components.

        Args:
            text: Raw text string to process

        Returns:
            List of Token objects with pos_tag and is_content_word populated

        Examples:
            >>> pos_tagger = POSTagger()
            >>> tokens = pos_tagger.tag_text_directly("The cat sat")
            >>> [(t.text, t.pos_tag) for t in tokens]
            [('The', 'DET'), ('cat', 'NOUN'), ('sat', 'VERB')]
        """
        doc = self.nlp(text)

        tokens = []
        for spacy_token in doc:
            token = Token(
                text=spacy_token.text,
                pos_tag=spacy_token.pos_,
                phonetic="",  # Not populated yet
                is_content_word=self.is_content_word_from_pos(spacy_token.pos_),
                syllable_count=0,  # Not populated yet
            )
            tokens.append(token)

        return tokens
