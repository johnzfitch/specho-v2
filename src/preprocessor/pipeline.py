"""Linguistic Preprocessor Pipeline for SpecHO watermark detection system.

This module provides the orchestrator that chains all preprocessor components
together to transform raw text into fully annotated linguistic structures.
The LinguisticPreprocessor is the entry point for the entire detection pipeline.

Component 1 of the SpecHO detection system.

From architecture.md:
"The Linguistic Preprocessor transforms raw text into structured linguistic
representations that subsequent components can analyze. It annotates text with
linguistic information that watermark detection requires: POS tags, dependency
trees, phonetic transcriptions, syllable counts, and content-word status."

Tier: 1 (MVP)
Task: 2.5
Dependencies: Tasks 2.1-2.4 (Tokenizer, POSTagger, DependencyParser, PhoneticTranscriber)
"""

import logging
from typing import List, Tuple
from spacy.tokens import Doc as SpacyDoc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Token
from preprocessor.tokenizer import Tokenizer
from preprocessor.pos_tagger import POSTagger
from preprocessor.dependency_parser import DependencyParser
from preprocessor.phonetic import PhoneticTranscriber


class LinguisticPreprocessor:
    """Orchestrator that chains all preprocessor components together.

    The LinguisticPreprocessor is the entry point for linguistic analysis in the
    SpecHO pipeline. It receives raw text and returns fully enriched Token objects
    along with the dependency parse tree.

    Architecture Pattern: Orchestrator
    - Minimal orchestration logic, delegates all work to subcomponents
    - Sequential chaining: DependencyParser → Tokenizer → POSTagger → PhoneticTranscriber
    - Returns both Token list and spaCy Doc for downstream use
    - DependencyParser runs FIRST to create canonical spaCy doc for POS tagging alignment

    Data Flow:
        Raw text (str)
            ↓
        DependencyParser: text → spacy.Doc (dependency tree, canonical tokenization)
            ↓
        Tokenizer: text → List[Token] (text populated only)
            ↓
        POSTagger: tokens + spacy.Doc → List[Token] (+pos_tag, +is_content_word)
            ↓
        PhoneticTranscriber: tokens → List[Token] (+phonetic, +syllable_count)
            ↓
        Output: (List[Token], spacy.Doc)

    The Token list has all fields populated:
    - text: from Tokenizer
    - pos_tag: from POSTagger
    - phonetic: from PhoneticTranscriber
    - is_content_word: from POSTagger
    - syllable_count: from PhoneticTranscriber

    The spacy.Doc provides:
    - Dependency tree for clause boundary detection
    - Sentence boundaries
    - Syntactic structure information

    Tier 1 Implementation:
    - Simple sequential processing
    - No error recovery or fallbacks
    - No caching or optimization
    - Synchronous execution

    Attributes:
        tokenizer: Tokenizer instance
        pos_tagger: POSTagger instance
        dependency_parser: DependencyParser instance
        phonetic_transcriber: PhoneticTranscriber instance
    """

    def __init__(self):
        """Initialize the LinguisticPreprocessor with all subcomponents.

        Creates instances of all four preprocessor components. This happens
        once during initialization to avoid recreating NLP models on every
        call to process().

        Examples:
            >>> preprocessor = LinguisticPreprocessor()
            >>> tokens, doc = preprocessor.process("The cat sat on the mat.")
        """
        logging.info("Initializing LinguisticPreprocessor (Tier 1)")

        # Initialize all subcomponents
        self.tokenizer = Tokenizer()
        self.pos_tagger = POSTagger()
        self.dependency_parser = DependencyParser()
        self.phonetic_transcriber = PhoneticTranscriber()

        logging.info("LinguisticPreprocessor initialized with 4 components")

    def process(self, text: str) -> Tuple[List[Token], SpacyDoc]:
        """Process raw text through the complete preprocessing pipeline.

        This is the main entry point for linguistic preprocessing. It chains
        all four subcomponents in sequence to produce fully annotated tokens
        and dependency parse trees.

        Processing Steps:
        1. Dependency Parsing: Build syntactic dependency tree (creates canonical spaCy doc)
        2. Tokenization: Split text into Token objects
        3. POS Tagging: Add part-of-speech tags using canonical doc (ensures alignment)
        4. Phonetic Transcription: Add phonetic representations and syllable counts

        Args:
            text: Raw text string to process

        Returns:
            Tuple of (enriched_tokens, dependency_doc):
            - enriched_tokens: List[Token] with all fields populated
            - dependency_doc: spacy.tokens.Doc with dependency parse

        Examples:
            >>> preprocessor = LinguisticPreprocessor()
            >>> tokens, doc = preprocessor.process("The cat sat.")
            >>>
            >>> # Verify all Token fields are populated
            >>> tokens[1].text  # "cat"
            'cat'
            >>> tokens[1].pos_tag  # "NOUN"
            'NOUN'
            >>> tokens[1].phonetic  # "K AE1 T"
            'K AE1 T'
            >>> tokens[1].is_content_word  # True
            True
            >>> tokens[1].syllable_count  # 1
            1
            >>>
            >>> # Dependency parse available
            >>> len(list(doc.sents))  # Number of sentences
            1

        Notes:
            Tier 1 implementation performs no validation or error handling
            beyond what the subcomponents provide. If any component fails,
            the exception propagates to the caller.

            The Token list and spacy.Doc may have different tokenization
            in edge cases (e.g., contractions). The POSTagger handles this
            with a fallback method. For most purposes, use the Token list
            for content analysis and the Doc for structural analysis.

        Raises:
            Any exceptions from subcomponents (typically spaCy errors)
        """
        if not text or not text.strip():
            logging.warning("Empty text provided to preprocessor")
            return ([], self.dependency_parser.parse(""))

        logging.debug(f"Processing text: {len(text)} characters, {len(text.split())} words")

        # Step 1: Dependency Parsing (FIRST to create canonical spaCy doc)
        # Creates spaCy Doc with full syntactic analysis that will be shared
        dependency_doc = self.dependency_parser.parse(text)
        logging.debug(f"Dependency Parsing: {len(list(dependency_doc.sents))} sentences")

        # Step 2: Tokenization
        # Creates Token objects with only 'text' field populated
        tokens = self.tokenizer.tokenize(text)
        logging.debug(f"Tokenization: {len(tokens)} tokens")

        # Step 3: POS Tagging (using dependency_doc for perfect alignment)
        # Enriches tokens with 'pos_tag' and 'is_content_word' fields
        tagged_tokens = self.pos_tagger.tag(tokens, spacy_doc=dependency_doc)
        logging.debug(f"POS Tagging: {sum(1 for t in tagged_tokens if t.is_content_word)} content words")

        # Step 4: Phonetic Transcription
        # Enriches tokens with 'phonetic' and 'syllable_count' fields
        enriched_tokens = self.phonetic_transcriber.transcribe_tokens(tagged_tokens)
        logging.debug(f"Phonetic Transcription: {sum(t.syllable_count for t in enriched_tokens)} total syllables")

        # Verify data quality (Tier 1: simple checks only)
        self._validate_output(enriched_tokens, dependency_doc)

        logging.info(f"Preprocessing complete: {len(enriched_tokens)} tokens, {len(list(dependency_doc.sents))} sentences")

        return (enriched_tokens, dependency_doc)

    def _validate_output(self, tokens: List[Token], doc: SpacyDoc) -> None:
        """Validate that preprocessing produced reasonable output.

        Tier 1 validation performs simple sanity checks to catch obvious
        errors. This is not comprehensive error handling, just basic quality
        control.

        Args:
            tokens: Enriched token list
            doc: Dependency parse doc

        Logs warnings if validation issues are detected.
        Does not raise exceptions in Tier 1.
        """
        # Check token list is non-empty for non-empty docs
        if doc and len(list(doc)) > 0 and len(tokens) == 0:
            logging.warning("Dependency parse succeeded but token list is empty")

        # Check that most tokens have populated fields
        if tokens:
            fields_populated = sum(1 for t in tokens if t.pos_tag != "" and t.syllable_count > 0)
            population_rate = fields_populated / len(tokens)

            if population_rate < 0.5:
                logging.warning(f"Low field population rate: {population_rate:.1%}")

        # Check that some content words were identified
        if tokens:
            content_words = sum(1 for t in tokens if t.is_content_word)
            content_rate = content_words / len(tokens) if len(tokens) > 0 else 0

            # Expect 30-60% content words in typical English text
            if content_rate < 0.2 or content_rate > 0.8:
                logging.warning(f"Unusual content word rate: {content_rate:.1%}")

    def get_token_count(self, text: str) -> int:
        """Quick utility to get token count without full processing.

        Useful for estimating processing time or validating input.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens

        Examples:
            >>> preprocessor = LinguisticPreprocessor()
            >>> preprocessor.get_token_count("Hello world!")
            3
        """
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def process_batch(self, texts: List[str]) -> List[Tuple[List[Token], SpacyDoc]]:
        """Process multiple texts through the pipeline.

        Tier 1 implementation simply calls process() in a loop. No batching
        optimization. Tier 2 may add parallel processing or batch spaCy calls.

        Args:
            texts: List of text strings to process

        Returns:
            List of (enriched_tokens, dependency_doc) tuples

        Examples:
            >>> preprocessor = LinguisticPreprocessor()
            >>> results = preprocessor.process_batch(["Text 1.", "Text 2."])
            >>> len(results)
            2
        """
        logging.info(f"Processing batch of {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            logging.debug(f"Processing text {i+1}/{len(texts)}")
            result = self.process(text)
            results.append(result)

        logging.info(f"Batch processing complete: {len(results)} texts processed")
        return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_process(text: str) -> Tuple[List[Token], SpacyDoc]:
    """Convenience function for one-off preprocessing without creating LinguisticPreprocessor instance.

    Args:
        text: Text to process

    Returns:
        Tuple of (enriched_tokens, dependency_doc)

    Examples:
        >>> tokens, doc = quick_process("The cat sat.")
        >>> tokens[1].text
        'cat'
        >>> len(list(doc.sents))
        1
    """
    preprocessor = LinguisticPreprocessor()
    return preprocessor.process(text)
