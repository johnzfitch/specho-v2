"""Main watermark detection orchestrator.

This module implements the SpecHODetector class which coordinates the complete
watermark detection pipeline from raw text to statistical verdict.

The detector chains all five pipeline components:
1. LinguisticPreprocessor: Text → enriched tokens + dependency parse
2. ClauseIdentifier: Tokens → thematic clause pairs
3. EchoAnalysisEngine: Clause pairs → echo similarity scores
4. ScoringModule: Echo scores → document-level score
5. StatisticalValidator: Document score → z-score + confidence

Tier 1 is a simple orchestrator with error handling and logging at each stage.

Tier: 1 (MVP)
Task: 7.1
Dependencies: All prior tasks (1.1-6.4)
"""

import logging
from typing import Optional

from .models import DocumentAnalysis, ClausePair, EchoScore
from .preprocessor.pipeline import LinguisticPreprocessor
from .clause_identifier.pipeline import ClauseIdentifier
from .echo_engine.pipeline import EchoAnalysisEngine
from .scoring.pipeline import ScoringModule
from .validator.pipeline import StatisticalValidator


class SpecHODetector:
    """Main entry point for Echo Rule watermark detection.

    The SpecHODetector orchestrates the complete detection pipeline, chaining
    all five components to analyze text and produce a statistical verdict on
    whether AI watermarking is present.

    Tier 1 Implementation:
    - Simple sequential pipeline execution
    - Error handling at each stage with fallback to empty results
    - Logging of intermediate results for debugging
    - Returns complete DocumentAnalysis with all intermediate data

    The orchestrator pattern keeps the main detector simple while delegating
    all processing logic to specialized components. This makes the system
    modular, testable, and maintainable.

    Pipeline Flow:
        Input (raw text)
            ↓
        1. LinguisticPreprocessor
            ↓ (List[Token], spacy.Doc)
        2. ClauseIdentifier
            ↓ (List[ClausePair])
        3. EchoAnalysisEngine
            ↓ (List[EchoScore])
        4. ScoringModule
            ↓ (float: document_score)
        5. StatisticalValidator
            ↓ (float: z_score, float: confidence)
        Output (DocumentAnalysis)

    Examples:
        >>> detector = SpecHODetector()
        >>> text = "The sky darkened. But hope remained."
        >>> analysis = detector.analyze(text)
        >>> print(f"Score: {analysis.final_score:.3f}")
        Score: 0.450
        >>> print(f"Z-Score: {analysis.z_score:.2f}")
        Z-Score: 1.23
        >>> print(f"Confidence: {analysis.confidence:.1%}")
        Confidence: 89.1%

        >>> # Empty text returns zero scores
        >>> analysis = detector.analyze("")
        >>> analysis.final_score
        0.0

    Attributes:
        preprocessor: LinguisticPreprocessor instance
        clause_identifier: ClauseIdentifier instance
        echo_engine: EchoAnalysisEngine instance
        scoring_module: ScoringModule instance
        validator: StatisticalValidator instance
        baseline_path: Path to baseline statistics file
    """

    def __init__(
        self,
        baseline_path: str = "data/baseline/baseline_stats.pkl",
        semantic_model_path: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the detection pipeline with all components.

        Args:
            baseline_path: Path to baseline statistics file for validation.
                          Default: "data/baseline/baseline_stats.pkl"
            semantic_model_path: Path to semantic embeddings model OR model name.
                               For Sentence Transformers: Use model name (e.g., 'all-MiniLM-L6-v2')
                               For gensim: Use file path to embeddings
                               Default: "all-MiniLM-L6-v2" (Sentence Transformer)

        Raises:
            FileNotFoundError: If baseline_path doesn't exist

        Tier 1 uses default configurations for all components. Configuration
        support will be added in Tier 2.

        Examples:
            >>> detector = SpecHODetector()
            >>> detector = SpecHODetector("data/baseline/custom_baseline.pkl")
            >>> detector = SpecHODetector(semantic_model_path="all-mpnet-base-v2")
        """
        logging.info("Initializing SpecHODetector pipeline...")

        # Initialize all five components
        self.preprocessor = LinguisticPreprocessor()
        self.clause_identifier = ClauseIdentifier()
        self.echo_engine = EchoAnalysisEngine(semantic_model_path=semantic_model_path)
        self.scoring_module = ScoringModule()
        self.validator = StatisticalValidator(baseline_path)

        self.baseline_path = baseline_path
        self.semantic_model_path = semantic_model_path

        logging.info("SpecHODetector initialization complete")

    def analyze(self, text: str) -> DocumentAnalysis:
        """Analyze text for Echo Rule watermark presence.

        This is the main entry point for the detection system. It executes
        the complete pipeline and returns comprehensive analysis results.

        Algorithm (Tier 1):
        1. Validate input (non-empty string)
        2. Preprocess: Tokenize, POS tag, parse dependencies, phonetic transcribe
        3. Identify clause pairs: Apply Rules A, B, C and extract zones
        4. Analyze echoes: Calculate phonetic, structural, semantic similarities
        5. Score document: Weight and aggregate all echo scores
        6. Validate: Compare to baseline and calculate z-score + confidence
        7. Package results into DocumentAnalysis object

        Error Handling:
        - Empty text → returns zero analysis
        - Component failures → logged, empty results passed to next stage
        - Complete pipeline failure → raises exception after logging

        Args:
            text: Raw text to analyze for watermark presence

        Returns:
            DocumentAnalysis object containing:
            - Original text
            - All clause pairs identified
            - Echo scores for each pair
            - Final document score (0.0-1.0)
            - Z-score (standard deviations from human baseline)
            - Confidence level (0.0-1.0)

        Raises:
            ValueError: If text is None
            Exception: If pipeline encounters unrecoverable error

        Examples:
            >>> detector = SpecHODetector()
            >>> text = "The conference ended. However, discussions continued."
            >>> analysis = detector.analyze(text)
            >>> analysis.final_score  # Document-level echo score
            0.752
            >>> analysis.z_score  # Statistical significance
            2.43
            >>> analysis.confidence  # Watermark confidence
            0.992

            >>> # Minimal valid text
            >>> analysis = detector.analyze("Hello world.")
            >>> len(analysis.clause_pairs)
            0

            >>> # Empty text handling
            >>> analysis = detector.analyze("")
            >>> analysis.final_score
            0.0
        """
        # Validate input
        if text is None:
            raise ValueError("Input text cannot be None")

        # Handle empty text early
        if not text.strip():
            logging.warning("Empty text provided, returning zero analysis")
            return self._create_empty_analysis(text)

        logging.info(f"Analyzing text ({len(text)} characters)...")

        try:
            # Stage 1: Linguistic Preprocessing
            logging.debug("Stage 1: Linguistic preprocessing...")
            tokens, doc = self.preprocessor.process(text)
            logging.debug(f"  → {len(tokens)} tokens, {len(list(doc.sents))} sentences")

            # Stage 2: Clause Pair Identification
            logging.debug("Stage 2: Identifying clause pairs...")
            clause_pairs = self.clause_identifier.identify_pairs(tokens, doc)
            logging.debug(f"  → {len(clause_pairs)} clause pairs identified")

            # Stage 3: Echo Analysis
            logging.debug("Stage 3: Analyzing echoes...")
            echo_scores = []
            for idx, pair in enumerate(clause_pairs):
                try:
                    echo_score = self.echo_engine.analyze_pair(pair)
                    echo_scores.append(echo_score)
                    logging.debug(
                        f"  → Pair {idx+1}: phonetic={echo_score.phonetic_score:.3f}, "
                        f"structural={echo_score.structural_score:.3f}, "
                        f"semantic={echo_score.semantic_score:.3f}"
                    )
                except Exception as e:
                    logging.error(f"Error analyzing pair {idx}: {e}")
                    # Continue with remaining pairs

            logging.debug(f"  → {len(echo_scores)} echo scores computed")

            # Stage 4: Document Scoring
            logging.debug("Stage 4: Scoring document...")
            final_score = self.scoring_module.score_document(echo_scores)
            logging.debug(f"  → Final document score: {final_score:.4f}")

            # Stage 5: Statistical Validation
            logging.debug("Stage 5: Statistical validation...")
            z_score, confidence = self.validator.validate(final_score)
            logging.debug(f"  → Z-score: {z_score:.4f}, Confidence: {confidence:.4f}")

            # Package complete analysis
            analysis = DocumentAnalysis(
                text=text,
                clause_pairs=clause_pairs,
                echo_scores=echo_scores,
                final_score=final_score,
                z_score=z_score,
                confidence=confidence
            )

            logging.info(
                f"Analysis complete: score={final_score:.3f}, z={z_score:.2f}, "
                f"conf={confidence:.1%}"
            )

            return analysis

        except Exception as e:
            logging.error(f"Pipeline failure: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _create_empty_analysis(self, text: str) -> DocumentAnalysis:
        """Create DocumentAnalysis with zero values for empty/invalid input.

        Args:
            text: Original text (may be empty)

        Returns:
            DocumentAnalysis with empty clause pairs, zero scores

        This helper ensures consistent return type even for edge cases.
        """
        return DocumentAnalysis(
            text=text,
            clause_pairs=[],
            echo_scores=[],
            final_score=0.0,
            z_score=0.0,
            confidence=0.5  # Neutral confidence for empty input
        )

    def get_pipeline_info(self) -> dict:
        """Get information about the configured pipeline.

        Returns:
            Dictionary with pipeline configuration details

        Examples:
            >>> detector = SpecHODetector()
            >>> info = detector.get_pipeline_info()
            >>> info['baseline_path']
            'data/baseline/baseline_stats.pkl'
            >>> info['semantic_model_path']
            'all-MiniLM-L6-v2'
            >>> info['semantic_model_loaded']
            True
            >>> info['components']
            ['LinguisticPreprocessor', 'ClauseIdentifier', 'EchoAnalysisEngine',
             'ScoringModule', 'StatisticalValidator']
        """
        return {
            'components': [
                'LinguisticPreprocessor',
                'ClauseIdentifier',
                'EchoAnalysisEngine',
                'ScoringModule',
                'StatisticalValidator'
            ],
            'baseline_path': self.baseline_path,
            'baseline_stats': self.validator.get_baseline_info(),
            'semantic_model_path': self.semantic_model_path,
            'semantic_model_loaded': self.echo_engine.semantic_analyzer.model is not None,
            'semantic_model_type': self.echo_engine.semantic_analyzer.model_type,
            'tier': 1,
            'version': '0.1.0'
        }
