"""
Extractor Test Suite

Tests for the modular extractor architecture.
Run with: pytest tests/test_extractors.py -v
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.extractors import get_registry, reset_registry
from src.extractors.base import BaseExtractor, ExtractorResult
from src.extractors.lightweight import LightweightExtractor
from src.extractors.connectors import ConnectorExtractor
from src.extractors.punctuation import PunctuationExtractor
from src.extractors.epistemic import EpistemicExtractor
from src.extractors.transitions import TransitionExtractor


# Sample texts for testing
HUMAN_SAMPLE = """
I've been thinking about this problem for a while now, and honestly?
I'm not entirely sure what the best approach is. Maybe we could try
something different—like starting from scratch and seeing where it leads.

But here's the thing: every time I try a new method, I run into the
same issues. It's frustrating! Still, I think there might be a way
forward if we just... keep experimenting?

What do you think? Have you ever dealt with something like this before?
"""

AI_SAMPLE = """
The implementation of effective strategies requires careful consideration
of multiple factors. First, it is important to establish clear objectives.
Second, the methodology should be systematically evaluated. Third, the
results must be thoroughly analyzed.

Furthermore, the data indicates that optimal outcomes are achieved through
consistent application of established principles. Additionally, research
has demonstrated the efficacy of structured approaches in similar contexts.

In conclusion, the evidence strongly supports the adoption of comprehensive
frameworks for addressing complex challenges. Therefore, organizations
should prioritize the development of robust systems.
"""


class TestBaseExtractor:
    """Tests for BaseExtractor interface."""

    def test_extractor_result_access(self):
        """Test ExtractorResult data access."""
        result = ExtractorResult(
            group="test",
            features={"a": 1.0, "b": 2.0},
            extraction_time_ms=0.5,
            success=True
        )
        assert result["a"] == 1.0
        assert result["b"] == 2.0
        assert result["nonexistent"] == 0.0  # Default

    def test_extractor_result_to_dict(self):
        """Test ExtractorResult serialization."""
        result = ExtractorResult(
            group="test",
            features={"x": 0.5},
            extraction_time_ms=1.0,
            success=True
        )
        d = result.to_dict()
        assert d["group"] == "test"
        assert d["features"]["x"] == 0.5
        assert d["success"] is True


class TestLightweightExtractor:
    """Tests for LightweightExtractor (zero dependencies)."""

    def setup_method(self):
        self.extractor = LightweightExtractor()

    def test_feature_count(self):
        """Should have 10 features."""
        assert self.extractor.n_features == 10

    def test_group_name(self):
        """Should be in 'lightweight' group."""
        assert self.extractor.group == "lightweight"

    def test_no_dependencies(self):
        """Should have no dependencies."""
        assert self.extractor.dependencies == []

    def test_extraction_success(self):
        """Should successfully extract from valid text."""
        result = self.extractor.extract(HUMAN_SAMPLE)
        assert result.success is True
        assert len(result.features) == 10

    def test_short_text_handling(self):
        """Should handle very short text gracefully."""
        result = self.extractor.extract("Hi")
        assert result.success is True
        # Should return zeros for short text
        assert all(v == 0.0 for v in result.features.values())

    def test_sentence_length_cv_differs(self):
        """Human text should have higher sentence length CV."""
        human_result = self.extractor.extract(HUMAN_SAMPLE)
        ai_result = self.extractor.extract(AI_SAMPLE)
        # Human text has more varied sentence lengths
        assert human_result["sentence_length_cv"] > ai_result["sentence_length_cv"]

    def test_type_token_ratio(self):
        """Should calculate type-token ratio correctly."""
        result = self.extractor.extract("the cat sat on the mat")
        # 5 unique / 6 total = 0.833...
        assert 0.8 < result["type_token_ratio"] < 0.9


class TestConnectorExtractor:
    """Tests for ConnectorExtractor."""

    def setup_method(self):
        self.extractor = ConnectorExtractor()

    def test_feature_count(self):
        """Should have 8 features."""
        assert self.extractor.n_features == 8

    def test_no_dependencies(self):
        """Should have no dependencies."""
        assert self.extractor.dependencies == []

    def test_and_but_ratio(self):
        """Should detect and/but usage."""
        text_with_and = "I like apples and oranges and bananas."
        text_with_but = "I like apples but not oranges but maybe bananas."

        result_and = self.extractor.extract(text_with_and)
        result_but = self.extractor.extract(text_with_but)

        # More "and" = higher ratio
        assert result_and["and_but_ratio"] > result_but["and_but_ratio"]

    def test_however_detection(self):
        """Should detect 'however' usage."""
        text = "The idea is good. However, the execution is poor. However, we can improve."
        result = self.extractor.extract(text)
        assert result["however_rate"] > 0


class TestPunctuationExtractor:
    """Tests for PunctuationExtractor."""

    def setup_method(self):
        self.extractor = PunctuationExtractor()

    def test_feature_count(self):
        """Should have 8 features."""
        assert self.extractor.n_features == 8

    def test_question_rate(self):
        """Should detect questions."""
        text_with_questions = "What do you think? Is this right? I'm not sure."
        text_no_questions = "This is a statement. Another statement. Final statement."

        result_q = self.extractor.extract(text_with_questions)
        result_no_q = self.extractor.extract(text_no_questions)

        assert result_q["question_rate"] > result_no_q["question_rate"]

    def test_em_dash_detection(self):
        """Should detect em-dashes."""
        text = "The result—surprisingly—was positive. This—as we expected—worked well."
        result = self.extractor.extract(text)
        assert result["em_dash_rate"] > 0


class TestEpistemicExtractor:
    """Tests for EpistemicExtractor."""

    def setup_method(self):
        self.extractor = EpistemicExtractor()

    def test_feature_count(self):
        """Should have 6 features."""
        assert self.extractor.n_features == 6

    def test_hedge_detection(self):
        """Should detect hedging language."""
        text_hedged = "I think maybe this could possibly work, perhaps."
        text_certain = "This definitely works. It is clearly correct."

        result_h = self.extractor.extract(text_hedged)
        result_c = self.extractor.extract(text_certain)

        assert result_h["hedge_density"] > result_c["hedge_density"]

    def test_certainty_score(self):
        """Should calculate certainty score."""
        text_certain = "This is definitely correct. It is obviously true. Clearly the best."
        result = self.extractor.extract(text_certain)
        # More boosters = higher certainty
        assert result["certainty_score"] > 0.5


class TestTransitionExtractor:
    """Tests for TransitionExtractor."""

    def setup_method(self):
        self.extractor = TransitionExtractor()

    def test_feature_count(self):
        """Should have 6 features."""
        assert self.extractor.n_features == 6

    def test_transition_starters(self):
        """Should detect transition word starters."""
        text_transitions = "First, we begin. Furthermore, we continue. Finally, we end."
        text_normal = "We begin here. We continue on. We end now."

        result_t = self.extractor.extract(text_transitions)
        result_n = self.extractor.extract(text_normal)

        assert result_t["transition_starter_rate"] > result_n["transition_starter_rate"]


class TestRegistry:
    """Tests for ExtractorRegistry."""

    def setup_method(self):
        reset_registry()
        self.registry = get_registry(quiet=True)

    def test_all_extractors_registered(self):
        """Should have all extractors registered."""
        groups = self.registry.groups
        # At minimum should have these zero-dependency groups
        assert "lightweight" in groups
        assert "connectors" in groups
        assert "punctuation" in groups
        assert "epistemic" in groups
        assert "transitions" in groups

    def test_extract_all(self):
        """Should extract from all extractors."""
        result = self.registry.extract_all(HUMAN_SAMPLE)
        assert result.n_features > 0
        assert result.total_time_ms >= 0

    def test_extract_group(self):
        """Should extract from single group."""
        result = self.registry.extract_group("lightweight", HUMAN_SAMPLE)
        assert result.n_features == 10

    def test_extract_single_feature(self):
        """Should extract single feature by name."""
        value = self.registry.extract_feature("sentence_length_cv", HUMAN_SAMPLE)
        assert isinstance(value, float)

    def test_failed_extractor_isolation(self):
        """Failed extractors shouldn't stop others."""
        result = self.registry.extract_all(HUMAN_SAMPLE)
        # Even if one fails, we should still get features from others
        assert result.n_features > 0

    def test_diagnostic_report(self):
        """Should generate diagnostic report."""
        report = self.registry.test_all()
        assert "summary" in report
        assert "extractors" in report
        assert report["summary"]["total_features"] > 0


class TestHumanVsAI:
    """Comparative tests showing human vs AI text differences."""

    def setup_method(self):
        reset_registry()
        self.registry = get_registry(quiet=True)

    def test_overall_feature_differences(self):
        """Human and AI text should produce different features."""
        human_result = self.registry.extract_all(HUMAN_SAMPLE)
        ai_result = self.registry.extract_all(AI_SAMPLE)

        # At least some features should differ significantly
        differences = 0
        for feature in human_result.features:
            h_val = human_result.features[feature]
            a_val = ai_result.features[feature]
            if abs(h_val - a_val) > 0.1:
                differences += 1

        # Expect at least 5 features to differ meaningfully
        assert differences >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
