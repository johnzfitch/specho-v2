#!/usr/bin/env python3
"""
SpecHO Unified Fingerprint System

Combines three systems into one coherent fingerprint:
1. specHO Core Pipeline - Echo analysis with clause pairs, zones, scores
2. 39D Cognitive Fingerprint - Trajectory, epistemic, transitions, syntactic
3. Micro Distributions - Full per-model fingerprinting data

Output: Complete fingerprint suitable for both:
- Binary classification (human vs AI)
- Model attribution (distinguishing between AI models)

Every dimension matters for fingerprinting - we preserve ALL signals.
"""

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

# Add specHO to path
SPECHO_PATH = Path(__file__).parent.parent / "specho_core"
sys.path.insert(0, str(SPECHO_PATH))
sys.path.insert(0, str(SPECHO_PATH / "specHO"))

FINGERPRINT_PATH = Path(__file__).parent.parent / "corpus" / "src"
sys.path.insert(0, str(FINGERPRINT_PATH))


# ============================================================================
# DATA MODELS - Complete fingerprint structure
# ============================================================================

@dataclass
class PairTypeStats:
    """Statistics for a specific pair type (punctuation/conjunction/transition)."""
    count: int = 0
    percentage: float = 0.0
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    structural_mean: float = 0.0
    structural_std: float = 0.0
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    combined_mean: float = 0.0


@dataclass
class ClauseTypeStats:
    """Distribution of clause types."""
    main_count: int = 0
    main_pct: float = 0.0
    subordinate_count: int = 0
    subordinate_pct: float = 0.0
    coordinate_count: int = 0
    coordinate_pct: float = 0.0


@dataclass
class ZoneStats:
    """Zone extraction statistics."""
    avg_zone_a_tokens: float = 0.0
    avg_zone_b_tokens: float = 0.0
    zone_a_content_ratio: float = 0.0
    zone_b_content_ratio: float = 0.0


@dataclass
class EchoDistribution:
    """Full echo score distribution for fingerprinting."""
    # Overall statistics
    total_pairs: int = 0
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    phonetic_max: float = 0.0
    phonetic_min: float = 0.0
    structural_mean: float = 0.0
    structural_std: float = 0.0
    structural_max: float = 0.0
    structural_min: float = 0.0
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    semantic_max: float = 0.0
    semantic_min: float = 0.0
    combined_mean: float = 0.0
    combined_std: float = 0.0
    combined_max: float = 0.0
    
    # Derived statistics
    cooccurrence_rate: float = 0.0  # All three high simultaneously
    geometric_mean: float = 0.0
    burstiness: float = 0.0  # std/mean ratio
    
    # Percentiles for distribution shape
    phonetic_p25: float = 0.0
    phonetic_p75: float = 0.0
    semantic_p25: float = 0.0
    semantic_p75: float = 0.0


@dataclass
class CognitiveFingerprint39D:
    """The 39-dimensional cognitive fingerprint."""
    # Layer A: Trajectory (5D)
    concept_jump_mean: float = 0.0
    concept_jump_variance: float = 0.0
    path_tortuosity: float = 1.0
    turning_angle_mean: float = 0.0
    return_rate: float = 0.0
    
    # Layer B: SpecHO Extended (15D)
    phonetic_mean: float = 0.0
    phonetic_std: float = 0.0
    phonetic_max: float = 0.0
    structural_mean: float = 0.0
    structural_std: float = 0.0
    structural_max: float = 0.0
    semantic_mean: float = 0.0
    semantic_std: float = 0.0
    semantic_max: float = 0.0
    cooccurrence_rate: float = 0.0
    geometric_mean: float = 0.0
    overall_mean: float = 0.0
    overall_std: float = 0.0
    overall_max: float = 0.0
    burstiness: float = 0.0
    
    # Layer C: Epistemic (6D)
    hedge_density: float = 0.0
    hedge_clustering: float = 0.0
    hedge_position_bias: float = 0.5
    confidence_mean: float = 0.5
    confidence_variance: float = 0.0
    confidence_arc: float = 0.0
    
    # Layer C: Transitions (6D)
    additive_rate: float = 0.0
    contrastive_rate: float = 0.0
    causal_rate: float = 0.0
    temporal_rate: float = 0.0
    exemplifying_rate: float = 0.0
    reformulating_rate: float = 0.0
    
    # Layer D: Syntactic (7D)
    clause_length_mean: float = 0.0
    clause_length_std: float = 0.0
    clause_rhythm_autocorr: float = 0.0
    sentence_complexity: float = 1.0
    comma_density: float = 0.0
    semicolon_rate: float = 0.0
    parenthetical_rate: float = 0.0


@dataclass
class MicroDistributions:
    """Micro-level distributions for model fingerprinting."""
    # Pair type distribution
    pair_punctuation: PairTypeStats = field(default_factory=PairTypeStats)
    pair_conjunction: PairTypeStats = field(default_factory=PairTypeStats)
    pair_transition: PairTypeStats = field(default_factory=PairTypeStats)
    
    # Clause type distribution
    clause_types: ClauseTypeStats = field(default_factory=ClauseTypeStats)
    
    # Zone statistics
    zones: ZoneStats = field(default_factory=ZoneStats)
    
    # Echo distribution
    echo: EchoDistribution = field(default_factory=EchoDistribution)
    
    # POS distribution (for micro-fingerprinting)
    pos_noun_ratio: float = 0.0
    pos_verb_ratio: float = 0.0
    pos_adj_ratio: float = 0.0
    pos_adv_ratio: float = 0.0
    pos_propn_ratio: float = 0.0
    
    # Syllable distribution
    syllable_mean: float = 0.0
    syllable_std: float = 0.0
    syllable_cv: float = 0.0  # coefficient of variation


@dataclass
class UnifiedFingerprint:
    """
    Complete unified fingerprint for a text sample.
    
    Contains ALL information needed for both:
    - Binary classification (human vs AI)
    - Model attribution (distinguishing between AI models)
    """
    # Metadata
    text_length: int = 0
    token_count: int = 0
    sentence_count: int = 0
    clause_count: int = 0
    pair_count: int = 0
    extraction_time_ms: float = 0.0
    
    # Core specHO output
    final_score: float = 0.0
    z_score: float = 0.0
    confidence: float = 0.5
    
    # 39D Cognitive Fingerprint
    fingerprint_39d: CognitiveFingerprint39D = field(default_factory=CognitiveFingerprint39D)
    
    # Micro distributions for fingerprinting
    micro: MicroDistributions = field(default_factory=MicroDistributions)
    
    def to_flat_vector(self) -> Dict[str, float]:
        """
        Convert to flat feature vector for ML models.
        Returns all features as a flat dictionary.
        """
        features = {}
        
        # Metadata features
        features['text_length'] = float(self.text_length)
        features['token_count'] = float(self.token_count)
        features['sentence_count'] = float(self.sentence_count)
        features['clause_count'] = float(self.clause_count)
        features['pair_count'] = float(self.pair_count)
        
        # Core scores
        features['final_score'] = self.final_score
        features['z_score'] = self.z_score
        features['confidence'] = self.confidence
        
        # 39D fingerprint
        fp = self.fingerprint_39d
        for key in ['concept_jump_mean', 'concept_jump_variance', 'path_tortuosity',
                    'turning_angle_mean', 'return_rate',
                    'phonetic_mean', 'phonetic_std', 'phonetic_max',
                    'structural_mean', 'structural_std', 'structural_max',
                    'semantic_mean', 'semantic_std', 'semantic_max',
                    'cooccurrence_rate', 'geometric_mean',
                    'overall_mean', 'overall_std', 'overall_max', 'burstiness',
                    'hedge_density', 'hedge_clustering', 'hedge_position_bias',
                    'confidence_mean', 'confidence_variance', 'confidence_arc',
                    'additive_rate', 'contrastive_rate', 'causal_rate',
                    'temporal_rate', 'exemplifying_rate', 'reformulating_rate',
                    'clause_length_mean', 'clause_length_std', 'clause_rhythm_autocorr',
                    'sentence_complexity', 'comma_density', 'semicolon_rate',
                    'parenthetical_rate']:
            features[f'fp39_{key}'] = getattr(fp, key)
        
        # Micro distributions
        micro = self.micro
        
        # Pair type percentages
        features['micro_pair_punct_pct'] = micro.pair_punctuation.percentage
        features['micro_pair_conj_pct'] = micro.pair_conjunction.percentage
        features['micro_pair_trans_pct'] = micro.pair_transition.percentage
        
        # Pair type scores
        for pair_type, stats in [('punct', micro.pair_punctuation),
                                  ('conj', micro.pair_conjunction),
                                  ('trans', micro.pair_transition)]:
            features[f'micro_{pair_type}_phonetic'] = stats.phonetic_mean
            features[f'micro_{pair_type}_structural'] = stats.structural_mean
            features[f'micro_{pair_type}_semantic'] = stats.semantic_mean
        
        # Clause types
        features['micro_clause_main_pct'] = micro.clause_types.main_pct
        features['micro_clause_sub_pct'] = micro.clause_types.subordinate_pct
        features['micro_clause_coord_pct'] = micro.clause_types.coordinate_pct
        
        # Zone stats
        features['micro_zone_a_tokens'] = micro.zones.avg_zone_a_tokens
        features['micro_zone_b_tokens'] = micro.zones.avg_zone_b_tokens
        
        # POS ratios
        features['micro_pos_noun'] = micro.pos_noun_ratio
        features['micro_pos_verb'] = micro.pos_verb_ratio
        features['micro_pos_adj'] = micro.pos_adj_ratio
        features['micro_pos_adv'] = micro.pos_adv_ratio
        features['micro_pos_propn'] = micro.pos_propn_ratio
        
        # Syllable stats
        features['micro_syll_mean'] = micro.syllable_mean
        features['micro_syll_std'] = micro.syllable_std
        features['micro_syll_cv'] = micro.syllable_cv
        
        # Echo distribution details
        echo = micro.echo
        features['micro_echo_pairs'] = float(echo.total_pairs)
        features['micro_echo_phonetic_range'] = echo.phonetic_max - echo.phonetic_min
        features['micro_echo_semantic_range'] = echo.semantic_max - echo.semantic_min
        features['micro_echo_phonetic_iqr'] = echo.phonetic_p75 - echo.phonetic_p25
        features['micro_echo_semantic_iqr'] = echo.semantic_p75 - echo.semantic_p25
        
        return features
    
    def to_dict(self) -> Dict:
        """Convert to nested dictionary for JSON serialization."""
        return {
            'metadata': {
                'text_length': self.text_length,
                'token_count': self.token_count,
                'sentence_count': self.sentence_count,
                'clause_count': self.clause_count,
                'pair_count': self.pair_count,
                'extraction_time_ms': self.extraction_time_ms,
            },
            'scores': {
                'final_score': self.final_score,
                'z_score': self.z_score,
                'confidence': self.confidence,
            },
            'fingerprint_39d': asdict(self.fingerprint_39d),
            'micro': {
                'pair_types': {
                    'punctuation': asdict(self.micro.pair_punctuation),
                    'conjunction': asdict(self.micro.pair_conjunction),
                    'transition': asdict(self.micro.pair_transition),
                },
                'clause_types': asdict(self.micro.clause_types),
                'zones': asdict(self.micro.zones),
                'echo': asdict(self.micro.echo),
                'pos_ratios': {
                    'noun': self.micro.pos_noun_ratio,
                    'verb': self.micro.pos_verb_ratio,
                    'adj': self.micro.pos_adj_ratio,
                    'adv': self.micro.pos_adv_ratio,
                    'propn': self.micro.pos_propn_ratio,
                },
                'syllables': {
                    'mean': self.micro.syllable_mean,
                    'std': self.micro.syllable_std,
                    'cv': self.micro.syllable_cv,
                },
            },
        }


# ============================================================================
# UNIFIED EXTRACTOR
# ============================================================================

class SpecHOUnified:
    """
    Unified SpecHO Fingerprint Extractor.
    
    Combines:
    1. specHO Core Pipeline (clause detection, echo analysis)
    2. 39D Cognitive Fingerprint (trajectory, epistemic, transitions, syntactic)
    3. Micro Distributions (pair types, clause types, POS, syllables)
    
    Usage:
        extractor = SpecHOUnified()
        fingerprint = extractor.extract("Your text here...")
        
        # For ML: get flat feature vector
        features = fingerprint.to_flat_vector()
        
        # For storage: get nested dict
        data = fingerprint.to_dict()
    """
    
    def __init__(
        self,
        baseline_path: str = None,
        semantic_model: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True
    ):
        """
        Initialize unified extractor.
        
        Args:
            baseline_path: Path to baseline stats (optional, creates default if None)
            semantic_model: Sentence transformer model name
            lazy_load: If True, delay loading heavy models until first use
        """
        self.baseline_path = baseline_path
        self.semantic_model = semantic_model
        self.lazy_load = lazy_load
        
        self._detector = None
        self._cognitive_extractor = None
        self._nlp = None
        
        if not lazy_load:
            self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        if self._detector is None:
            logging.info("Initializing SpecHO detector...")
            try:
                from detector import SpecHODetector
                
                # Use default baseline if not specified
                if self.baseline_path and Path(self.baseline_path).exists():
                    self._detector = SpecHODetector(
                        baseline_path=self.baseline_path,
                        semantic_model_path=self.semantic_model
                    )
                else:
                    # Create detector without baseline validation
                    self._detector = SpecHODetector.__new__(SpecHODetector)
                    from preprocessor.pipeline import LinguisticPreprocessor
                    from clause_identifier.pipeline import ClauseIdentifier
                    from echo_engine.pipeline import EchoAnalysisEngine
                    from scoring.pipeline import ScoringModule
                    
                    self._detector.preprocessor = LinguisticPreprocessor()
                    self._detector.clause_identifier = ClauseIdentifier()
                    self._detector.echo_engine = EchoAnalysisEngine(
                        semantic_model_path=self.semantic_model
                    )
                    self._detector.scoring_module = ScoringModule()
                    self._detector.validator = None  # Skip validation
                    
            except Exception as e:
                logging.error(f"Failed to initialize SpecHO detector: {e}")
                raise
        
        if self._cognitive_extractor is None:
            logging.info("Initializing Cognitive Fingerprint extractor...")
            try:
                from fingerprint.extractor import CognitiveExtractor
                self._cognitive_extractor = CognitiveExtractor(
                    embedding_model=self.semantic_model
                )
            except Exception as e:
                logging.warning(f"Cognitive extractor not available: {e}")
                # Try simplified trajectory analyzer
                try:
                    from fingerprint.trajectory import TrajectoryAnalyzer
                    self._trajectory_analyzer = TrajectoryAnalyzer(self.semantic_model)
                except:
                    self._trajectory_analyzer = None
                self._cognitive_extractor = None
    
    def extract(self, text: str) -> UnifiedFingerprint:
        """
        Extract complete unified fingerprint from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            UnifiedFingerprint with all features populated
        """
        start_time = time.time()
        
        # Ensure components are initialized
        self._init_components()
        
        # Initialize result
        result = UnifiedFingerprint()
        result.text_length = len(text)
        
        if not text.strip():
            return result
        
        try:
            # ================================================================
            # STAGE 1: Core specHO Pipeline
            # ================================================================
            
            # Get preprocessed tokens and doc
            tokens, doc = self._detector.preprocessor.process(text)
            result.token_count = len(tokens)
            result.sentence_count = len(list(doc.sents))
            
            # Get clause pairs
            clause_pairs = self._detector.clause_identifier.identify_pairs(tokens, doc)
            result.pair_count = len(clause_pairs)
            
            # Count clauses (unique clauses in all pairs)
            seen_clauses = set()
            for pair in clause_pairs:
                seen_clauses.add((pair.clause_a.start_idx, pair.clause_a.end_idx))
                seen_clauses.add((pair.clause_b.start_idx, pair.clause_b.end_idx))
            result.clause_count = len(seen_clauses)
            
            # Calculate echo scores
            echo_scores = []
            for pair in clause_pairs:
                try:
                    score = self._detector.echo_engine.analyze_pair(pair)
                    echo_scores.append(score)
                except Exception as e:
                    logging.debug(f"Echo analysis failed for pair: {e}")
            
            # Aggregate scores
            if echo_scores:
                result.final_score = self._detector.scoring_module.score_document(echo_scores)
            
            # Statistical validation (if available)
            if self._detector.validator:
                result.z_score, result.confidence = self._detector.validator.validate(
                    result.final_score
                )
            
            # ================================================================
            # STAGE 2: Extract Micro Distributions
            # ================================================================
            
            self._extract_micro_distributions(
                result, tokens, doc, clause_pairs, echo_scores
            )
            
            # ================================================================
            # STAGE 3: 39D Cognitive Fingerprint
            # ================================================================
            
            if self._cognitive_extractor:
                try:
                    cognitive_fp = self._cognitive_extractor.extract_from_specho(
                        text, echo_scores
                    )
                    # Copy all fields
                    result.fingerprint_39d = CognitiveFingerprint39D(
                        **cognitive_fp.to_dict()
                    )
                except Exception as e:
                    logging.warning(f"Cognitive fingerprint extraction failed: {e}")
                    # Fall back to extracting what we can from echo scores
                    self._extract_basic_39d(result, echo_scores)
            else:
                # Extract basic 39D from available data
                self._extract_basic_39d(result, echo_scores)
            
        except Exception as e:
            logging.error(f"Fingerprint extraction failed: {e}")
            raise
        
        result.extraction_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _extract_micro_distributions(
        self,
        result: UnifiedFingerprint,
        tokens: List,
        doc,
        clause_pairs: List,
        echo_scores: List
    ):
        """Extract micro-level distributions from analysis results."""
        micro = result.micro
        
        # ================================================================
        # Pair Type Distribution
        # ================================================================
        
        pair_type_counts = defaultdict(int)
        pair_type_scores = defaultdict(lambda: {'phonetic': [], 'structural': [], 'semantic': []})
        
        for pair, score in zip(clause_pairs, echo_scores):
            ptype = pair.pair_type
            pair_type_counts[ptype] += 1
            pair_type_scores[ptype]['phonetic'].append(score.phonetic_score)
            pair_type_scores[ptype]['structural'].append(score.structural_score)
            pair_type_scores[ptype]['semantic'].append(score.semantic_score)
        
        total_pairs = sum(pair_type_counts.values()) or 1
        
        for ptype, stats_obj in [
            ('punctuation', micro.pair_punctuation),
            ('conjunction', micro.pair_conjunction),
            ('transition', micro.pair_transition)
        ]:
            count = pair_type_counts.get(ptype, 0)
            stats_obj.count = count
            stats_obj.percentage = (count / total_pairs) * 100
            
            scores = pair_type_scores.get(ptype, {})
            if scores.get('phonetic'):
                stats_obj.phonetic_mean = float(np.mean(scores['phonetic']))
                stats_obj.phonetic_std = float(np.std(scores['phonetic']))
                stats_obj.structural_mean = float(np.mean(scores['structural']))
                stats_obj.structural_std = float(np.std(scores['structural']))
                stats_obj.semantic_mean = float(np.mean(scores['semantic']))
                stats_obj.semantic_std = float(np.std(scores['semantic']))
                stats_obj.combined_mean = float(np.mean([
                    (p + st + se) / 3 
                    for p, st, se in zip(scores['phonetic'], scores['structural'], scores['semantic'])
                ]))
        
        # ================================================================
        # Clause Type Distribution
        # ================================================================
        
        clause_type_counts = defaultdict(int)
        for pair in clause_pairs:
            clause_type_counts[pair.clause_a.clause_type] += 1
            clause_type_counts[pair.clause_b.clause_type] += 1
        
        total_clauses = sum(clause_type_counts.values()) or 1
        micro.clause_types.main_count = clause_type_counts.get('main', 0)
        micro.clause_types.main_pct = (micro.clause_types.main_count / total_clauses) * 100
        micro.clause_types.subordinate_count = clause_type_counts.get('subordinate', 0)
        micro.clause_types.subordinate_pct = (micro.clause_types.subordinate_count / total_clauses) * 100
        micro.clause_types.coordinate_count = clause_type_counts.get('coordinate', 0)
        micro.clause_types.coordinate_pct = (micro.clause_types.coordinate_count / total_clauses) * 100
        
        # ================================================================
        # Zone Statistics
        # ================================================================
        
        zone_a_sizes = []
        zone_b_sizes = []
        zone_a_content = []
        zone_b_content = []
        
        for pair in clause_pairs:
            zone_a_sizes.append(len(pair.zone_a_tokens))
            zone_b_sizes.append(len(pair.zone_b_tokens))
            
            # Content word ratio in zones
            if pair.zone_a_tokens:
                zone_a_content.append(
                    sum(1 for t in pair.zone_a_tokens if t.is_content_word) / len(pair.zone_a_tokens)
                )
            if pair.zone_b_tokens:
                zone_b_content.append(
                    sum(1 for t in pair.zone_b_tokens if t.is_content_word) / len(pair.zone_b_tokens)
                )
        
        if zone_a_sizes:
            micro.zones.avg_zone_a_tokens = float(np.mean(zone_a_sizes))
            micro.zones.avg_zone_b_tokens = float(np.mean(zone_b_sizes))
        if zone_a_content:
            micro.zones.zone_a_content_ratio = float(np.mean(zone_a_content))
        if zone_b_content:
            micro.zones.zone_b_content_ratio = float(np.mean(zone_b_content))
        
        # ================================================================
        # Echo Distribution Details
        # ================================================================
        
        if echo_scores:
            echo = micro.echo
            echo.total_pairs = len(echo_scores)
            
            phonetic = [s.phonetic_score for s in echo_scores]
            structural = [s.structural_score for s in echo_scores]
            semantic = [s.semantic_score for s in echo_scores]
            combined = [(p + st + se) / 3 for p, st, se in zip(phonetic, structural, semantic)]
            
            echo.phonetic_mean = float(np.mean(phonetic))
            echo.phonetic_std = float(np.std(phonetic)) if len(phonetic) > 1 else 0.0
            echo.phonetic_max = float(np.max(phonetic))
            echo.phonetic_min = float(np.min(phonetic))
            echo.phonetic_p25 = float(np.percentile(phonetic, 25))
            echo.phonetic_p75 = float(np.percentile(phonetic, 75))
            
            echo.structural_mean = float(np.mean(structural))
            echo.structural_std = float(np.std(structural)) if len(structural) > 1 else 0.0
            echo.structural_max = float(np.max(structural))
            echo.structural_min = float(np.min(structural))
            
            echo.semantic_mean = float(np.mean(semantic))
            echo.semantic_std = float(np.std(semantic)) if len(semantic) > 1 else 0.0
            echo.semantic_max = float(np.max(semantic))
            echo.semantic_min = float(np.min(semantic))
            echo.semantic_p25 = float(np.percentile(semantic, 25))
            echo.semantic_p75 = float(np.percentile(semantic, 75))
            
            echo.combined_mean = float(np.mean(combined))
            echo.combined_std = float(np.std(combined)) if len(combined) > 1 else 0.0
            echo.combined_max = float(np.max(combined))
            
            # Cooccurrence: all three high
            threshold = 0.6
            echo.cooccurrence_rate = sum(
                1 for p, st, se in zip(phonetic, structural, semantic)
                if p > threshold and st > threshold and se > threshold
            ) / len(echo_scores)
            
            # Geometric mean
            geometric = [pow(max(0.001, p * st * se), 1/3) 
                        for p, st, se in zip(phonetic, structural, semantic)]
            echo.geometric_mean = float(np.mean(geometric))
            
            # Burstiness
            echo.burstiness = echo.combined_std / echo.combined_mean if echo.combined_mean > 0 else 0.0
        
        # ================================================================
        # POS Distribution
        # ================================================================
        
        pos_counts = defaultdict(int)
        total_tokens = len(tokens) or 1
        
        for token in tokens:
            pos_counts[token.pos_tag] += 1
        
        micro.pos_noun_ratio = pos_counts.get('NOUN', 0) / total_tokens
        micro.pos_verb_ratio = pos_counts.get('VERB', 0) / total_tokens
        micro.pos_adj_ratio = pos_counts.get('ADJ', 0) / total_tokens
        micro.pos_adv_ratio = pos_counts.get('ADV', 0) / total_tokens
        micro.pos_propn_ratio = pos_counts.get('PROPN', 0) / total_tokens
        
        # ================================================================
        # Syllable Distribution
        # ================================================================
        
        syllables = [t.syllable_count for t in tokens if t.syllable_count > 0]
        if syllables:
            micro.syllable_mean = float(np.mean(syllables))
            micro.syllable_std = float(np.std(syllables)) if len(syllables) > 1 else 0.0
            micro.syllable_cv = micro.syllable_std / micro.syllable_mean if micro.syllable_mean > 0 else 0.0
    
    def _extract_basic_39d(self, result: UnifiedFingerprint, echo_scores: List):
        """Extract basic 39D features from echo scores when cognitive extractor unavailable."""
        fp = result.fingerprint_39d
        
        if not echo_scores:
            return
        
        # Extract SpecHO Extended features (Layer B) from echo scores
        phonetic = [s.phonetic_score for s in echo_scores]
        structural = [s.structural_score for s in echo_scores]
        semantic = [s.semantic_score for s in echo_scores]
        combined = [(p + st + se) / 3 for p, st, se in zip(phonetic, structural, semantic)]
        
        fp.phonetic_mean = float(np.mean(phonetic))
        fp.phonetic_std = float(np.std(phonetic)) if len(phonetic) > 1 else 0.0
        fp.phonetic_max = float(np.max(phonetic))
        
        fp.structural_mean = float(np.mean(structural))
        fp.structural_std = float(np.std(structural)) if len(structural) > 1 else 0.0
        fp.structural_max = float(np.max(structural))
        
        fp.semantic_mean = float(np.mean(semantic))
        fp.semantic_std = float(np.std(semantic)) if len(semantic) > 1 else 0.0
        fp.semantic_max = float(np.max(semantic))
        
        fp.overall_mean = float(np.mean(combined))
        fp.overall_std = float(np.std(combined)) if len(combined) > 1 else 0.0
        fp.overall_max = float(np.max(combined))
        
        # Cooccurrence
        threshold = 0.6
        fp.cooccurrence_rate = sum(
            1 for p, st, se in zip(phonetic, structural, semantic)
            if p > threshold and st > threshold and se > threshold
        ) / len(echo_scores)
        
        # Geometric mean
        geometric = [pow(max(0.001, p * st * se), 1/3) 
                    for p, st, se in zip(phonetic, structural, semantic)]
        fp.geometric_mean = float(np.mean(geometric))
        
        # Burstiness
        fp.burstiness = fp.overall_std / fp.overall_mean if fp.overall_mean > 0 else 0.0
    
    def extract_batch(self, texts: List[str]) -> List[UnifiedFingerprint]:
        """Extract fingerprints for multiple texts."""
        return [self.extract(text) for text in texts]
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names in flat vector."""
        # Extract a dummy fingerprint to get feature names
        dummy = UnifiedFingerprint()
        return list(dummy.to_flat_vector().keys())


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for unified fingerprint extraction."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="SpecHO Unified Fingerprint Extractor")
    parser.add_argument("text", nargs="?", help="Text to analyze (or use --file)")
    parser.add_argument("--file", "-f", help="File containing text to analyze")
    parser.add_argument("--output", "-o", help="Output file for JSON results")
    parser.add_argument("--flat", action="store_true", help="Output flat feature vector")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Get text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide text or --file")
        sys.exit(1)
    
    # Extract fingerprint
    extractor = SpecHOUnified()
    fingerprint = extractor.extract(text)
    
    # Output
    if args.flat:
        output = fingerprint.to_flat_vector()
    else:
        output = fingerprint.to_dict()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
