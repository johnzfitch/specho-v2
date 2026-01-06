"""
ASAP (Automated Student Assessment Prize) Essay Collector

Source: Kaggle ASAP dataset
Contains: ~13,000 student essays from 2012
License: Competition data (research use)
Pre-LLM: Yes (2012)

This is one of the cleanest academic essay datasets available.
Essays are graded, so we can optionally stratify by quality.
"""

import csv
from pathlib import Path
from typing import Iterator, Optional
from datetime import date
import logging

from ..schema import CorpusSample, Domain, Subdomain, DomainConfig
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register
class ASAPCollector(BaseCollector):
    """Collector for ASAP essay dataset."""
    
    DOMAIN = Domain.ACADEMIC
    SUBDOMAIN = Subdomain.ESSAYS
    SOURCE_NAME = "asap"
    SOURCE_VERSION = "1.0"
    LICENSE = "kaggle_competition"
    
    # ASAP was from 2012
    CONTENT_DATE = date(2012, 1, 1)
    
    # Essay set descriptions (for metadata)
    ESSAY_SET_INFO = {
        1: {"topic": "computers_effects", "type": "persuasive"},
        2: {"topic": "censorship_libraries", "type": "persuasive"},
        3: {"topic": "source_dependent_response", "type": "analytical"},
        4: {"topic": "source_dependent_response", "type": "analytical"},
        5: {"topic": "source_dependent_response", "type": "analytical"},
        6: {"topic": "source_dependent_response", "type": "analytical"},
        7: {"topic": "patience_narrative", "type": "narrative"},
        8: {"topic": "laughter_narrative", "type": "narrative"},
    }
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        min_score_percentile: float = 0.0,  # Filter low-quality essays
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default path - user should download from Kaggle
        self.data_path = data_path or self.cache_dir / "training_set_rel3.tsv"
        self.min_score_percentile = min_score_percentile
        
        if not self.data_path.exists():
            logger.warning(
                f"ASAP data not found at {self.data_path}. "
                f"Download from: https://www.kaggle.com/c/asap-aes/data"
            )
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield essays from ASAP dataset."""
        
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        with open(self.data_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                if collected >= target:
                    break
                
                try:
                    essay_id = row.get('essay_id', row.get('id', ''))
                    essay_set = int(row.get('essay_set', 1))
                    essay_text = row.get('essay', '')
                    
                    # Skip empty
                    if not essay_text or not essay_text.strip():
                        continue
                    
                    # Clean text
                    text = self._clean_essay(essay_text)
                    
                    # Create sample
                    sample = self.create_sample(
                        text=text,
                        original_id=str(essay_id),
                        content_date=self.CONTENT_DATE,
                        url="https://www.kaggle.com/c/asap-aes",
                    )
                    
                    # Add essay set info to metadata
                    set_info = self.ESSAY_SET_INFO.get(essay_set, {})
                    
                    # Validate
                    if self.validate_sample(sample):
                        self._seen_hashes.add(sample.content_hash)
                        collected += 1
                        yield sample
                        
                        if collected % 500 == 0:
                            logger.info(f"Collected {collected}/{target} ASAP essays")
                
                except Exception as e:
                    logger.warning(f"Error processing essay {essay_id}: {e}")
                    continue
        
        logger.info(f"ASAP collection complete: {collected} essays")
    
    def _clean_essay(self, text: str) -> str:
        """Clean and normalize essay text."""
        # Remove weird artifacts from OCR/data entry
        text = text.replace('@CAPS', '')
        text = text.replace('@PERSON', '[PERSON]')
        text = text.replace('@LOCATION', '[LOCATION]')
        text = text.replace('@ORGANIZATION', '[ORG]')
        text = text.replace('@DATE', '[DATE]')
        text = text.replace('@TIME', '[TIME]')
        text = text.replace('@MONEY', '[MONEY]')
        text = text.replace('@PERCENT', '[PERCENT]')
        text = text.replace('@NUM', '[NUM]')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


# Additional collector for Persuade 2.0 dataset
@DomainRegistry.register  
class Persuade2Collector(BaseCollector):
    """Collector for Persuade 2.0 argumentative essay corpus."""
    
    DOMAIN = Domain.ACADEMIC
    SUBDOMAIN = Subdomain.ESSAYS
    SOURCE_NAME = "persuade2"
    SOURCE_VERSION = "2.0"
    LICENSE = "research"
    
    # Persuade 2.0 essays are from 2020-2021
    CONTENT_DATE = date(2021, 1, 1)
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path or self.cache_dir / "persuade_2.0_human_scores_demo_id_github.csv"
        
        if not self.data_path.exists():
            logger.warning(
                f"Persuade 2.0 data not found at {self.data_path}. "
                f"Download from: https://github.com/scrosseern/persuade_corpus_2.0"
            )
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield essays from Persuade 2.0."""
        
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if collected >= target:
                    break
                
                try:
                    essay_id = row.get('essay_id_comp', row.get('essay_id', ''))
                    text = row.get('full_text', '')
                    
                    if not text or not text.strip():
                        continue
                    
                    # Clean
                    text = ' '.join(text.split())
                    
                    sample = self.create_sample(
                        text=text,
                        original_id=str(essay_id),
                        content_date=self.CONTENT_DATE,
                        url="https://github.com/scrosseern/persuade_corpus_2.0",
                    )
                    
                    if self.validate_sample(sample):
                        self._seen_hashes.add(sample.content_hash)
                        collected += 1
                        yield sample
                        
                        if collected % 500 == 0:
                            logger.info(f"Collected {collected}/{target} Persuade essays")
                
                except Exception as e:
                    logger.warning(f"Error processing essay: {e}")
                    continue
        
        logger.info(f"Persuade 2.0 collection complete: {collected} essays")
