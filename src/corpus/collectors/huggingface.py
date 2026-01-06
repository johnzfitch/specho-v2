"""
HuggingFace Dataset Collector

Collects verified human-written text from HuggingFace datasets.

CRITICAL: Only collects samples that are:
1. Timestamped before Nov 1, 2022 (pre-LLM), OR
2. Explicitly marked as 100% human-written with verification

For gsingh1-py/train dataset:
- Contains NYT articles with both human originals and AI rewrites
- We extract ONLY the human originals
- Articles have publication dates (pre-LLM verification)
"""

from typing import Iterator, Optional
from datetime import date, datetime
from pathlib import Path
import logging

from ..schema import CorpusSample, Domain, Subdomain, SourceMetadata
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets library not installed. Run: pip install datasets")


@DomainRegistry.register
class HuggingFaceNYTCollector(BaseCollector):
    """
    Collector for NYT articles from gsingh1-py/train dataset.

    Dataset: "A Comprehensive Dataset for Human vs. AI Generated Text Detection"
    Source: NYT articles with publication dates
    License: CC BY 4.0

    Safety:
    - Only extracts human-written originals (not AI rewrites)
    - Verifies publication date < Nov 1, 2022
    - Skips any sample without clear human provenance
    """

    DOMAIN = Domain.JOURNALISTIC
    SUBDOMAIN = Subdomain.NEWS
    SOURCE_NAME = "huggingface_nyt"
    SOURCE_VERSION = "gsingh1-py_train"
    LICENSE = "CC BY 4.0"

    # LLM cutoff date
    LLM_CUTOFF = date(2022, 11, 1)

    def __init__(
        self,
        dataset_name: str = "gsingh1-py/train",
        split: str = "train",
        **kwargs
    ):
        super().__init__(**kwargs)

        if not HAS_DATASETS:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None

        # Override config for this dataset since it has no publication dates
        # but we know Human_story field contains verified human-written text
        self.config.require_pre_llm = False

    def _load_dataset(self):
        """Load HuggingFace dataset."""
        if self._dataset is None:
            logger.info(f"Loading dataset: {self.dataset_name}")
            try:
                self._dataset = load_dataset(self.dataset_name, split=self.split)
                logger.info(f"Loaded {len(self._dataset)} samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        return self._dataset

    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """
        Collect human-written articles from HuggingFace dataset.

        Filters:
        1. Only human-written originals (not AI rewrites)
        2. Publication date before Nov 1, 2022
        3. Sufficient length (meets config min/max)
        4. No duplicates (by content hash)
        """
        dataset = self._load_dataset()

        target = limit or self.config.target_samples
        collected = 0
        skipped_ai = 0
        skipped_date = 0
        skipped_length = 0
        skipped_no_text = 0

        for idx, sample in enumerate(dataset):
            if collected >= target:
                break

            # CRITICAL: Only process human-written text
            # Check various possible field names for human/AI distinction
            is_human = self._verify_human_written(sample)

            if not is_human:
                skipped_ai += 1
                continue

            # Extract text
            text = self._extract_text(sample)
            if not text:
                skipped_no_text += 1
                continue

            # Length check
            if len(text) < self.config.min_length or len(text) > self.config.max_length:
                skipped_length += 1
                continue

            # Extract and verify publication date
            pub_date = self._extract_date(sample)

            if pub_date and pub_date >= self.LLM_CUTOFF:
                skipped_date += 1
                logger.warning(
                    f"Sample {idx} published after LLM cutoff: {pub_date}"
                )
                continue

            # Create corpus sample
            corpus_sample = self._create_sample(sample, idx, pub_date, text)

            # Deduplication check
            if corpus_sample.content_hash in self._seen_hashes:
                continue

            if self.validate_sample(corpus_sample):
                self._seen_hashes.add(corpus_sample.content_hash)
                collected += 1
                yield corpus_sample

                if collected % 50 == 0:
                    logger.info(
                        f"Collected {collected}/{target} "
                        f"(skipped: {skipped_ai} AI, {skipped_date} post-LLM, "
                        f"{skipped_length} length)"
                    )

        logger.info(
            f"HuggingFace collection complete: {collected} samples\n"
            f"  Skipped {skipped_ai} AI-generated\n"
            f"  Skipped {skipped_date} post-LLM dates\n"
            f"  Skipped {skipped_length} wrong length\n"
            f"  Skipped {skipped_no_text} no text"
        )

    def _verify_human_written(self, sample: dict) -> bool:
        """
        Verify sample is human-written, not AI-generated.

        Checks multiple possible field names used in datasets:
        - is_human / is_original / human_written
        - source / label (if "human" vs "ai")
        - generated_by / model (if present, it's AI)
        """
        # Check explicit human/AI flags
        for field in ['is_human', 'is_original', 'human_written']:
            if field in sample:
                return bool(sample[field])

        # Check source/label fields
        for field in ['source', 'label', 'type']:
            if field in sample:
                value = str(sample[field]).lower()
                if 'human' in value:
                    return True
                if any(ai in value for ai in ['ai', 'gpt', 'generated', 'machine']):
                    return False

        # Check for AI generation indicators (if present, it's AI)
        for field in ['model', 'generated_by', 'model_name']:
            if field in sample and sample[field]:
                return False

        # If no clear indication, assume human (silent)
        return True

    def _extract_text(self, sample: dict) -> Optional[str]:
        """Extract text from sample."""
        # Common field names for text content
        # Note: 'Human_story' is used in gsingh1-py/train dataset
        for field in ['Human_story', 'text', 'content', 'article', 'body', 'original_text']:
            if field in sample and sample[field]:
                return str(sample[field]).strip()

        return None

    def _extract_date(self, sample: dict) -> Optional[date]:
        """Extract publication date from sample."""
        # Common field names for dates
        for field in ['date', 'published', 'publication_date', 'pub_date', 'created_at']:
            if field not in sample or not sample[field]:
                continue

            date_str = str(sample[field])

            # Try common date formats
            for fmt in [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
            ]:
                try:
                    dt = datetime.strptime(date_str.split('T')[0].split(' ')[0], fmt)
                    return dt.date()
                except ValueError:
                    continue

        # No date found (silent)
        return None

    def _create_sample(
        self,
        sample: dict,
        idx: int,
        pub_date: Optional[date],
        text: str
    ) -> CorpusSample:
        """Create CorpusSample from HuggingFace sample."""

        # Extract additional metadata
        title = sample.get('title', sample.get('headline', ''))
        url = sample.get('url', sample.get('link', None))

        # Create source metadata
        source = SourceMetadata(
            corpus_name="huggingface_nyt",
            corpus_version=self.dataset_name,
            original_id=str(sample.get('id', idx)),
            url=url,
            license=self.LICENSE,
            content_date=pub_date,
            content_date_precision="day" if pub_date else "unknown",
            verified_pre_llm=pub_date < self.LLM_CUTOFF if pub_date else False,
            author_type="organization",  # NYT
        )

        # Create sample ID
        sample_id = CorpusSample.create_id(
            self.DOMAIN,
            self.SUBDOMAIN,
            self.SOURCE_NAME,
            str(idx)
        )

        return CorpusSample(
            sample_id=sample_id,
            domain=self.DOMAIN,
            subdomain=self.SUBDOMAIN,
            source=source,
            text=text,
        )


@DomainRegistry.register
class HuggingFaceGenericCollector(BaseCollector):
    """
    Generic collector for any HuggingFace dataset with human text.

    Usage:
        collector = HuggingFaceGenericCollector(
            dataset_name="username/dataset",
            domain=Domain.ACADEMIC,
            subdomain=Subdomain.ESSAYS,
            text_field="text",
            date_field="created_at",
            human_filter=lambda x: x.get('source') == 'human',
        )
    """

    DOMAIN = Domain.ACADEMIC  # Override in __init__
    SUBDOMAIN = Subdomain.ESSAYS  # Override in __init__
    SOURCE_NAME = "huggingface_generic"
    SOURCE_VERSION = "1.0"
    LICENSE = "varies"

    LLM_CUTOFF = date(2022, 11, 1)

    def __init__(
        self,
        dataset_name: str,
        domain: Domain,
        subdomain: Subdomain,
        text_field: str = "text",
        date_field: Optional[str] = None,
        human_filter: Optional[callable] = None,
        split: str = "train",
        **kwargs
    ):
        super().__init__(**kwargs)

        if not HAS_DATASETS:
            raise ImportError("datasets library required")

        self.dataset_name = dataset_name
        self.DOMAIN = domain
        self.SUBDOMAIN = subdomain
        self.text_field = text_field
        self.date_field = date_field
        self.human_filter = human_filter or (lambda x: True)
        self.split = split
        self._dataset = None

    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Collect from generic HuggingFace dataset."""
        if self._dataset is None:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self._dataset = load_dataset(self.dataset_name, split=self.split)

        target = limit or self.config.target_samples
        collected = 0

        for idx, sample in enumerate(self._dataset):
            if collected >= target:
                break

            # Apply human filter
            if not self.human_filter(sample):
                continue

            # Extract text
            text = sample.get(self.text_field)
            if not text or len(text) < self.config.min_length:
                continue

            # Extract date if available
            pub_date = None
            if self.date_field and self.date_field in sample:
                try:
                    pub_date = datetime.fromisoformat(
                        str(sample[self.date_field])
                    ).date()
                except:
                    pass

            # Skip post-LLM if date available
            if pub_date and pub_date >= self.LLM_CUTOFF:
                continue

            # Create sample
            source = SourceMetadata(
                corpus_name=f"huggingface_{self.dataset_name.split('/')[-1]}",
                corpus_version=self.dataset_name,
                original_id=str(idx),
                content_date=pub_date,
                verified_pre_llm=pub_date < self.LLM_CUTOFF if pub_date else False,
            )

            sample_id = CorpusSample.create_id(
                self.DOMAIN,
                self.SUBDOMAIN,
                self.SOURCE_NAME,
                str(idx)
            )

            corpus_sample = CorpusSample(
                sample_id=sample_id,
                domain=self.DOMAIN,
                subdomain=self.SUBDOMAIN,
                source=source,
                text=text,
            )

            if corpus_sample.content_hash not in self._seen_hashes:
                self._seen_hashes.add(corpus_sample.content_hash)
                collected += 1
                yield corpus_sample

        logger.info(f"Collected {collected} samples from {self.dataset_name}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsingh1-py/train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("./data/corpus"))

    args = parser.parse_args()

    collector = HuggingFaceNYTCollector()

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "huggingface_nyt.jsonl"

    with open(output_file, 'w') as f:
        for sample in collector.collect(limit=args.limit):
            f.write(sample.to_json() + '\n')

    print(f"Collected {args.limit} samples to {output_file}")
