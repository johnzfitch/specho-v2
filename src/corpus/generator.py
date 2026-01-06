"""
PACT AI Rewrite Generator

Takes human-written corpus samples and generates AI-rewritten
variants across multiple models, prompts, and temperatures.

This creates the paired data needed for SpecHO training and validation.
"""

import asyncio
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

from .schema import CorpusSample, Domain, Subdomain, SourceMetadata

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for a rewrite prompt."""
    id: int
    text: str
    style: str
    hypothesis: str = ""


@dataclass
class ModelConfig:
    """Configuration for a model to use for rewrites."""
    name: str                   # Full model name (e.g., "gpt-4o-2024-08-06")
    family: str                 # Provider family (openai, anthropic, google, xai)
    temperatures: list[float]   # Temperatures to test
    max_output_tokens: int = 4096
    
    @property
    def short_name(self) -> str:
        """Short name for IDs."""
        return self.name.split('/')[-1].replace('-', '_')


# Default prompt configurations
DEFAULT_PROMPTS = [
    PromptConfig(
        id=0,
        text="Rewrite this:",
        style="minimal",
        hypothesis="Even minimal prompts should trigger watermarking",
    ),
    PromptConfig(
        id=1,
        text="Rewrite this text in your own words while preserving the meaning.",
        style="neutral",
        hypothesis="Baseline detection scenario",
    ),
    PromptConfig(
        id=2,
        text="Rewrite this text to improve clarity, flow, and readability.",
        style="improvement",
        hypothesis="Improvement prompts may increase stylistic changes",
    ),
    PromptConfig(
        id=3,
        text="Rewrite this text in a more formal, academic tone suitable for publication.",
        style="formal",
        hypothesis="Register constraints may affect echo patterns",
    ),
    PromptConfig(
        id=4,
        text="Rewrite this text in a casual, conversational tone for a general audience.",
        style="casual",
        hypothesis="Casual register may reduce structural echoing",
    ),
]


# Default model configurations
DEFAULT_MODELS = [
    ModelConfig(
        name="gpt-4o-2024-08-06",
        family="openai",
        temperatures=[0.7, 1.0],
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        family="anthropic",
        temperatures=[0.7, 1.0],
    ),
    ModelConfig(
        name="gemini-1.5-pro",
        family="google",
        temperatures=[0.7, 1.0],
    ),
    ModelConfig(
        name="grok-2",
        family="xai",
        temperatures=[0.7, 1.0],
    ),
    ModelConfig(
        name="meta-llama/Llama-3.1-70B-Instruct",
        family="meta",
        temperatures=[0.7],
        max_output_tokens=2048,
    ),
]


class RewriteGenerator:
    """
    Generates AI rewrites of human-written text.
    
    Usage:
        generator = RewriteGenerator(
            output_dir="./rewrites",
            models=[...],  # Optional, uses defaults
            prompts=[...], # Optional, uses defaults
        )
        
        # Generate rewrites for a corpus
        await generator.process_corpus(
            corpus_path="./corpus/academic.jsonl",
            runs_per_config=3,
        )
    """
    
    def __init__(
        self,
        output_dir: Path | str,
        models: Optional[list[ModelConfig]] = None,
        prompts: Optional[list[PromptConfig]] = None,
        rate_limit_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = models or DEFAULT_MODELS
        self.prompts = prompts or DEFAULT_PROMPTS
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize API clients
        self._clients = {}
        self._init_clients()
        
        # Stats tracking
        self.stats = {
            'total_generated': 0,
            'by_model': {},
            'by_prompt': {},
            'errors': 0,
        }
    
    def _init_clients(self):
        """Initialize API clients for each model family."""
        
        # OpenAI
        if any(m.family == 'openai' for m in self.models):
            try:
                from openai import OpenAI
                self._clients['openai'] = OpenAI()
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
        
        # Anthropic
        if any(m.family == 'anthropic' for m in self.models):
            try:
                from anthropic import Anthropic
                self._clients['anthropic'] = Anthropic()
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")
        
        # Google
        if any(m.family == 'google' for m in self.models):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                self._clients['google'] = genai
            except Exception as e:
                logger.warning(f"Could not initialize Google client: {e}")
        
        # xAI (uses OpenAI-compatible API)
        if any(m.family == 'xai' for m in self.models):
            try:
                from openai import OpenAI
                self._clients['xai'] = OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=os.environ.get('XAI_API_KEY'),
                )
            except Exception as e:
                logger.warning(f"Could not initialize xAI client: {e}")
        
        # Meta (via Together, Replicate, or local)
        if any(m.family == 'meta' for m in self.models):
            try:
                from openai import OpenAI
                self._clients['meta'] = OpenAI(
                    base_url="https://api.together.xyz/v1",
                    api_key=os.environ.get('TOGETHER_API_KEY'),
                )
            except Exception as e:
                logger.warning(f"Could not initialize Meta/Together client: {e}")
    
    async def process_corpus(
        self,
        corpus_path: Path | str,
        runs_per_config: int = 3,
        limit: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Process a corpus file and generate rewrites.
        
        Args:
            corpus_path: Path to JSONL corpus file
            runs_per_config: Number of runs per model/prompt/temp combination
            limit: Max samples to process (None = all)
            resume_from: sample_id to resume from (for crash recovery)
        """
        corpus_path = Path(corpus_path)
        
        if not corpus_path.exists():
            logger.error(f"Corpus not found: {corpus_path}")
            return
        
        # Output file
        output_file = self.output_dir / f"rewrites_{corpus_path.stem}.jsonl"
        
        # Track what we've already done
        done_ids = set()
        if resume_from:
            # Load existing output to find done IDs
            if output_file.exists():
                with open(output_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            done_ids.add(data.get('parent_id'))
                        except:
                            continue
        
        processed = 0
        skipped = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'a', encoding='utf-8') as f_out:
                for line in f_in:
                    if limit and processed >= limit:
                        break
                    
                    try:
                        data = json.loads(line)
                        sample = CorpusSample.from_dict(data)
                        
                        # Skip if already processed
                        if sample.sample_id in done_ids:
                            skipped += 1
                            continue
                        
                        # Generate all variants
                        variants = await self.generate_variants(
                            sample,
                            runs_per_config=runs_per_config,
                        )
                        
                        # Write variants
                        for variant in variants:
                            f_out.write(variant.to_json() + '\n')
                        
                        processed += 1
                        
                        if processed % 10 == 0:
                            logger.info(f"Processed {processed} samples ({skipped} skipped)")
                        
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}")
                        self.stats['errors'] += 1
                        continue
        
        logger.info(f"Generation complete: {processed} samples processed, {self.stats['total_generated']} variants generated")
        self._save_stats()
    
    async def generate_variants(
        self,
        original: CorpusSample,
        runs_per_config: int = 3,
    ) -> list[CorpusSample]:
        """Generate all model × prompt × temp × run variants for one sample."""
        
        variants = []
        
        for model in self.models:
            if model.family not in self._clients:
                logger.debug(f"Skipping {model.name} - no client")
                continue
            
            for prompt in self.prompts:
                for temp in model.temperatures:
                    for run in range(runs_per_config):
                        try:
                            variant = await self._generate_single(
                                original=original,
                                model=model,
                                prompt=prompt,
                                temperature=temp,
                                run_number=run + 1,
                            )
                            
                            if variant:
                                variants.append(variant)
                                self.stats['total_generated'] += 1
                                self.stats['by_model'][model.name] = self.stats['by_model'].get(model.name, 0) + 1
                                self.stats['by_prompt'][prompt.style] = self.stats['by_prompt'].get(prompt.style, 0) + 1
                            
                            # Rate limiting
                            await asyncio.sleep(self.rate_limit_delay)
                            
                        except Exception as e:
                            logger.warning(f"Error generating variant: {e}")
                            self.stats['errors'] += 1
                            continue
        
        return variants
    
    async def _generate_single(
        self,
        original: CorpusSample,
        model: ModelConfig,
        prompt: PromptConfig,
        temperature: float,
        run_number: int,
    ) -> Optional[CorpusSample]:
        """Generate a single rewrite."""
        
        full_prompt = f"{prompt.text}\n\n{original.text}"
        
        client = self._clients.get(model.family)
        if not client:
            return None
        
        rewritten_text = None
        
        try:
            if model.family == 'anthropic':
                response = client.messages.create(
                    model=model.name,
                    max_tokens=model.max_output_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                rewritten_text = response.content[0].text
                
            elif model.family in ['openai', 'xai', 'meta']:
                response = client.chat.completions.create(
                    model=model.name,
                    max_tokens=model.max_output_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                rewritten_text = response.choices[0].message.content
                
            elif model.family == 'google':
                gen_model = client.GenerativeModel(model.name)
                response = gen_model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': model.max_output_tokens,
                    }
                )
                rewritten_text = response.text
            
        except Exception as e:
            logger.warning(f"API error for {model.name}: {e}")
            return None
        
        if not rewritten_text:
            return None
        
        # Create variant sample
        variant_id = (
            f"{original.sample_id}_"
            f"{model.short_name}_"
            f"p{prompt.id}_"
            f"t{int(temperature*10)}_"
            f"r{run_number}"
        )
        
        # Copy source metadata
        source = SourceMetadata(
            corpus_name=f"rewrite_{model.family}",
            corpus_version=model.name,
            original_id=original.sample_id,
            url=original.source.url,
            license=original.source.license,
            content_date=original.source.content_date,
            verified_pre_llm=False,  # AI-generated is never pre-LLM
        )
        
        return CorpusSample(
            sample_id=variant_id,
            domain=original.domain,
            subdomain=original.subdomain,
            source=source,
            text=rewritten_text,
            is_original=False,
            parent_id=original.sample_id,
            model=model.name,
            model_family=model.family,
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            prompt_style=prompt.style,
            generation_temp=temperature,
            generation_timestamp=datetime.utcnow(),
            run_number=run_number,
        )
    
    def _save_stats(self) -> None:
        """Save generation statistics."""
        stats_file = self.output_dir / "generation_stats.json"
        
        self.stats['generation_completed'] = datetime.utcnow().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


# CLI interface
async def main():
    """Command-line interface for rewrite generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AI rewrites of corpus")
    parser.add_argument("corpus", type=Path, help="Path to corpus JSONL file")
    parser.add_argument("-o", "--output", type=Path, default=Path("./rewrites"), help="Output directory")
    parser.add_argument("-r", "--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("-l", "--limit", type=int, help="Max samples to process")
    parser.add_argument("--resume", type=str, help="Resume from sample ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = RewriteGenerator(output_dir=args.output)
    
    await generator.process_corpus(
        corpus_path=args.corpus,
        runs_per_config=args.runs,
        limit=args.limit,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
