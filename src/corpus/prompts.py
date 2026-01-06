"""
PACT Corpus - Prompt Taxonomy & Embedding Space

This module implements a two-axis prompt taxonomy for systematic testing
of AI watermark detection across different cognitive operations.

Axis 1: TRANSFORMATION STYLE
    How the model should transform/rewrite text
    minimal → neutral → improvement → formal → casual

Axis 2: COGNITIVE OPERATION  
    What cognitive task the model should perform
    explain → qa → extract → critique → synthesize

The hypothesis: Echo Rule watermarks should appear in ANY generative output,
not just paraphrasing. If watermarks only appear in transformation prompts
but not cognitive prompts, that's a critical detection gap.

Usage:
    from corpus.prompts import PromptTaxonomy, PromptSpace
    
    # Get all prompts for a category
    taxonomy = PromptTaxonomy()
    prompts = taxonomy.get_prompts("transform", "minimal")
    
    # Sample diverse prompts across the space
    space = PromptSpace()
    space.load_taxonomy(taxonomy)
    diverse_sample = space.sample_diverse(n=50)
    
    # Find coverage gaps
    gaps = space.find_coverage_gaps()
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import json
import random
from pathlib import Path


class PromptAxis(str, Enum):
    """Primary axis of prompt classification."""
    TRANSFORM = "transform"      # Produces rewritten/paraphrased text
    COGNITIVE = "cognitive"      # Produces analysis/synthesis
    HYBRID = "hybrid"            # Combines transformation with cognition


class TransformStyle(str, Enum):
    """Transformation style variants."""
    MINIMAL = "minimal"          # Terse, lazy prompts
    NEUTRAL = "neutral"          # Standard paraphrase requests
    IMPROVEMENT = "improvement"  # Edit for clarity/quality
    FORMAL = "formal"            # Academic/professional register
    CASUAL = "casual"            # Conversational register


class CognitiveOp(str, Enum):
    """Cognitive operation types."""
    EXPLAIN = "explain"          # Explain/summarize meaning
    QA = "qa"                    # Answer questions about text
    EXTRACT = "extract"          # Extract structured information
    CRITIQUE = "critique"        # Analyze weaknesses/issues
    SYNTHESIZE = "synthesize"    # Generate derivative artifacts


@dataclass
class Prompt:
    """A single prompt with full metadata."""
    id: int
    text: str
    axis: PromptAxis
    category: str                    # TransformStyle or CognitiveOp value
    basis: list[str] = field(default_factory=list)  # For hybrid prompts
    hypothesis: str = ""             # What we're testing
    output_type: str = "paraphrase"  # Expected output type
    
    def format(self, input_text: str) -> str:
        """Format prompt with input text."""
        # Handle different prompt formats
        if self.text.endswith(":"):
            return f"{self.text}\n\n{input_text}"
        elif "{text}" in self.text:
            return self.text.format(text=input_text)
        else:
            return f"{self.text}\n\n{input_text}"


class PromptTaxonomy:
    """
    Complete taxonomy of prompts across both axes.
    
    Contains 100 transformation prompts (20 per style)
    and 100 cognitive prompts (50 base + 50 hybrid combinations).
    """
    
    def __init__(self):
        self.prompts: dict[str, dict[str, list[Prompt]]] = {
            "transform": {},
            "cognitive": {},
            "hybrid": {},
        }
        self._load_transform_prompts()
        self._load_cognitive_prompts()
        self._load_hybrid_prompts()
    
    def _load_transform_prompts(self):
        """Load transformation-style prompts."""
        
        # MINIMAL - Terse, lazy prompts (20)
        self.prompts["transform"]["minimal"] = [
            Prompt(1, "Rewrite this:", PromptAxis.TRANSFORM, "minimal", 
                   hypothesis="Even minimal prompts trigger watermarking"),
            Prompt(2, "Rewrite.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(3, "Reword this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(4, "Rephrase this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(5, "Paraphrase this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(6, "Fix the wording.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(7, "Clean this up.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(8, "Make this better.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(9, "Edit this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(10, "Polish this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(11, "Rewrite the below:", PromptAxis.TRANSFORM, "minimal"),
            Prompt(12, "Rewrite the following.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(13, "Rework this.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(14, "Say this differently.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(15, "New wording:", PromptAxis.TRANSFORM, "minimal"),
            Prompt(16, "Rewrite it.", PromptAxis.TRANSFORM, "minimal"),
            Prompt(17, "Rephrase:", PromptAxis.TRANSFORM, "minimal"),
            Prompt(18, "Paraphrase:", PromptAxis.TRANSFORM, "minimal"),
            Prompt(19, "Rewrite my text:", PromptAxis.TRANSFORM, "minimal"),
            Prompt(20, "Rewrite this sentence:", PromptAxis.TRANSFORM, "minimal"),
        ]
        
        # NEUTRAL - Standard paraphrase requests (20)
        self.prompts["transform"]["neutral"] = [
            Prompt(21, "Rewrite this in your own words.", PromptAxis.TRANSFORM, "neutral",
                   hypothesis="Baseline detection scenario"),
            Prompt(22, "Can you rewrite this in your own words?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(23, "Please rephrase this for me.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(24, "Can you reword this without changing the meaning?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(25, "Rewrite this while keeping the same meaning.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(26, "Could you paraphrase this text?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(27, "Rewrite this so it reads differently.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(28, "Rephrase this with similar meaning.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(29, "Rewrite this in a clear way.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(30, "Can you rewrite this paragraph?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(31, "Rewrite this passage.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(32, "Please rewrite the following text.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(33, "Could you rewrite this and keep it accurate?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(34, "Rewrite this with different phrasing.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(35, "Can you restate this?", PromptAxis.TRANSFORM, "neutral"),
            Prompt(36, "Rewrite this so it sounds natural.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(37, "Please reword this section.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(38, "Rewrite this to avoid repetition.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(39, "Rewrite this in a fresh way.", PromptAxis.TRANSFORM, "neutral"),
            Prompt(40, "Rewrite the below text in your own words.", PromptAxis.TRANSFORM, "neutral"),
        ]
        
        # IMPROVEMENT - Edit for quality (20)
        self.prompts["transform"]["improvement"] = [
            Prompt(41, "Improve the clarity and flow of this.", PromptAxis.TRANSFORM, "improvement",
                   hypothesis="Improvement prompts may increase stylistic changes"),
            Prompt(42, "Edit for clarity and readability.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(43, "Tighten this up and make it clearer.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(44, "Make this more concise without losing meaning.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(45, "Improve the structure and wording.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(46, "Polish this for better readability.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(47, "Rewrite this to be clearer and smoother.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(48, "Remove redundancy and improve flow.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(49, "Simplify this while keeping the key points.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(50, "Make this easier to understand.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(51, "Streamline this text.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(52, "Make this more readable.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(53, "Improve the writing quality.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(54, "Clean up the grammar and style.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(55, "Make this flow better.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(56, "Refine this text.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(57, "Enhance the clarity.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(58, "Fix awkward phrasing.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(59, "Make this more professional.", PromptAxis.TRANSFORM, "improvement"),
            Prompt(60, "Improve without changing the meaning.", PromptAxis.TRANSFORM, "improvement"),
        ]
        
        # FORMAL - Academic/professional register (20)
        self.prompts["transform"]["formal"] = [
            Prompt(61, "Rewrite this in a formal academic tone.", PromptAxis.TRANSFORM, "formal",
                   hypothesis="Register constraints may affect echo patterns"),
            Prompt(62, "Make this sound more professional.", PromptAxis.TRANSFORM, "formal"),
            Prompt(63, "Rewrite for a scholarly audience.", PromptAxis.TRANSFORM, "formal"),
            Prompt(64, "Convert to formal English.", PromptAxis.TRANSFORM, "formal"),
            Prompt(65, "Make this appropriate for a business report.", PromptAxis.TRANSFORM, "formal"),
            Prompt(66, "Rewrite in third person formal style.", PromptAxis.TRANSFORM, "formal"),
            Prompt(67, "Make this suitable for publication.", PromptAxis.TRANSFORM, "formal"),
            Prompt(68, "Rewrite for an executive summary.", PromptAxis.TRANSFORM, "formal"),
            Prompt(69, "Convert to academic writing style.", PromptAxis.TRANSFORM, "formal"),
            Prompt(70, "Make this more authoritative.", PromptAxis.TRANSFORM, "formal"),
            Prompt(71, "Rewrite for a professional context.", PromptAxis.TRANSFORM, "formal"),
            Prompt(72, "Formalize this text.", PromptAxis.TRANSFORM, "formal"),
            Prompt(73, "Make this sound more official.", PromptAxis.TRANSFORM, "formal"),
            Prompt(74, "Rewrite in business English.", PromptAxis.TRANSFORM, "formal"),
            Prompt(75, "Convert to formal register.", PromptAxis.TRANSFORM, "formal"),
            Prompt(76, "Make this appropriate for a journal.", PromptAxis.TRANSFORM, "formal"),
            Prompt(77, "Rewrite for a technical audience.", PromptAxis.TRANSFORM, "formal"),
            Prompt(78, "Make this more serious in tone.", PromptAxis.TRANSFORM, "formal"),
            Prompt(79, "Rewrite as if for a legal document.", PromptAxis.TRANSFORM, "formal"),
            Prompt(80, "Convert to professional prose.", PromptAxis.TRANSFORM, "formal"),
        ]
        
        # CASUAL - Conversational register (20)
        self.prompts["transform"]["casual"] = [
            Prompt(81, "Rewrite this in a casual, conversational tone.", PromptAxis.TRANSFORM, "casual",
                   hypothesis="Casual register may reduce structural echoing"),
            Prompt(82, "Make this sound more friendly.", PromptAxis.TRANSFORM, "casual"),
            Prompt(83, "Rewrite like you're talking to a friend.", PromptAxis.TRANSFORM, "casual"),
            Prompt(84, "Make this less formal.", PromptAxis.TRANSFORM, "casual"),
            Prompt(85, "Rewrite for social media.", PromptAxis.TRANSFORM, "casual"),
            Prompt(86, "Make this more approachable.", PromptAxis.TRANSFORM, "casual"),
            Prompt(87, "Rewrite in plain English.", PromptAxis.TRANSFORM, "casual"),
            Prompt(88, "Make this easier to read.", PromptAxis.TRANSFORM, "casual"),
            Prompt(89, "Rewrite for a general audience.", PromptAxis.TRANSFORM, "casual"),
            Prompt(90, "Make this sound natural and relaxed.", PromptAxis.TRANSFORM, "casual"),
            Prompt(91, "Rewrite like a blog post.", PromptAxis.TRANSFORM, "casual"),
            Prompt(92, "Make this conversational.", PromptAxis.TRANSFORM, "casual"),
            Prompt(93, "Rewrite in everyday language.", PromptAxis.TRANSFORM, "casual"),
            Prompt(94, "Make this feel more human.", PromptAxis.TRANSFORM, "casual"),
            Prompt(95, "Rewrite like explaining to a teenager.", PromptAxis.TRANSFORM, "casual"),
            Prompt(96, "Remove jargon and simplify.", PromptAxis.TRANSFORM, "casual"),
            Prompt(97, "Make this fun to read.", PromptAxis.TRANSFORM, "casual"),
            Prompt(98, "Rewrite with personality.", PromptAxis.TRANSFORM, "casual"),
            Prompt(99, "Make this less stuffy.", PromptAxis.TRANSFORM, "casual"),
            Prompt(100, "Rewrite for a text message.", PromptAxis.TRANSFORM, "casual"),
        ]
    
    def _load_cognitive_prompts(self):
        """Load cognitive operation prompts - these produce analysis, not paraphrase."""
        
        # EXPLAIN - Summarize/explain meaning (10)
        self.prompts["cognitive"]["explain"] = [
            Prompt(201, "Summarize the key points without rewriting:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis", hypothesis="Does explanation trigger watermarking?"),
            Prompt(202, "Explain what this text is saying in one paragraph:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(203, "What is the main argument here? Explain briefly:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(204, "Explain the core message to someone unfamiliar with the topic:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(205, "Break down the key ideas in this text:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(206, "Explain using a causal chain: A leads to B leads to C:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(207, "What is being optimized for here (time, cost, quality, risk)?", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(208, "Explain which parts are descriptive vs prescriptive:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(209, "Map this to: inputs, process, outputs, constraints, risks:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
            Prompt(210, "Explain the implicit timeline and sequencing:", PromptAxis.COGNITIVE, "explain",
                   output_type="analysis"),
        ]
        
        # QA - Answer questions about text (10)
        self.prompts["cognitive"]["qa"] = [
            Prompt(211, "What is the single most important point? Quote it:", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis", hypothesis="Q&A may not trigger watermarking"),
            Prompt(212, "What assumptions does this text make? List them:", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(213, "What questions does this leave unanswered?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(214, "What would change your mind about the main claim?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(215, "What is the primary constraint stated here? Quote it:", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(216, "What is the measurable success condition, if any?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(217, "What would be a reasonable counterexample?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(218, "What are the non-negotiables stated here?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(219, "What is the primary deliverable implied?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
            Prompt(220, "What must be true for this to work?", PromptAxis.COGNITIVE, "qa",
                   output_type="analysis"),
        ]
        
        # EXTRACT - Pull structured information (10)
        self.prompts["cognitive"]["extract"] = [
            Prompt(221, "Extract all named entities (people, places, organizations):", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction", hypothesis="Extraction may bypass watermarking"),
            Prompt(222, "List all claims made, with supporting quotes:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(223, "Extract dates, deadlines, and time references:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(224, "Pull out all numerical data and statistics:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(225, "Extract requirements as a numbered list:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(226, "List all actions or recommendations mentioned:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(227, "Extract a glossary: term, definition, where it appears:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(228, "Pull out all URLs, file paths, commands, or identifiers:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(229, "Extract as JSON: {entities, dates, requirements, actions}:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
            Prompt(230, "Build a concept graph: nodes are concepts, edges are relationships:", PromptAxis.COGNITIVE, "extract",
                   output_type="extraction"),
        ]
        
        # CRITIQUE - Analyze weaknesses (10)
        self.prompts["cognitive"]["critique"] = [
            Prompt(231, "What are the weaknesses in this argument? Be specific:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis", hypothesis="Critical analysis may trigger different patterns"),
            Prompt(232, "Find logical gaps: where conclusions lack support:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(233, "Spot contradictions or internal inconsistencies:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(234, "List potential objections a reader might have:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(235, "Check for scope creep: where does this try to do too much?", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(236, "Find unsupported claims. Label as: needs data, needs citation, or opinion:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(237, "Evaluate for bias, loaded language, or manipulative framing:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(238, "Identify missing definitions that cause confusion:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(239, "Assess risk: what could go wrong if someone followed this literally?", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
            Prompt(240, "Rate each claim as strong, weak, or unsupported:", PromptAxis.COGNITIVE, "critique",
                   output_type="analysis"),
        ]
        
        # SYNTHESIZE - Generate derivative artifacts (10)
        self.prompts["cognitive"]["synthesize"] = [
            Prompt(241, "Turn this into an implementation plan with milestones:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis", hypothesis="Synthesis may show strongest watermarking"),
            Prompt(242, "Convert to a decision memo: context, options, tradeoffs, recommendation:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(243, "Generate a test plan: test cases, expected outcomes, edge cases:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(244, "Produce a checklist someone could follow:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(245, "Create a rubric to evaluate whether these goals were achieved:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(246, "Generate a JSON schema to store this information:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(247, "Produce a risk register: risk, cause, impact, mitigation:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(248, "Draft a project brief: objective, users, constraints, success metrics:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(249, "Create an FAQ anticipating confusion points:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
            Prompt(250, "Generate acceptance criteria and test cases:", PromptAxis.COGNITIVE, "synthesize",
                   output_type="synthesis"),
        ]
    
    def _load_hybrid_prompts(self):
        """Load hybrid prompts that combine transformation with cognition."""
        
        self.prompts["hybrid"]["explain_critique"] = [
            Prompt(301, "Explain the intended meaning, then list potential misreadings:", 
                   PromptAxis.HYBRID, "explain_critique", 
                   basis=["explain", "critique"], output_type="analysis"),
            Prompt(302, "Summarize the argument, then identify its weakest points:",
                   PromptAxis.HYBRID, "explain_critique",
                   basis=["explain", "critique"], output_type="analysis"),
        ]
        
        self.prompts["hybrid"]["extract_synthesize"] = [
            Prompt(311, "Extract requirements, then generate acceptance criteria:",
                   PromptAxis.HYBRID, "extract_synthesize",
                   basis=["extract", "synthesize"], output_type="synthesis"),
            Prompt(312, "Pull out the key claims, then create a validation checklist:",
                   PromptAxis.HYBRID, "extract_synthesize",
                   basis=["extract", "synthesize"], output_type="synthesis"),
        ]
        
        self.prompts["hybrid"]["qa_critique"] = [
            Prompt(321, "Answer: what is weakest here? Then critique with evidence:",
                   PromptAxis.HYBRID, "qa_critique",
                   basis=["qa", "critique"], output_type="analysis"),
            Prompt(322, "What must be true for this to work? Evaluate each assumption:",
                   PromptAxis.HYBRID, "qa_critique",
                   basis=["qa", "critique"], output_type="analysis"),
        ]
        
        self.prompts["hybrid"]["explain_synthesize"] = [
            Prompt(331, "Explain the goal, then propose a concrete 7-day plan:",
                   PromptAxis.HYBRID, "explain_synthesize",
                   basis=["explain", "synthesize"], output_type="synthesis"),
            Prompt(332, "Summarize the intent, then generate implementation tasks:",
                   PromptAxis.HYBRID, "explain_synthesize",
                   basis=["explain", "synthesize"], output_type="synthesis"),
        ]
        
        self.prompts["hybrid"]["critique_synthesize"] = [
            Prompt(341, "Critique failure modes, then produce mitigations:",
                   PromptAxis.HYBRID, "critique_synthesize",
                   basis=["critique", "synthesize"], output_type="synthesis"),
            Prompt(342, "Identify risks, then create a monitoring checklist:",
                   PromptAxis.HYBRID, "critique_synthesize",
                   basis=["critique", "synthesize"], output_type="synthesis"),
        ]
    
    def get_prompts(
        self, 
        axis: str, 
        category: str
    ) -> list[Prompt]:
        """Get all prompts for a given axis and category."""
        return self.prompts.get(axis, {}).get(category, [])
    
    def get_all_prompts(self) -> list[Prompt]:
        """Get all prompts across all axes."""
        all_prompts = []
        for axis_prompts in self.prompts.values():
            for category_prompts in axis_prompts.values():
                all_prompts.extend(category_prompts)
        return all_prompts
    
    def get_random_prompt(
        self, 
        axis: Optional[str] = None,
        category: Optional[str] = None
    ) -> Prompt:
        """Get a random prompt, optionally filtered."""
        if axis and category:
            prompts = self.get_prompts(axis, category)
        elif axis:
            prompts = []
            for cat_prompts in self.prompts.get(axis, {}).values():
                prompts.extend(cat_prompts)
        else:
            prompts = self.get_all_prompts()
        
        return random.choice(prompts)
    
    def get_experimental_matrix(self) -> list[dict]:
        """
        Generate the full experimental matrix.
        
        Returns configurations for testing each prompt axis/category
        combination to measure watermark detection rates.
        """
        matrix = []
        
        for axis, categories in self.prompts.items():
            for category, prompts in categories.items():
                matrix.append({
                    "axis": axis,
                    "category": category,
                    "prompt_count": len(prompts),
                    "output_type": prompts[0].output_type if prompts else "unknown",
                    "hypothesis": prompts[0].hypothesis if prompts else "",
                    "sample_prompts": [p.text for p in prompts[:3]],
                })
        
        return matrix
    
    def to_json(self) -> str:
        """Export taxonomy as JSON."""
        data = {
            "metadata": {
                "total_prompts": len(self.get_all_prompts()),
                "axes": list(self.prompts.keys()),
            },
            "prompts": {}
        }
        
        for axis, categories in self.prompts.items():
            data["prompts"][axis] = {}
            for category, prompts in categories.items():
                data["prompts"][axis][category] = [
                    {
                        "id": p.id,
                        "text": p.text,
                        "output_type": p.output_type,
                        "hypothesis": p.hypothesis,
                    }
                    for p in prompts
                ]
        
        return json.dumps(data, indent=2)
    
    def save(self, path: Path):
        """Save taxonomy to file."""
        path.write_text(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> 'PromptTaxonomy':
        """Load taxonomy from file."""
        # For now, just create fresh - could implement JSON loading
        return cls()


class PromptSpace:
    """
    Embedding space analysis for prompts.
    
    Maps prompts to a continuous vector space for:
    - Finding coverage gaps
    - Sampling diverse prompts
    - Clustering similar prompts
    - Generating interpolated prompts
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.prompts: list[Prompt] = []
        self.embeddings = None
        self._labels = []
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for PromptSpace. "
                    "Install with: pip install sentence-transformers"
                )
    
    def load_taxonomy(self, taxonomy: PromptTaxonomy):
        """Load prompts from taxonomy and compute embeddings."""
        self._load_model()
        
        self.prompts = taxonomy.get_all_prompts()
        self._labels = [(p.axis.value, p.category) for p in self.prompts]
        
        texts = [p.text for p in self.prompts]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
    
    def add_prompts(self, prompts: list[Prompt]):
        """Add additional prompts and recompute embeddings."""
        self._load_model()
        
        self.prompts.extend(prompts)
        self._labels.extend([(p.axis.value, p.category) for p in prompts])
        
        texts = [p.text for p in self.prompts]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
    
    def find_similar(self, prompt_text: str, n: int = 5) -> list[tuple[Prompt, float]]:
        """Find prompts most similar to given text."""
        self._load_model()
        import numpy as np
        
        query_embedding = self.model.encode([prompt_text])[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        return [(self.prompts[i], float(similarities[i])) for i in top_indices]
    
    def find_coverage_gaps(self, n_clusters: int = 20) -> list[dict]:
        """
        Find regions of prompt space with poor coverage.
        
        Returns cluster centroids that have few assigned prompts.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.embeddings)
        
        # Count prompts per cluster
        counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        
        # Find sparse clusters (fewer than 3 prompts)
        sparse_indices = np.where(counts < 3)[0]
        
        gaps = []
        for i in sparse_indices:
            # Find the prompt nearest to centroid to characterize the gap
            cluster_mask = kmeans.labels_ == i
            if cluster_mask.sum() > 0:
                cluster_prompts = [p for j, p in enumerate(self.prompts) if cluster_mask[j]]
                sample = cluster_prompts[0].text if cluster_prompts else "unknown"
            else:
                sample = "empty cluster"
            
            gaps.append({
                "cluster_id": int(i),
                "centroid": kmeans.cluster_centers_[i].tolist(),
                "prompt_count": int(counts[i]),
                "sample_prompt": sample,
            })
        
        return gaps
    
    def sample_diverse(self, n: int = 50) -> list[Prompt]:
        """
        Sample prompts for maximum diversity using k-means.
        
        Picks one prompt nearest to each of n cluster centroids.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        n = min(n, len(self.prompts))
        
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(self.embeddings)
        
        selected = []
        for i in range(n):
            cluster_mask = kmeans.labels_ == i
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_prompts = [p for j, p in enumerate(self.prompts) if cluster_mask[j]]
            
            if len(cluster_prompts) == 0:
                continue
            
            # Find nearest to centroid
            distances = np.linalg.norm(
                cluster_embeddings - kmeans.cluster_centers_[i], 
                axis=1
            )
            nearest_idx = np.argmin(distances)
            selected.append(cluster_prompts[nearest_idx])
        
        return selected
    
    def sample_stratified(
        self, 
        n_per_category: int = 5
    ) -> dict[str, list[Prompt]]:
        """
        Sample diverse prompts within each category.
        
        Ensures representation across all axes/categories while
        maximizing diversity within each.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        samples = {}
        
        # Group by (axis, category)
        categories = {}
        for i, (axis, cat) in enumerate(self._labels):
            key = f"{axis}_{cat}"
            if key not in categories:
                categories[key] = []
            categories[key].append(i)
        
        for key, indices in categories.items():
            if len(indices) <= n_per_category:
                samples[key] = [self.prompts[i] for i in indices]
                continue
            
            # Cluster within category
            cat_embeddings = self.embeddings[indices]
            n_clusters = min(n_per_category, len(indices))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(cat_embeddings)
            
            selected = []
            for i in range(n_clusters):
                cluster_mask = kmeans.labels_ == i
                cluster_indices = [indices[j] for j in range(len(indices)) if cluster_mask[j]]
                
                if cluster_indices:
                    # Pick one from cluster
                    selected.append(self.prompts[cluster_indices[0]])
            
            samples[key] = selected
        
        return samples
    
    def visualize_2d(self, output_path: Optional[Path] = None):
        """
        Generate 2D visualization of prompt space.
        
        Returns matplotlib figure or saves to file.
        """
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Reduce to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Color by axis
        colors = {
            "transform": "blue",
            "cognitive": "red", 
            "hybrid": "green",
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for axis in colors:
            mask = [l[0] == axis for l in self._labels]
            if any(mask):
                points = embeddings_2d[mask]
                ax.scatter(
                    points[:, 0], 
                    points[:, 1], 
                    c=colors[axis], 
                    label=axis,
                    alpha=0.6
                )
        
        ax.legend()
        ax.set_title("Prompt Embedding Space (t-SNE)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def get_category_centroids(self) -> dict[str, list[float]]:
        """Get centroid embedding for each category."""
        import numpy as np
        
        centroids = {}
        
        categories = {}
        for i, (axis, cat) in enumerate(self._labels):
            key = f"{axis}_{cat}"
            if key not in categories:
                categories[key] = []
            categories[key].append(i)
        
        for key, indices in categories.items():
            cat_embeddings = self.embeddings[indices]
            centroids[key] = np.mean(cat_embeddings, axis=0).tolist()
        
        return centroids


# Convenience functions

def get_diverse_prompt_sample(n: int = 50) -> list[Prompt]:
    """Quick function to get a diverse sample of prompts."""
    taxonomy = PromptTaxonomy()
    space = PromptSpace()
    space.load_taxonomy(taxonomy)
    return space.sample_diverse(n)


def get_experimental_prompts() -> dict[str, list[Prompt]]:
    """Get prompts organized for experimental design."""
    taxonomy = PromptTaxonomy()
    space = PromptSpace()
    space.load_taxonomy(taxonomy)
    return space.sample_stratified(n_per_category=5)


if __name__ == "__main__":
    # Demo
    taxonomy = PromptTaxonomy()
    
    print("=== Prompt Taxonomy ===")
    print(f"Total prompts: {len(taxonomy.get_all_prompts())}")
    
    print("\n=== Experimental Matrix ===")
    for config in taxonomy.get_experimental_matrix():
        print(f"  {config['axis']}/{config['category']}: {config['prompt_count']} prompts")
        print(f"    Output type: {config['output_type']}")
        print(f"    Hypothesis: {config['hypothesis'][:60]}...")
    
    # Test embedding space (requires sentence-transformers)
    try:
        print("\n=== Embedding Space Analysis ===")
        space = PromptSpace()
        space.load_taxonomy(taxonomy)
        
        diverse = space.sample_diverse(10)
        print(f"Diverse sample ({len(diverse)} prompts):")
        for p in diverse[:5]:
            print(f"  [{p.axis.value}/{p.category}] {p.text[:50]}...")
        
        gaps = space.find_coverage_gaps()
        print(f"\nCoverage gaps: {len(gaps)} sparse clusters")
        
    except ImportError:
        print("\n(Install sentence-transformers for embedding analysis)")
