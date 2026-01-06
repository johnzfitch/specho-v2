#!/usr/bin/env python3
"""
SpecHO v2 Unified TUI

Complete terminal interface for:
- Dynamic feature visualization (auto-discovers from registry)
- Live text analysis
- Tiered classification (1/2/3)
- AI detection with confidence
- Model comparison
- Corpus statistics

Features are dynamically loaded from the registry - add new extractors
and they appear automatically!
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Static, DataTable, TabbedContent, TabPane,
    Input, Button, Label, TextArea, ProgressBar
)
from textual.binding import Binding
from textual.reactive import reactive

# Import registry for dynamic feature discovery
try:
    from src.extractors.registry import get_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


def _build_dimension_layers_from_registry() -> Dict:
    """
    Dynamically build DIMENSION_LAYERS from the registry.

    This replaces the hardcoded 45D structure with auto-discovered features.
    """
    if not REGISTRY_AVAILABLE:
        return _LEGACY_DIMENSION_LAYERS

    try:
        registry = get_registry(quiet=True)
    except Exception:
        return _LEGACY_DIMENSION_LAYERS

    layers = {}

    # Map groups to tier/trust/deps based on known patterns
    TIER_MAP = {
        # Tier 0: No deps
        "statistical": (1, "HIGH", "none (<5ms)"),
        "lexical": (1, "HIGH", "none (<5ms)"),
        "structural": (1, "HIGH", "none (<5ms)"),
        "punctuation": (1, "MEDIUM", "none (<5ms)"),
        "connectors": (1, "MEDIUM", "none (<5ms)"),
        "epistemic": (1, "MEDIUM", "none (<5ms)"),
        "transitions": (1, "MEDIUM", "none (<5ms)"),
        "rhythm": (1, "MEDIUM", "none (<5ms)"),
        "lightweight": (1, "HIGH", "none (<5ms)"),
        "alignment": (1, "HIGH", "none (<5ms)"),
        "noise": (1, "HIGH", "none (<5ms)"),
        "dynamics": (1, "MEDIUM", "none (<5ms)"),
        "fractal": (1, "MEDIUM", "none (<5ms)"),
        # Tier 1: spaCy
        "pos": (2, "MEDIUM-HIGH", "spaCy"),
        "dependency": (2, "MEDIUM", "spaCy"),
        "information": (2, "MEDIUM-HIGH", "spaCy"),
        "topology": (2, "MEDIUM", "spaCy"),
        # Tier 2: transformers/tiktoken
        "trajectory": (3, "HIGH", "sentence-transformers"),
        "echo": (2, "MEDIUM-HIGH", "spaCy (fallback available)"),
        "tokenization": (2, "MEDIUM", "tiktoken"),
    }

    for i, group in enumerate(sorted(registry.groups)):
        extractors = registry._by_group[group]
        features = []
        for ext_name in extractors:
            ext = registry._extractors[ext_name]
            features.extend(ext.feature_names)

        tier, trust, deps = TIER_MAP.get(group, (1, "MEDIUM", "unknown"))

        layers[f"LAYER_{chr(65+i)}_{group.upper()}"] = {
            "short": chr(65+i),
            "name": group.title(),
            "description": f"{group.title()} features ({len(features)}D)",
            "trust": trust,
            "tier": tier,
            "deps": deps,
            "dimensions": features,
        }

    return layers


# =============================================================================
# DIMENSION METADATA - Legacy 45D (fallback)
# =============================================================================

_LEGACY_DIMENSION_LAYERS = {
    "LAYER_A_TRAJECTORY": {
        "short": "A",
        "name": "Trajectory",
        "description": "Semantic Navigation Geometry",
        "trust": "HIGH",
        "tier": 3,
        "deps": "sentence-transformers",
        "dimensions": [
            "concept_jump_mean",
            "concept_jump_variance",
            "path_tortuosity",
            "turning_angle_mean",
            "return_rate",
        ],
    },
    "LAYER_B_ECHO": {
        "short": "B",
        "name": "Echo",
        "description": "Clause Boundary Patterns",
        "trust": "MEDIUM-HIGH",
        "tier": 2,
        "deps": "spaCy (fallback available)",
        "dimensions": [
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max", "burstiness",
        ],
    },
    "LAYER_C_EPISTEMIC": {
        "short": "C.1",
        "name": "Epistemic",
        "description": "Hedging & Confidence Patterns",
        "trust": "MEDIUM",
        "tier": 2,
        "deps": "none",
        "dimensions": [
            "hedge_density", "hedge_clustering", "hedge_position_bias",
            "confidence_mean", "confidence_variance", "confidence_arc",
        ],
    },
    "LAYER_C_TRANSITIONS": {
        "short": "C.2",
        "name": "Transitions",
        "description": "Discourse Connectors",
        "trust": "MEDIUM",
        "tier": 2,
        "deps": "none",
        "dimensions": [
            "additive_rate", "contrastive_rate", "causal_rate",
            "temporal_rate", "exemplifying_rate", "reformulating_rate",
        ],
    },
    "LAYER_D_SYNTACTIC": {
        "short": "D",
        "name": "Syntactic",
        "description": "Structural Rhythm",
        "trust": "LOWER",
        "tier": 3,
        "deps": "spaCy",
        "dimensions": [
            "clause_length_mean", "clause_length_std", "clause_rhythm_autocorr",
            "sentence_complexity", "comma_density", "semicolon_rate", "parenthetical_rate",
        ],
    },
    "LAYER_E_LIGHTWEIGHT": {
        "short": "E",
        "name": "Lightweight",
        "description": "Fast Structural Features (NO DEPS)",
        "trust": "HIGH",
        "tier": 1,
        "deps": "none (<1ms)",
        "dimensions": [
            "sentence_length_cv",     # d=-1.47
            "paragraph_length_cv",    # d=-1.10
            "type_token_ratio",       # d=-1.07
            "compression_ratio",      # d=-0.99
            "list_marker_rate",       # d=+0.93
            "hapax_legomena_ratio",   # d=-0.75
        ],
    },
}

# Build DIMENSION_LAYERS dynamically from registry (falls back to legacy)
DIMENSION_LAYERS = _build_dimension_layers_from_registry()

MODEL_DISPLAY_NAMES = {
    "HUMAN": "Human",
    "gemma-2-9b": "Gemma-2-9B",
    "mistral-7B": "Mistral-7B",
    "qwen-2-72B": "Qwen-2-72B",
    "llama-8B": "Llama-3-8B",
    "accounts/yi-01-ai/models/yi-large": "Yi-Large",
    "GPT_4-o": "GPT-4o",
    "claude-sonnet-4": "Claude Sonnet",
    "claude-opus-4": "Claude Opus",
}

# Reference statistics for Layer E (from validation)
LAYER_E_REFERENCE = {
    "human": {
        "sentence_length_cv": {"mean": 0.89, "std": 0.15},
        "paragraph_length_cv": {"mean": 0.65, "std": 0.20},
        "type_token_ratio": {"mean": 0.72, "std": 0.08},
        "compression_ratio": {"mean": 0.71, "std": 0.05},
        "list_marker_rate": {"mean": 0.001, "std": 0.003},
        "hapax_legomena_ratio": {"mean": 0.52, "std": 0.08},
    },
    "ai": {
        "sentence_length_cv": {"mean": 0.35, "std": 0.10},
        "paragraph_length_cv": {"mean": 0.30, "std": 0.12},
        "type_token_ratio": {"mean": 0.58, "std": 0.10},
        "compression_ratio": {"mean": 0.64, "std": 0.04},
        "list_marker_rate": {"mean": 0.015, "std": 0.02},
        "hapax_legomena_ratio": {"mean": 0.38, "std": 0.10},
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_deviation_category(percent_diff: float) -> Tuple[str, str]:
    """Categorize deviation magnitude."""
    abs_diff = abs(percent_diff)
    if abs_diff < 10:
        return ("NEAR", "green")
    elif abs_diff < 25:
        return ("MINOR", "yellow")
    elif abs_diff < 50:
        return ("MODERATE", "orange1")
    else:
        return ("MAJOR", "red bold")


def format_dimension_name(dim: str) -> str:
    """Format dimension name for display."""
    return dim.replace("_", " ").title()


def create_bar(value: float, min_val: float, max_val: float, width: int = 20) -> str:
    """Create horizontal bar chart."""
    if max_val == min_val:
        filled = width // 2
    else:
        normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
        filled = int(normalized * width)
    return "█" * filled + "░" * (width - filled)


def get_tier_color(tier: int) -> str:
    """Get color for tier display."""
    return {1: "green", 2: "yellow", 3: "cyan"}.get(tier, "white")


# =============================================================================
# LIGHTWEIGHT FEATURE EXTRACTION (Tier 1)
# =============================================================================

def extract_layer_e(text: str) -> Dict[str, float]:
    """
    Extract Layer E features (6D) with NO dependencies.
    This is the fast path - <1ms.
    """
    import re
    from collections import Counter

    # Basic text stats
    words = text.split()
    word_count = len(words)

    if word_count < 5:
        return {dim: 0.0 for dim in DIMENSION_LAYERS["LAYER_E_LIGHTWEIGHT"]["dimensions"]}

    # Sentence length CV
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]

    if len(sent_lengths) > 1:
        sent_mean = statistics.mean(sent_lengths)
        sent_std = statistics.stdev(sent_lengths)
        sentence_length_cv = sent_std / sent_mean if sent_mean > 0 else 0
    else:
        sentence_length_cv = 0.0

    # Paragraph length CV
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        para_lengths = [len(p.split()) for p in paragraphs]
        para_mean = statistics.mean(para_lengths)
        para_std = statistics.stdev(para_lengths)
        paragraph_length_cv = para_std / para_mean if para_mean > 0 else 0
    else:
        paragraph_length_cv = 0.0

    # Type-token ratio
    word_lower = [w.lower() for w in words]
    unique_words = set(word_lower)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0

    # Compression ratio (approximate)
    text_bytes = len(text.encode('utf-8'))
    try:
        import zlib
        compressed = len(zlib.compress(text.encode('utf-8')))
        compression_ratio = compressed / text_bytes if text_bytes > 0 else 1
    except:
        compression_ratio = 0.7  # Default

    # List marker rate
    list_patterns = re.findall(r'(?:^|\n)\s*(?:\d+[.)]|[-*•])\s', text)
    list_marker_rate = len(list_patterns) / word_count if word_count > 0 else 0

    # Hapax legomena ratio
    word_freq = Counter(word_lower)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_legomena_ratio = hapax / len(unique_words) if unique_words else 0

    return {
        "sentence_length_cv": sentence_length_cv,
        "paragraph_length_cv": paragraph_length_cv,
        "type_token_ratio": type_token_ratio,
        "compression_ratio": compression_ratio,
        "list_marker_rate": list_marker_rate,
        "hapax_legomena_ratio": hapax_legomena_ratio,
    }


def classify_tier1(features: Dict[str, float]) -> Tuple[str, float, Dict]:
    """
    Tier 1 classification using Layer E only.
    Returns: (prediction, confidence, details)
    """
    scores = {"human": 0, "ai": 0}
    evidence = []

    for dim, value in features.items():
        human_ref = LAYER_E_REFERENCE["human"].get(dim, {"mean": 0.5, "std": 0.2})
        ai_ref = LAYER_E_REFERENCE["ai"].get(dim, {"mean": 0.5, "std": 0.2})

        # Distance from each distribution
        human_dist = abs(value - human_ref["mean"]) / max(human_ref["std"], 0.01)
        ai_dist = abs(value - ai_ref["mean"]) / max(ai_ref["std"], 0.01)

        if human_dist < ai_dist:
            scores["human"] += 1
            evidence.append((dim, "human", value))
        else:
            scores["ai"] += 1
            evidence.append((dim, "ai", value))

    total = scores["human"] + scores["ai"]
    if total == 0:
        return "unknown", 0.5, {}

    human_prob = scores["human"] / total
    ai_prob = scores["ai"] / total

    prediction = "human" if human_prob > ai_prob else "ai"
    confidence = max(human_prob, ai_prob)

    return prediction, confidence, {
        "human_prob": human_prob,
        "ai_prob": ai_prob,
        "evidence": evidence,
        "features": features,
    }


# =============================================================================
# TUI VIEWS
# =============================================================================

class ArchitectureView(Static):
    """Visual representation of the 45D architecture."""

    def compose(self) -> ComposeResult:
        arch_text = """
┌─────────────────────────────────────────────────────────────────────────┐
│                         SPECHO V2: 45D ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TIER 1 (<1ms)          TIER 2 (~50ms)         TIER 3 (~100ms)        │
│   ┌───────────┐          ┌───────────┐          ┌───────────┐          │
│   │  LAYER E  │          │  LAYER B  │          │  LAYER A  │          │
│   │    6D     │──────────│    15D    │──────────│    5D     │          │
│   │ Lightwt   │          │   Echo    │          │ Trajectory│          │
│   │ (NO DEPS) │          │           │          │           │          │
│   └─────┬─────┘          │  LAYER C  │          │  LAYER D  │          │
│         │                │    12D    │          │    7D     │          │
│         │                │ Epist+Trn │          │ Syntactic │          │
│         │                └─────┬─────┘          └─────┬─────┘          │
│         │                      │                      │                │
│         └──────────────────────┴──────────────────────┘                │
│                                │                                        │
│                                ▼                                        │
│                    ┌──────────────────────┐                            │
│                    │    45D FINGERPRINT   │                            │
│                    │  AI Detection: 98.6% │                            │
│                    │  Model ID: ~75%      │                            │
│                    └──────────────────────┘                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  A=Trajectory  B=Echo  C=Epistemic+Transitions  D=Syntactic  E=Light   │
└─────────────────────────────────────────────────────────────────────────┘
        """
        yield Static(Panel(arch_text, title="Architecture", border_style="cyan"))


class LayerOverview(Static):
    """Overview of all 5 layers with their properties."""

    def compose(self) -> ComposeResult:
        table = Table(
            title="45D Layer Overview",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Layer", style="bold", width=8)
        table.add_column("Name", width=12)
        table.add_column("Dims", justify="right", width=5)
        table.add_column("Trust", width=12)
        table.add_column("Tier", justify="center", width=5)
        table.add_column("Dependencies", width=25)
        table.add_column("Description", width=30)

        for layer_key, info in DIMENSION_LAYERS.items():
            tier_style = get_tier_color(info["tier"])
            table.add_row(
                info["short"],
                info["name"],
                str(len(info["dimensions"])),
                info["trust"],
                Text(str(info["tier"]), style=tier_style),
                info["deps"],
                info["description"],
            )

        # Totals row
        total_dims = sum(len(info["dimensions"]) for info in DIMENSION_LAYERS.values())
        table.add_row(
            "TOTAL",
            "",
            Text(str(total_dims), style="bold green"),
            "",
            "",
            "",
            "",
            style="bold",
        )

        yield Static(table)


class LiveAnalysisView(Static):
    """Live text analysis with Tier 1 detection."""

    def __init__(self):
        super().__init__()
        self.result_text = ""

    def compose(self) -> ComposeResult:
        yield Static(Panel(
            "Paste or type text below, then press [bold]Analyze[/bold] for Tier 1 detection.",
            title="Live Analysis (Tier 1)",
            border_style="green"
        ))

        yield TextArea(id="input_text", language=None)
        yield Button("Analyze", id="analyze_btn", variant="primary")
        yield Static(id="analysis_result")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze_btn":
            text_area = self.query_one("#input_text", TextArea)
            result_widget = self.query_one("#analysis_result", Static)

            text = text_area.text
            if not text.strip():
                result_widget.update("Enter some text to analyze.")
                return

            # Extract and classify
            start = time.perf_counter()
            features = extract_layer_e(text)
            prediction, confidence, details = classify_tier1(features)
            elapsed = (time.perf_counter() - start) * 1000

            # Build result display
            pred_color = "green" if prediction == "human" else "red"
            conf_bar = create_bar(confidence, 0, 1, width=20)

            result = Text()
            result.append(f"\n{'='*60}\n", style="dim")
            result.append("TIER 1 RESULT", style="bold cyan")
            result.append(f" ({elapsed:.2f}ms)\n\n", style="dim")

            result.append("Prediction: ", style="bold")
            result.append(f"{prediction.upper()}\n", style=f"bold {pred_color}")

            result.append("Confidence: ", style="bold")
            result.append(f"{confidence:.1%} ", style=pred_color)
            result.append(f"[{conf_bar}]\n\n", style="dim")

            result.append("Layer E Features:\n", style="bold")
            for dim, value in features.items():
                human_ref = LAYER_E_REFERENCE["human"].get(dim, {}).get("mean", 0.5)
                ai_ref = LAYER_E_REFERENCE["ai"].get(dim, {}).get("mean", 0.5)

                bar = create_bar(value, min(human_ref, ai_ref) * 0.5, max(human_ref, ai_ref) * 1.5, width=15)
                closer_to = "H" if abs(value - human_ref) < abs(value - ai_ref) else "A"

                result.append(f"  {format_dimension_name(dim):25}", style="dim")
                result.append(f" {value:.3f} ", style="bold")
                result.append(f"[{bar}] ", style="dim")
                result.append(f"({closer_to})\n", style="green" if closer_to == "H" else "red")

            result_widget.update(result)


class LayerDetailView(Static):
    """Detailed view for a specific layer with model comparison."""

    def __init__(self, results: Optional[Dict], layer_key: str):
        super().__init__()
        self.results = results
        self.layer_key = layer_key
        self.layer_info = DIMENSION_LAYERS[layer_key]

    def compose(self) -> ComposeResult:
        # Layer header
        header_text = Text()
        header_text.append(f"LAYER {self.layer_info['short']}: ", style="bold cyan")
        header_text.append(f"{self.layer_info['name'].upper()}\n", style="bold")
        header_text.append(f"{self.layer_info['description']} ", style="italic")
        header_text.append(f"| Trust: {self.layer_info['trust']} ", style="dim")
        header_text.append(f"| Tier: {self.layer_info['tier']} ", style=get_tier_color(self.layer_info['tier']))
        header_text.append(f"| Deps: {self.layer_info['deps']}", style="dim")

        yield Static(Panel(header_text, box=box.DOUBLE))

        # Dimension list
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Dimension", style="bold", width=30)
        table.add_column("Description", width=50)

        for i, dim in enumerate(self.layer_info["dimensions"], 1):
            table.add_row(str(i), format_dimension_name(dim), dim)

        yield Static(table)

        # If we have results data, show comparison
        if self.results and "by_model" in self.results:
            yield Static("\n[bold]Model Comparison:[/bold]")

            comp_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
            comp_table.add_column("Dimension", width=25)
            comp_table.add_column("Human", justify="right", width=10)

            # Add model columns dynamically
            model_keys = list(self.results.get("by_model", {}).keys())[:5]
            for mk in model_keys:
                display = MODEL_DISPLAY_NAMES.get(mk, mk[:12])
                comp_table.add_column(display, justify="right", width=10)

            human = self.results.get("human_baseline", {})

            for dim in self.layer_info["dimensions"]:
                row = [format_dimension_name(dim)]

                # Human value
                human_vals = human.get("cognitive_features", {}).get(dim, [])
                human_mean = statistics.mean(human_vals) if human_vals else 0
                row.append(f"{human_mean:.3f}")

                # Model values
                for mk in model_keys:
                    model_info = self.results["by_model"].get(mk, {})
                    model_vals = model_info.get("cognitive_features", {}).get(dim, [])
                    if model_vals:
                        model_mean = statistics.mean(model_vals)
                        # Color by deviation
                        if human_mean != 0:
                            pct_diff = ((model_mean - human_mean) / human_mean) * 100
                            _, style = calculate_deviation_category(pct_diff)
                            row.append(Text(f"{model_mean:.3f}", style=style))
                        else:
                            row.append(f"{model_mean:.3f}")
                    else:
                        row.append("N/A")

                comp_table.add_row(*row)

            yield Static(comp_table)


class StatsView(Static):
    """Corpus and validation statistics."""

    def __init__(self, results: Optional[Dict] = None):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        yield Static(Panel(
            "[bold]Corpus & Validation Statistics[/bold]",
            border_style="cyan"
        ))

        if not self.results:
            yield Static("[dim]No validation data loaded. Run validation to populate.[/dim]")
            return

        # Summary stats
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Metric", width=30)
        table.add_column("Value", justify="right", width=20)

        human = self.results.get("human_baseline", {})
        models = self.results.get("by_model", {})

        table.add_row("Human Samples", str(human.get("samples", 0)))
        table.add_row("AI Models", str(len(models)))
        table.add_row("Total AI Samples", str(sum(m.get("samples", 0) for m in models.values())))

        # Performance (if available)
        if "accuracy" in self.results:
            table.add_row("Accuracy", f"{self.results['accuracy']*100:.1f}%")
        if "human_recognition" in self.results:
            table.add_row("Human Recognition", f"{self.results['human_recognition']*100:.1f}%")
        if "false_positive_rate" in self.results:
            table.add_row("False Positive Rate", f"{self.results['false_positive_rate']*100:.1f}%")

        yield Static(table)


# =============================================================================
# MAIN TUI APP
# =============================================================================

class SpecHOTUI(App):
    """SpecHO v2 Unified Terminal Interface."""

    CSS = """
    TabbedContent {
        height: 100%;
    }

    TabPane {
        padding: 1 2;
    }

    TextArea {
        height: 10;
        margin: 1 0;
    }

    Button {
        margin: 1 0;
    }
    """

    TITLE = "SpecHO v2 - 45D AI Text Detection"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "focus_analyze", "Analyze"),
    ]

    def __init__(self, results_path: Optional[Path] = None):
        super().__init__()
        self.results_path = results_path
        self.results = None

        if results_path and results_path.exists():
            with results_path.open('r') as f:
                self.results = json.load(f)

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent():
            with TabPane("Overview", id="overview"):
                yield ArchitectureView()
                yield LayerOverview()

            with TabPane("Analyze", id="analyze"):
                yield LiveAnalysisView()

            # Layer tabs
            for layer_key, layer_info in DIMENSION_LAYERS.items():
                tab_name = f"Layer {layer_info['short']}"
                with TabPane(tab_name, id=layer_key.lower()):
                    yield LayerDetailView(self.results, layer_key)

            with TabPane("Stats", id="stats"):
                yield StatsView(self.results)

        yield Footer()

    def action_refresh(self) -> None:
        """Reload results."""
        if self.results_path and self.results_path.exists():
            with self.results_path.open('r') as f:
                self.results = json.load(f)
        self.refresh()

    def action_focus_analyze(self) -> None:
        """Switch to analyze tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "analyze"


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SpecHO v2 Terminal UI - 45D AI Text Detection"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to validation results JSON (optional)",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze text from file (non-interactive)",
    )

    args = parser.parse_args()

    # Non-interactive analysis mode
    if args.analyze:
        text = Path(args.analyze).read_text() if Path(args.analyze).exists() else args.analyze

        console = Console()
        console.print("\n[bold cyan]SpecHO v2 - Tier 1 Analysis[/bold cyan]\n")

        start = time.perf_counter()
        features = extract_layer_e(text)
        prediction, confidence, details = classify_tier1(features)
        elapsed = (time.perf_counter() - start) * 1000

        pred_color = "green" if prediction == "human" else "red"
        console.print(f"Prediction: [{pred_color}]{prediction.upper()}[/{pred_color}]")
        console.print(f"Confidence: {confidence:.1%}")
        console.print(f"Time: {elapsed:.2f}ms\n")

        console.print("[bold]Layer E Features:[/bold]")
        for dim, value in features.items():
            console.print(f"  {format_dimension_name(dim):25} {value:.4f}")

        return

    # Interactive TUI mode
    results_path = None
    if args.results:
        results_path = Path(args.results)
    else:
        # Try default locations
        defaults = [
            Path("data/validation_39d_results.json"),
            Path(__file__).parent / "data/validation/validation_39d_results.json",
        ]
        for p in defaults:
            if p.exists():
                results_path = p
                break

    app = SpecHOTUI(results_path)
    app.run()


if __name__ == "__main__":
    main()
