#!/usr/bin/env python3
"""
39D Cognitive Fingerprint TUI Visualization

Textual-based terminal UI for comparing human baseline vs AI model fingerprints.
Designed to scale beyond 39D as new fingerprinting techniques are added.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich import box
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Header, Footer, Static, DataTable, TabbedContent, TabPane
from textual.binding import Binding


# Dimension metadata - extensible for future additions
DIMENSION_LAYERS = {
    "TRAJECTORY": {
        "description": "Semantic Navigation Geometry",
        "trust": "HIGH",
        "dimensions": [
            "concept_jump_mean",
            "concept_jump_variance",
            "path_tortuosity",
            "turning_angle_mean",
            "return_rate",
        ],
    },
    "SPECHO_EXTENDED": {
        "description": "Echo Pattern Statistics",
        "trust": "MEDIUM-HIGH",
        "dimensions": [
            "phonetic_mean", "phonetic_std", "phonetic_max",
            "structural_mean", "structural_std", "structural_max",
            "semantic_mean", "semantic_std", "semantic_max",
            "cooccurrence_rate", "geometric_mean",
            "overall_mean", "overall_std", "overall_max", "burstiness",
        ],
    },
    "EPISTEMIC": {
        "description": "Hedging & Confidence Patterns",
        "trust": "MEDIUM",
        "dimensions": [
            "hedge_density", "hedge_clustering", "hedge_position_bias",
            "confidence_mean", "confidence_variance", "confidence_arc",
        ],
    },
    "TRANSITIONS": {
        "description": "Discourse Connectors",
        "trust": "MEDIUM",
        "dimensions": [
            "additive_rate", "contrastive_rate", "causal_rate",
            "temporal_rate", "exemplifying_rate", "reformulating_rate",
        ],
    },
    "SYNTACTIC": {
        "description": "Structural Rhythm",
        "trust": "LOWER",
        "dimensions": [
            "clause_length_mean", "clause_length_std", "clause_rhythm_autocorr",
            "sentence_complexity", "comma_density", "semicolon_rate", "parenthetical_rate",
        ],
    },
}

MODEL_DISPLAY_NAMES = {
    "gemma-2-9b": "Gemini-2-9B",
    "mistral-7B": "Mistral-7B",
    "qwen-2-72B": "Qwen-2-72B",
    "llama-8B": "Llama-3-8B",
    "accounts/yi-01-ai/models/yi-large": "Yi-Large",
    "GPT_4-o": "GPT-4o",
}


def calculate_deviation_category(percent_diff: float) -> Tuple[str, str]:
    """
    Categorize deviation magnitude.

    Returns: (category, style)
    """
    abs_diff = abs(percent_diff)

    if abs_diff < 10:
        return ("NEAR", "green")
    elif abs_diff < 25:
        return ("MINOR", "yellow")
    elif abs_diff < 50:
        return ("MODERATE", "orange")
    else:
        return ("MAJOR", "red bold")


def format_dimension_name(dim: str) -> str:
    """Format dimension name for display."""
    return dim.replace("_", " ").title()


def create_bar(value: float, min_val: float, max_val: float, width: int = 20) -> str:
    """Create horizontal bar chart using Rich's safe unicode."""
    if max_val == min_val:
        filled = width // 2
    else:
        normalized = (value - min_val) / (max_val - min_val)
        filled = int(normalized * width)

    # Use Rich-safe block characters
    return "█" * filled + "░" * (width - filled)


def identify_top_differentiators(results: Dict, top_n: int = 5) -> List[Tuple[str, float, str]]:
    """
    Identify dimensions with highest variance across models.

    Returns: List of (dimension_name, variance, layer_name)
    """
    human = results["human_baseline"]
    model_data = results["by_model"]

    dimension_variances = []

    # Iterate through all layers
    for layer_name, layer_info in DIMENSION_LAYERS.items():
        for dim in layer_info["dimensions"]:
            if dim not in human["cognitive_features"]:
                continue

            values = []

            # Human value
            human_vals = human["cognitive_features"][dim]
            if human_vals:
                values.append(statistics.mean(human_vals))

            # AI model values
            for model_info in model_data.values():
                if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                    model_vals = model_info["cognitive_features"][dim]
                    if model_vals:
                        values.append(statistics.mean(model_vals))

            # Calculate variance
            if len(values) > 1:
                variance = statistics.variance(values)
                dimension_variances.append((dim, variance, layer_name))

    # Sort by variance descending
    dimension_variances.sort(key=lambda x: x[1], reverse=True)

    return dimension_variances[:top_n]


class SummaryView(Static):
    """Summary view showing top differentiators."""

    def __init__(self, results: Dict):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        # Create summary content
        console = Console()

        # Top differentiators
        top_diff = identify_top_differentiators(self.results, top_n=5)

        table = Table(
            title="Top 5 Key Differentiators (Highest Variance Across Models)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Rank", style="dim", width=4)
        table.add_column("Dimension", style="bold", width=30)
        table.add_column("Layer", width=20)
        table.add_column("Human", justify="right", width=10)
        table.add_column("AI Range", justify="right", width=15)
        table.add_column("Distribution", width=20)

        human = self.results["human_baseline"]
        model_data = self.results["by_model"]

        for rank, (dim, variance, layer) in enumerate(top_diff, 1):
            # Get human value
            human_vals = human["cognitive_features"][dim]
            human_mean = statistics.mean(human_vals) if human_vals else 0.0

            # Get AI model values
            ai_values = []
            for model_info in model_data.values():
                if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                    vals = model_info["cognitive_features"][dim]
                    if vals:
                        ai_values.append(statistics.mean(vals))

            if ai_values:
                ai_min = min(ai_values)
                ai_max = max(ai_values)
                ai_range = f"{ai_min:.2f}-{ai_max:.2f}"

                # Create distribution bar
                all_vals = [human_mean] + ai_values
                dist_bar = create_bar(human_mean, min(all_vals), max(all_vals), width=15)

                table.add_row(
                    f"{rank}.",
                    format_dimension_name(dim),
                    layer,
                    f"{human_mean:.3f}",
                    ai_range,
                    dist_bar,
                )

        yield Static(table)


class LayerDetailView(Static):
    """Detailed view for a specific layer."""

    def __init__(self, results: Dict, layer_name: str):
        super().__init__()
        self.results = results
        self.layer_name = layer_name
        self.layer_info = DIMENSION_LAYERS[layer_name]

    def compose(self) -> ComposeResult:
        console = Console()

        # Layer header
        header_text = Text()
        header_text.append(f"{self.layer_name}\n", style="bold cyan")
        header_text.append(f"{self.layer_info['description']} ", style="italic")
        header_text.append(f"(Trust: {self.layer_info['trust']})", style="dim")

        yield Static(Panel(header_text, box=box.DOUBLE))

        # Create comparison table
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            expand=True,
        )

        table.add_column("Dimension", style="bold", width=30)
        table.add_column("Human", justify="right", width=10)

        # Add model columns
        model_keys = list(MODEL_DISPLAY_NAMES.keys())
        for model_key in model_keys:
            display_name = MODEL_DISPLAY_NAMES[model_key]
            table.add_column(display_name, justify="right", width=12)

        # Populate table
        human = self.results["human_baseline"]
        model_data = self.results["by_model"]

        for dim in self.layer_info["dimensions"]:
            if dim not in human["cognitive_features"]:
                continue

            row_data = [format_dimension_name(dim)]

            # Human value
            human_vals = human["cognitive_features"][dim]
            human_mean = statistics.mean(human_vals) if human_vals else 0.0
            row_data.append(f"{human_mean:.3f}")

            # Model values
            for model_key in model_keys:
                if model_key not in model_data:
                    row_data.append("N/A")
                    continue

                model_info = model_data[model_key]
                if model_info["samples"] > 0 and dim in model_info["cognitive_features"]:
                    vals = model_info["cognitive_features"][dim]
                    if vals:
                        model_mean = statistics.mean(vals)

                        # Calculate deviation
                        if human_mean != 0:
                            percent_diff = ((model_mean - human_mean) / human_mean) * 100
                        else:
                            percent_diff = 0

                        category, style = calculate_deviation_category(percent_diff)

                        # Format with deviation indicator
                        value_str = f"{model_mean:.3f}"
                        if abs(percent_diff) >= 10:
                            arrow = "↑" if percent_diff > 0 else "↓"
                            value_str += arrow

                        row_data.append(Text(value_str, style=style))
                    else:
                        row_data.append("N/A")
                else:
                    row_data.append("N/A")

            table.add_row(*row_data)

        yield Static(table)


class ModelComparisonView(Static):
    """Side-by-side comparison of all models."""

    def __init__(self, results: Dict):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        # Create model signature summary
        table = Table(
            title="Model Signature Comparison",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Model", style="bold", width=15)
        table.add_column("Samples", justify="right", width=8)
        table.add_column("Key Characteristics", width=60)

        # Human baseline
        human = self.results["human_baseline"]
        table.add_row(
            "Human",
            str(human["samples"]),
            "[dim]Baseline reference for all comparisons[/dim]",
        )

        # AI models
        model_data = self.results["by_model"]

        for model_key in MODEL_DISPLAY_NAMES.keys():
            if model_key not in model_data:
                continue

            model_info = model_data[model_key]
            display_name = MODEL_DISPLAY_NAMES[model_key]

            # Generate characteristic summary (simple, data-driven)
            characteristics = self._generate_characteristics(model_key, model_info, human)

            table.add_row(
                display_name,
                str(model_info["samples"]),
                characteristics,
            )

        yield Static(table)

    def _generate_characteristics(self, model_key: str, model_info: Dict, human: Dict) -> str:
        """Generate simple characteristic summary based on major deviations."""
        characteristics = []

        # Check key dimensions for major deviations
        key_dims = [
            "path_tortuosity",
            "parenthetical_rate",
            "hedge_density",
            "clause_length_mean",
            "burstiness",
        ]

        for dim in key_dims:
            if dim not in model_info["cognitive_features"] or dim not in human["cognitive_features"]:
                continue

            model_vals = model_info["cognitive_features"][dim]
            human_vals = human["cognitive_features"][dim]

            if not model_vals or not human_vals:
                continue

            model_mean = statistics.mean(model_vals)
            human_mean = statistics.mean(human_vals)

            if human_mean != 0:
                percent_diff = ((model_mean - human_mean) / human_mean) * 100

                if abs(percent_diff) >= 50:
                    direction = "high" if percent_diff > 0 else "low"
                    dim_display = format_dimension_name(dim)
                    characteristics.append(f"{dim_display}: [bold]{direction}[/bold]")

        if characteristics:
            return " | ".join(characteristics[:3])  # Limit to top 3
        else:
            return "[dim]Similar to human baseline[/dim]"


class FingerprintTUI(App):
    """Textual TUI for 39D cognitive fingerprint visualization."""

    CSS = """
    TabbedContent {
        height: 100%;
    }

    TabPane {
        padding: 1 2;
    }
    """

    TITLE = "39D Cognitive Fingerprint Analysis"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, results_path: Path):
        super().__init__()
        self.results_path = results_path

        # Load results
        with results_path.open('r') as f:
            self.results = json.load(f)

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent():
            with TabPane("Summary", id="summary"):
                yield SummaryView(self.results)

            with TabPane("Model Comparison", id="models"):
                yield ModelComparisonView(self.results)

            # Add tab for each layer
            for layer_name in DIMENSION_LAYERS.keys():
                layer_display = layer_name.replace("_", " ").title()
                with TabPane(layer_display, id=layer_name.lower()):
                    yield LayerDetailView(self.results, layer_name)

        yield Footer()

    def action_refresh(self) -> None:
        """Reload results and refresh display."""
        with self.results_path.open('r') as f:
            self.results = json.load(f)
        self.refresh()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="39D Cognitive Fingerprint TUI Visualization"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="data/validation_39d_results.json",
        help="Path to results JSON file",
    )

    args = parser.parse_args()

    results_path = Path(__file__).parent.parent / args.results

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    app = FingerprintTUI(results_path)
    app.run()


if __name__ == "__main__":
    main()
