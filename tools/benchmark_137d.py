#!/usr/bin/env python3
"""
Benchmark 137D Features on 500 Samples

Extracts all 137 features from human and AI text pairs,
calculates Cohen's d effect sizes, and reports top discriminators.

Uses rich for beautiful TUI display.
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich import box
from rich.text import Text

from extractors.registry import get_registry

console = Console()


def load_samples(path: str) -> List[Dict]:
    """Load samples from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", data)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def effect_size_interpretation(d: float) -> Tuple[str, str]:
    """Interpret Cohen's d magnitude with color."""
    abs_d = abs(d)
    if abs_d >= 1.0:
        return "LARGE", "bright_green" if d > 0 else "bright_red"
    elif abs_d >= 0.8:
        return "large", "green" if d > 0 else "red"
    elif abs_d >= 0.5:
        return "medium", "yellow"
    elif abs_d >= 0.2:
        return "small", "dim"
    else:
        return "negligible", "dim white"


def create_effect_bar(d: float, width: int = 20) -> Text:
    """Create a visual bar representing effect size."""
    abs_d = min(abs(d), 2.0)  # Cap at 2.0 for display
    filled = int((abs_d / 2.0) * width)

    if d > 0:
        bar = "█" * filled + "░" * (width - filled)
        color = "green" if abs_d >= 0.5 else "dim green"
        direction = "→ HUMAN"
    else:
        bar = "░" * (width - filled) + "█" * filled
        color = "red" if abs_d >= 0.5 else "dim red"
        direction = "AI ←"

    return Text(f"[{bar}] {direction}", style=color)


def main():
    console.clear()

    # Header
    console.print(Panel.fit(
        "[bold cyan]137D Feature Benchmark[/bold cyan]\n"
        "[dim]Analyzing Human vs AI Text Patterns[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Load registry
    with console.status("[bold green]Loading extractors..."):
        registry = get_registry(quiet=True)

    console.print(f"[green]✓[/green] Loaded [bold]{len(registry._extractors)}[/bold] extractors with [bold]{registry.n_features}[/bold] features")

    # Load samples
    samples_path = Path(__file__).parent.parent / "data/corpus/samples_500.json"

    with console.status(f"[bold green]Loading samples from {samples_path.name}..."):
        samples = load_samples(samples_path)

    console.print(f"[green]✓[/green] Loaded [bold]{len(samples)}[/bold] samples\n")

    # Storage for feature values
    human_features = defaultdict(list)
    ai_features = defaultdict(list)

    # Process samples with progress bar
    console.print("[bold]Extracting features...[/bold]")

    processed = 0
    errors = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing samples...", total=len(samples))

        for i, sample in enumerate(samples):
            progress.update(task, advance=1)

            human_text = sample.get("human", "")
            ai_variants = sample.get("ai_variants", {})

            if not human_text or len(human_text) < 100:
                continue

            # Extract human features
            try:
                human_result = registry.extract_all(human_text)
                for name, value in human_result.features.items():
                    human_features[name].append(value)
            except Exception:
                errors += 1
                continue

            # Extract AI features (use first available variant)
            for model_name, ai_text in ai_variants.items():
                if ai_text and len(ai_text) >= 100:
                    try:
                        ai_result = registry.extract_all(ai_text)
                        for name, value in ai_result.features.items():
                            ai_features[name].append(value)
                    except Exception:
                        errors += 1
                    break

            processed += 1

    elapsed = time.time() - start_time
    console.print(f"\n[green]✓[/green] Processed [bold]{processed}[/bold] samples in [bold]{elapsed:.1f}s[/bold] ({elapsed/max(processed,1):.2f}s/sample)")
    if errors:
        console.print(f"[yellow]⚠[/yellow] Errors: {errors}")

    # Calculate Cohen's d for each feature
    effect_sizes = []
    for feature in sorted(human_features.keys()):
        h_vals = human_features[feature]
        a_vals = ai_features[feature]

        if len(h_vals) < 10 or len(a_vals) < 10:
            continue

        d = cohens_d(h_vals, a_vals)
        h_mean = sum(h_vals) / len(h_vals)
        a_mean = sum(a_vals) / len(a_vals)

        effect_sizes.append((feature, d, h_mean, a_mean))

    effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)

    # TOP DISCRIMINATORS TABLE
    console.print()
    table = Table(
        title="[bold cyan]Top 30 Discriminating Features[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", style="dim", width=4)
    table.add_column("Feature", style="white", width=35)
    table.add_column("Cohen's d", justify="right", width=10)
    table.add_column("Effect Size", width=30)
    table.add_column("Human μ", justify="right", width=8)
    table.add_column("AI μ", justify="right", width=8)
    table.add_column("Interpretation", width=12)

    for rank, (feature, d, h_mean, a_mean) in enumerate(effect_sizes[:30], 1):
        interp, color = effect_size_interpretation(d)
        bar = create_effect_bar(d)

        d_style = "bold green" if d > 0 else "bold red"

        table.add_row(
            str(rank),
            feature,
            f"[{d_style}]{d:+.3f}[/{d_style}]",
            bar,
            f"{h_mean:.3f}",
            f"{a_mean:.3f}",
            f"[{color}]{interp}[/{color}]"
        )

    console.print(table)

    # GROUP SUMMARY TABLE
    console.print()
    group_effects = defaultdict(list)
    for feature, d, _, _ in effect_sizes:
        group = "unknown"
        for ext_name, ext in registry._extractors.items():
            if feature in ext.feature_names:
                group = ext.group
                break
        group_effects[group].append((feature, d))

    group_table = Table(
        title="[bold cyan]Summary by Extractor Group[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    group_table.add_column("Group", style="white", width=20)
    group_table.add_column("Features", justify="right", width=10)
    group_table.add_column("Avg |d|", justify="right", width=10)
    group_table.add_column("Max |d|", justify="right", width=10)
    group_table.add_column("Best Feature", width=30)

    sorted_groups = sorted(group_effects.items(), key=lambda x: -max(abs(d) for _, d in x[1]))

    for group, features in sorted_groups:
        effects = [abs(d) for _, d in features]
        avg_d = sum(effects) / len(effects)
        max_d = max(effects)
        best_feature = max(features, key=lambda x: abs(x[1]))[0]

        color = "green" if max_d >= 0.8 else "yellow" if max_d >= 0.5 else "dim"

        group_table.add_row(
            f"[{color}]{group}[/{color}]",
            str(len(features)),
            f"{avg_d:.3f}",
            f"[bold]{max_d:.3f}[/bold]",
            best_feature
        )

    console.print(group_table)

    # SUMMARY PANEL
    strong = sum(1 for _, d, _, _ in effect_sizes if abs(d) >= 1.0)
    medium = sum(1 for _, d, _, _ in effect_sizes if 0.5 <= abs(d) < 1.0)
    small = sum(1 for _, d, _, _ in effect_sizes if 0.2 <= abs(d) < 0.5)
    negligible = sum(1 for _, d, _, _ in effect_sizes if abs(d) < 0.2)

    human_biased = sum(1 for _, d, _, _ in effect_sizes if d > 0.5)
    ai_biased = sum(1 for _, d, _, _ in effect_sizes if d < -0.5)

    summary = f"""
[bold green]Strong discriminators (|d| ≥ 1.0):[/bold green] {strong}
[bold yellow]Medium discriminators (|d| 0.5-1.0):[/bold yellow] {medium}
[dim]Small discriminators (|d| 0.2-0.5):[/dim] {small}
[dim white]Negligible (|d| < 0.2):[/dim white] {negligible}

[bold]Direction of effect:[/bold]
  [green]→ Human-biased (d > 0.5):[/green] {human_biased} features
  [red]← AI-biased (d < -0.5):[/red] {ai_biased} features

[bold]Total features analyzed:[/bold] {len(effect_sizes)}
[bold]Samples processed:[/bold] {processed} human + {processed} AI
"""

    console.print(Panel(summary, title="[bold cyan]Summary Statistics[/bold cyan]", border_style="cyan"))

    # INTERPRETATION GUIDE
    guide = """
[bold]How to read Cohen's d:[/bold]
  • [green]Positive d (+)[/green] = Feature is HIGHER in human text
  • [red]Negative d (−)[/red] = Feature is HIGHER in AI text
  • |d| ≥ 0.8 = Large effect (reliable discriminator)
  • |d| 0.5-0.8 = Medium effect (useful signal)
  • |d| 0.2-0.5 = Small effect (weak signal)
  • |d| < 0.2 = Negligible (not useful)
"""
    console.print(Panel(guide, title="[bold]Interpretation Guide[/bold]", border_style="dim"))


if __name__ == "__main__":
    main()
