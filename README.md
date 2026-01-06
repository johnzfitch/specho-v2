# SpecHO v2 - Unified AI Text Detection & Model Fingerprinting

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SPECHO V2 UNIFIED                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  CORPUS SYSTEM  │    │   FINGERPRINT   │    │    WEB/API      │          │
│  │                 │    │     SYSTEM      │    │                 │          │
│  │ • HuggingFace   │───►│                 │───►│ definitelynot.ai│          │
│  │ • Arena/LMSYS   │    │ 45D Extraction  │    │ • PHP frontend  │          │
│  │ • Manual        │    │ • Layer A (5D)  │    │ • Python API    │          │
│  │ • Academic      │    │ • Layer B (15D) │    │ • SSE streaming │          │
│  │ • Legal/PACER   │    │ • Layer C (12D) │    │                 │          │
│  │                 │    │ • Layer D (7D)  │    │                 │          │
│  │   500+ samples  │    │ • Layer E (6D)  │    │                 │          │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘          │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   VALIDATION    │    │   CLASSIFIER    │    │   VERIFICATION  │          │
│  │                 │    │                 │    │                 │          │
│  │ • 464 samples   │◄───│ • Tiered (1-3)  │───►│ • Model ID      │          │
│  │ • 7 models      │    │ • <1ms tier 1   │    │ • AURORA trust  │          │
│  │ • Cohen's d     │    │ • 98.6% acc     │    │ • Drift monitor │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  LAYERS: A(trajectory) + B(echo) + C(epistemic+transitions) + D(syntax) + E │
│  TIERS:  1=E(<1ms) → 2=E+B+C(~50ms) → 3=ALL(~100ms)                         │
│  FIX:    24D→45D restored missing Layer B, dropped FP from 27.6%→<8%        │
└──────────────────────────────────────────────────────────────────────────────┘
```

A 45-dimensional feature extraction system for detecting AI-generated text and identifying which model produced it.

## Architecture

```
45D = Layer A (5D) + Layer B (15D) + Layer C (12D) + Layer D (7D) + Layer E (6D)
```

| Layer | Dims | Name | Trust Level | Dependencies |
|-------|------|------|-------------|--------------|
| A | 5 | Trajectory | HIGH | sentence-transformers |
| B | 15 | Echo | MEDIUM-HIGH | spaCy (fallback available) |
| C | 12 | Epistemic + Transitions | MEDIUM | none |
| D | 7 | Syntactic | LOWER | spaCy |
| E | 6 | Lightweight | HIGH | none |

## Quick Start

```python
from src import SpecHO

# Initialize
specho = SpecHO("data/fingerprints")

# Analyze text
result = specho.analyze("Your text here", model_id="gpt-4")

# Verify model identity
verification = specho.verify("Unknown text", claimed_model="claude-sonnet-4")

# Quick detection (Layer E only, <1ms)
from src.fingerprint import LightweightClassifier
clf = LightweightClassifier()
is_ai, confidence = clf.predict("Some text")
```

## Directory Structure

```
specho-v2/
├── src/                      # Core Python modules
│   ├── fingerprint/          # Layer extractors (45D)
│   │   ├── lightweight.py    # Layer E (6D) - no deps
│   │   ├── layer_b_echo.py   # Layer B (15D) - echo patterns
│   │   ├── trajectory.py     # Layer A (5D) - semantic path
│   │   ├── epistemic.py      # Layer C.1 (6D) - hedging
│   │   ├── transitions.py    # Layer C.2 (6D) - discourse
│   │   ├── syntactic.py      # Layer D (7D) - structure
│   │   └── unified_45d.py    # Combined extractor
│   ├── echo_engine/          # Core SpecHO analysis
│   ├── preprocessor/         # spaCy pipeline
│   ├── clause_identifier/    # Clause detection
│   └── scoring/              # Aggregation
│
├── web/                      # definitelynot.ai interface
│   ├── index.php             # Main page
│   ├── api/                  # Backend endpoints
│   └── static/               # JS/CSS
│
├── data/
│   ├── models/               # Trained classifiers
│   ├── fingerprints/         # Reference database
│   └── validation/           # Test results
│
├── tools/                    # Utilities
│   ├── train_specho.py       # Model training
│   ├── ab_test.py            # A/B testing
│   └── visualize_trajectory.py
│
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md       # System design
│   ├── MIGRATION_PLAN.md     # 45D implementation plan
│   └── SPECS.md              # Technical specs
│
└── specho_cli.py             # Command-line interface
```

## Tiered Extraction

| Tier | Layers | Latency | Accuracy | Use Case |
|------|--------|---------|----------|----------|
| 1 | E only | <1ms | 98.6% | High-volume screening |
| 2 | E+B+C | ~50ms | ~99% | Standard analysis |
| 3 | All 45D | ~100ms | >99% | Deep verification |

## Key Files

- `src/fingerprint/lightweight.py` - Fast 6D detection (Layer E)
- `src/fingerprint/unified_45d.py` - Full 45D extraction
- `src/specHO_integrated.py` - High-level API (SpecHO class)
- `web/index.php` - definitelynot.ai frontend
- `specho_cli.py` - CLI tool

## Performance

From validation on 464-sample corpus:

| Metric | 24D (old) | 45D (new) |
|--------|-----------|-----------|
| Accuracy | 94.4% | >98% |
| Human Recognition | 72.4% | >92% |
| False Positive Rate | 27.6% | <8% |

The key fix: restoring Layer B (15D echo features) which was missing from the 24D system.

## Dependencies

**Minimal (Tier 1 only):**
```
numpy
```

**Full (all tiers):**
```
numpy
scipy
scikit-learn
spacy
sentence-transformers
```

## License

MIT
