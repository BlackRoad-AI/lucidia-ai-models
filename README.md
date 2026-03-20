# lucidia-ai-models

> **Lucidia AI Models** — Universal AI model memory hub with registry, version tracking, and performance benchmarking.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![BlackRoad AI](https://img.shields.io/badge/BlackRoad-AI-FF1D6C)](https://blackroad.ai)
[![License](https://img.shields.io/badge/license-Proprietary-black)](LICENSE)

---

## Overview

`lucidia-ai-models` is Lucidia's universal AI model memory hub. It provides a structured
registry for tracking AI models from any provider (HuggingFace, OpenAI, Anthropic, Meta,
local), managing model versions with checkpoint tracking, and recording benchmark results
for performance comparisons — all persisted in a local SQLite database.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    lucidia-ai-models                            │
├────────────────┬───────────────────┬───────────────────────────┤
│  CLI Layer     │  Python API       │  Persistence (SQLite)      │
│                │                   │                            │
│  ai-models     │  AIModelRegistry  │  ~/.blackroad/ai_models.db │
│  register      │                   │                            │
│  list          │  ModelEntry       │  ┌────────┐ ┌──────────┐  │
│  version       │  ModelVersion     │  │models  │ │versions  │  │
│  benchmark     │  BenchmarkResult  │  ├────────┤ ├──────────┤  │
│  metrics       │  ModelMetrics     │  │bench-  │ │          │  │
│  get           │                   │  │marks   │ │          │  │
└────────────────┴───────────────────┴──┴────────┴─┴──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Model

```
ModelEntry (1) ──────────── (N) ModelVersion
     │                              │
     │                              │ (FK: model_id)
     └────────────── (N) BenchmarkResult
                            │
                    (FK: model_id, version_id)
```

---

## Features

- 📚 **Model Registry** — register models from any provider with metadata
- 🏷️ **Tag Support** — flexible tagging (chat, instruct, rlhf, code, etc.)
- 📌 **Version Tracking** — track model checkpoints with training step counts
- 🏆 **Benchmarking** — record MMLU, HellaSwag, HumanEval scores with latency metrics
- 📊 **Metrics Aggregation** — best score, avg latency, version history per model
- 🔍 **Provider Filtering** — list models by provider
- 🗄️ **Zero-config DB** — auto-creates `~/.blackroad/ai_models.db`
- 🖥️ **Rich CLI** — formatted table output with colored metrics bar

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| pytest | ≥ 7.0 | Testing |

No external AI library dependencies — pure stdlib + sqlite3.

---

## Installation

```bash
git clone https://github.com/BlackRoad-AI/lucidia-ai-models.git
cd lucidia-ai-models

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LUCIDIA_MODELS_DB` | `~/.blackroad/ai_models.db` | SQLite DB path |

### Supported Providers

The registry is provider-agnostic. Common values:

| Provider | Example Models |
|----------|---------------|
| `huggingface` | Mistral, Falcon, Zephyr |
| `meta` | LLaMA-2, LLaMA-3, CodeLlama |
| `openai` | GPT-4, GPT-3.5-turbo |
| `anthropic` | Claude-3, Claude-2 |
| `deepseek` | DeepSeek-Coder, DeepSeek-V2 |
| `alibaba` | Qwen-2.5, Qwen-72B |
| `blackroad` | Custom fine-tuned models |

---

## Usage

### CLI

#### Register a model

```bash
python src/ai_models.py register \
    --name "LLaMA-3 8B Instruct" \
    --provider meta \
    --arch transformer \
    --params 8B \
    --tags chat instruct rlhf
```

**Output:**
```
✓ Registered LLaMA-3 8B Instruct [a3f2c1e8] provider=meta
```

#### List all models

```bash
python src/ai_models.py list
```

**Output:**
```
──────────────────────────────────────────────────────────
│ ID       │ Name                  │ Provider  │ Params │ Tags             │
──────────────────────────────────────────────────────────
│ a3f2c1e8 │ LLaMA-3 8B Instruct  │ meta      │ 8B     │ chat,instruct    │
│ b1c4d2e0 │ Mistral 7B v0.2      │ hf        │ 7B     │ chat             │
│ c9f3a2d1 │ Qwen-2.5 72B         │ alibaba   │ 72B    │ instruct,code    │
──────────────────────────────────────────────────────────
```

#### Filter by provider

```bash
python src/ai_models.py list --provider meta
```

#### Add a version checkpoint

```bash
python src/ai_models.py version \
    --model-id a3f2c1e8 \
    --version 1.2.0 \
    --steps 50000 \
    --path /models/llama3-8b/checkpoint-50000 \
    --notes "Post-RLHF fine-tune, improved instruction following"
```

**Output:**
```
✓ Version 1.2.0 added for model a3f2c1e8
```

#### Record a benchmark

```bash
python src/ai_models.py benchmark \
    --model-id a3f2c1e8 \
    --task mmlu \
    --score 68.4 \
    --p50 112.0 \
    --p99 380.0 \
    --tps 14.2
```

**Output:**
```
✓ Benchmark [mmlu] score=68.4 ██████░░░░ p50=112.0ms
```

#### View model metrics

```bash
python src/ai_models.py metrics --model-id a3f2c1e8
```

**Output:**
```
── Metrics: a3f2c1e8 ──────────
  model_id               a3f2c1e8
  total_versions         3
  total_benchmarks       5
  best_score             72.1
  avg_latency_ms         145.3
  last_updated           2025-01-15T10:32:11
```

#### Get full model details

```bash
python src/ai_models.py get a3f2c1e8
```

**Output:**
```
── LLaMA-3 8B Instruct ──────────────────
  model_id             a3f2c1e8
  name                 LLaMA-3 8B Instruct
  provider             meta
  architecture         transformer
  parameter_count      8B
  license              apache-2.0
  tags                 ['chat', 'instruct', 'rlhf']
  created_at           2025-01-14T09:00:00
```

---

### Python API

```python
from pathlib import Path
from src.ai_models import (
    AIModelRegistry, ModelEntry, ModelVersion, BenchmarkResult
)

registry = AIModelRegistry()

# Register a model
entry = registry.register_model(ModelEntry(
    name="LLaMA-3 8B",
    provider="meta",
    architecture="transformer",
    parameter_count="8B",
    tags=["chat", "instruct"],
))

# Add a version
registry.add_version(ModelVersion(
    model_id=entry.model_id,
    version="1.0.0",
    training_steps=100_000,
    notes="Base model",
))

# Record a benchmark
registry.benchmark_model(BenchmarkResult(
    model_id=entry.model_id,
    task="mmlu",
    score=68.4,
    latency_p50_ms=112.0,
    latency_p99_ms=380.0,
    throughput_rps=14.2,
))

# Query metrics
metrics = registry.get_metrics(entry.model_id)
print(f"Best score: {metrics.best_score}")

# List all
for model in registry.list_models(provider="meta"):
    print(f"{model.model_id}: {model.name}")

registry.close()
```

---

## API Reference

### `AIModelRegistry`

| Method | Returns | Description |
|--------|---------|-------------|
| `register_model(entry)` | `ModelEntry` | Register a new model |
| `get_model(model_id)` | `Optional[ModelEntry]` | Fetch model by ID |
| `list_models(provider?)` | `List[ModelEntry]` | List all (or filter) models |
| `add_version(version)` | `ModelVersion` | Add a version checkpoint |
| `get_versions(model_id)` | `List[ModelVersion]` | Get all versions |
| `benchmark_model(result)` | `BenchmarkResult` | Record benchmark result |
| `get_metrics(model_id)` | `Optional[ModelMetrics]` | Get aggregate metrics |
| `close()` | `None` | Close DB connection |

### `ModelEntry` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | auto-UUID8 | Unique ID |
| `name` | `str` | required | Full model name |
| `provider` | `str` | `""` | Source provider |
| `architecture` | `str` | `""` | Architecture type |
| `parameter_count` | `str` | `""` | Parameter count (e.g. "7B") |
| `license` | `str` | `"apache-2.0"` | License SPDX |
| `tags` | `List[str]` | `[]` | Searchable tags |

---

## Running Tests

```bash
pytest tests/test_ai_models.py -v

# Expected: 18 passed
```

---

## Database Schema

```sql
-- ~/.blackroad/ai_models.db

CREATE TABLE models (
    model_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    provider        TEXT,
    architecture    TEXT,
    parameter_count TEXT,
    license         TEXT,
    tags_json       TEXT,   -- JSON array
    created_at      TEXT
);

CREATE TABLE versions (
    version_id      TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    version         TEXT NOT NULL,
    checkpoint_path TEXT,
    training_steps  INTEGER DEFAULT 0,
    is_latest       INTEGER DEFAULT 1,
    notes           TEXT,
    created_at      TEXT
);

CREATE TABLE benchmarks (
    bench_id        TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    version_id      TEXT,
    task            TEXT,
    score           REAL,
    latency_p50_ms  REAL,
    latency_p99_ms  REAL,
    throughput_rps  REAL,
    gpu_memory_gb   REAL,
    notes           TEXT,
    created_at      TEXT
);
```

---

## Related Repos

| Repo | Role |
|------|------|
| `lucidia-ai-models-enhanced` | Quantization & LoRA pipeline |
| `blackroad-vllm-mvp` | Inference server wrapper |
| `blackroad-ai-cluster` | Distributed cluster orchestration |
| `blackroad-ai-memory-bridge` | Agent semantic memory |

---

*© BlackRoad OS, Inc. All rights reserved. Proprietary — not open-access.*
