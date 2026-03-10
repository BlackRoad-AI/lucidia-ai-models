# Lucidia AI Models

<div align="center">

### Click a portal. See it work. That's it.

[![Open Lucidia](https://img.shields.io/badge/✨%20Open%20Lucidia-lucidia.earth-FF1D6C?style=for-the-badge&logoColor=white)](https://lucidia.earth)
[![BlackRoad AI](https://img.shields.io/badge/🖤%20BlackRoad%20AI-blackroad.io-F5A623?style=for-the-badge&logoColor=white)](https://blackroad.io)
[![Live Portal](https://img.shields.io/badge/🌐%20Live%20Portal-Open%20Now-2979FF?style=for-the-badge&logoColor=white)](https://lucidia.earth)

</div>

---

## 🚀 Live Portals — No Setup Required

> **No terminal. No installs. No jargon. Just click.**

| Portal | What It Does | Open |
|--------|-------------|------|
| 🌍 **[lucidia.earth](https://lucidia.earth)** | The full Lucidia AI experience | **[→ Open Now](https://lucidia.earth)** |
| 🖤 **[blackroad.io](https://blackroad.io)** | BlackRoad AI platform | **[→ Open Now](https://blackroad.io)** |
| ⚙️ **[GitHub Source](https://github.com/BlackRoad-AI/lucidia-ai-models)** | Browse the source code | **[→ View Code](https://github.com/BlackRoad-AI/lucidia-ai-models)** |
| 📧 **[Contact Us](mailto:blackroad.systems@gmail.com)** | Get help, say hello | **[→ Email](mailto:blackroad.systems@gmail.com)** |

---

## ✨ What Is This?

**Lucidia AI Models** is the intelligence engine that remembers your AI models so you don't have to.

- Track every model you've ever used — from OpenAI to Meta to your own custom builds
- Know which version of which model performed best and why
- Never lose a benchmark result again
- Everything persisted automatically — no cloud required

**You don't need to touch any code to use the portals above.** This repo is the engine under the hood.

---

## 🎯 What You Can Do Right Now

1. **Go to [lucidia.earth](https://lucidia.earth)** — explore the live AI platform
2. **Go to [blackroad.io](https://blackroad.io)** — see the full BlackRoad AI ecosystem
3. **Star this repo** — get updates when new portals go live

That's literally it for most people. ☝️

---

## 🧠 What's Inside (For the Curious)

| Feature | Description |
|---------|-------------|
| 📚 **Model Registry** | Register models from any provider with metadata |
| 📌 **Version Tracking** | Track checkpoints, training steps, notes |
| 🏆 **Benchmarking** | MMLU, HellaSwag, HumanEval scores + latency |
| 📊 **Metrics Aggregation** | Best scores, averages, full history per model |
| 🗄️ **Zero-Config DB** | Auto-creates `~/.blackroad/ai_models.db` — just works |
| 🔒 **Your Data** | Local SQLite, no external cloud needed |

---

## 💻 For Developers (Optional)

<details>
<summary>Click to expand developer setup</summary>

### Requirements

- Python ≥ 3.9

### Quick Start

```bash
git clone https://github.com/BlackRoad-AI/lucidia-ai-models.git
cd lucidia-ai-models
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Register a Model

```bash
python src/ai_models.py register \
    --name "LLaMA-3 8B Instruct" \
    --provider meta \
    --arch transformer \
    --params 8B \
    --tags chat instruct
```

### List Models

```bash
python src/ai_models.py list
```

### Run Tests

```bash
pytest tests/test_ai_models.py -v
```

### Python API

```python
from src.ai_models import AIModelRegistry, ModelEntry

registry = AIModelRegistry()
entry = registry.register_model(ModelEntry(
    name="LLaMA-3 8B",
    provider="meta",
    architecture="transformer",
    parameter_count="8B",
    tags=["chat", "instruct"],
))
registry.close()
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LUCIDIA_MODELS_DB` | `~/.blackroad/ai_models.db` | SQLite DB path |

</details>

---

## 🌐 Related Portals

| Portal | Role |
|--------|------|
| [lucidia.earth](https://lucidia.earth) | Main Lucidia experience |
| [blackroad.io](https://blackroad.io) | BlackRoad AI platform |
| [github.com/BlackRoad-AI](https://github.com/BlackRoad-AI) | All open repos |

---

<div align="center">

**[✨ Open Lucidia](https://lucidia.earth)** · **[🖤 BlackRoad AI](https://blackroad.io)** · **[📧 Contact](mailto:blackroad.systems@gmail.com)**

*© BlackRoad OS, Inc. All rights reserved.*

</div>
