"""Tests for src/ai_models.py — Lucidia AI Model Registry."""
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_models import (
    ModelEntry, ModelVersion, BenchmarkResult,
    ModelMetrics, AIModelRegistry,
    OllamaRouter, OLLAMA_ALIASES,
)


@pytest.fixture
def registry(tmp_path):
    reg = AIModelRegistry(db_path=tmp_path / "test_registry.db")
    yield reg
    reg.close()


@pytest.fixture
def reg_with_model(registry):
    entry = ModelEntry(
        model_id="llama3-test", name="LLaMA-3 Test",
        provider="meta", architecture="transformer",
        parameter_count="8B",
    )
    registry.register_model(entry)
    return registry, entry


# ── dataclass defaults ────────────────────────────────────────────────────────
def test_model_entry_defaults():
    entry = ModelEntry()
    assert entry.license == "apache-2.0"
    assert entry.tags == []
    assert len(entry.model_id) == 8


def test_model_version_defaults():
    ver = ModelVersion()
    assert ver.version == "1.0.0"
    assert ver.is_latest is True
    assert ver.training_steps == 0


# ── registration ──────────────────────────────────────────────────────────────
def test_register_model_returns_entry(registry):
    entry = ModelEntry(model_id="m1", name="Model One", provider="openai")
    result = registry.register_model(entry)
    assert result.model_id == "m1"


def test_get_model_after_register(registry):
    entry = ModelEntry(model_id="get-test", name="GetTest", provider="google")
    registry.register_model(entry)
    fetched = registry.get_model("get-test")
    assert fetched is not None
    assert fetched.name == "GetTest"
    assert fetched.provider == "google"


def test_get_model_not_found(registry):
    assert registry.get_model("nonexistent") is None


def test_tags_preserved_round_trip(registry):
    entry = ModelEntry(
        model_id="tagged", name="Tagged", provider="hf",
        tags=["chat", "instruct", "rlhf"]
    )
    registry.register_model(entry)
    fetched = registry.get_model("tagged")
    assert fetched.tags == ["chat", "instruct", "rlhf"]


# ── listing ───────────────────────────────────────────────────────────────────
def test_list_models_empty(registry):
    assert registry.list_models() == []


def test_list_models_count(registry):
    for i in range(3):
        registry.register_model(ModelEntry(model_id=f"m{i}", name=f"M{i}", provider="meta"))
    assert len(registry.list_models()) == 3


def test_list_models_filter_by_provider(registry):
    registry.register_model(ModelEntry(model_id="a1", name="A1", provider="openai"))
    registry.register_model(ModelEntry(model_id="b1", name="B1", provider="google"))
    registry.register_model(ModelEntry(model_id="c1", name="C1", provider="openai"))
    openai = registry.list_models(provider="openai")
    assert len(openai) == 2
    assert all(m.provider == "openai" for m in openai)


# ── versioning ────────────────────────────────────────────────────────────────
def test_add_version(reg_with_model):
    registry, entry = reg_with_model
    ver = ModelVersion(model_id=entry.model_id, version="1.0.0", training_steps=1000)
    result = registry.add_version(ver)
    assert result.version == "1.0.0"
    versions = registry.get_versions(entry.model_id)
    assert len(versions) == 1


def test_latest_version_flag_on_second_add(reg_with_model):
    registry, entry = reg_with_model
    registry.add_version(ModelVersion(model_id=entry.model_id, version="1.0.0"))
    registry.add_version(ModelVersion(model_id=entry.model_id, version="2.0.0"))
    versions = registry.get_versions(entry.model_id)
    # Most recent first
    assert versions[0].is_latest is True
    assert versions[1].is_latest is False


def test_get_versions_empty_model(registry):
    assert registry.get_versions("no-model") == []


# ── benchmarking ──────────────────────────────────────────────────────────────
def test_benchmark_model_stores_result(reg_with_model):
    registry, entry = reg_with_model
    bench = BenchmarkResult(
        model_id=entry.model_id, task="mmlu",
        score=72.5, latency_p50_ms=120.0, latency_p99_ms=450.0,
        throughput_rps=8.5,
    )
    result = registry.benchmark_model(bench)
    assert result.score == 72.5


def test_benchmark_score_clamped_display(reg_with_model):
    """Score bar should not crash on 0 or 100."""
    registry, entry = reg_with_model
    registry.benchmark_model(BenchmarkResult(model_id=entry.model_id, task="t1", score=0.0))
    registry.benchmark_model(BenchmarkResult(model_id=entry.model_id, task="t2", score=100.0))


# ── metrics ───────────────────────────────────────────────────────────────────
def test_get_metrics_with_data(reg_with_model):
    registry, entry = reg_with_model
    registry.add_version(ModelVersion(model_id=entry.model_id, version="1.0.0"))
    registry.benchmark_model(BenchmarkResult(
        model_id=entry.model_id, task="mmlu",
        score=80.0, latency_p50_ms=100.0
    ))
    metrics = registry.get_metrics(entry.model_id)
    assert metrics is not None
    assert metrics.total_versions == 1
    assert metrics.total_benchmarks == 1
    assert metrics.best_score == 80.0


def test_get_metrics_no_data(registry):
    assert registry.get_metrics("no-such-model") is None


def test_get_metrics_best_score_is_max(reg_with_model):
    registry, entry = reg_with_model
    registry.add_version(ModelVersion(model_id=entry.model_id, version="1.0.0"))
    registry.benchmark_model(BenchmarkResult(model_id=entry.model_id, task="t1", score=60.0))
    registry.benchmark_model(BenchmarkResult(model_id=entry.model_id, task="t2", score=85.0))
    registry.benchmark_model(BenchmarkResult(model_id=entry.model_id, task="t3", score=70.0))
    metrics = registry.get_metrics(entry.model_id)
    assert metrics.best_score == 85.0


# ── OllamaRouter ──────────────────────────────────────────────────────────────

def test_ollama_aliases_contains_expected():
    assert "@copilot" in OLLAMA_ALIASES
    assert "@lucidia" in OLLAMA_ALIASES
    assert "@blackboxprogramming" in OLLAMA_ALIASES
    assert "@ollama" in OLLAMA_ALIASES


def test_resolve_alias_known():
    for alias in ("@copilot", "@lucidia", "@blackboxprogramming", "@ollama"):
        assert OllamaRouter.resolve_alias(alias) is True


def test_resolve_alias_case_insensitive():
    assert OllamaRouter.resolve_alias("@Copilot") is True
    assert OllamaRouter.resolve_alias("@LUCIDIA") is True
    assert OllamaRouter.resolve_alias("@BlackboxProgramming") is True


def test_resolve_alias_unknown():
    assert OllamaRouter.resolve_alias("@chatgpt") is False
    assert OllamaRouter.resolve_alias("@claude") is False
    assert OllamaRouter.resolve_alias("@openai") is False
    assert OllamaRouter.resolve_alias("") is False


def test_ollama_router_chat_success():
    router = OllamaRouter(base_url="http://localhost:11434")
    fake_response = json.dumps({"response": "Hello from Ollama!"}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = router.chat("Hello", model="llama3", alias="@ollama")
        assert result["response"] == "Hello from Ollama!"
        mock_open.assert_called_once()


def test_ollama_router_chat_all_aliases():
    router = OllamaRouter(base_url="http://localhost:11434")
    fake_response = json.dumps({"response": "ok"}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    for alias in ("@copilot", "@lucidia", "@blackboxprogramming", "@ollama"):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = router.chat("test", alias=alias)
            assert result["response"] == "ok"


def test_ollama_router_chat_invalid_alias():
    router = OllamaRouter()
    with pytest.raises(ValueError, match="Unknown alias"):
        router.chat("Hello", alias="@chatgpt")


def test_ollama_router_uses_local_url_by_default():
    router = OllamaRouter()
    assert "localhost" in router.base_url or "127.0.0.1" in router.base_url


def test_ollama_router_custom_base_url():
    router = OllamaRouter(base_url="http://192.168.1.100:11434")
    assert router.base_url == "http://192.168.1.100:11434"


def test_ollama_router_rejects_external_url():
    with pytest.raises(ValueError, match="local/private address"):
        OllamaRouter(base_url="http://evil.example.com:11434")


def test_ollama_router_payload_sent_correctly():
    router = OllamaRouter(base_url="http://localhost:11434")
    fake_response = json.dumps({"response": "pong"}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        router.chat("ping", model="mistral", alias="@lucidia")
        call_args = mock_open.call_args[0][0]
        payload = json.loads(call_args.data.decode())
        assert payload["model"] == "mistral"
        assert payload["prompt"] == "ping"
        assert payload["stream"] is False
