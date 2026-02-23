"""
Lucidia AI Models — Universal AI model memory hub.
Model registry, version tracking, and performance benchmarking.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "ai_models.db"


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class ModelEntry:
    model_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    provider: str = ""
    architecture: str = ""
    parameter_count: str = ""
    license: str = "apache-2.0"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ModelVersion:
    version_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    version: str = "1.0.0"
    checkpoint_path: str = ""
    training_steps: int = 0
    is_latest: bool = True
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BenchmarkResult:
    bench_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    version_id: str = ""
    task: str = ""
    score: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    gpu_memory_gb: float = 0.0
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ModelMetrics:
    model_id: str
    total_versions: int
    total_benchmarks: int
    best_score: float
    avg_latency_ms: float
    last_updated: str


# ── Core class ────────────────────────────────────────────────────────────────
class AIModelRegistry:
    """Universal AI model memory hub with version tracking and benchmarking."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT,
                architecture TEXT,
                parameter_count TEXT,
                license TEXT,
                tags_json TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                checkpoint_path TEXT,
                training_steps INTEGER DEFAULT 0,
                is_latest INTEGER DEFAULT 1,
                notes TEXT,
                created_at TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );
            CREATE TABLE IF NOT EXISTS benchmarks (
                bench_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version_id TEXT,
                task TEXT,
                score REAL,
                latency_p50_ms REAL,
                latency_p99_ms REAL,
                throughput_rps REAL,
                gpu_memory_gb REAL,
                notes TEXT,
                created_at TEXT
            );
        """)
        self._conn.commit()

    def register_model(self, entry: ModelEntry) -> ModelEntry:
        """Register a new model in the hub."""
        self._conn.execute(
            "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.model_id, entry.name, entry.provider, entry.architecture,
             entry.parameter_count, entry.license,
             json.dumps(entry.tags), entry.created_at)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Registered {BOLD}{entry.name}{NC} "
              f"[{C}{entry.model_id}{NC}] provider={entry.provider}")
        return entry

    def add_version(self, version: ModelVersion) -> ModelVersion:
        """Track a new model version."""
        # Mark previous versions as not-latest
        self._conn.execute(
            "UPDATE versions SET is_latest=0 WHERE model_id=?",
            (version.model_id,)
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (version.version_id, version.model_id, version.version,
             version.checkpoint_path, version.training_steps,
             int(version.is_latest), version.notes, version.created_at)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Version {BOLD}{version.version}{NC} added for model "
              f"{C}{version.model_id}{NC}")
        return version

    def benchmark_model(self, result: BenchmarkResult) -> BenchmarkResult:
        """Record a benchmark result for a model."""
        self._conn.execute(
            "INSERT OR REPLACE INTO benchmarks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (result.bench_id, result.model_id, result.version_id, result.task,
             result.score, result.latency_p50_ms, result.latency_p99_ms,
             result.throughput_rps, result.gpu_memory_gb,
             result.notes, result.created_at)
        )
        self._conn.commit()
        bar = "█" * int(result.score / 10) + "░" * (10 - int(result.score / 10))
        print(f"{G}✓{NC} Benchmark [{result.task}] score={Y}{result.score:.1f}{NC} "
              f"{C}{bar}{NC} p50={result.latency_p50_ms}ms")
        return result

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Retrieve a model by ID."""
        row = self._conn.execute(
            "SELECT * FROM models WHERE model_id=?", (model_id,)
        ).fetchone()
        if not row:
            return None
        return ModelEntry(
            model_id=row[0], name=row[1], provider=row[2],
            architecture=row[3], parameter_count=row[4],
            license=row[5], tags=json.loads(row[6] or "[]"), created_at=row[7]
        )

    def list_models(self, provider: Optional[str] = None) -> List[ModelEntry]:
        """List all registered models, optionally filtered by provider."""
        query = "SELECT * FROM models"
        params: tuple = ()
        if provider:
            query += " WHERE provider=?"
            params = (provider,)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            ModelEntry(model_id=r[0], name=r[1], provider=r[2],
                       architecture=r[3], parameter_count=r[4],
                       license=r[5], tags=json.loads(r[6] or "[]"),
                       created_at=r[7])
            for r in rows
        ]

    def get_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions for a model."""
        rows = self._conn.execute(
            "SELECT * FROM versions WHERE model_id=? ORDER BY created_at DESC",
            (model_id,)
        ).fetchall()
        return [
            ModelVersion(version_id=r[0], model_id=r[1], version=r[2],
                         checkpoint_path=r[3], training_steps=r[4],
                         is_latest=bool(r[5]), notes=r[6], created_at=r[7])
            for r in rows
        ]

    def get_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get aggregate metrics for a model."""
        ver_count = self._conn.execute(
            "SELECT COUNT(*) FROM versions WHERE model_id=?", (model_id,)
        ).fetchone()[0]
        bench = self._conn.execute(
            "SELECT COUNT(*), MAX(score), AVG(latency_p50_ms) FROM benchmarks WHERE model_id=?",
            (model_id,)
        ).fetchone()
        last = self._conn.execute(
            "SELECT MAX(created_at) FROM versions WHERE model_id=?", (model_id,)
        ).fetchone()[0]
        if not ver_count:
            return None
        return ModelMetrics(
            model_id=model_id, total_versions=ver_count,
            total_benchmarks=bench[0] or 0,
            best_score=round(bench[1] or 0.0, 2),
            avg_latency_ms=round(bench[2] or 0.0, 2),
            last_updated=last or "",
        )

    def close(self) -> None:
        self._conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_table(headers: List[str], rows: List[tuple]) -> None:
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "─" * (sum(widths) + len(widths) * 3 + 1)
    print(f"{B}{sep}{NC}")
    hdr = "│ " + " │ ".join(f"{C}{h:<{widths[i]}}{NC}" for i, h in enumerate(headers)) + " │"
    print(hdr)
    print(f"{B}{sep}{NC}")
    for row in rows:
        line = "│ " + " │ ".join(f"{str(row[i]):<{widths[i]}}" for i in range(len(headers))) + " │"
        print(line)
    print(f"{B}{sep}{NC}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-models", description="Lucidia AI Models — Universal model hub"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Register a model")
    reg.add_argument("--name", required=True)
    reg.add_argument("--provider", default="huggingface")
    reg.add_argument("--arch", default="transformer")
    reg.add_argument("--params", default="7B")
    reg.add_argument("--tags", nargs="*", default=[])

    lst = sub.add_parser("list", help="List models")
    lst.add_argument("--provider", default=None)

    ver = sub.add_parser("version", help="Add a model version")
    ver.add_argument("--model-id", required=True)
    ver.add_argument("--version", default="1.0.0")
    ver.add_argument("--steps", type=int, default=0)
    ver.add_argument("--path", default="")
    ver.add_argument("--notes", default="")

    bch = sub.add_parser("benchmark", help="Record a benchmark result")
    bch.add_argument("--model-id", required=True)
    bch.add_argument("--task", required=True)
    bch.add_argument("--score", type=float, required=True)
    bch.add_argument("--p50", type=float, default=0.0)
    bch.add_argument("--p99", type=float, default=0.0)
    bch.add_argument("--tps", type=float, default=0.0)

    met = sub.add_parser("metrics", help="Show model metrics")
    met.add_argument("--model-id", required=True)

    get = sub.add_parser("get", help="Get model details")
    get.add_argument("model_id")

    args = parser.parse_args()
    registry = AIModelRegistry()

    try:
        if args.cmd == "register":
            entry = ModelEntry(
                name=args.name, provider=args.provider,
                architecture=args.arch, parameter_count=args.params,
                tags=args.tags,
            )
            registry.register_model(entry)

        elif args.cmd == "list":
            models = registry.list_models(provider=args.provider)
            if not models:
                print(f"{Y}No models registered.{NC}")
                return
            _print_table(
                ["ID", "Name", "Provider", "Params", "Tags"],
                [(m.model_id, m.name, m.provider, m.parameter_count,
                  ",".join(m.tags)) for m in models]
            )

        elif args.cmd == "version":
            v = ModelVersion(
                model_id=args.model_id, version=args.version,
                training_steps=args.steps, checkpoint_path=args.path,
                notes=args.notes,
            )
            registry.add_version(v)

        elif args.cmd == "benchmark":
            br = BenchmarkResult(
                model_id=args.model_id, task=args.task, score=args.score,
                latency_p50_ms=args.p50, latency_p99_ms=args.p99,
                throughput_rps=args.tps,
            )
            registry.benchmark_model(br)

        elif args.cmd == "metrics":
            m = registry.get_metrics(args.model_id)
            if not m:
                print(f"{R}No data for model {args.model_id}{NC}")
                return
            print(f"\n{BOLD}{B}── Metrics: {args.model_id} ──────────{NC}")
            for k, v in asdict(m).items():
                print(f"  {C}{k:<22}{NC} {Y}{v}{NC}")

        elif args.cmd == "get":
            m = registry.get_model(args.model_id)
            if not m:
                print(f"{R}Model not found: {args.model_id}{NC}")
                return
            print(f"\n{BOLD}{B}── {m.name} ──────────────────────{NC}")
            for k, v in asdict(m).items():
                print(f"  {C}{k:<20}{NC} {v}")

    finally:
        registry.close()


if __name__ == "__main__":
    main()
