# Last edited: 2026-10-11 12:00
"""Save and load benchmark result records as JSON."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchRecord:
    model: str
    context_tokens: int
    batch_size: int
    prefill_tps: float
    decode_tps: float
    time_to_first_token_s: float
    peak_memory_gb: float
    perplexity: float | None = None


def save_records(records: list[BenchRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def load_records(path: Path) -> list[BenchRecord]:
    return [BenchRecord(**row) for row in json.loads(path.read_text())]
