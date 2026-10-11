# Last edited: 2026-10-11 12:00
"""Save and load benchmark result records as JSON."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchRecord:
    model: str  # Model path as a string
    context_tokens: int  # Number of tokens in the prompt
    batch_size: int  # Number of sequences ran together in a batch
    prefill_tps: float  # The tokens per second that were calculated for the prefill phase
    decode_tps: float  # tokens per second for the decode step
    # Similar to prefill_tps, but it is a duration rather than a rate,
    # and it also takes into account system overheads.
    time_to_first_token_s: float
    peak_memory_gb: float  # The peak amount of memory used during the run
    # Measure of how perplexed the model was by the real answers to the prompts
    # (the real next tokens in the prompt it was supposed to generate as an answer).
    # Lower is better because that means it was less surprised.
    perplexity: float | None = None


def save_records(records: list[BenchRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def load_records(path: Path) -> list[BenchRecord]:
    return [BenchRecord(**row) for row in json.loads(path.read_text())]
