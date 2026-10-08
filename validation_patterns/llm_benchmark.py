"""Small offline LLM inference accounting example, without model adapters.

Do not mistake token throughput for model quality or schema-enforcement proof.
Cross-backend first-token timing may not be comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class InferenceSample:
    case_id: str
    prompt_tokens: int
    generated_tokens: int
    first_token_seconds: float
    total_seconds: float
    finish_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id is required")
        for name in ("prompt_tokens", "generated_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.generated_tokens:
            raise ValueError("At least one generated token is required")
        for name in ("first_token_seconds", "total_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.first_token_seconds >= self.total_seconds:
            raise ValueError("Total latency must exceed first-token latency")
        if self.finish_reason not in ("stop", "length", "error"):
            raise ValueError("Unknown finish reasons cannot be recorded as completed")

    @property
    def decode_tokens_per_second(self) -> float | None:
        # The first emitted token belongs to the prefill/first-token boundary.
        return ((self.generated_tokens - 1) / (self.total_seconds - self.first_token_seconds)
                if self.generated_tokens > 1 else None)


@dataclass(frozen=True)
class LLMReport:
    cases: int
    completed: int
    truncated: int
    errors: int
    completion_rate: float
    median_total_seconds: float | None
    median_first_token_seconds: float | None
    median_decode_tokens_per_second: float | None
    throughput_samples: int

    def as_dict(self) -> dict[str, int | float | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_llm(samples: list[InferenceSample]) -> LLMReport:
    """Keep truncation and runtime error visible; calculate only observed speeds.

    Medians are computed on *successfully completed* cases only, so their
    selection bias must be reported next to completion_rate.
    """
    if not samples:
        raise ValueError("No inference observations; do not report a model winner")
    ids = [row.case_id for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs would distort the completion rate")

    complete = [row for row in samples if row.finish_reason == "stop"]
    speeds = [row.decode_tokens_per_second for row in complete]
    speeds = [value for value in speeds if value is not None]
    return LLMReport(
        cases=len(samples),
        completed=len(complete),
        truncated=sum(row.finish_reason == "length" for row in samples),
        errors=sum(row.finish_reason == "error" for row in samples),
        completion_rate=len(complete) / len(samples),
        median_total_seconds=median([row.total_seconds for row in complete]) if complete else None,
        median_first_token_seconds=(median([row.first_token_seconds for row in complete])
                                    if complete else None),
        median_decode_tokens_per_second=median(speeds) if speeds else None,
        throughput_samples=len(speeds),
    )
