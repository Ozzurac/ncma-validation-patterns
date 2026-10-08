"""Pure-Python illustration of a bounded speech-recognition evaluation.

This is independent example code, NOT the private STT lab runner. Differences
from a real experiment's transcript normalizer must be recorded explicitly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from statistics import median


def normalize_transcript(value: str) -> str:
    """NFKC + casefold + punctuation-to-spaces + whitespace collapse.

    Accents are intentionally preserved. Tokenization is whitespace-based.
    Other policies (numbers, contractions, accents) will change WER.
    """
    if not isinstance(value, str):
        raise TypeError("Transcript must be a string")
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(value.split())


def _edits(reference: list[str], hypothesis: list[str]) -> int:
    """Minimum insertions, deletions and substitutions (Levenshtein)."""
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, start=1):
        current = [i]
        for j, b in enumerate(hypothesis, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _p95(values: list[float]) -> float:
    """Nearest-rank p95; defined only for a nonempty sample."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


@dataclass(frozen=True)
class SpeechSample:
    clip_id: str
    reference: str
    hypothesis: str
    audio_seconds: float
    inference_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise ValueError("clip_id must be a nonempty string")
        if not isinstance(self.reference, str) or not normalize_transcript(self.reference):
            raise ValueError("A nonempty reference transcript is required")
        if not isinstance(self.hypothesis, str):
            raise ValueError("hypothesis must be a string")
        for name in ("audio_seconds", "inference_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ASRReport:
    clips: int
    word_errors: int
    reference_words: int
    char_errors: int
    reference_chars: int
    exact_clips: int
    wer: float
    cer: float
    audio_seconds: float
    inference_seconds: float
    realtime_factor: float
    median_inference_seconds: float
    p95_inference_seconds: float
    warm_median_seconds: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_asr(samples: list[SpeechSample]) -> ASRReport:
    """Score fixed-order clips without using ground truth to tune a model.

    WER and CER are *micro-averaged* across the corpus (total edits divided by
    total reference tokens). The warm median excludes the FIRST measured clip.
    That single-clip exclusion is a diagnostic, not a cold-boot SLA.
    """
    if not samples:
        raise ValueError("No evaluation samples; zero examples is not a pass")
    ids = [row.clip_id for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated clip IDs invalidate corpus accounting")

    word_errors = reference_words = char_errors = reference_chars = exact = 0
    for row in samples:
        ref = normalize_transcript(row.reference)
        hyp = normalize_transcript(row.hypothesis)
        ref_words, hyp_words = ref.split(), hyp.split()
        word_errors += _edits(ref_words, hyp_words)
        reference_words += len(ref_words)
        # Character error rate excludes whitespace in this illustrative policy.
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        char_errors += _edits(ref_chars, hyp_chars)
        reference_chars += len(ref_chars)
        exact += ref == hyp

    timings = [row.inference_seconds for row in samples]
    audio_total = sum(row.audio_seconds for row in samples)
    inference_total = sum(timings)
    return ASRReport(
        clips=len(samples),
        word_errors=word_errors,
        reference_words=reference_words,
        char_errors=char_errors,
        reference_chars=reference_chars,
        exact_clips=exact,
        wer=word_errors / reference_words,
        cer=char_errors / reference_chars,
        audio_seconds=audio_total,
        inference_seconds=inference_total,
        realtime_factor=inference_total / audio_total,
        median_inference_seconds=median(timings),
        p95_inference_seconds=_p95(timings),
        warm_median_seconds=median(timings[1:]) if len(timings) > 1 else None,
    )
