"""Demonstrate basic train/dev/holdout isolation and dev-only selection.

Exact hashes and group IDs do not detect semantic paraphrase leakage; a real
ML pipeline needs stronger provenance and an independently held-out oracle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class Split(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    HOLDOUT = "HOLDOUT"


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Example:
    record_id: str
    split: Split
    group_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, Split):
            raise ValueError("Unrecognized dataset split")
        for name in ("record_id", "group_id", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


def _fingerprint(prompt: str) -> str:
    text = " ".join(prompt.casefold().split()).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


@dataclass(frozen=True)
class SplitReport:
    status: AuditStatus
    records: int
    duplicate_ids: int
    cross_split_groups: int
    cross_split_prompts: int
    present_splits: tuple[Split, ...]


def audit_splits(rows: list[Example]) -> SplitReport:
    """Audit exact/group overlap without exposing the original prompt text."""
    by_group: dict[str, set[Split]] = {}
    by_prompt: dict[str, set[Split]] = {}
    ids: set[str] = set()
    duplicate_ids = 0
    present: set[Split] = set()
    for row in rows:
        duplicate_ids += row.record_id in ids
        ids.add(row.record_id)
        present.add(row.split)
        by_group.setdefault(row.group_id, set()).add(row.split)
        by_prompt.setdefault(_fingerprint(row.prompt), set()).add(row.split)
    groups = sum(len(s) > 1 for s in by_group.values())
    prompts = sum(len(s) > 1 for s in by_prompt.values())
    if duplicate_ids or groups or prompts:
        status = AuditStatus.FAIL
    elif present != set(Split):
        status = AuditStatus.INCONCLUSIVE
    else:
        status = AuditStatus.PASS
    return SplitReport(status, len(rows), duplicate_ids, groups, prompts,
                       tuple(sorted(present, key=lambda s: s.value)))


@dataclass(frozen=True)
class DevCheckpoint:
    epoch: int
    dev_entity_exact: int
    dev_total: int
    dev_wer: float
    general_wer: float

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("Invalid epoch")
        if type(self.dev_total) is not int or self.dev_total < 1:
            raise ValueError("Invalid dev sample count")
        if type(self.dev_entity_exact) is not int or not 0 <= self.dev_entity_exact <= self.dev_total:
            raise ValueError("Invalid dev exact count")
        for name in ("dev_wer", "general_wer"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid {name}")


def select_checkpoint(rows: list[DevCheckpoint], *, max_general_wer: float) -> DevCheckpoint:
    """Choose from DEV and general-regression evidence, never HOLDOUT scores.

    A held-out set is evaluated once after the winner is frozen. These
    thresholds and scoring rules are illustrative, not NCMA training policy.
    """
    if type(max_general_wer) not in (int, float) or not 0 <= max_general_wer < float("inf"):
        raise ValueError("Invalid generalization threshold")
    eligible = [r for r in rows if r.general_wer <= max_general_wer]
    if not eligible:
        raise ValueError("No eligible checkpoint; do not select by training loss")
    if len({r.epoch for r in rows}) != len(rows):
        raise ValueError("Duplicate epoch numbers")
    return max(eligible, key=lambda r: (r.dev_entity_exact / r.dev_total,
                                       -r.dev_wer, -r.general_wer, -r.epoch))
