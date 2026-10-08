"""Asymmetric regression comparison with explicit inconclusive states.

Pure, standalone adaptation of a testing principle used in ModForge V2.
No private validator, oracle implementation or domain-specific rules are here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    ACCEPT = "ACCEPT"
    REFUSE = "REFUSE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class Change(str, Enum):
    BOTH_ACCEPT = "BOTH_ACCEPT"
    BOTH_REFUSE = "BOTH_REFUSE"
    MISSED_REFUSAL = "MISSED_REFUSAL"
    NEW_REFUSAL = "NEW_REFUSAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class Gate(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    before: State | None
    after: State | None
    change: Change


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    cases: tuple[CaseResult, ...]

    def count(self, change: Change) -> int:
        return sum(item.change is change for item in self.cases)

    @property
    def gate(self) -> Gate:
        # Failures must not be concealed by unrelated UNKNOWNs.
        if self.count(Change.MISSED_REFUSAL):
            return Gate.FAIL
        if not self.cases or self.count(Change.INCONCLUSIVE):
            return Gate.INCONCLUSIVE
        # More refusals are not automatically better: could be false positives.
        if self.count(Change.NEW_REFUSAL):
            return Gate.REVIEW
        return Gate.PASS

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate.value,
            "cases": len(self.cases),
            "unchanged_accepted": self.count(Change.BOTH_ACCEPT),
            "unchanged_refused": self.count(Change.BOTH_REFUSE),
            "missed_refusal": self.count(Change.MISSED_REFUSAL),
            "new_refusal_needs_review": self.count(Change.NEW_REFUSAL),
            "inconclusive": self.count(Change.INCONCLUSIVE),
        }


def _parse_state(value: object) -> State | None:
    if isinstance(value, State):
        return value
    if isinstance(value, str):
        try:
            return State(value)
        except ValueError:
            return None
    return None


def _compare(before: State | None, after: State | None) -> Change:
    if before is None or after is None:
        return Change.INCONCLUSIVE
    if before in (State.UNKNOWN, State.ERROR) or after in (State.UNKNOWN, State.ERROR):
        return Change.INCONCLUSIVE
    if before is State.REFUSE and after is State.ACCEPT:
        return Change.MISSED_REFUSAL
    if before is State.ACCEPT and after is State.REFUSE:
        return Change.NEW_REFUSAL
    if before is State.REFUSE and after is State.REFUSE:
        return Change.BOTH_REFUSE
    return Change.BOTH_ACCEPT


def compare_cases(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> ComparisonReport:
    """Compare matching synthetic case IDs, treating missing data as unknown.

    The legacy reference is a regression floor only, not ground truth about
    the new validator. A newly refused input requires separate confirmation.
    """
    ids = set(reference) | set(candidate)
    if any(not isinstance(key, str) or not key.strip() for key in ids):
        raise ValueError("Case IDs must be nonblank strings")

    rows = []
    for case_id in sorted(ids):
        before = _parse_state(reference.get(case_id))
        after = _parse_state(candidate.get(case_id))
        rows.append(CaseResult(case_id, before, after, _compare(before, after)))
    return ComparisonReport(tuple(rows))