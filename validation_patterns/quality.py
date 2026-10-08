"""Detection-quality metrics: failures and missing evidence remain visible.

A tiny educational metric model, not a published ModForge reliability result.
All examples use synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QualityGate(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class Counts:
    """Confusion counts for a validator tested against an independent oracle.

    detected_defect: known-bad input refused (TP)
    missed_defect: known-bad input accepted (FN)
    false_alarm: known-good input refused (FP)
    accepted_clean: known-good input accepted (TN)
    unknown: inputs with no independently established ground truth
    """

    detected_defect: int
    missed_defect: int
    false_alarm: int
    accepted_clean: int
    unknown: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (
            self.detected_defect,
            self.missed_defect,
            self.false_alarm,
            self.accepted_clean,
            self.unknown,
        )):
            raise ValueError("All counts must be non-negative integers")

    @property
    def known(self) -> int:
        return (self.detected_defect + self.missed_defect
                + self.false_alarm + self.accepted_clean)

    @property
    def total(self) -> int:
        return self.known + self.unknown


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class QualityReport:
    counts: Counts
    recall: float | None
    precision: float | None
    false_positive_rate: float | None
    evidence_coverage: float | None
    gate: QualityGate

    def to_dict(self) -> dict[str, object]:
        return {
            "known_cases": self.counts.known,
            "unknown_cases": self.counts.unknown,
            "recall": self.recall,
            "precision": self.precision,
            "false_positive_rate": self.false_positive_rate,
            "evidence_coverage": self.evidence_coverage,
            "gate": self.gate.value,
        }


def assess_quality(
    counts: Counts,
    *,
    min_known_cases: int = 10,
    min_recall: float = 0.95,
    max_false_positive_rate: float = 0.05,
) -> QualityReport:
    """Assess a bounded validation sample without asserting universal accuracy.

    A sample without both known-good and known-bad inputs cannot pass.
    Unknown ground truth and undersized samples remain INCONCLUSIVE.
    """
    if type(min_known_cases) is not int or min_known_cases < 1:
        raise ValueError("min_known_cases must be positive")
    if not (0.0 <= min_recall <= 1.0 and 0.0 <= max_false_positive_rate <= 1.0):
        raise ValueError("Thresholds must be in [0, 1]")

    recall = _divide(
        counts.detected_defect, counts.detected_defect + counts.missed_defect
    )
    precision = _divide(
        counts.detected_defect, counts.detected_defect + counts.false_alarm
    )
    fpr = _divide(
        counts.false_alarm, counts.false_alarm + counts.accepted_clean
    )
    coverage = _divide(counts.known, counts.total)

    if (recall is not None and recall < min_recall) or (
        fpr is not None and fpr > max_false_positive_rate
    ):
        gate = QualityGate.FAIL
    elif (
        counts.known < min_known_cases
        or counts.unknown > 0
        or recall is None
        or fpr is None
    ):
        gate = QualityGate.INCONCLUSIVE
    else:
        gate = QualityGate.PASS

    return QualityReport(counts, recall, precision, fpr, coverage, gate)