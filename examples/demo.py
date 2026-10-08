"""Demonstrate all components with fictional, synthetic input data."""

from __future__ import annotations

from pprint import pprint

from validation_patterns.differential import compare_cases
from validation_patterns.errors import parse_known_error
from validation_patterns.quality import Counts, assess_quality


def main() -> None:
    print("EXAMPLE 1 | Public error projection (fictional payload)")
    payload = {
        "error": {
            "code": "STATE_CONFLICT",
            "detail": "private upstream stack trace must never be echoed",
            "retryable": True,
        }
    }
    public = parse_known_error(409, payload)
    assert public is not None
    pprint(public.to_dict())
    assert "stack trace" not in str(public.to_dict())

    print("\nEXAMPLE 2 | Migration regression report (fictional cases)")
    old = {"already_bad": "REFUSE", "new_case": "ACCEPT"}
    new = {"already_bad": "ACCEPT", "new_case": "REFUSE"}
    comparison = compare_cases(old, new)
    pprint(comparison.to_dict())
    assert comparison.gate.value == "FAIL"

    print("\nEXAMPLE 3 | Sample quality evidence (SYNTHETIC counts)")
    quality = assess_quality(Counts(19, 1, 0, 20), min_recall=0.90)
    pprint(quality.to_dict())
    assert quality.gate.value == "PASS"


if __name__ == "__main__":
    main()