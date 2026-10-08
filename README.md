# NCMA Validation Patterns

**Practical Python examples for safer AI tooling, API integrations and evidence-led delivery.**

This repository contains three **standalone, reduced adaptations** of engineering lessons from the private NCMA Systems lab. These are **not copies of operational modules** or a production framework.

| Component | Problem | Demonstration |
| --- | --- | --- |
| [`errors.py`](validation_patterns/errors.py) | Upstream exceptions can expose secrets | Public error messages built from a strict allowlist |
| [`differential.py`](validation_patterns/differential.py) | New validators can silently lose refusal coverage | Asymmetric migration testing; uncertainty is not PASS |
| [`quality.py`](validation_patterns/quality.py) | Green dashboards hide false positives and missed defects | Recall, precision, FPR and evidence coverage |

## Run it

Requires **Python 3.11+**. The examples have **no third-party runtime dependencies**.

```bash
git clone https://github.com/Ozzurac/ncma-validation-patterns.git
cd ncma-validation-patterns
python -m unittest discover -s tests -v
python -m examples.demo
```

## Example: no secret-bearing error projection

```python
from validation_patterns.errors import parse_known_error

response = parse_known_error(409, {
    "error": {
        "code": "STATE_CONFLICT",
        "detail": "A private stack trace that must never reach users",
        "retryable": True,
    }
})
assert response is not None
print(response.to_dict())
# Only fixed safe fields are emitted.
```

Unrecognized errors return `None`, which is *not* a signal of success.

## Example: a regression is still a regression

```python
from validation_patterns.differential import compare_cases

old = {"case1": "REFUSE", "case2": "ACCEPT"}
new = {"case1": "ACCEPT", "case2": "REFUSE"}

report = compare_cases(old, new)
assert report.gate.value == "FAIL"
```

A newly refused case is not automatically a win. It could be a false positive; it needs external confirmation.

## Example: a pass needs both clean and broken samples

```python
from validation_patterns.quality import Counts, assess_quality

# Purely SYNTHETIC counts, not NCMA benchmark results:
quality = assess_quality(Counts(19, 1, 0, 20), min_recall=0.90)
assert quality.gate.value == "PASS"
```

Zero false alarms without any defect tests cannot qualify a validator.

## Engineering boundaries

- [Design decisions](DESIGN_NOTES.md)
- [Publication/sanitization policy](PUBLICATION_SCOPE.md)
- [Tests](tests/)
- [GitHub Actions](.github/workflows/tests.yml)

**All examples, case IDs and quality counts are synthetic.** The repository has no model, real gateway endpoints, customer data, secrets, file-changing adapter, game resources, private corpus or proprietary runtime code. It does not claim to qualify an AI model or product.

**Related:** [NCMA Engineering](https://github.com/Ozzurac/ncma-engineering) · [Agent Governance Lab](https://github.com/Ozzurac/agent-governance-lab) · [Professional portfolio](https://ncmasystems.com)

**Author:** [Valter Lourenço Junior](https://github.com/Ozzurac)

**License:** MIT (original standalone examples only).