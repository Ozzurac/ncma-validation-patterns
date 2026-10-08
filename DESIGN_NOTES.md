# Design decisions

## Public error transport

The demo recognizes four fictional application codes and responds with fixed user-facing text. It never mirrors arbitrary upstream `detail`, headers, request IDs, paths, tracebacks or raw object fields. Unknown shapes remain unclassified.

**Not included:** actual service authentication, transport adapters, the private gateway schema, operational routing or audit storage.

## Asymmetric migration checks

An earlier validator's known refusals define a *minimum regression floor*, not objective ground truth:

- Old refuses, new accepts: **FAIL** (refusal coverage lost).
- Old accepts, new refuses: **REVIEW** (possible new detection or false positive).
- Both agree: **PASS**, but only for the bounded comparison.
- Missing data, invalid states or execution errors: **INCONCLUSIVE**.

The demo refuses to call an empty evaluation a pass.

## Detection quality

A useful validator must measure both **known-bad samples** (recall) and **known-good samples** (false-positive rate). Unverifiable cases remain visible as `unknown`.

A test corpus with no broken inputs might report zero false positives while proving nothing about defect detection. A corpus with no clean inputs says nothing about false alarms. Neither can produce a passing quality gate.

Metrics in the demonstration are computed from synthetic counts. A bounded sample-level PASS is never proof of production readiness, external correctness or statistical generalization.

## Why these components

They show interview-relevant Python engineering behaviors: strict types, explicit failure states, deterministic order, stable contracts, small functions, negative testing and documented limitations. Private system controllers, domain engines and infrastructure were deliberately excluded.