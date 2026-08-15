# Offline Decision Lab

Learning and evaluating decision policies purely from logged data, with no live
environment access.

This README is a placeholder. Per the specification it is written last, once
results exist, and it will then carry the headline results table, the validation
harness figures, the distribution-shift finding, the architecture diagram, and
the stated limitations.

Until then, the authoritative documents are:

- `offline-decision-lab-spec.md` — the full specification.
- `docs/decisions.md` — dated log of ambiguity calls and spec-versus-reality
  discrepancies.

## Setup

```
make setup      # venv + editable install with dev extras
make test       # pytest with coverage
make lint       # ruff
make typecheck  # mypy --strict on src/odl
```

## Build status

Phase 0 (steps 0–1: OBD availability, types, schema validation, scaffolding, CI)
and Phase 1 (steps 2–5: synthetic environment, uniform and epsilon-greedy
policies, IPS and SNIPS, bootstrap intervals, minimal validation harness) are
complete. The Phase 1 gate passes: under uniform logging IPS recovers the true
policy value with +0.41% relative bias and 97.5% CI coverage.

Phase 2 (reward model, Direct Method, doubly robust with cross-fitting, full
sweeps) is next. See `docs/build-plan.md`.
