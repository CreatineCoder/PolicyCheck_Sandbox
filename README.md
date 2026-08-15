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

Step 0 (OBD availability) and step 1 (types, schema validation, scaffolding, CI)
of the build order are complete. Everything after that is unbuilt.
