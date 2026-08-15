# Decision log

Dated entries recording ambiguity calls and places where reality contradicted
the specification. Per spec section 16, decisions are recorded rather than
silently made, and where reality disagrees with the spec, reality wins.

---

## 2026-08-15 — OBD endpoint verified before any other work

Spec section 13 Step 0. A HEAD request to
`https://research.zozo.com/data_release/open_bandit_dataset.zip` returned
HTTP 200, `Content-Type: application/zip`, `Content-Length: 412931917`
(~394 MiB). The project's only external dependency is live and needs no
registration. No redesign required; the MIND/MSLR fallback stays unused.

The schema claims in spec section 6.4 have not yet been re-verified against the
downloaded files — that happens when the loader is built (step 9), and any
discrepancy gets its own entry here.

## 2026-08-15 — Python 3.12 instead of 3.11

The spec asks for Python 3.11; the available interpreter is 3.12.10. Nothing in
the dependency set requires 3.11, so `requires-python` is `>=3.11,<3.13` and
development proceeds on 3.12. The upper bound is deliberate: `obp` and `torch`
wheel availability is the practical constraint, not language features.

## 2026-08-15 — `uv` unavailable, using `venv` + `pip`

`uv` is not installed on this machine. The spec permits `venv` + `pip` as the
fallback. `make setup` uses the standard library `venv` and an editable install.
Dependencies are pinned by lower bound in `pyproject.toml` with upper bounds
only where a known incompatibility exists.

## 2026-08-15 — `obp` isolated to an optional extra

`obp` is a cross-check dependency only (spec section 7.4) and pins older
`numpy`/`scikit-learn` ranges that would drag the whole project backwards. It
lives in the `crosscheck` extra and is installed into a separate environment for
the agreement test. This keeps the from-scratch implementations — which are the
point — on current numerics.

## 2026-08-15 — Validation raises rather than asserts

`src/odl/types.py` uses an explicit `_require` helper that raises `ValueError`
instead of bare `assert`. Assertions vanish under `python -O`, and the ground
truth leakage guard (spec section 0.4) must be structural, not conditional on
interpreter flags.

## 2026-08-15 — Ground-truth access funnelled through two accessors

`BanditFeedback.expected_reward` is never read directly outside
`types.py`. All access goes through `require_ground_truth(caller)` or is
forbidden by `require_no_ground_truth(caller)`. This makes leakage a crash with
an attributable caller name rather than a silently inflated result, satisfying
the spec's demand that the guard be structural rather than by convention.

## 2026-08-15 — `Trajectory.done` stored, not inferred

The spec's `Trajectory` carries a single `done` flag. It is kept as stored state
rather than derived from the state array, because FQI's bootstrapping target
distinguishes true termination from horizon truncation, and inferring it would
quietly bias the value of long episodes.
