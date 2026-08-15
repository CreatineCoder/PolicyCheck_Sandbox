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

## 2026-08-15 — Bootstrap resamples records, not per-sample values

The spec describes the bootstrap as resampling the per-sample values. For
mean-type estimators such as IPS that is exactly equivalent to resampling
records. For ratio-type estimators such as SNIPS it is not: every per-record
value shares a denominator computed from the whole sample, so averaging a
resampled subset of them is not a valid replicate and would understate the
interval.

`ope/bootstrap.py` therefore resamples *indices* and recomputes the estimator,
which reduces to the spec's description for IPS and stays correct for SNIPS and
for doubly-robust later. `SNIPSEstimator.per_sample_values` is still provided
for diagnostics, with a docstring stating that it must not be averaged over a
resample.

## 2026-08-15 — Nonlinearity uses `sin(3 z)`, not `sin(z)`

The spec suggests `nonlinear(x, a) = sin(w_a · x)`. With contexts drawn from
`N(0, I)` and weights scaled so `w_a · x` is O(1), `sin(z) ≈ z` over the bulk of
the distribution, which makes the nonlinear term nearly collinear with the
linear one. A linear reward model would then absorb most of it, and the
misspecification sweep would show far less DM-versus-DR separation than `beta`
implies — which section 15 flags as a symptom of the nonlinearity being too
weak.

The frequency is therefore exposed as `nonlinear_frequency`, defaulting to 3.0.
The final value will be set by whatever visibly separates DM from DR in the
Phase 2 sweep; this default is a starting point, not a tuned result.

## 2026-08-15 — Environment weights scaled by `1/sqrt(d)`

Without scaling, `w_a · x` grows with the context dimension, the sigmoid
saturates, and every expected reward collapses toward 0 or 1 — a trivial
problem with no estimation difficulty left in it. Scaling by `1/sqrt(d)` keeps
the logit O(1) as `d` grows. A test asserts the mean expected reward stays
inside `(0.2, 0.8)` with non-trivial spread at `d = 50`, so the property is
enforced rather than assumed.

## 2026-08-15 — The oracle logging policy lives in `envs/`, not `policies/`

`OracleEpsilonPolicy` is greedy on the environment's true `mu`, so it reads
ground truth. Placing it in `envs/synthetic_bandit.py` rather than the general
`policies/` package makes it structurally impossible to reach for when working
with real logs, and keeps every class in `policies/` free of ground-truth
access. This is the same containment principle as the accessors on
`BanditFeedback`.

## 2026-08-15 — Ridge reward model raises on an unobserved action

`RidgeRewardModel.fit` raises when any action has zero logged samples rather
than falling back to the ridge prior of zero. The prior is a well-defined
number, which is precisely the danger: `r_hat(x, a) = 0` for an unlogged action
is a plausible-looking value that silently biases every downstream estimate.
This will matter again in Part 2, where the logging policy is deliberately
narrow; the intended fix there is a conservative method, not a silent default.

## 2026-08-15 — Phase 1 reward model is closed-form ridge, not XGBoost

The spec specifies XGBoost with cross-fitting for the Direct Method and doubly
robust estimators. That still holds for Phase 2. The epsilon-greedy *policy* in
Phase 1, however, uses a per-action closed-form ridge regression, because it is
exactly deterministic given the data and adds no second source of randomness to
control while the determinism guarantees are being established. The estimator
reward models in Phase 2 are a separate component and will use XGBoost as
specified.

## 2026-08-15 — Harness seeding via `SeedSequence.spawn`

`run_validation` derives every random draw from a single root seed through
`np.random.SeedSequence.spawn`: one substream for the ground-truth integral, and
per replication one for data generation and one for the bootstrap. Two
consequences are deliberate. Replications are independent rather than correlated
through a shared generator, and changing the number of bootstrap replicates `B`
does not perturb the generated logs, so results across different `B` settings
remain directly comparable.

## 2026-08-15 — `Trajectory.done` stored, not inferred

The spec's `Trajectory` carries a single `done` flag. It is kept as stored state
rather than derived from the state array, because FQI's bootstrapping target
distinguishes true termination from horizon truncation, and inferring it would
quietly bias the value of long episodes.
