# Phase-wise build plan

Derived from spec section 13. The spec's seventeen steps are grouped into eight
phases, each with an explicit **exit gate**: a condition that must hold before
the next phase starts. The gates are not ceremony — several of them exist
because proceeding past a broken component silently corrupts everything
downstream, and the failure surfaces much later as a wrong headline number.

Phase ordering is fixed. The one place where reordering is tempting — running
estimators on real data before the synthetic harness passes — is precisely the
failure mode the spec forbids (section 0.4).

Time estimates are in focused working sessions of roughly three hours, and
assume no prior code exists for the component in question.

---

## Phase 0 — Foundations *(complete)*

**Spec steps:** 0, 1
**Effort:** 1 session
**Status:** done, committed.

| Deliverable | State |
|---|---|
| OBD endpoint verified (HTTP 200, 413 MB zip) | done |
| `src/odl/types.py` — `BanditFeedback`, `Trajectory` | done |
| `src/odl/data/schema.py` — loud validation | done |
| Repo layout, `pyproject.toml`, `Makefile`, four configs, CI | done |
| `scripts/download_obd.sh` | done, sample path untested until Phase 3 |

**Gate (met):** 35 tests pass, `ruff` clean, `mypy --strict` clean, ground-truth
guards covered by tests in both directions.

---

## Phase 1 — The estimation core, and the gate that protects it

**Spec steps:** 2, 3, 4, 5
**Effort:** 2–3 sessions
**Why grouped:** these four steps are the minimum viable loop — an environment
that knows the truth, a policy to evaluate, an estimator, and a check that the
estimator recovers the truth. Nothing downstream is meaningful until this loop
closes.

**Build**

1. `envs/synthetic_bandit.py` — contexts `x ~ N(0, I_10)`, per-action `w_a`,
   `b_a` from a seeded RNG, `mu(x, a) = sigmoid(w_a·x + b_a + beta·sin(w_a·x))`,
   Bernoulli rewards. Exact `true_policy_value(policy)` by averaging
   `sum_a pi(a|x) mu(x, a)` over a large fresh context sample.
   Configurable logging policy with an `epsilon` overlap knob.
2. `policies/base.py` (`Policy` ABC), `policies/uniform.py`,
   `policies/epsilon_greedy.py`. `action_dist` returns a full distribution, not
   an argmax — OPE needs `pi(a|x)` for every logged action.
3. `ope/base.py` (`Estimator` ABC with `estimate` and `per_sample_values`),
   `ope/ips.py` — IPS, SNIPS, optional clipping at `tau`, ESS reported
   alongside every IPS estimate.
4. `ope/bootstrap.py` — B = 1000 percentile CIs over per-sample values.
5. `eval/validation_harness.py`, minimal version: generate logs, compute
   `V_true`, compute each `V_hat` with CI.

**Tests**

- Every estimator against a hand-computed toy example with a closed-form answer.
- Property (`hypothesis`): with `pi_e == pi_b`, IPS recovers the empirical mean
  reward to floating-point tolerance; uniform target under uniform logging
  equals the mean reward.
- `action_dist` rows sum to 1 within tolerance, for every policy.
- Determinism: same seed, byte-identical output across two runs.

**Exit gate — the hard one.** Under uniform logging, IPS recovers the true
policy value within Monte Carlo error, and its bootstrap CI covers truth at
roughly the nominal rate. **Do not proceed until this passes.** If IPS cannot
recover truth in the easiest possible regime, the harness is wrong, and every
number produced after this point inherits that error.

**Risk:** a `true_policy_value` that is subtly biased (too few context samples,
or reusing the logged contexts instead of fresh ones) looks like an estimator
bug and can burn a session. Compute it on fresh contexts and check its own
Monte Carlo error first.

---

## Phase 2 — The full estimator suite and E01

**Spec steps:** 6, 7
**Effort:** 2–3 sessions

**Build**

1. Reward model `r_hat(x, a)` — XGBoost, wrapped so cross-fitting is the only
   way to call it on data used for fitting.
2. `ope/direct.py` — Direct Method.
3. `ope/doubly_robust.py` — DR with **K-fold cross-fitting**. Without it the
   reward model overfits the same data used for correction and DR loses its
   guarantee. This is the detail a technical reviewer checks first.
4. Full harness: relative bias, RMSE over R = 100 log replications, and **CI
   coverage** — the fraction of replications whose 95% interval contained
   `V_true`.
5. Three sweeps → `experiments/e01_ope_validation.py` → table + figures in
   `reports/`.

**Sweeps**

| Sweep | Varies | Expected finding |
|---|---|---|
| Overlap | logging `epsilon` from 1.0 to 0.05 | IPS variance blows up, ESS collapses, coverage falls well below 95%; DR holds longer |
| Misspecification | `beta` from 0 to 4 | DM bias grows; DR stays near truth — the doubly-robust property made visible |
| Sample size | n from 1e3 to 5e5, log scale | convergence rates |

**Tests**

- A test verifies the reward model never predicts on data it was fitted to.
  Cross-fitting that is claimed but not applied is the single most likely
  silent bug in this project.
- Statistical test at reduced scale: DR relative bias below threshold under
  good overlap.

**Exit gate:** under good overlap and correct specification, DR coverage is
within a few points of 95%. If it is not, the implementation is wrong — debug
rather than reporting the number. Separately, the misspecification sweep must
*visibly* separate DM from DR; if it does not, `beta` is too small, so increase
the nonlinearity until the effect is real.

**Note:** the figures produced here are the primary interview artifact. Budget
time for them to be legible, not merely correct.

---

## Phase 3 — Real data, and the strongest result in Part 1

**Spec steps:** 8, 9, 10, 11
**Effort:** 3–4 sessions

**Build**

1. `policies/linucb.py` — per-arm `A_a`, `b_a` with Sherman-Morrison rank-1
   inverse updates. LinUCB is deterministic, so convert to a distribution by
   softmax over scores or an epsilon-mix; **document which and why** — a
   deterministic target makes IPS variance explode and handling that is part of
   the story.
2. `policies/thompson.py` — Bayesian linear regression per arm; `action_dist`
   by K = 1000 Monte Carlo argmax draws, cached per context batch.
3. `data/obd_loader.py` — sample and full paths. Re-verify the section 6.4
   schema claims against the actual files and record any discrepancy in
   `docs/decisions.md`; reality wins over the spec. One-hot the hashed
   `user_feature_*` categoricals. Filter to a single position (option (a)) and
   state the two-thirds data cost in the README.
4. **E02** — LinUCB and Thompson policy value on OBD with bootstrap CIs and ESS.
5. **E02b — the cross-policy validation.** Compute the true BTS value as the
   on-policy mean click rate in the `bts` logs; independently estimate it from
   the `random` logs with BTS as target; compare. Then the reverse, harder
   direction. Report per-estimator absolute and relative error plus ESS.
6. `obp` cross-check test — same inputs, agreement within tolerance, proving the
   from-scratch implementations are correct rather than merely self-consistent.

**Operational facts that shape this phase**

- CTR is roughly 0.4%. The 10k sample contains about 38 clicks. It is far too
  small for any reported number — sample for tests and CI, full 26M dataset for
  every result that appears in the README.
- `random` propensity is a constant 0.0125 = 1/80, giving perfect overlap. Start
  here.
- `bts` has minimum propensity 4.5e-05, implying importance weights near 278
  against a uniform target. Run this split specifically to demonstrate IPS
  instability on real data. Clipping and ESS reporting are mandatory here.

**Exit gate:** E02b produces a measured error against a real ground-truth
number, in both directions, and the `obp` cross-check passes.

**Checkpoint.** Part 1 is now a complete, standalone, presentable project.
If time runs short, stop here and polish rather than leaving Part 2 half-built.
Skip to Phase 7 in that case.

---

## Phase 4 — Sequential environment and the distribution-shift finding

**Spec steps:** 12, 13
**Effort:** 2–3 sessions

**Build**

1. `envs/session_mdp.py` — interest vector (d = 8), step index, fatigue scalar;
   10 content categories each with an `(engagement, fatigue_cost)` profile,
   including at least one clickbait action; interest drifts toward the shown
   category, fatigue accumulates, early termination probability rises with
   fatigue; T = 20. `true_policy_value(policy, gamma, n_rollouts)` by Monte
   Carlo.
2. **Verify before building anything on top of it:** compute the optimal greedy
   and optimal discounted policies by value iteration on a discretized version
   and assert their values differ meaningfully. Log that gap as a number.
   If the gap is small the whole of Part 2 proves nothing, because a bandit
   would solve the environment.
3. Narrow logging policy — a mediocre heuristic touching only 4 of 10 actions,
   with per-step propensities saved. This is the realistic production case.
4. `offline_rl/fqi.py` — standard batch Fitted Q-Iteration, XGBoost regressor,
   config-switchable between per-action models and one model over
   `(s, one_hot(a))`. Track and plot the Bellman residual per iteration.
5. **E03** — train FQI on the narrow logs, then compare `mean(max_a Q(s,a))` —
   what the model believes it will achieve — against the true value of the
   induced greedy policy.

**Expected finding:** FQI substantially overestimates, and the overestimation
concentrates on actions the logging policy rarely took, because there is no data
to correct the extrapolation. Deliver a bar chart of estimated vs true Q grouped
by logged action frequency, and a scatter of `Q_est − Q_true` against that
frequency.

**Exit gate:** the myopic-vs-far-sighted gap is logged as a number, and the FQI
overestimation is quantified and correlated against logged action frequency.

---

## Phase 5 — Conservative Q-Learning and honest sequential OPE

**Spec steps:** 14, 15
**Effort:** 3–4 sessions
**This is the longest-running phase in wall-clock terms** — the alpha sweep is
5 values by 5 seeds, or 25 training runs.

**Build**

1. `offline_rl/cql.py` — PyTorch, discrete actions. MLP Q-network, two hidden
   layers of 256 with ReLU, target network with Polyak averaging.
   `L_CQL = alpha · E_s[logsumexp_a Q(s,a) − E_{a~data}[Q(s,a)]] + L_Bellman`.
2. Alpha sweep over `[0.0, 0.1, 1.0, 5.0, 10.0]`. `alpha = 0` reduces to plain
   offline DQN, which gives the ablation for free.
3. **E04** — true policy value for FQI vs CQL across alpha, reported as
   mean ± std over 5 fixed seeds. RL is unstable; single-seed results are not
   results.
4. `offline_rl/sequential_ope.py` — per-decision importance sampling and its
   weighted variant. Report ESS per horizon length and show it collapsing;
   report PDIS distance from the environment's true value at T = 5, 10, 20.

**On the PDIS result:** weights compound multiplicatively over the horizon, so
variance grows exponentially with T. That is a genuine limitation, and it goes
in the report with numbers attached rather than being tuned away. Naming the
limits of one's own method is the point of the exercise.

**Exit gate:** CQL sweep complete across 5 seeds with variance reported, and
PDIS variance collapse measured at all three horizons.

---

## Phase 6 — The serving loop

**Spec step:** 16
**Effort:** 1–2 sessions

**Build**

- `service/app.py` — `POST /recommend` returning
  `{action, propensity, decision_id, policy_version}`; `POST /reward` joining
  delayed feedback; `GET /healthz`, `GET /metrics`.
- `service/logging_middleware.py` — append-only Parquet sink under
  `data/serving_logs/`, written in the exact `BanditFeedback` schema.
- Policies loaded from a versioned artifact directory. No retraining in the
  request path.
- Dockerfile, `make serve`.

**The one idea this phase exists to demonstrate:** the response and the log
record both carry the propensity. A system that does not log its own
propensities cannot be evaluated later, at any price, by any method. Everything
else about the service is scaffolding around that sentence.

**Exit gate:** the round-trip integration test passes — serve decisions, log
them, load them back as `BanditFeedback`, run OPE on them.

---

## Phase 7 — Reporting

**Spec step:** 17
**Effort:** 1–2 sessions

Written last, once results exist. Structure it for a reviewer spending ninety
seconds: the problem in three sentences, the headline results table with
intervals, then the validation harness *first* among the technical content
because it is the differentiator, then the distribution-shift figure, the
architecture diagram, plainly stated limitations, and exact reproduction
commands.

**Exit gate:** every box in spec section 15 is checked, `make test` is green with
coverage above 80% on `ope/` and `policies/`, `mypy --strict` is clean, and two
identical seeded runs produce identical output.

---

## Summary

| Phase | Spec steps | Sessions | Produces |
|---|---|---|---|
| 0 Foundations | 0–1 | 1 *(done)* | types, schema, scaffolding, CI |
| 1 Estimation core | 2–5 | 2–3 | synthetic env, IPS/SNIPS, minimal harness |
| 2 Full suite + E01 | 6–7 | 2–3 | DM, DR with cross-fitting, three sweeps |
| 3 Real data + E02/E02b | 8–11 | 3–4 | LinUCB, Thompson, OBD, cross-policy validation |
| 4 Sequential + E03 | 12–13 | 2–3 | session MDP, FQI, distribution-shift finding |
| 5 CQL + PDIS + E04 | 14–15 | 3–4 | CQL alpha sweep, sequential OPE limits |
| 6 Service | 16 | 1–2 | FastAPI with propensity logging, round-trip test |
| 7 Reporting | 17 | 1–2 | README, figures, final report |

Roughly 15–22 sessions end to end. Phases 0–3 alone, about 8–11 sessions,
constitute a complete and defensible project.

## Standing rules across all phases

- Every reported number carries a bootstrap CI. Every RL number carries seed
  variance. No bare point estimates anywhere in the repo.
- Every ambiguity call gets a dated entry in `docs/decisions.md`. Where reality
  contradicts the spec, reality wins and the discrepancy is recorded.
- A result that looks too good is a suspected bug until proven otherwise. Check
  these four first: ground-truth leakage, cross-fitting not actually applied,
  evaluation on training data, and a target policy accidentally equal to the
  logging policy.
- A result that looks bad is either a bug (fix it) or a genuine limitation (it
  goes in the report with a number attached). Do not tune the setup until an
  inconvenient limitation disappears.
- Commit in small logical increments. After each spec step, report what was
  built, what the tests show, and any decision recorded — one step per report,
  not batched.
