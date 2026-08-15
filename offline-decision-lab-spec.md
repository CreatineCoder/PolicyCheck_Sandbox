# Offline Decision Lab — Implementation Specification

**Project codename:** `offline-decision-lab`
**Audience:** implementing agent (Claude Code)
**Author context:** portfolio project for campus placements; target roles are Data/ML/Quant and SDE.

---

## 0. Vision and intent — read this before writing any code

### 0.1 What this project actually is

Most reinforcement learning portfolio projects train an agent inside a simulator and report that it scored well. That is the easy half of the problem, and it is not the half industry struggles with. Companies can build models. What they cannot easily do is **decide whether a new model is safe to deploy, using only records of what the old model did.** Switching on an untested policy costs real money and real users, and there are always more candidate policies than there are A/B test slots.

This project is about that second problem. The full name for it is **off-policy evaluation** in the single-step case and **offline reinforcement learning** in the sequential case. The unifying constraint is: **you learn and evaluate entirely from logged data, and you never interact with the live system.**

### 0.2 The thesis the code must demonstrate

> Estimating a policy's value from logged data is easy to do and hard to do *correctly*. The estimators can be silently, badly wrong, and on real data you can never directly check — because the true value is precisely the thing you cannot observe. Therefore the estimators themselves must be validated against known ground truth before anyone trusts a number they produce.

Every design decision in this spec follows from that sentence. When the spec is ambiguous or silent, resolve the ambiguity in whichever direction better supports that thesis.

### 0.3 What "good" looks like here

The deliverable is **not** a high score. There is no leaderboard. The deliverable is a system whose numbers a reasonable person would believe, plus the evidence that justifies believing them.

Concretely, this reorders the usual priorities:

| Priority | Why |
|---|---|
| 1. Correctness of the estimators | A wrong estimator invalidates everything downstream. Unit-test against hand-computed values. |
| 2. Honest uncertainty | Every reported number carries a confidence interval. Every RL number carries seed variance. No bare point estimates anywhere in the repo. |
| 3. Evidence that the estimators work | The validation harness (§7) and the real-data cross-policy check (§6.4). These are the project. |
| 4. Reproducibility | One command, fixed seed, identical output. Non-determinism is a bug, not a quirk. |
| 5. Named limitations | Where the methods break is a finding, not an embarrassment. Measure it and report it. |
| 6. Raw performance | Genuinely last. A modest, well-validated lift beats an impressive unvalidated one. |

### 0.4 Failure modes to actively avoid

These are the ways this project most plausibly goes wrong. Treat each as a hard constraint.

- **Silent fallbacks.** Never substitute a default when data is missing or malformed. Raise. A pipeline that quietly drops rows produces a confident wrong number, which is worse than a crash.
- **Ground-truth leakage.** The synthetic environments expose `expected_reward`. Any code path that reads it while evaluating *real* data is a critical bug. Guard it structurally, not by convention — assert on it.
- **Estimating before validating.** Do not run estimators on real data until they have passed the synthetic harness. The build order in §13 enforces this; do not reorder it for convenience.
- **Hiding variance.** If an estimator has a 40% relative error under poor overlap, that number goes in the report. Suppressing it defeats the purpose of the project.
- **Scope creep into deep RL.** No Atari, no Gym control tasks, no benchmark chasing. If a change makes the project look more like a standard deep-RL repo, it is the wrong change.
- **Notebook-driven results.** Every reported number comes from a script in `experiments/`, runnable from the Makefile. Notebooks are for exploration only and nothing imports from them.

### 0.5 Who reads the output

Two audiences, and the repo must serve both:

- A **technical reviewer** who will read `src/odl/ope/` and check whether doubly robust is implemented with cross-fitting. Code quality and correctness are for them.
- A **hiring manager** who spends roughly ninety seconds on the README. The headline results table and the validation figures are for them.

Write for both. Neither is served by an undocumented repo of clever code, nor by a polished README over unverified numbers.

### 0.6 One-paragraph summary

Build a research-grade but production-shaped system for learning and evaluating decision policies purely from logged data, with no live environment access. Part 1 covers the single-step case (contextual bandits) with rigorous off-policy evaluation. Part 2 covers the sequential case (offline RL) with a demonstration of the distribution-shift failure mode and its mitigation. The distinguishing feature is not the algorithms — those are standard and well documented — but the **validation machinery that verifies the evaluation methods themselves against known ground truth, on both synthetic data and real logs, before any number is trusted.**

---

## 1. Notation

Used consistently throughout this document and required in docstrings.

| Symbol | Meaning |
|---|---|
| `x` | context / state features |
| `a` | action, in `[0, n_actions)` |
| `r` | observed reward |
| `pi_b(a\|x)` | **logging** (behaviour) policy — generated the data |
| `pi_e(a\|x)` | **target** (evaluation) policy — the one being assessed |
| `p_i` | logged propensity, `= pi_b(a_i\|x_i)` |
| `w_i` | importance weight, `= pi_e(a_i\|x_i) / p_i` |
| `V(pi)` | policy value, `= E[r]` under `pi` — the estimand |
| `V_hat` | an estimate of `V` |
| `r_hat(x,a)` | reward-model prediction |
| `gamma` | discount factor (Part 2 only) |
| `ESS` | effective sample size, `(sum w)^2 / sum(w^2)` |

**Overlap** (also called support) means: for every action `pi_e` would take with non-trivial probability, `pi_b` had non-zero probability of taking it too. Where overlap fails, importance weighting is not merely noisy — it is undefined. Detecting and reporting overlap violations is required behaviour, not an optional diagnostic.

---

## 2. Objectives and non-goals

### Objectives

1. Implement contextual bandit policies (LinUCB, Thompson sampling, epsilon-greedy) that train from logged data.
2. Implement OPE estimators (IPS, SNIPS, Direct Method, Doubly Robust) **from scratch**, with bootstrapped confidence intervals.
3. Build a synthetic environment with known ground truth, and use it to **validate the OPE estimators** — proving they recover the true policy value before applying them to real logs.
4. Apply the validated pipeline to a real logged-bandit dataset and report policy value with uncertainty.
5. Extend to sequential decisions: implement Fitted Q-Iteration (FQI) and Conservative Q-Learning (CQL) on logged trajectories.
6. Empirically demonstrate **distribution shift** — FQI overestimating the value of actions absent from the logs — and show CQL correcting it.
7. Ship a FastAPI service that serves policy decisions and logs its own propensities, closing the loop for future evaluation.

### Non-goals

- No live/online training. Nothing interacts with real users.
- No deep RL benchmark chasing (no Atari, no MuJoCo, no Gym control tasks).
- No attempt at state-of-the-art results. Correctness, honesty of evaluation, and reproducibility are the deliverables.
- No distributed training. Everything must run on a single machine, CPU-only, with GPU as optional acceleration for Part 2.

---

## 3. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Env / deps | `uv` (fallback: `venv` + `pip`) | Pin everything in `pyproject.toml` |
| Numerics | `numpy`, `scipy` | |
| Data | `pandas`, `pyarrow` | Parquet for all intermediate artifacts |
| Classical ML | `scikit-learn`, `xgboost` | Reward models, FQI regressors |
| Deep learning | `pytorch` | CQL only |
| Reference OPE | `obp` (Open Bandit Pipeline) | **Cross-check only** — see §6.4 |
| API | `fastapi`, `uvicorn`, `pydantic` | |
| Config | `hydra-core` or `pydantic-settings` | Prefer hydra for experiment sweeps |
| Tracking | `mlflow` (local file backend) | Avoid W&B to keep it offline/free |
| Testing | `pytest`, `hypothesis` | |
| Plotting | `matplotlib` | No seaborn; keep deps lean |
| Lint/format | `ruff` | |

**Constraint:** every experiment must be reproducible from a single command with a fixed seed. Non-determinism is a bug.

---

## 4. Repository layout

```
offline-decision-lab/
├── pyproject.toml
├── README.md                      # written LAST, see §11
├── Makefile                       # make setup / test / part1 / part2 / serve
├── docs/
│   └── decisions.md               # dated log of ambiguity calls, see §16
├── configs/
│   ├── synthetic_bandit.yaml
│   ├── real_bandit.yaml
│   ├── synthetic_mdp.yaml
│   └── serve.yaml
├── src/odl/
│   ├── __init__.py
│   ├── types.py                   # dataclasses for BanditFeedback, Trajectory
│   ├── envs/
│   │   ├── synthetic_bandit.py    # ground-truth-known bandit env
│   │   └── session_mdp.py         # ground-truth-known sequential env
│   ├── policies/
│   │   ├── base.py                # Policy ABC
│   │   ├── uniform.py
│   │   ├── epsilon_greedy.py
│   │   ├── linucb.py
│   │   └── thompson.py
│   ├── ope/
│   │   ├── base.py                # Estimator ABC
│   │   ├── ips.py                 # IPS + SNIPS + clipping
│   │   ├── direct.py              # reward-model based
│   │   ├── doubly_robust.py
│   │   └── bootstrap.py           # CI machinery
│   ├── offline_rl/
│   │   ├── fqi.py
│   │   ├── cql.py
│   │   └── sequential_ope.py      # per-decision importance sampling
│   ├── data/
│   │   ├── obd_loader.py          # Open Bandit Dataset
│   │   └── schema.py              # validation of log format
│   ├── eval/
│   │   ├── validation_harness.py  # THE centerpiece — see §7
│   │   └── metrics.py
│   └── service/
│       ├── app.py                 # FastAPI
│       └── logging_middleware.py  # writes propensity-logged decisions
├── experiments/
│   ├── e01_ope_validation.py
│   ├── e02_real_bandit.py
│   ├── e03_distribution_shift.py
│   └── e04_cql_comparison.py
├── tests/
├── reports/                       # generated figures + markdown results
└── data/                          # gitignored; download scripts only
```

---

## 5. Core data contracts

Define these in `src/odl/types.py` as frozen dataclasses with validation. **Every module reads and writes these types — no ad-hoc dicts.**

### 5.1 `BanditFeedback` (single-step)

```python
@dataclass(frozen=True)
class BanditFeedback:
    context: np.ndarray        # (n, d_context) float64
    action: np.ndarray         # (n,) int64, in [0, n_actions)
    reward: np.ndarray         # (n,) float64
    propensity: np.ndarray     # (n,) float64 in (0, 1]
    n_actions: int
    action_context: np.ndarray | None = None   # (n_actions, d_action)
    # Ground truth, ONLY populated by synthetic envs. Must be None for real data.
    expected_reward: np.ndarray | None = None  # (n, n_actions)
```

The `expected_reward` field is the key to the validation harness: with it you can compute a policy's true value exactly; without it you can only estimate. **Any code path that reads `expected_reward` when evaluating real data is a bug and must raise.**

### 5.2 `Trajectory` (sequential)

```python
@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray         # (T+1, d_state) — includes terminal state
    actions: np.ndarray        # (T,) int64
    rewards: np.ndarray        # (T,) float64
    propensities: np.ndarray   # (T,) float64
    done: bool
```

Store collections of trajectories as Parquet with an `episode_id` column; provide converters both ways.

---

## 6. Part 1 — Contextual bandits with off-policy evaluation

### 6.1 Synthetic environment (`envs/synthetic_bandit.py`)

Purpose: an environment where the true expected reward of every (context, action) pair is known by construction.

Design:
- Sample context `x ~ N(0, I_d)`, `d = 10`.
- For each action `a`, fix a weight vector `w_a` and bias `b_a` drawn once from a seeded RNG.
- Define `mu(x, a) = sigmoid(w_a · x + b_a + beta * nonlinear(x, a))`.
  - Include a `nonlinear` term (e.g. `sin(w_a · x)`) with a tunable coefficient `beta`. **This matters:** with `beta = 0` a linear reward model is perfectly specified and the Direct Method looks great; with `beta > 0` it is misspecified and you can demonstrate that Doubly Robust degrades gracefully while DM does not. That contrast is a headline result.
- Reward is Bernoulli with probability `mu(x, a)`.
- Expose `true_policy_value(policy) -> float`, computed by averaging `sum_a pi(a|x) * mu(x, a)` over a large fresh context sample.

Provide a configurable **logging policy** with an `epsilon` mixing parameter so you can control overlap between logging and target policies. Low overlap should break IPS — demonstrating that is an experiment, not a failure.

### 6.2 Policies (`policies/`)

All implement:

```python
class Policy(ABC):
    def fit(self, feedback: BanditFeedback) -> None: ...
    def action_dist(self, context: np.ndarray) -> np.ndarray:  # (n, n_actions), rows sum to 1
        ...
```

Returning a full **distribution**, not an argmax, is required — OPE estimators need `pi(a|x)` for every logged action.

- **`UniformPolicy`** — baseline, uniform over actions.
- **`EpsilonGreedyPolicy`** — fits a reward model, mixes greedy with uniform at rate `epsilon`.
- **`LinUCBPolicy`** — maintain per-arm `A_a = I*lambda + sum x x^T` and `b_a = sum r*x`. Score is `theta_a · x + alpha * sqrt(x^T A_a^-1 x)`. Since LinUCB is deterministic, convert to a distribution by softmax over scores with a temperature parameter, or by an epsilon-mix. **Document which you chose and why** — a deterministic target policy makes IPS variance explode, and handling that is part of the story.
- **`ThompsonSamplingPolicy`** — Bayesian linear regression per arm with Normal-Inverse-Gamma or a Gaussian approximation. To get `action_dist`, run Monte Carlo: draw `K = 1000` parameter samples, count argmax frequencies. Cache per context batch.

Use `numpy` linear algebra with Sherman-Morrison rank-1 updates for `A_a^-1`; do not invert from scratch each step.

### 6.3 OPE estimators (`ope/`)

Interface:

```python
class Estimator(ABC):
    def estimate(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,          # (n, n_actions) from target policy
        reward_model_preds: np.ndarray | None = None,  # (n, n_actions)
    ) -> float: ...
    def per_sample_values(self, ...) -> np.ndarray:  # (n,) — needed for bootstrap
```

Implement, with the estimand being `V(pi) = E[r]` under the target policy:

**IPS.** `V = mean( (pi(a_i|x_i) / p_i) * r_i )`. Add optional weight clipping at `tau` and expose `tau` as config. Report the effective sample size `ESS = (sum w)^2 / sum(w^2)` alongside every IPS estimate — it is the single best diagnostic for whether the estimate is trustworthy.

**SNIPS.** Self-normalized: `V = sum(w_i r_i) / sum(w_i)`. Biased but far lower variance.

**Direct Method.** Fit `\hat{r}(x, a)` on the logs (use `xgboost` with cross-fitting), then `V = mean( sum_a pi(a|x_i) * \hat{r}(x_i, a) )`.

**Doubly Robust.**
```
V = mean( sum_a pi(a|x_i) * r_hat(x_i, a)
          + (pi(a_i|x_i) / p_i) * (r_i - r_hat(x_i, a_i)) )
```
Implement **cross-fitting** (K-fold): fit the reward model on K−1 folds, apply to the held-out fold. Without this the reward model overfits the same data used for correction and DR loses its guarantee. This detail is worth calling out explicitly in the README.

**Bootstrap CIs (`ope/bootstrap.py`).** Resample the per-sample values with replacement B = 1000 times, report the 2.5th and 97.5th percentiles. Every reported number in this project carries an interval. No bare point estimates anywhere.

### 6.4 Real data (`data/obd_loader.py`)

Use the **Open Bandit Dataset** (ZOZO). Availability was verified on 2026-08-15; the facts below are confirmed against the actual files, not inferred from documentation. **Trust these over anything the model recalls about OBD.**

#### Two sources, use both

| Source | How | Size | Use for |
|---|---|---|---|
| Sample | `git clone --depth 1 https://github.com/st-tech/zr-obp` → `obd/` | 10,000 rows per (policy, campaign); 12 MB each | Unit tests, CI, smoke runs |
| Full | Direct zip, no registration: `https://research.zozo.com/data_release/open_bandit_dataset.zip` | ~26M rows total | All reported results |

Write `scripts/download_obd.sh` for both. Gitignore `data/`. Commit the sample-based tests so CI runs without the large download.

#### Verified file layout

```
obd/{behavior_policy}/{campaign}/{campaign}.csv
obd/{behavior_policy}/{campaign}/item_context.csv
```
`behavior_policy ∈ {random, bts}`, `campaign ∈ {all, men, women}`.

#### Verified schema (`all` campaign)

Main CSV: unnamed integer index column (read with `index_col=0`), then `timestamp`, `item_id`, `position`, `click`, `propensity_score`, `user_feature_0..3`, `user-item_affinity_0..79`.

Note the exact naming: `user_feature_*` uses underscores, `user-item_affinity_*` uses a **hyphen**. Get this wrong and column selection silently returns nothing. Also note the docs claim `user feature 0-4` (five features); **there are actually four** — `user_feature_0` through `user_feature_3`. Trust the file.

`item_context.csv` is 80 rows × 5 columns: `item_id`, `item_feature_0..3`.

#### Critical properties confirmed by inspection

- **80 actions**, 3 positions in the `all` campaign.
- **`user_feature_*` are hashed strings, not numbers.** In the 10k sample, `user_feature_0` takes only 3 distinct hash values. These are categorical IDs. You must hash-encode or one-hot them; feeding them to a numeric model as-is will fail or silently produce garbage. The 80 affinity columns are genuine floats and are the useful signal, though heavily zero-valued.
- **`random` logging propensity is a constant 0.0125** = 1/80, exactly uniform. This gives perfect overlap with any target policy — ideal for the first real-data experiment.
- **`bts` has 7,883 distinct propensities in 10k rows, minimum 4.5e-05.** Against a uniform target that is a maximum importance weight around 278. Weight clipping and ESS reporting are not optional here; run the `bts` split specifically to demonstrate IPS instability on real data.
- **CTR is roughly 0.4%** (38 clicks in the 10,000-row random sample). This is the single most important operational fact: **the sample is far too small for meaningful results.** Point estimates on 38 positive examples will have intervals wider than any effect. Use the sample for tests only and the full 26M dataset for every reported number.

#### The upgrade: real-data ground truth

The most valuable property of OBD, and the reason to prefer it over every alternative, is that **it contains logs from two different policies collected on the same platform during a single A/B test.** That enables something the synthetic harness can only approximate:

1. Compute the **true** value of the Bernoulli TS policy directly, as the on-policy mean click rate in the `bts` logs. This is ground truth from a real system — no modelling assumption.
2. Independently **estimate** that same quantity by running your OPE estimators on the `random` logs, treating BTS as the target policy.
3. Compare. The gap is your estimator's real-world error, measured against a real number.

Then do it in reverse (estimate Random's value from `bts` logs) — the harder direction, since the logging policy is non-uniform and overlap is poorer.

**Elevate this to a first-class experiment (E02b).** Synthetic validation proves the estimators are implemented correctly; this proves they work on real logged data with real ground truth. Very few portfolio projects can make that second claim, and it is a far stronger interview answer than any lift number.

#### Position handling — decide this early

`position ∈ {1, 2, 3}` means each impression is a **three-slot slate**, not a single action. Three defensible options:

- **(a)** Filter to one position and treat it as a standard contextual bandit. Simplest, loses two-thirds of the data. **Recommended default.**
- **(b)** Treat `(item_id, position)` as a compound action over 240 arms. Larger action space, sparser data.
- **(c)** Use a slate/ranking estimator such as the independent-IPS or pseudo-inverse estimator.

Pick (a) for the main pipeline. State the choice and its cost explicitly in the README; an interviewer who knows OBD will ask, and having a reasoned answer is worth more than silently dropping the column.

#### Validation on load

Enforce in `data/schema.py`: propensities in `(0, 1]`, `item_id` within campaign range, no NaNs, `click ∈ {0, 1}`, row count matches expectation. Assert `expected_reward is None` for all real data. **Fail loudly** — silent schema drift is the most likely source of a wrong headline number.

#### Fallback

If the ZOZO endpoint is down, the GitHub sample still supports the full pipeline at reduced statistical power. A distant second choice is **MIND** or **MSLR-WEB10K** with a *simulated* logging policy — but flag prominently that propensities are then synthetic, and note that this forfeits the two-policy ground-truth comparison entirely, which is most of the reason for choosing OBD.

---

## 7. The validation harness — the centerpiece

**This is the part of the project that distinguishes it. Give it the most care.**

### 7.1 The problem it solves

OPE estimators are models. They can be silently wrong. On real data you can never check, because the true policy value is unobservable — that is the entire premise. So: verify the estimators on synthetic data where truth is known, characterize when they break, and only then apply them to real logs.

### 7.2 Protocol (`eval/validation_harness.py`)

Implement a function that takes a synthetic env config, a logging policy, a target policy, and a list of estimators, and:

1. Generates logged data from the logging policy.
2. Computes `V_true(target)` exactly using `expected_reward`.
3. Computes each estimator's `V_hat` with bootstrap CI.
4. Reports per estimator: **relative bias** `(V_hat − V_true)/V_true`, **RMSE** across R independent log replications (R = 100), and **CI coverage** — the fraction of the 100 replications where the 95% interval contained `V_true`.

Coverage is the sharpest diagnostic. A well-behaved estimator should cover close to 95% of the time. IPS under low overlap will cover far less, and showing that number is more convincing than any argument.

### 7.3 Required sweeps

Run the harness across:

- **Overlap.** Vary logging-policy epsilon from 1.0 (uniform) down to 0.05 (near-deterministic). Expect IPS variance to blow up and ESS to collapse; expect DR to hold up longer.
- **Reward model misspecification.** Vary the `beta` nonlinearity coefficient. Expect DM bias to grow while DR stays close to truth — this is the doubly-robust property made visible.
- **Sample size.** n from 1,000 to 500,000, log scale. Show convergence rates.

Produce a figure per sweep in `reports/`. These figures are the interview artifact.

### 7.4 Cross-check against `obp`

Once your from-scratch estimators pass the harness, run the `obp` library's implementations on identical inputs and assert agreement within tolerance in a test. This proves your implementations are correct rather than merely self-consistent. **Do not use `obp` as the primary implementation** — implementing from scratch is the point.

---

## 8. Part 2 — Sequential offline RL

### 8.1 Sequential environment (`envs/session_mdp.py`)

A small, fully-known MDP modelling a user session, chosen so that dynamic effects genuinely matter:

- **State:** user interest vector (continuous, `d = 8`), session step index, fatigue scalar.
- **Actions:** a small discrete set (say 10 content categories), each with a `(engagement, fatigue_cost)` profile. Include at least one "clickbait" action with high immediate reward and high fatigue cost.
- **Dynamics:** interest drifts toward the shown category; fatigue accumulates; the episode terminates early with probability increasing in fatigue.
- **Reward:** immediate engagement.
- **Horizon:** T = 20 with early termination.

The clickbait action makes the myopic and far-sighted optimal policies **different**, which is essential — otherwise a bandit would solve it and Part 2 proves nothing. Verify this before proceeding: compute the optimal greedy policy and the optimal discounted policy via value iteration on a discretized version, and assert their values differ meaningfully. Log that gap; it is the justification for the whole of Part 2.

Provide `true_policy_value(policy, gamma, n_rollouts)` by Monte Carlo rollout — again, ground truth available by construction.

### 8.2 Logged trajectory generation

Generate logs with a deliberately **narrow** logging policy: a mediocre heuristic that only ever selects a subset (say 4 of 10) of actions. This is the realistic case — production systems do not try everything — and it sets up the distribution-shift demonstration.

Save propensities per step.

### 8.3 Fitted Q-Iteration (`offline_rl/fqi.py`)

Standard batch algorithm:

```
Q_0 = 0
repeat for k = 1..K:
    targets_i = r_i + gamma * (1 - done_i) * max_a Q_{k-1}(s'_i, a)
    fit Q_k on (s_i, a_i) -> targets_i
```

Use `XGBRegressor` per action, or a single regressor over concatenated `(s, one_hot(a))`. Config-switchable. Track the Bellman residual per iteration and plot it.

### 8.4 Demonstrating distribution shift (`experiments/e03`)

This is the experiment that produces the most memorable result. Steps:

1. Train FQI on the narrow logs.
2. For each state in a held-out set, compute `max_a Q(s, a)` and record which action attains it.
3. Compute the true value of the greedy policy induced by `Q` using the environment.
4. Compare `mean(max_a Q(s,a))` — what the model *believes* it will achieve — against the true value.

Expected result: FQI substantially overestimates, and the overestimation concentrates on actions the logging policy rarely or never took, because there is no data to correct the extrapolation. Produce:

- A bar chart of estimated vs true Q-value, grouped by how often the logging policy took each action.
- A scatter plot of `Q_estimated − Q_true` against logged action frequency.

Write the finding as a numbered result in `reports/`.

### 8.5 Conservative Q-Learning (`offline_rl/cql.py`)

PyTorch implementation for discrete actions. The CQL loss adds a regularizer to standard Bellman error:

```
L_CQL = alpha * E_s[ logsumexp_a Q(s,a) − E_{a ~ data}[Q(s,a)] ]  +  L_Bellman
```

The first term pushes down Q on all actions while pushing up Q on actions actually present in the data, keeping the learned policy anchored to the data distribution.

Requirements:
- MLP Q-network, 2 hidden layers, 256 units, ReLU.
- Target network with Polyak averaging.
- `alpha` is the key hyperparameter — sweep it over `[0.0, 0.1, 1.0, 5.0, 10.0]`. Note that `alpha = 0` reduces to plain offline DQN, giving you the ablation for free.
- Fixed seeds; report mean ± std over 5 seeds. **RL is unstable; single-seed results are not results.**

### 8.6 Sequential OPE (`offline_rl/sequential_ope.py`)

Implement **per-decision importance sampling** and its weighted variant:

```
w_t = prod_{k=0..t} pi(a_k|s_k) / p_k
V_PDIS = mean_over_episodes( sum_t gamma^t * w_t * r_t )
```

Then be honest about it: importance weights compound multiplicatively over the horizon, so variance grows exponentially with T. Report ESS per horizon length and show it collapsing. **Document this as a genuine limitation rather than hiding it** — the reason model-based evaluation and conservative methods exist is precisely that PDIS does not scale in horizon. Interviewers respect a candidate who names the limits of their own method.

Use the environment's true value as the reference; report how far PDIS is from it at T = 5, 10, 20.

---

## 9. Serving layer (`service/`)

A minimal FastAPI app demonstrating the production loop:

- `POST /recommend` — body `{context: [float], policy_id: str}`. Returns `{action: int, propensity: float, decision_id: str, policy_version: str}`.
- **Critical:** the response and the log record must both include the propensity. A system that does not log its own propensities cannot be evaluated later. This single design choice is the whole point of the service and should be stated as such in the README.
- `POST /reward` — body `{decision_id: str, reward: float}`. Joins delayed feedback to the decision.
- `GET /healthz`, `GET /metrics` — decisions served, action distribution, mean propensity.
- Append-only Parquet or JSONL sink under `data/serving_logs/`, written in the exact `BanditFeedback` schema so logs collected by the service are directly consumable by the OPE pipeline. Demonstrate this round-trip in a test.
- Load policies from a versioned artifact directory; no retraining inside the request path.

Dockerfile, and `make serve`.

---

## 10. Experiments to run and results to record

Each script writes a markdown file plus figures into `reports/`.

| ID | Script | Headline output |
|---|---|---|
| E01 | `e01_ope_validation.py` | Bias / RMSE / CI-coverage table for all 4 estimators across overlap, misspecification, and sample-size sweeps |
| E02 | `e02_real_bandit.py` | Policy value of LinUCB and Thompson vs logged policy on OBD, with bootstrap CIs and ESS |
| E02b | `e02b_cross_policy_validation.py` | **OPE error against real ground truth**: estimate BTS value from `random` logs, compare to on-policy BTS mean; then the reverse direction. Report per-estimator absolute and relative error, plus ESS. |
| E03 | `e03_distribution_shift.py` | FQI overestimation quantified; scatter of error vs logged action frequency |
| E04 | `e04_cql_comparison.py` | True policy value for FQI vs CQL across alpha, mean ± std over 5 seeds |

Every table reports uncertainty. Every RL number reports seed variance.

---

## 11. README requirements

Write this **last**, once results exist. A hiring reviewer spends about ninety seconds on it, so structure accordingly:

1. **The problem in three sentences** — you have logs from an old policy, you want to know if a new policy is better, you cannot run it live.
2. **Headline results table** — the four experiment outcomes, with numbers and intervals.
3. **The validation harness** — explain that the estimators were verified against known ground truth before being trusted, and show the coverage figure. Lead with this; it is the differentiator.
4. **The distribution-shift finding** — the FQI-vs-CQL figure with a one-paragraph explanation.
5. **Architecture diagram** — data flow from logs through policies and estimators to the service.
6. **Limitations, stated plainly** — synthetic sequential environment, PDIS horizon variance, single dataset, no live validation. Do not hide these; naming them is a positive signal.
7. **Reproduce** — exact commands.

No emoji. No badges beyond CI status. Prose over bullet-fragments in the explanatory sections.

---

## 12. Testing requirements

- **Unit:** every estimator on a hand-computed toy example with known closed-form answer. Every policy's `action_dist` sums to 1 within tolerance.
- **Property-based** (`hypothesis`): IPS with target policy equal to logging policy must recover the empirical mean reward. Uniform target with uniform logging must equal the mean reward. These invariants catch most implementation errors.
- **Statistical:** the validation harness itself, run at reduced scale, asserting DR relative bias below a threshold under good overlap.
- **Integration:** service round-trip — serve decisions, log them, load them as `BanditFeedback`, run OPE on them.
- **Determinism:** same seed produces byte-identical results across two runs.

Target coverage above 80% on `src/odl/ope/` and `src/odl/policies/`.

---

## 13. Build order

Strictly sequential; each phase must pass its tests before the next begins.

**Step 0 — do this first, before any other work.** Verify the OBD download link responds and the schema matches §6.4. This is the project's only external dependency and the only thing that could force a redesign. Confirming it takes ten minutes; discovering it broken at step 9 costs days.

1. `types.py`, schema validation, project scaffolding, CI.
2. Synthetic bandit env with exact `true_policy_value`.
3. Uniform and epsilon-greedy policies.
4. IPS and SNIPS + bootstrap. Unit tests.
5. **Validation harness, minimal version.** Confirm IPS recovers truth under uniform logging. *Do not proceed until this passes* — everything downstream depends on the harness being correct.
6. Reward model, Direct Method, Doubly Robust with cross-fitting.
7. Full validation sweeps → E01.
8. LinUCB and Thompson sampling.
9. OBD loader (verify download + schema on day one, not at this step) → E02.
10. Cross-policy real-data validation → E02b. **This is the strongest single result in Part 1.**
11. `obp` cross-check test.
12. Sequential env, verify myopic-vs-farsighted value gap.
13. FQI → E03.
14. CQL → E04.
15. Sequential OPE.
16. FastAPI service + round-trip test.
17. README, figures, final report.

**Checkpoint after step 11:** Part 1 is a complete, standalone, presentable project at that point. If time runs short, stop there and polish rather than leaving Part 2 half-built.

---

## 14. Style and quality constraints

- Type-hint everything; `mypy --strict` on `src/odl/`.
- Docstrings on public functions state the estimand or algorithm and cite the source paper where relevant (Dudík et al. 2011 for DR; Li et al. 2010 for LinUCB; Kumar et al. 2020 for CQL).
- No notebooks in the main pipeline. Notebooks may exist under `notebooks/` for exploration only, and nothing imports from them.
- No magic numbers in code; all hyperparameters live in `configs/`.
- Commit in logical increments with meaningful messages. Commit history is visible to reviewers and a single "initial commit" dump is a negative signal.

---

## 15. Definition of done

The project is complete when every box below is true. Anything unchecked is unfinished work, not a nice-to-have.

**Part 1**
- [ ] Four estimators implemented from scratch, each unit-tested against a hand-computed toy case.
- [ ] Property tests pass: with `pi_e == pi_b`, IPS recovers the empirical mean reward exactly (to floating-point tolerance).
- [ ] Doubly robust uses K-fold cross-fitting; a test verifies the reward model never predicts on data it was fitted to.
- [ ] Validation harness reports bias, RMSE, and CI coverage across all three sweeps (overlap, misspecification, sample size).
- [ ] Under good overlap and correct specification, DR coverage is within a few points of 95%. If it is not, the implementation is wrong — debug rather than reporting the number.
- [ ] The misspecification sweep visibly separates DM from DR. If it does not, `beta` is too small; increase the nonlinearity until the effect is real.
- [ ] `obp` cross-check test passes within tolerance on identical inputs.
- [ ] Real-data cross-policy validation (E02b) produces a measured error against true BTS value.
- [ ] Every reported number has a bootstrap CI and a stated ESS.

**Part 2**
- [ ] Sequential environment verified to have a meaningful myopic-vs-farsighted value gap, with that gap logged as a number.
- [ ] FQI overestimation demonstrated and quantified, with the error correlated against logged action frequency.
- [ ] CQL alpha sweep run over 5 seeds; results reported as mean ± std.
- [ ] PDIS variance collapse measured and reported at three horizon lengths.

**Infrastructure**
- [ ] `make test` green; coverage above 80% on `ope/` and `policies/`.
- [ ] `mypy --strict` clean on `src/odl/`.
- [ ] Two identical runs with the same seed produce identical output.
- [ ] Service round-trip test passes: serve → log → load as `BanditFeedback` → run OPE.
- [ ] README complete with headline results table and figures.

---

## 16. Working agreement

**When the spec is silent or ambiguous.** Choose the option that better supports the thesis in §0.2 — usually the more conservative, more measurable, or more honest one. Record the decision and its rationale in `docs/decisions.md` as a short dated entry. Do not silently pick and move on.

**When something in the spec turns out to be wrong.** The spec was written from verified inspection of the actual dataset, but details drift. If reality contradicts this document, **reality wins.** Note the discrepancy in `docs/decisions.md` and proceed. Do not contort the code to match a stale spec.

**When a result looks too good.** Treat it as a suspected bug until proven otherwise. The most common causes are ground-truth leakage, cross-fitting not actually applied, evaluating on training data, and a target policy accidentally equal to the logging policy. Check those four before celebrating.

**When a result looks bad.** Distinguish two cases. A *bug* means fix it. A *genuine limitation* — IPS failing under poor overlap, PDIS variance exploding over long horizons — is a finding, and it goes in the report with a number attached. Do not tune the setup until the limitation disappears; the limitations are among the most valuable content in the project.

**Commits.** Small, logical, meaningfully messaged. The history is visible to reviewers and a single squashed dump is a negative signal.

**Progress reporting.** After each numbered step in §13, state what was built, what the tests show, and any decision recorded. Do not batch several steps into one report.

---

## 17. Resume bullets this produces

Constraints: one line each, no wrapping, maximum four per project entry.

Suggested set (adjust once real numbers land):

- Built offline policy-evaluation system implementing IPS, SNIPS, DM and doubly-robust estimators from scratch with bootstrapped confidence intervals
- Validated estimators against known ground truth on synthetic benchmarks, achieving 94% CI coverage and under 2% relative bias for DR
- Validated OPE against real ground truth on 26M-record ZOZO Open Bandit Dataset, recovering true policy value within X% using two-policy A/B logs
- Demonstrated Y% Q-value overestimation from distribution shift in offline RL, reduced to Z% via Conservative Q-Learning across 5 seeds

The fourth bullet is the differentiator; keep it even if something else has to go.
