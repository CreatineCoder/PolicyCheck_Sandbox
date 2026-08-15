"""The validation harness (spec section 7) -- minimal version.

The problem this solves: off-policy estimators are models, and models can be
silently wrong. On real logs the error can never be measured, because the true
policy value is the unobservable quantity by definition. So the estimators are
checked here first, on synthetic data where truth is known by construction, and
only afterwards applied to real logs.

Protocol, per replication:

1. Generate logged data from the logging policy.
2. Compute ``V_true(target)`` exactly from ``expected_reward``.
3. Compute each estimator's ``V_hat`` with a bootstrap CI.
4. Record relative error, ESS, and whether the interval covered the truth.

Across R replications the harness reports relative bias, RMSE, and CI coverage
per estimator.

This is the minimal version: it runs one configuration. The parameter sweeps
over overlap, misspecification, and sample size build on top of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from odl.envs.synthetic_bandit import SyntheticBanditEnv
from odl.eval.metrics import coverage, mean_ci_width, relative_bias, rmse
from odl.ope.base import Estimator, effective_sample_size, importance_weights
from odl.ope.bootstrap import ConfidenceInterval, bootstrap_estimator
from odl.policies.base import Policy

__all__ = [
    "EstimatorSummary",
    "HarnessResult",
    "ReplicationResult",
    "run_validation",
]


@dataclass(frozen=True)
class ReplicationResult:
    """One estimator's outcome on one simulated log."""

    estimator: str
    interval: ConfidenceInterval
    ess: float

    @property
    def estimate(self) -> float:
        return self.interval.point


@dataclass(frozen=True)
class EstimatorSummary:
    """An estimator's aggregate behaviour across R replications.

    Attributes:
        estimator: estimator name.
        n_replications: R.
        v_true: the exact policy value being recovered.
        mean_estimate: mean of ``V_hat`` across replications.
        relative_bias: ``(mean(V_hat) - V_true) / V_true``.
        rmse: root mean squared error about ``V_true``.
        coverage: fraction of 95% intervals containing ``V_true``. The headline
            number: an estimator whose coverage is far below nominal is
            misreporting its own uncertainty.
        mean_ci_width: mean interval width, read alongside coverage so that
            vacuously wide intervals are not mistaken for well-calibrated ones.
        mean_ess: mean effective sample size of the importance weights.
    """

    estimator: str
    n_replications: int
    v_true: float
    mean_estimate: float
    relative_bias: float
    rmse: float
    coverage: float
    mean_ci_width: float
    mean_ess: float

    def format_row(self) -> str:
        """Return a fixed-width summary line for console and report output."""
        return (
            f"{self.estimator:<16} "
            f"V_hat={self.mean_estimate:.5f}  "
            f"rel_bias={self.relative_bias:+.4f}  "
            f"rmse={self.rmse:.5f}  "
            f"coverage={self.coverage:.2%}  "
            f"ci_width={self.mean_ci_width:.5f}  "
            f"ESS={self.mean_ess:.1f}"
        )


@dataclass(frozen=True)
class HarnessResult:
    """The full outcome of a harness run."""

    v_true: float
    v_true_stderr: float
    n_rounds: int
    n_replications: int
    summaries: tuple[EstimatorSummary, ...]
    replications: tuple[tuple[ReplicationResult, ...], ...]

    def summary_for(self, estimator: str) -> EstimatorSummary:
        """Return the summary for a named estimator, raising if absent."""
        for summary in self.summaries:
            if summary.estimator == estimator:
                return summary
        available = [s.estimator for s in self.summaries]
        raise KeyError(f"no estimator named {estimator!r}; available: {available}")

    def format_table(self) -> str:
        """Return the multi-line report table."""
        header = (
            f"V_true = {self.v_true:.6f} (+/- {self.v_true_stderr:.6f} MC stderr), "
            f"n = {self.n_rounds}, replications = {self.n_replications}"
        )
        return "\n".join([header, *(s.format_row() for s in self.summaries)])


def run_validation(
    env: SyntheticBanditEnv,
    logging_policy: Policy,
    target_policy: Policy,
    estimators: Sequence[Estimator],
    n_rounds: int,
    seed: int,
    n_replications: int = 100,
    n_true_value_samples: int = 100_000,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> HarnessResult:
    """Run the validation protocol and return per-estimator summaries.

    Args:
        env: synthetic environment supplying ground truth.
        logging_policy: ``pi_b``, which generates the logs. Its overlap with the
            target is the property the harness is ultimately probing.
        target_policy: ``pi_e``, the policy being evaluated. Assumed already
            fitted; the harness does not train it, so that the target stays
            identical across replications and the only thing varying is the log.
        estimators: estimators to compare on identical inputs.
        n_rounds: log size per replication.
        seed: root seed. Every random draw in the run derives from it, so two
            runs with the same seed produce identical output.
        n_replications: R, the number of independent logs.
        n_true_value_samples: fresh contexts used for the exact value integral.
        n_bootstrap: bootstrap replicates per interval, B.
        alpha: total tail mass; 0.05 gives 95% intervals.

    Returns:
        A :class:`HarnessResult` carrying the per-replication detail and the
        aggregate bias, RMSE, and coverage per estimator.
    """
    if n_replications < 1:
        raise ValueError(f"n_replications must be >= 1, got {n_replications}")
    if len(estimators) == 0:
        raise ValueError("at least one estimator is required")

    names = [e.name for e in estimators]
    if len(set(names)) != len(names):
        raise ValueError(
            f"estimator names must be unique so results can be attributed; got {names}"
        )

    # Independent, reproducible substreams: one for the ground-truth integral
    # and one per replication. Deriving them from a single SeedSequence keeps
    # the whole run determined by `seed` alone, while ensuring the replications
    # are not accidentally correlated through a shared generator.
    root = np.random.SeedSequence(seed)
    truth_seed, replication_seed = root.spawn(2)
    replication_streams = np.random.SeedSequence(
        replication_seed.entropy, spawn_key=replication_seed.spawn_key
    ).spawn(n_replications)

    v_true, v_true_stderr = env.true_policy_value_with_stderr(
        target_policy, n_true_value_samples, np.random.default_rng(truth_seed)
    )
    if v_true == 0.0:
        raise ValueError(
            "the target policy has a true value of exactly zero, so relative "
            "bias is undefined; choose a different target or environment"
        )

    all_replications: list[tuple[ReplicationResult, ...]] = []
    for stream in replication_streams:
        # Two generators per replication: one for data generation, one for the
        # bootstrap. Keeping them separate means changing B does not perturb the
        # logs, so results across bootstrap settings stay comparable.
        data_seed, bootstrap_seed = stream.spawn(2)
        data_rng = np.random.default_rng(data_seed)
        bootstrap_rng = np.random.default_rng(bootstrap_seed)

        feedback = env.generate_feedback(logging_policy, n_rounds, data_rng)
        action_dist = target_policy.action_dist_checked(feedback.context)
        weights = importance_weights(feedback, action_dist)
        ess = effective_sample_size(weights)

        results: list[ReplicationResult] = []
        for estimator in estimators:
            interval = bootstrap_estimator(
                estimator,
                feedback,
                action_dist,
                rng=bootstrap_rng,
                n_resamples=n_bootstrap,
                alpha=alpha,
            )
            results.append(
                ReplicationResult(estimator=estimator.name, interval=interval, ess=ess)
            )
        all_replications.append(tuple(results))

    summaries: list[EstimatorSummary] = []
    for position, estimator in enumerate(estimators):
        per_rep = [rep[position] for rep in all_replications]
        estimates = [r.estimate for r in per_rep]
        intervals = [r.interval for r in per_rep]
        summaries.append(
            EstimatorSummary(
                estimator=estimator.name,
                n_replications=n_replications,
                v_true=v_true,
                mean_estimate=float(np.mean(estimates)),
                relative_bias=relative_bias(estimates, v_true),
                rmse=rmse(estimates, v_true),
                coverage=coverage(intervals, v_true),
                mean_ci_width=mean_ci_width(intervals),
                mean_ess=float(np.mean([r.ess for r in per_rep])),
            )
        )

    return HarnessResult(
        v_true=v_true,
        v_true_stderr=v_true_stderr,
        n_rounds=n_rounds,
        n_replications=n_replications,
        summaries=tuple(summaries),
        replications=tuple(all_replications),
    )
