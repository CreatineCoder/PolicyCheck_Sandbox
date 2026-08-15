"""Bootstrap confidence intervals (spec section 6.3).

Every reported number in this project carries an interval. There are no bare
point estimates anywhere in the repository, because a point estimate from a
handful of effective samples looks exactly like a point estimate from a million
of them, and the difference is the entire question.

Method: nonparametric percentile bootstrap. Records are resampled with
replacement ``n_resamples`` times and the estimator is recomputed on each
resample; the interval is the empirical 2.5th and 97.5th percentile of those
replicates.

The spec describes this as resampling the per-sample values. For mean-type
estimators such as IPS the two are identical. For ratio-type estimators such as
SNIPS they are not: the per-record values share a denominator computed from the
whole sample, so averaging a resampled subset of them is not a valid replicate.
Resampling *indices* and recomputing the estimator is the generalisation that is
correct for both, so that is what is implemented here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from odl.ope.base import Estimator
from odl.types import BanditFeedback

__all__ = ["ConfidenceInterval", "bootstrap_ci", "bootstrap_estimator"]


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate together with its bootstrap interval.

    Attributes:
        point: the estimate computed on the observed sample.
        lower: lower percentile bound.
        upper: upper percentile bound.
        alpha: total tail mass; ``0.05`` gives a 95% interval.
        n_resamples: number of bootstrap replicates used.
    """

    point: float
    lower: float
    upper: float
    alpha: float
    n_resamples: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if self.lower > self.upper:
            raise ValueError(f"lower bound {self.lower} exceeds upper bound {self.upper}")

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower

    def covers(self, value: float) -> bool:
        """Whether the interval contains ``value``.

        The harness calls this against the known true policy value to measure
        coverage, which is the sharpest available diagnostic of whether an
        estimator's stated uncertainty is honest.
        """
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        confidence = 100.0 * (1.0 - self.alpha)
        return f"{self.point:.6f} [{self.lower:.6f}, {self.upper:.6f}] ({confidence:.0f}% CI)"


def bootstrap_ci(
    estimate_fn: Callable[[np.ndarray], float],
    n_rounds: int,
    rng: np.random.Generator,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for an arbitrary index-based estimator.

    Args:
        estimate_fn: maps an integer index array of length ``n_rounds`` to a
            scalar estimate. Called once on ``arange(n_rounds)`` for the point
            estimate and once per resample.
        n_rounds: number of logged records.
        rng: generator supplying the resampling indices. Determinism of the
            whole interval follows from the state of this generator.
        n_resamples: number of bootstrap replicates, B.
        alpha: total tail mass.
    """
    if n_rounds < 2:
        raise ValueError(f"bootstrap requires at least 2 records, got {n_rounds}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be >= 2, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    point = estimate_fn(np.arange(n_rounds, dtype=np.int64))

    replicates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        index = rng.integers(0, n_rounds, size=n_rounds, dtype=np.int64)
        replicates[b] = estimate_fn(index)

    if not np.isfinite(replicates).all():
        raise ValueError(
            "bootstrap produced non-finite replicates; the estimator is undefined "
            "on at least one resample, which usually means the target policy has "
            "near-zero overlap with the logs"
        )

    lower, upper = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return ConfidenceInterval(
        point=float(point),
        lower=float(lower),
        upper=float(upper),
        alpha=alpha,
        n_resamples=n_resamples,
    )


def bootstrap_estimator(
    estimator: Estimator,
    feedback: BanditFeedback,
    action_dist: np.ndarray,
    rng: np.random.Generator,
    reward_model_preds: np.ndarray | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Bootstrap interval for an :class:`~odl.ope.base.Estimator`.

    Resamples logged records, carrying the target action distribution and any
    reward-model predictions along with them so that every replicate stays
    internally consistent.
    """
    def estimate_fn(index: np.ndarray) -> float:
        return estimator.estimate(
            feedback.select(index),
            action_dist[index],
            None if reward_model_preds is None else reward_model_preds[index],
        )

    return bootstrap_ci(
        estimate_fn,
        n_rounds=feedback.n_rounds,
        rng=rng,
        n_resamples=n_resamples,
        alpha=alpha,
    )
