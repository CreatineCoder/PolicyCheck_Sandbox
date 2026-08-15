"""Off-policy estimator interface and shared weight machinery (spec section 6.3).

The estimand throughout is ``V(pi_e) = E[r]`` under the target policy.

Everything in this module operates on logged feedback plus a target action
distribution. Nothing here reads ``expected_reward``: estimators must work
identically on synthetic and real data, and the only way to guarantee that is
for them never to have access to the truth in the first place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from odl.policies.base import validate_action_dist
from odl.types import BanditFeedback

__all__ = [
    "Estimator",
    "effective_sample_size",
    "importance_weights",
    "target_propensity",
]


def target_propensity(action_dist: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return ``pi_e(a_i|x_i)``, the target probability of each *logged* action."""
    return np.asarray(action_dist[np.arange(action.shape[0]), action], dtype=np.float64)


def importance_weights(
    feedback: BanditFeedback, action_dist: np.ndarray, clip_tau: float | None = None
) -> np.ndarray:
    """Return ``w_i = pi_e(a_i|x_i) / p_i``, optionally clipped above at ``clip_tau``.

    Clipping trades variance for bias: it caps the influence of any single
    logged record, at the cost of systematically shrinking the estimate toward
    the logging policy's value. It is a deliberate choice, so it is never
    applied by default -- ``clip_tau=None`` means no clipping.

    Args:
        feedback: logged bandit data supplying ``a_i`` and ``p_i``.
        action_dist: ``(n, n_actions)`` target distribution ``pi_e(a|x)``.
        clip_tau: upper bound on the weights, or ``None`` to leave them raw.
    """
    validate_action_dist(action_dist, feedback.n_actions, n_rounds=feedback.n_rounds)
    weights = target_propensity(action_dist, feedback.action) / feedback.propensity

    if clip_tau is not None:
        if clip_tau <= 0.0:
            raise ValueError(f"clip_tau must be positive, got {clip_tau}")
        weights = np.minimum(weights, clip_tau)
    return np.asarray(weights, dtype=np.float64)


def effective_sample_size(weights: np.ndarray) -> float:
    """Return ``ESS = (sum w)^2 / sum(w^2)``.

    The single best diagnostic for whether an importance-weighted estimate can
    be trusted. It is the number of equally-weighted samples that would carry
    the same information: with ``n = 10000`` and ``ESS = 12``, the estimate
    rests on twelve records regardless of how many were logged, and its
    confidence interval should be read with that in mind.

    Equals ``n`` exactly when all weights are equal, and approaches 1 as the
    weight distribution concentrates on a single record.
    """
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "importance weights sum to zero, so ESS is undefined; the target "
            "policy assigns no probability to any logged action, which means "
            "the logs carry no information about it"
        )
    return total**2 / float(np.square(weights).sum())


class Estimator(ABC):
    """Abstract off-policy estimator of ``V(pi_e) = E[r]``.

    Subclasses implement :meth:`per_sample_values`, returning one contribution
    per logged record. The default :meth:`estimate` averages them, which is
    correct for every estimator whose value is a plain sample mean. Ratio-type
    estimators such as SNIPS override :meth:`estimate`.
    """

    name: str

    @abstractmethod
    def per_sample_values(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,
        reward_model_preds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the ``(n,)`` per-record contributions to the estimate."""

    def estimate(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,
        reward_model_preds: np.ndarray | None = None,
    ) -> float:
        """Return the point estimate ``V_hat``."""
        return float(
            self.per_sample_values(feedback, action_dist, reward_model_preds).mean()
        )

    @staticmethod
    def _check_reward_model_preds(
        feedback: BanditFeedback, reward_model_preds: np.ndarray | None
    ) -> np.ndarray:
        """Validate and return required reward-model predictions.

        Raises when they are absent. Estimators that need ``r_hat`` do not
        substitute zeros or a mean when it is missing -- a Direct Method quietly
        run against an all-zero reward model returns 0.0, which is a plausible
        looking number and a completely wrong one.
        """
        if reward_model_preds is None:
            raise ValueError(
                "this estimator requires reward_model_preds of shape "
                f"({feedback.n_rounds}, {feedback.n_actions}), but none were supplied"
            )
        if reward_model_preds.shape != (feedback.n_rounds, feedback.n_actions):
            raise ValueError(
                f"reward_model_preds must have shape "
                f"{(feedback.n_rounds, feedback.n_actions)}, got {reward_model_preds.shape}"
            )
        if not np.isfinite(reward_model_preds).all():
            raise ValueError("reward_model_preds contains NaN or infinite values")
        return reward_model_preds.astype(np.float64, copy=False)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
