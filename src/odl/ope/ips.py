"""Inverse propensity scoring and its self-normalised variant (spec section 6.3).

IPS is unbiased whenever the logging policy has support wherever the target
policy does. Its variance, however, is governed by the importance weights, and
those are unbounded: a single logged record with a small propensity and a large
target probability can dominate the entire estimate. Reporting ESS alongside
every IPS number is therefore mandatory in this project, not decorative.

References:
    Horvitz and Thompson (1952), the origin of the estimator.
    Swaminathan and Joachims (2015), on self-normalisation and the propensity
    overfitting it corrects.
"""

from __future__ import annotations

import numpy as np

from odl.ope.base import Estimator, effective_sample_size, importance_weights
from odl.types import BanditFeedback

__all__ = ["IPSEstimator", "SNIPSEstimator"]


class IPSEstimator(Estimator):
    """``V_hat = mean( w_i * r_i )`` with ``w_i = pi_e(a_i|x_i) / p_i``.

    Unbiased under overlap and unclipped. With ``clip_tau`` set, the estimator
    becomes deliberately biased in exchange for bounded variance; the bias is
    always downward toward the logging policy's value, since clipping can only
    reduce a weight.
    """

    def __init__(self, clip_tau: float | None = None, name: str | None = None) -> None:
        if clip_tau is not None and clip_tau <= 0.0:
            raise ValueError(f"clip_tau must be positive, got {clip_tau}")
        self.clip_tau = clip_tau
        default = "ips" if clip_tau is None else f"ips_clip{clip_tau:g}"
        self.name = name if name is not None else default

    def per_sample_values(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,
        reward_model_preds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``w_i * r_i``. ``reward_model_preds`` is unused by IPS."""
        weights = importance_weights(feedback, action_dist, clip_tau=self.clip_tau)
        return np.asarray(weights * feedback.reward, dtype=np.float64)

    def effective_sample_size(
        self, feedback: BanditFeedback, action_dist: np.ndarray
    ) -> float:
        """Return the ESS of this estimate's importance weights.

        Reported alongside every IPS estimate. An ESS far below ``n`` means the
        confidence interval is being computed from far fewer effective records
        than the row count suggests.
        """
        return effective_sample_size(importance_weights(feedback, action_dist, self.clip_tau))


class SNIPSEstimator(Estimator):
    """``V_hat = sum(w_i r_i) / sum(w_i)``, the self-normalised estimator.

    Biased at finite ``n``, but consistent, and with far lower variance than
    plain IPS. The normalisation divides out the random fluctuation in the total
    weight, which is the dominant error term when a handful of records carry
    most of the mass.

    Because the estimate is a ratio rather than a mean, its confidence interval
    must be built by resampling *records* and recomputing the ratio, not by
    resampling per-record values and averaging them. The bootstrap in
    :mod:`odl.ope.bootstrap` resamples indices for exactly this reason.
    """

    def __init__(self, clip_tau: float | None = None, name: str | None = None) -> None:
        if clip_tau is not None and clip_tau <= 0.0:
            raise ValueError(f"clip_tau must be positive, got {clip_tau}")
        self.clip_tau = clip_tau
        default = "snips" if clip_tau is None else f"snips_clip{clip_tau:g}"
        self.name = name if name is not None else default

    def per_sample_values(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,
        reward_model_preds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``n * w_i * r_i / sum(w)``, whose mean is the SNIPS estimate.

        Provided for diagnostics and for interface symmetry. These values are
        *not* independent across records -- every one of them depends on the
        whole sample through the denominator -- so averaging a resampled subset
        of them does not give a valid bootstrap replicate.
        """
        weights = importance_weights(feedback, action_dist, clip_tau=self.clip_tau)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(
                "importance weights sum to zero; SNIPS is undefined because the "
                "target policy assigns no probability to any logged action"
            )
        scaled = weights * feedback.reward * (feedback.n_rounds / total)
        return np.asarray(scaled, dtype=np.float64)

    def estimate(
        self,
        feedback: BanditFeedback,
        action_dist: np.ndarray,
        reward_model_preds: np.ndarray | None = None,
    ) -> float:
        """Return ``sum(w_i r_i) / sum(w_i)``."""
        weights = importance_weights(feedback, action_dist, clip_tau=self.clip_tau)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(
                "importance weights sum to zero; SNIPS is undefined because the "
                "target policy assigns no probability to any logged action"
            )
        return float(np.dot(weights, feedback.reward) / total)

    def effective_sample_size(
        self, feedback: BanditFeedback, action_dist: np.ndarray
    ) -> float:
        """Return the ESS of this estimate's importance weights."""
        return effective_sample_size(importance_weights(feedback, action_dist, self.clip_tau))
