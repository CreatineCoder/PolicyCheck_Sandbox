"""Policy interface (spec section 6.2).

Every policy exposes a full action *distribution* rather than an argmax. This is
not stylistic: off-policy estimators need ``pi_e(a|x)`` evaluated at the logged
action ``a_i``, which an argmax cannot provide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from odl.types import BanditFeedback

__all__ = ["Policy", "sample_actions", "validate_action_dist"]

# Rows of an action distribution must sum to 1 within this tolerance. Tight
# enough to catch a genuine normalisation bug, loose enough to tolerate the
# float64 error of a softmax over a few hundred actions.
_SIMPLEX_TOLERANCE = 1e-9


def validate_action_dist(
    action_dist: np.ndarray, n_actions: int, n_rounds: int | None = None
) -> None:
    """Raise unless ``action_dist`` is a valid ``(n, n_actions)`` row-stochastic matrix.

    A distribution that does not sum to 1, or that assigns negative mass, makes
    every downstream importance weight meaningless. Checked at every boundary
    rather than trusted.
    """
    if not isinstance(action_dist, np.ndarray):
        raise ValueError(f"action_dist must be a numpy.ndarray, got {type(action_dist)!r}")
    if action_dist.ndim != 2:
        raise ValueError(f"action_dist must be 2-dimensional, got shape {action_dist.shape}")
    if action_dist.shape[1] != n_actions:
        raise ValueError(
            f"action_dist has {action_dist.shape[1]} columns but n_actions is {n_actions}"
        )
    if n_rounds is not None and action_dist.shape[0] != n_rounds:
        raise ValueError(
            f"action_dist has {action_dist.shape[0]} rows but the feedback has {n_rounds}"
        )
    if not np.isfinite(action_dist).all():
        raise ValueError("action_dist contains NaN or infinite values")
    if (action_dist < 0.0).any():
        raise ValueError("action_dist contains negative probabilities")

    row_sums = action_dist.sum(axis=1)
    worst = float(np.abs(row_sums - 1.0).max())
    if worst > _SIMPLEX_TOLERANCE:
        raise ValueError(
            f"action_dist rows must sum to 1; worst deviation is {worst:.3e} "
            f"(tolerance {_SIMPLEX_TOLERANCE:.0e})"
        )


def sample_actions(
    action_dist: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one action per row and return ``(actions, propensities)``.

    The propensity returned is the probability the sampling distribution
    actually assigned to the drawn action -- it is read off the same matrix that
    produced the draw, never recomputed later. Recomputing is how logged
    propensities silently drift out of sync with the actions they describe.

    Vectorised inverse-CDF sampling, so results depend only on ``rng`` state and
    not on the number of actions or any per-row Python iteration.
    """
    n_rounds, n_actions = action_dist.shape
    cumulative = np.cumsum(action_dist, axis=1)
    # Guard the final edge against float64 round-off, so a draw of exactly 1.0
    # cannot fall past the last bin and produce an out-of-range action.
    cumulative[:, -1] = 1.0
    draws = np.asarray(rng.random(n_rounds), dtype=np.float64)
    actions = np.argmax(cumulative > draws[:, None], axis=1).astype(np.int64)
    propensities = np.asarray(action_dist[np.arange(n_rounds), actions], dtype=np.float64)

    if not (propensities > 0.0).all():
        raise ValueError(
            "sampled an action with zero propensity; this indicates a malformed "
            "action distribution, since a zero-probability action can never be drawn"
        )
    if not (actions < n_actions).all():
        raise ValueError("sampled an out-of-range action")
    return actions, propensities


class Policy(ABC):
    """Abstract contextual policy ``pi(a|x)``.

    Subclasses implement :meth:`fit` (train from logged feedback) and
    :meth:`action_dist` (return the full conditional distribution).
    """

    def __init__(self, n_actions: int, name: str) -> None:
        if n_actions < 2:
            raise ValueError(f"n_actions must be >= 2, got {n_actions}")
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def fit(self, feedback: BanditFeedback) -> None:
        """Train the policy from logged bandit feedback.

        Policies that require no training implement this as a no-op, but must
        still accept the call so that callers need not special-case them.
        """

    @abstractmethod
    def action_dist(self, context: np.ndarray) -> np.ndarray:
        """Return ``pi(a|x)`` as an ``(n, n_actions)`` row-stochastic matrix."""

    def action_dist_checked(self, context: np.ndarray) -> np.ndarray:
        """:meth:`action_dist` with the simplex contract enforced.

        Callers that feed the result straight into an estimator should use this
        rather than trusting a subclass to normalise correctly.
        """
        dist = self.action_dist(context)
        validate_action_dist(dist, self.n_actions, n_rounds=context.shape[0])
        return dist

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, n_actions={self.n_actions})"
