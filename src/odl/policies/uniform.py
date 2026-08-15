"""Uniform-random policy (spec section 6.2).

The reference baseline, and the logging policy under which off-policy
estimation is easiest: uniform logging gives perfect overlap with any target
policy, so an estimator that cannot recover truth here is broken rather than
merely noisy.
"""

from __future__ import annotations

import numpy as np

from odl.policies.base import Policy
from odl.types import BanditFeedback

__all__ = ["UniformPolicy"]


class UniformPolicy(Policy):
    """``pi(a|x) = 1 / n_actions`` for every context.

    Context-independent by construction, so :meth:`fit` is a documented no-op
    rather than an oversight.
    """

    def __init__(self, n_actions: int, name: str = "uniform") -> None:
        super().__init__(n_actions=n_actions, name=name)

    def fit(self, feedback: BanditFeedback) -> None:
        """No-op: a uniform policy has no parameters to learn."""

    def action_dist(self, context: np.ndarray) -> np.ndarray:
        """Return an ``(n, n_actions)`` matrix of ``1 / n_actions``."""
        if context.ndim != 2:
            raise ValueError(f"context must be 2-dimensional, got shape {context.shape}")
        return np.full(
            (context.shape[0], self.n_actions), 1.0 / self.n_actions, dtype=np.float64
        )
