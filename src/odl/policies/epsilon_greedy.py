"""Epsilon-greedy policy over a fitted reward model (spec section 6.2).

Fits ``r_hat(x, a)`` from logged feedback, then plays the argmax action with
probability ``1 - epsilon`` and mixes in uniform exploration at rate
``epsilon``. The uniform component is what keeps every action's probability
bounded away from zero, which is exactly what makes the resulting policy usable
as a *logging* policy: ``epsilon`` is the overlap knob the validation harness
sweeps.
"""

from __future__ import annotations

import numpy as np

from odl.policies.base import Policy
from odl.types import BanditFeedback

__all__ = ["EpsilonGreedyPolicy", "RidgeRewardModel"]


class RidgeRewardModel:
    """Per-action ridge regression ``r_hat(x, a) = theta_a . [x, 1]``.

    One independent linear model per action, fitted by the closed-form ridge
    solution. Chosen over an iterative learner for Phase 1 because it is exactly
    deterministic given the data -- reproducibility is a hard requirement, and a
    reward model with its own random seed is one more thing to control.

    Being linear, this model is *correctly specified* on the synthetic
    environment only when the nonlinearity coefficient ``beta`` is zero. That is
    deliberate: the gap between correct and misspecified is the contrast the
    Direct Method and doubly-robust comparison is built on.
    """

    def __init__(self, n_actions: int, ridge_lambda: float = 1.0) -> None:
        if ridge_lambda <= 0.0:
            raise ValueError(
                f"ridge_lambda must be positive, got {ridge_lambda}; a zero penalty "
                "makes the solution undefined for actions with rank-deficient data"
            )
        self.n_actions = n_actions
        self.ridge_lambda = ridge_lambda
        self._coefficients: np.ndarray | None = None

    @staticmethod
    def _design(context: np.ndarray) -> np.ndarray:
        """Append an intercept column to the context matrix."""
        return np.hstack([context, np.ones((context.shape[0], 1), dtype=np.float64)])

    def fit(self, context: np.ndarray, action: np.ndarray, reward: np.ndarray) -> None:
        """Fit one ridge model per action.

        Raises if any action has no logged data. An action the logs never took
        cannot have its reward estimated, and returning the ridge prior of zero
        for it would be a silent fallback that quietly biases every downstream
        value estimate.
        """
        design = self._design(context)
        d = design.shape[1]
        coefficients = np.zeros((self.n_actions, d), dtype=np.float64)
        penalty = self.ridge_lambda * np.eye(d, dtype=np.float64)

        for a in range(self.n_actions):
            mask = action == a
            count = int(mask.sum())
            if count == 0:
                raise ValueError(
                    f"action {a} has no logged samples, so r_hat(x, {a}) is unidentified. "
                    "Log with a policy that retains support over all actions, or reduce "
                    "the action space -- do not fall back to a default prediction."
                )
            design_a = design[mask]
            gram = design_a.T @ design_a + penalty
            coefficients[a] = np.linalg.solve(gram, design_a.T @ reward[mask])

        self._coefficients = coefficients

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Return the ``(n, n_actions)`` matrix of predicted rewards."""
        if self._coefficients is None:
            raise ValueError("RidgeRewardModel.predict called before fit")
        return np.asarray(self._design(context) @ self._coefficients.T, dtype=np.float64)


class EpsilonGreedyPolicy(Policy):
    """Greedy on a fitted reward model, mixed with uniform at rate ``epsilon``.

    ``epsilon = 1`` reduces to :class:`~odl.policies.uniform.UniformPolicy`;
    ``epsilon`` near zero approaches a deterministic policy, under which
    importance weights become extreme and IPS variance explodes. Both ends of
    that range are swept by the validation harness on purpose.
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float,
        ridge_lambda: float = 1.0,
        name: str = "epsilon_greedy",
    ) -> None:
        super().__init__(n_actions=n_actions, name=name)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
        self.epsilon = epsilon
        self.reward_model = RidgeRewardModel(n_actions=n_actions, ridge_lambda=ridge_lambda)
        self._is_fitted = False

    def fit(self, feedback: BanditFeedback) -> None:
        """Fit the reward model on logged ``(x, a, r)`` triples.

        Only the logged action's reward is observed, which is the whole
        difficulty: the model must extrapolate to actions not taken. Nothing
        here touches ``expected_reward``, so this is safe on real logs.
        """
        if feedback.n_actions != self.n_actions:
            raise ValueError(
                f"feedback has {feedback.n_actions} actions but this policy has {self.n_actions}"
            )
        self.reward_model.fit(feedback.context, feedback.action, feedback.reward)
        self._is_fitted = True

    def action_dist(self, context: np.ndarray) -> np.ndarray:
        """Return ``(1 - epsilon) * onehot(argmax_a r_hat) + epsilon / n_actions``.

        Ties in the argmax resolve to the lowest action index, which keeps the
        policy deterministic given the same reward model.
        """
        if not self._is_fitted:
            raise ValueError("EpsilonGreedyPolicy.action_dist called before fit")

        predictions = self.reward_model.predict(context)
        greedy = np.argmax(predictions, axis=1)

        dist = np.full(
            (context.shape[0], self.n_actions),
            self.epsilon / self.n_actions,
            dtype=np.float64,
        )
        dist[np.arange(context.shape[0]), greedy] += 1.0 - self.epsilon
        return dist
