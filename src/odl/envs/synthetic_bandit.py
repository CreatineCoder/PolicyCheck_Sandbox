"""Synthetic contextual bandit with known ground truth (spec section 6.1).

The point of this environment is that ``mu(x, a) = E[r | x, a]`` is known by
construction, so the true value of any policy can be computed rather than
estimated. That is what makes it possible to check whether an off-policy
estimator is right -- on real logs the true value is precisely the unobservable
quantity, which is the premise of the whole project.

Reward model::

    z(x, a)  = w_a . x + b_a
    mu(x, a) = sigmoid( z + beta * sin(nonlinear_frequency * z) )
    r        ~ Bernoulli( mu(x, a) )

The ``beta`` coefficient controls how badly a *linear* reward model is
misspecified. At ``beta = 0`` a linear model is exactly correct and the Direct
Method looks excellent; as ``beta`` grows, DM acquires bias while doubly-robust
estimation degrades gracefully. That contrast is a headline result, so the
nonlinearity is a first-class configurable rather than a fixed detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odl.policies.base import Policy, sample_actions, validate_action_dist
from odl.types import BanditFeedback

__all__ = ["OracleEpsilonPolicy", "SyntheticBanditConfig", "SyntheticBanditEnv"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    decay = np.exp(-np.abs(z))
    stable = np.where(z >= 0, 1.0 / (1.0 + decay), decay / (1.0 + decay))
    return np.asarray(stable, dtype=np.float64)


@dataclass(frozen=True)
class SyntheticBanditConfig:
    """Configuration for :class:`SyntheticBanditEnv`.

    Attributes:
        d_context: context dimensionality ``d``.
        n_actions: size of the action space.
        beta: coefficient on the nonlinear term. Zero makes a linear reward
            model perfectly specified.
        nonlinear_frequency: frequency inside the ``sin`` term. At frequency 1
            the nonlinearity is nearly collinear with the linear term over the
            bulk of the context distribution, which makes the misspecification
            far weaker than it appears; a frequency above 1 makes it genuine.
        seed: seeds the environment parameters ``w_a`` and ``b_a``. Distinct
            from the seeds used to draw contexts and rewards, so the same
            environment can be sampled repeatedly across replications.
    """

    d_context: int = 10
    n_actions: int = 10
    beta: float = 1.0
    nonlinear_frequency: float = 3.0
    seed: int = 20260815

    def __post_init__(self) -> None:
        if self.d_context < 1:
            raise ValueError(f"d_context must be >= 1, got {self.d_context}")
        if self.n_actions < 2:
            raise ValueError(f"n_actions must be >= 2, got {self.n_actions}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.nonlinear_frequency <= 0.0:
            raise ValueError(
                f"nonlinear_frequency must be positive, got {self.nonlinear_frequency}"
            )


class SyntheticBanditEnv:
    """A contextual bandit whose expected rewards are known exactly.

    Contexts are drawn ``x ~ N(0, I_d)``. Per-action parameters are drawn once
    from the configured seed and then fixed, so two environments built from the
    same config are identical.
    """

    def __init__(self, config: SyntheticBanditConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        # Scale the weights by 1/sqrt(d) so that w_a . x stays O(1) as the
        # context dimension grows; otherwise the sigmoid saturates, every
        # expected reward collapses to 0 or 1, and the problem becomes trivial.
        self.weights = rng.normal(size=(config.n_actions, config.d_context)) / np.sqrt(
            config.d_context
        )
        self.bias = rng.normal(scale=0.5, size=config.n_actions)

    @property
    def n_actions(self) -> int:
        return self.config.n_actions

    @property
    def d_context(self) -> int:
        return self.config.d_context

    def sample_context(self, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_rounds`` contexts from ``N(0, I_d)``."""
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        return rng.normal(size=(n_rounds, self.d_context))

    def expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Return the exact ``(n, n_actions)`` matrix of ``mu(x, a)``.

        This is ground truth. It is exposed on synthetic feedback only, and any
        real-data code path that reads it is a critical bug.
        """
        if context.ndim != 2 or context.shape[1] != self.d_context:
            raise ValueError(
                f"context must have shape (n, {self.d_context}), got {context.shape}"
            )
        linear = context @ self.weights.T + self.bias
        nonlinear = np.sin(self.config.nonlinear_frequency * linear)
        return _sigmoid(linear + self.config.beta * nonlinear)

    def generate_feedback(
        self, logging_policy: Policy, n_rounds: int, rng: np.random.Generator
    ) -> BanditFeedback:
        """Generate logged bandit feedback under ``logging_policy``.

        The returned feedback carries ``expected_reward``, which is what makes
        it usable by the validation harness. Propensities are read off the same
        distribution that produced the actions, never recomputed.
        """
        if logging_policy.n_actions != self.n_actions:
            raise ValueError(
                f"logging policy has {logging_policy.n_actions} actions, "
                f"environment has {self.n_actions}"
            )
        context = self.sample_context(n_rounds, rng)
        dist = logging_policy.action_dist_checked(context)
        action, propensity = sample_actions(dist, rng)

        mu = self.expected_reward(context)
        reward = rng.binomial(1, mu[np.arange(n_rounds), action]).astype(np.float64)

        return BanditFeedback(
            context=context,
            action=action,
            reward=reward,
            propensity=propensity,
            n_actions=self.n_actions,
            expected_reward=mu,
        )

    def true_policy_value(
        self, policy: Policy, n_samples: int, rng: np.random.Generator
    ) -> float:
        """Return ``V(pi) = E_x[ sum_a pi(a|x) mu(x, a) ]``, the estimand.

        Computed on **fresh** contexts, never on the logged ones. Reusing logged
        contexts would correlate the "truth" with the estimate being checked and
        would make the harness quietly optimistic about coverage.

        The remaining Monte Carlo error is reported by
        :meth:`true_policy_value_with_stderr`; use that when the size of the
        error matters relative to the bias being measured.
        """
        value, _ = self.true_policy_value_with_stderr(policy, n_samples, rng)
        return value

    def true_policy_value_with_stderr(
        self, policy: Policy, n_samples: int, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Return ``(V(pi), standard_error)`` over ``n_samples`` fresh contexts.

        The standard error is the Monte Carlo error of the ground truth itself.
        It bounds how small a bias the harness can resolve: an estimator cannot
        be shown to be biased by less than the error in the number it is being
        compared against.
        """
        context = self.sample_context(n_samples, rng)
        dist = policy.action_dist_checked(context)
        mu = self.expected_reward(context)
        per_context_value = np.sum(dist * mu, axis=1)
        value = float(per_context_value.mean())
        stderr = float(per_context_value.std(ddof=1) / np.sqrt(n_samples))
        return value, stderr

    def make_logging_policy(self, epsilon: float, name: str | None = None) -> OracleEpsilonPolicy:
        """Build a logging policy whose overlap is controlled by ``epsilon``.

        ``epsilon = 1.0`` gives uniform logging and therefore perfect overlap
        with any target policy. Smaller values concentrate mass on the oracle's
        best action, degrading overlap until importance weighting breaks down.
        Sweeping this parameter is how the harness measures where the estimators
        stop working.
        """
        return OracleEpsilonPolicy(env=self, epsilon=epsilon, name=name)


class OracleEpsilonPolicy(Policy):
    """Greedy on the environment's true ``mu``, mixed with uniform at ``epsilon``.

    Lives in the ``envs`` module rather than ``policies`` deliberately: it reads
    ground truth, so it is a property of the synthetic environment and can never
    be applied to real logs. Keeping the ground-truth-reading policy out of the
    general policy package makes that boundary structural.
    """

    def __init__(
        self, env: SyntheticBanditEnv, epsilon: float, name: str | None = None
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")
        super().__init__(
            n_actions=env.n_actions, name=name if name is not None else f"oracle_eps{epsilon:g}"
        )
        self.env = env
        self.epsilon = epsilon

    def fit(self, feedback: BanditFeedback) -> None:
        """No-op: an oracle policy has nothing to learn from data."""

    def action_dist(self, context: np.ndarray) -> np.ndarray:
        """Return the epsilon-mixed greedy distribution over true ``mu(x, a)``."""
        mu = self.env.expected_reward(context)
        greedy = np.argmax(mu, axis=1)
        dist = np.full(
            (context.shape[0], self.n_actions), self.epsilon / self.n_actions, dtype=np.float64
        )
        dist[np.arange(context.shape[0]), greedy] += 1.0 - self.epsilon
        validate_action_dist(dist, self.n_actions, n_rounds=context.shape[0])
        return dist
