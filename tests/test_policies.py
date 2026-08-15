"""Tests for the policy interface and concrete policies (spec sections 6.2, 12)."""

from __future__ import annotations

import numpy as np
import pytest

from odl.policies.base import sample_actions, validate_action_dist
from odl.policies.epsilon_greedy import EpsilonGreedyPolicy, RidgeRewardModel
from odl.policies.uniform import UniformPolicy
from odl.types import BanditFeedback

N_ACTIONS = 5
D_CONTEXT = 4


def make_logged_feedback(n: int = 500, seed: int = 0) -> BanditFeedback:
    """Uniformly logged feedback with a context-dependent reward signal."""
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n, D_CONTEXT))
    action = rng.integers(0, N_ACTIONS, size=n).astype(np.int64)
    # Action a is good when context dimension (a mod D_CONTEXT) is large, so a
    # linear model has real signal to find.
    signal = context[np.arange(n), action % D_CONTEXT]
    reward = (rng.random(n) < 1.0 / (1.0 + np.exp(-signal))).astype(np.float64)
    return BanditFeedback(
        context=context,
        action=action,
        reward=reward,
        propensity=np.full(n, 1.0 / N_ACTIONS),
        n_actions=N_ACTIONS,
    )


class TestValidateActionDist:
    def test_accepts_a_valid_distribution(self) -> None:
        validate_action_dist(np.full((3, N_ACTIONS), 1.0 / N_ACTIONS), N_ACTIONS)

    def test_rejects_rows_that_do_not_sum_to_one(self) -> None:
        dist = np.full((3, N_ACTIONS), 1.0 / N_ACTIONS)
        dist[1, 0] += 0.01
        with pytest.raises(ValueError, match="rows must sum to 1"):
            validate_action_dist(dist, N_ACTIONS)

    def test_rejects_negative_probabilities(self) -> None:
        dist = np.full((2, N_ACTIONS), 1.0 / N_ACTIONS)
        dist[0, 0] = -0.1
        dist[0, 1] += 0.1
        with pytest.raises(ValueError, match="negative probabilities"):
            validate_action_dist(dist, N_ACTIONS)

    def test_rejects_wrong_number_of_columns(self) -> None:
        with pytest.raises(ValueError, match="columns but n_actions"):
            validate_action_dist(np.full((2, 3), 1.0 / 3.0), N_ACTIONS)

    def test_rejects_row_count_mismatch(self) -> None:
        dist = np.full((3, N_ACTIONS), 1.0 / N_ACTIONS)
        with pytest.raises(ValueError, match="rows but the feedback has 7"):
            validate_action_dist(dist, N_ACTIONS, n_rounds=7)


class TestSampleActions:
    def test_propensities_match_the_distribution_that_produced_them(self) -> None:
        rng = np.random.default_rng(0)
        dist = rng.dirichlet(np.ones(N_ACTIONS), size=200)
        actions, propensities = sample_actions(dist, rng)
        np.testing.assert_allclose(propensities, dist[np.arange(200), actions])

    def test_deterministic_distribution_always_draws_its_action(self) -> None:
        dist = np.zeros((50, N_ACTIONS))
        dist[:, 2] = 1.0
        actions, propensities = sample_actions(dist, np.random.default_rng(1))
        assert set(actions.tolist()) == {2}
        np.testing.assert_allclose(propensities, 1.0)

    def test_empirical_frequencies_match_the_distribution(self) -> None:
        probabilities = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        dist = np.tile(probabilities, (200_000, 1))
        actions, _ = sample_actions(dist, np.random.default_rng(2))
        observed = np.bincount(actions, minlength=N_ACTIONS) / len(actions)
        np.testing.assert_allclose(observed, probabilities, atol=0.005)

    def test_actions_stay_in_range_under_float_roundoff(self) -> None:
        # A distribution whose cumulative sum lands fractionally below 1.0 is
        # the case that can push a draw past the final bin.
        dist = np.full((1000, 3), 1.0 / 3.0)
        actions, _ = sample_actions(dist, np.random.default_rng(3))
        assert actions.max() < 3


class TestUniformPolicy:
    def test_rows_sum_to_one(self) -> None:
        policy = UniformPolicy(n_actions=N_ACTIONS)
        dist = policy.action_dist_checked(np.zeros((10, D_CONTEXT)))
        np.testing.assert_allclose(dist.sum(axis=1), 1.0)
        np.testing.assert_allclose(dist, 1.0 / N_ACTIONS)

    def test_fit_is_a_no_op_and_does_not_change_behaviour(self) -> None:
        policy = UniformPolicy(n_actions=N_ACTIONS)
        context = np.zeros((4, D_CONTEXT))
        before = policy.action_dist(context)
        policy.fit(make_logged_feedback())
        np.testing.assert_array_equal(policy.action_dist(context), before)


class TestRidgeRewardModel:
    def test_recovers_a_known_linear_function(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        context = rng.normal(size=(n, 2))
        action = rng.integers(0, 2, size=n).astype(np.int64)
        # Action 0: r = 2*x0; action 1: r = -x1 + 1
        reward = np.where(action == 0, 2.0 * context[:, 0], -context[:, 1] + 1.0)

        model = RidgeRewardModel(n_actions=2, ridge_lambda=1e-6)
        model.fit(context, action, reward)
        predictions = model.predict(np.array([[1.0, 1.0], [0.0, 2.0]]))

        np.testing.assert_allclose(predictions[:, 0], [2.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(predictions[:, 1], [0.0, -1.0], atol=1e-3)

    def test_unobserved_action_raises_rather_than_defaulting(self) -> None:
        rng = np.random.default_rng(0)
        context = rng.normal(size=(50, 2))
        action = np.zeros(50, dtype=np.int64)  # action 1 never logged
        model = RidgeRewardModel(n_actions=2)
        with pytest.raises(ValueError, match="action 1 has no logged samples"):
            model.fit(context, action, rng.normal(size=50))

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="called before fit"):
            RidgeRewardModel(n_actions=2).predict(np.zeros((1, 2)))

    def test_zero_ridge_lambda_rejected(self) -> None:
        with pytest.raises(ValueError, match="ridge_lambda must be positive"):
            RidgeRewardModel(n_actions=2, ridge_lambda=0.0)


class TestEpsilonGreedyPolicy:
    def test_rows_sum_to_one(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.2)
        feedback = make_logged_feedback()
        policy.fit(feedback)
        dist = policy.action_dist_checked(feedback.context)
        np.testing.assert_allclose(dist.sum(axis=1), 1.0)

    def test_epsilon_one_reduces_to_uniform(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=1.0)
        feedback = make_logged_feedback()
        policy.fit(feedback)
        np.testing.assert_allclose(
            policy.action_dist(feedback.context), 1.0 / N_ACTIONS
        )

    def test_epsilon_zero_is_deterministic(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.0)
        feedback = make_logged_feedback()
        policy.fit(feedback)
        dist = policy.action_dist(feedback.context)
        assert set(np.unique(dist).tolist()) <= {0.0, 1.0}

    def test_greedy_action_carries_the_largest_probability(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.3)
        feedback = make_logged_feedback()
        policy.fit(feedback)
        dist = policy.action_dist(feedback.context)
        predictions = policy.reward_model.predict(feedback.context)
        np.testing.assert_array_equal(np.argmax(dist, axis=1), np.argmax(predictions, axis=1))

    def test_every_action_keeps_positive_support_when_epsilon_positive(self) -> None:
        # This is the property that makes epsilon-greedy usable as a logging
        # policy: no action can have zero propensity.
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.05)
        feedback = make_logged_feedback()
        policy.fit(feedback)
        assert (policy.action_dist(feedback.context) > 0.0).all()

    def test_action_dist_before_fit_raises(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.1)
        with pytest.raises(ValueError, match="called before fit"):
            policy.action_dist(np.zeros((2, D_CONTEXT)))

    def test_invalid_epsilon_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"epsilon must lie in \[0, 1\]"):
            EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=1.5)

    def test_action_count_mismatch_rejected(self) -> None:
        policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS + 1, epsilon=0.1)
        with pytest.raises(ValueError, match="feedback has 5 actions"):
            policy.fit(make_logged_feedback())

    def test_fit_is_deterministic(self) -> None:
        feedback = make_logged_feedback()
        dists = []
        for _ in range(2):
            policy = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.2)
            policy.fit(feedback)
            dists.append(policy.action_dist(feedback.context))
        np.testing.assert_array_equal(dists[0], dists[1])
