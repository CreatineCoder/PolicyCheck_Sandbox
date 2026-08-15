"""Tests for the synthetic bandit environment (spec sections 6.1, 12).

The environment is the source of ground truth for everything downstream, so its
own correctness is checked directly rather than inferred from the harness
passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from odl.envs.synthetic_bandit import SyntheticBanditConfig, SyntheticBanditEnv
from odl.policies.uniform import UniformPolicy

CONFIG = SyntheticBanditConfig(d_context=6, n_actions=5, beta=1.0, seed=7)


def make_env(**overrides: object) -> SyntheticBanditEnv:
    from dataclasses import replace

    return SyntheticBanditEnv(replace(CONFIG, **overrides))  # type: ignore[arg-type]


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("d_context", 0, "d_context must be >= 1"),
            ("n_actions", 1, "n_actions must be >= 2"),
            ("beta", -1.0, "beta must be non-negative"),
            ("nonlinear_frequency", 0.0, "nonlinear_frequency must be positive"),
        ],
    )
    def test_invalid_config_rejected(self, field: str, value: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            make_env(**{field: value})


class TestExpectedReward:
    def test_lies_strictly_inside_the_unit_interval(self) -> None:
        env = make_env()
        mu = env.expected_reward(env.sample_context(2000, np.random.default_rng(0)))
        assert mu.shape == (2000, env.n_actions)
        assert (mu > 0.0).all() and (mu < 1.0).all()

    def test_does_not_saturate_at_the_configured_dimension(self) -> None:
        # Weights are scaled by 1/sqrt(d) precisely so that the sigmoid stays in
        # its informative range as d grows. If this fails the problem is trivial.
        env = SyntheticBanditEnv(SyntheticBanditConfig(d_context=50, n_actions=10, seed=1))
        mu = env.expected_reward(env.sample_context(5000, np.random.default_rng(0)))
        assert 0.2 < float(mu.mean()) < 0.8
        assert float(mu.std()) > 0.05

    def test_beta_zero_is_exactly_a_linear_logit(self) -> None:
        env = make_env(beta=0.0)
        context = env.sample_context(100, np.random.default_rng(0))
        logits = context @ env.weights.T + env.bias
        np.testing.assert_allclose(env.expected_reward(context), 1.0 / (1.0 + np.exp(-logits)))

    def test_beta_changes_the_reward_surface(self) -> None:
        context = make_env().sample_context(500, np.random.default_rng(0))
        linear = make_env(beta=0.0).expected_reward(context)
        nonlinear = make_env(beta=2.0).expected_reward(context)
        assert float(np.abs(linear - nonlinear).mean()) > 0.05

    def test_wrong_context_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="context must have shape"):
            make_env().expected_reward(np.zeros((3, CONFIG.d_context + 1)))

    def test_same_config_gives_identical_parameters(self) -> None:
        np.testing.assert_array_equal(make_env().weights, make_env().weights)
        np.testing.assert_array_equal(make_env().bias, make_env().bias)


class TestGenerateFeedback:
    def test_populates_ground_truth_and_respects_the_contract(self) -> None:
        env = make_env()
        feedback = env.generate_feedback(
            UniformPolicy(env.n_actions), 500, np.random.default_rng(0)
        )
        assert feedback.n_rounds == 500
        assert feedback.has_ground_truth
        assert set(np.unique(feedback.reward).tolist()) <= {0.0, 1.0}
        np.testing.assert_allclose(feedback.propensity, 1.0 / env.n_actions)

    def test_logged_propensity_matches_the_logging_policy(self) -> None:
        env = make_env()
        logging_policy = env.make_logging_policy(epsilon=0.3)
        feedback = env.generate_feedback(logging_policy, 400, np.random.default_rng(1))
        dist = logging_policy.action_dist(feedback.context)
        expected = dist[np.arange(feedback.n_rounds), feedback.action]
        np.testing.assert_allclose(feedback.propensity, expected)

    def test_empirical_reward_rate_matches_mu_of_the_chosen_action(self) -> None:
        env = make_env()
        feedback = env.generate_feedback(
            UniformPolicy(env.n_actions), 200_000, np.random.default_rng(2)
        )
        truth = feedback.require_ground_truth("test")
        mu_chosen = truth[np.arange(feedback.n_rounds), feedback.action]
        assert abs(float(feedback.reward.mean() - mu_chosen.mean())) < 0.005

    def test_action_count_mismatch_raises(self) -> None:
        env = make_env()
        with pytest.raises(ValueError, match="logging policy has 3 actions"):
            env.generate_feedback(UniformPolicy(3), 10, np.random.default_rng(0))

    def test_generation_is_deterministic_given_the_seed(self) -> None:
        env = make_env()
        policy = UniformPolicy(env.n_actions)
        first = env.generate_feedback(policy, 100, np.random.default_rng(42))
        second = env.generate_feedback(policy, 100, np.random.default_rng(42))
        np.testing.assert_array_equal(first.action, second.action)
        np.testing.assert_array_equal(first.reward, second.reward)
        np.testing.assert_array_equal(first.context, second.context)


class TestTruePolicyValue:
    def test_uniform_policy_value_equals_the_mean_of_mu(self) -> None:
        env = make_env()
        policy = UniformPolicy(env.n_actions)
        rng = np.random.default_rng(0)
        value = env.true_policy_value(policy, 200_000, rng)
        reference = float(
            env.expected_reward(env.sample_context(200_000, np.random.default_rng(1))).mean()
        )
        assert abs(value - reference) < 0.005

    def test_oracle_policy_beats_uniform(self) -> None:
        env = make_env()
        rng = np.random.default_rng(0)
        uniform_value = env.true_policy_value(UniformPolicy(env.n_actions), 100_000, rng)
        oracle_value = env.true_policy_value(env.make_logging_policy(0.0), 100_000, rng)
        assert oracle_value > uniform_value + 0.05

    def test_stderr_shrinks_with_sample_size(self) -> None:
        env = make_env()
        policy = UniformPolicy(env.n_actions)
        _, small = env.true_policy_value_with_stderr(policy, 1_000, np.random.default_rng(0))
        _, large = env.true_policy_value_with_stderr(policy, 100_000, np.random.default_rng(0))
        assert large < small / 5.0

    def test_uses_fresh_contexts_not_the_logged_ones(self) -> None:
        # Two calls with different generators must disagree at Monte Carlo
        # scale; identical answers would mean contexts are being reused.
        env = make_env()
        policy = UniformPolicy(env.n_actions)
        first = env.true_policy_value(policy, 5_000, np.random.default_rng(0))
        second = env.true_policy_value(policy, 5_000, np.random.default_rng(1))
        assert first != second
        assert abs(first - second) < 0.02


class TestOracleEpsilonPolicy:
    def test_epsilon_one_is_uniform(self) -> None:
        env = make_env()
        dist = env.make_logging_policy(1.0).action_dist(
            env.sample_context(20, np.random.default_rng(0))
        )
        np.testing.assert_allclose(dist, 1.0 / env.n_actions)

    def test_epsilon_zero_puts_all_mass_on_the_best_action(self) -> None:
        env = make_env()
        context = env.sample_context(50, np.random.default_rng(0))
        dist = env.make_logging_policy(0.0).action_dist(context)
        np.testing.assert_array_equal(
            np.argmax(dist, axis=1), np.argmax(env.expected_reward(context), axis=1)
        )
        np.testing.assert_allclose(dist.max(axis=1), 1.0)

    def test_smaller_epsilon_gives_smaller_minimum_propensity(self) -> None:
        # This is the overlap knob: as epsilon falls, the rarest action's
        # logging probability falls with it and importance weights grow.
        env = make_env()
        context = env.sample_context(100, np.random.default_rng(0))
        wide = env.make_logging_policy(0.8).action_dist(context).min()
        narrow = env.make_logging_policy(0.05).action_dist(context).min()
        assert narrow < wide

    def test_invalid_epsilon_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"epsilon must lie in \[0, 1\]"):
            make_env().make_logging_policy(-0.1)
