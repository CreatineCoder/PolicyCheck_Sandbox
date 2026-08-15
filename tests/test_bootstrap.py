"""Tests for the bootstrap confidence-interval machinery (spec section 6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from odl.ope.bootstrap import ConfidenceInterval, bootstrap_ci, bootstrap_estimator
from odl.ope.ips import IPSEstimator, SNIPSEstimator
from odl.types import BanditFeedback
from tests.test_ope_ips import TOY_ACTION_DIST, toy_feedback


def make_feedback(n: int = 2000, seed: int = 0, p_reward: float = 0.3) -> BanditFeedback:
    rng = np.random.default_rng(seed)
    n_actions = 4
    return BanditFeedback(
        context=rng.normal(size=(n, 3)),
        action=rng.integers(0, n_actions, size=n).astype(np.int64),
        reward=rng.binomial(1, p_reward, size=n).astype(np.float64),
        propensity=np.full(n, 1.0 / n_actions),
        n_actions=n_actions,
    )


class TestConfidenceInterval:
    def test_covers_reports_containment(self) -> None:
        interval = ConfidenceInterval(point=0.5, lower=0.4, upper=0.6, alpha=0.05, n_resamples=100)
        assert interval.covers(0.5)
        assert interval.covers(0.4) and interval.covers(0.6)
        assert not interval.covers(0.39)
        assert interval.width == pytest.approx(0.2)

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="exceeds upper bound"):
            ConfidenceInterval(point=0.5, lower=0.6, upper=0.4, alpha=0.05, n_resamples=10)

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must lie"):
            ConfidenceInterval(point=0.5, lower=0.4, upper=0.6, alpha=1.0, n_resamples=10)


class TestBootstrapCI:
    def test_point_estimate_uses_the_full_sample(self) -> None:
        values = np.arange(100, dtype=np.float64)
        interval = bootstrap_ci(
            lambda idx: float(values[idx].mean()),
            n_rounds=100,
            rng=np.random.default_rng(0),
            n_resamples=200,
        )
        assert interval.point == pytest.approx(values.mean())

    def test_interval_brackets_the_point_estimate(self) -> None:
        values = np.random.default_rng(0).normal(size=500)
        interval = bootstrap_ci(
            lambda idx: float(values[idx].mean()),
            n_rounds=500,
            rng=np.random.default_rng(1),
            n_resamples=500,
        )
        assert interval.lower < interval.point < interval.upper

    def test_width_shrinks_as_the_sample_grows(self) -> None:
        rng = np.random.default_rng(0)
        widths = []
        for n in (200, 20_000):
            values = rng.normal(size=n)
            widths.append(
                bootstrap_ci(
                    lambda idx, v=values: float(v[idx].mean()),  # type: ignore[misc]
                    n_rounds=n,
                    rng=np.random.default_rng(2),
                    n_resamples=300,
                ).width
            )
        assert widths[1] < widths[0] / 5.0

    def test_recovers_the_known_standard_error_of_a_mean(self) -> None:
        # For a sample mean the bootstrap interval should match the analytic
        # normal interval closely. This is the check that the resampling itself
        # is right, independent of any estimator.
        rng = np.random.default_rng(0)
        values = rng.normal(loc=1.0, scale=2.0, size=5000)
        interval = bootstrap_ci(
            lambda idx: float(values[idx].mean()),
            n_rounds=5000,
            rng=np.random.default_rng(3),
            n_resamples=2000,
        )
        analytic = 1.96 * values.std(ddof=1) / np.sqrt(5000)
        assert interval.width == pytest.approx(2 * analytic, rel=0.1)

    def test_is_deterministic_given_the_generator_seed(self) -> None:
        values = np.random.default_rng(0).normal(size=300)
        intervals = [
            bootstrap_ci(
                lambda idx: float(values[idx].mean()),
                n_rounds=300,
                rng=np.random.default_rng(7),
                n_resamples=200,
            )
            for _ in range(2)
        ]
        assert intervals[0] == intervals[1]

    def test_rejects_degenerate_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least 2 records"):
            bootstrap_ci(lambda idx: 0.0, n_rounds=1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="n_resamples must be >= 2"):
            bootstrap_ci(lambda idx: 0.0, n_rounds=10, rng=np.random.default_rng(0), n_resamples=1)

    def test_non_finite_replicates_raise(self) -> None:
        with pytest.raises(ValueError, match="non-finite replicates"):
            bootstrap_ci(
                lambda idx: float("nan"),
                n_rounds=10,
                rng=np.random.default_rng(0),
                n_resamples=10,
            )


class TestBootstrapEstimator:
    def test_point_matches_the_direct_estimate(self) -> None:
        feedback, dist = make_feedback(), None
        dist = np.full((feedback.n_rounds, feedback.n_actions), 1.0 / feedback.n_actions)
        estimator = IPSEstimator()
        interval = bootstrap_estimator(
            estimator, feedback, dist, rng=np.random.default_rng(0), n_resamples=200
        )
        assert interval.point == pytest.approx(estimator.estimate(feedback, dist))

    def test_covers_the_truth_for_a_known_mean(self) -> None:
        # Uniform target under uniform logging estimates E[r], which is 0.3 here
        # by construction.
        feedback = make_feedback(n=20_000, p_reward=0.3)
        dist = np.full((feedback.n_rounds, feedback.n_actions), 1.0 / feedback.n_actions)
        interval = bootstrap_estimator(
            IPSEstimator(), feedback, dist, rng=np.random.default_rng(0), n_resamples=500
        )
        assert interval.covers(0.3)

    def test_handles_the_ratio_estimator_by_resampling_records(self) -> None:
        # SNIPS is not a sample mean, so a bootstrap that averaged per-sample
        # values would be wrong here. Resampling indices keeps it valid.
        feedback = make_feedback(n=5000)
        rng = np.random.default_rng(0)
        dist = rng.dirichlet(np.ones(feedback.n_actions), size=feedback.n_rounds)
        estimator = SNIPSEstimator()
        interval = bootstrap_estimator(
            estimator, feedback, dist, rng=np.random.default_rng(1), n_resamples=300
        )
        assert interval.point == pytest.approx(estimator.estimate(feedback, dist))
        assert interval.lower < interval.point < interval.upper

    def test_toy_fixture_point_is_the_hand_computed_value(self) -> None:
        interval = bootstrap_estimator(
            IPSEstimator(),
            toy_feedback(),
            TOY_ACTION_DIST,
            rng=np.random.default_rng(0),
            n_resamples=100,
        )
        assert interval.point == pytest.approx(1.4)

    def test_is_deterministic_given_the_generator_seed(self) -> None:
        feedback = make_feedback(n=500)
        dist = np.full((feedback.n_rounds, feedback.n_actions), 1.0 / feedback.n_actions)
        intervals = [
            bootstrap_estimator(
                IPSEstimator(), feedback, dist, rng=np.random.default_rng(11), n_resamples=100
            )
            for _ in range(2)
        ]
        assert intervals[0] == intervals[1]
