"""Tests for IPS, SNIPS and the shared weight machinery (spec sections 6.3, 12).

The toy cases here are hand-computed. That is the point: an estimator checked
only against another implementation of itself is self-consistent, not correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from odl.ope.base import effective_sample_size, importance_weights, target_propensity
from odl.ope.ips import IPSEstimator, SNIPSEstimator
from odl.policies.base import sample_actions
from odl.types import BanditFeedback

# ---------------------------------------------------------------------------
# Hand-computed fixture.
#
#   p          = [0.5, 0.5, 0.25, 0.25]
#   a          = [0,   1,   0,    1   ]
#   r          = [1,   0,   1,    1   ]
#   pi_e(a_i)  = [0.8, 0.2, 0.5,  0.5 ]
#   w          = [1.6, 0.4, 2.0,  2.0 ]
#
#   IPS   = mean(w * r) = (1.6 + 0 + 2.0 + 2.0) / 4 = 5.6 / 4 = 1.4
#   SNIPS = 5.6 / 6.0                                        = 0.9333...
#   ESS   = 6.0^2 / (1.6^2 + 0.4^2 + 2^2 + 2^2) = 36 / 10.72 = 3.35820895...
# ---------------------------------------------------------------------------

TOY_ACTION_DIST = np.array(
    [[0.8, 0.2], [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64
)
TOY_IPS = 1.4
TOY_SNIPS = 5.6 / 6.0
TOY_ESS = 36.0 / 10.72


def toy_feedback() -> BanditFeedback:
    return BanditFeedback(
        context=np.array([[0.0], [1.0], [2.0], [3.0]]),
        action=np.array([0, 1, 0, 1], dtype=np.int64),
        reward=np.array([1.0, 0.0, 1.0, 1.0]),
        propensity=np.array([0.5, 0.5, 0.25, 0.25]),
        n_actions=2,
    )


class TestWeightMachinery:
    def test_target_propensity_picks_the_logged_action(self) -> None:
        fb = toy_feedback()
        np.testing.assert_allclose(
            target_propensity(TOY_ACTION_DIST, fb.action), [0.8, 0.2, 0.5, 0.5]
        )

    def test_importance_weights_hand_computed(self) -> None:
        weights = importance_weights(toy_feedback(), TOY_ACTION_DIST)
        np.testing.assert_allclose(weights, [1.6, 0.4, 2.0, 2.0])

    def test_clipping_caps_weights_from_above_only(self) -> None:
        weights = importance_weights(toy_feedback(), TOY_ACTION_DIST, clip_tau=1.0)
        np.testing.assert_allclose(weights, [1.0, 0.4, 1.0, 1.0])

    def test_ess_hand_computed(self) -> None:
        assert effective_sample_size(
            importance_weights(toy_feedback(), TOY_ACTION_DIST)
        ) == pytest.approx(TOY_ESS)

    def test_ess_equals_n_when_all_weights_are_equal(self) -> None:
        assert effective_sample_size(np.full(37, 2.5)) == pytest.approx(37.0)

    def test_ess_approaches_one_when_a_single_weight_dominates(self) -> None:
        weights = np.array([1e6, 1.0, 1.0, 1.0])
        assert effective_sample_size(weights) == pytest.approx(1.0, abs=1e-4)

    def test_ess_undefined_when_all_weights_are_zero(self) -> None:
        with pytest.raises(ValueError, match="ESS is undefined"):
            effective_sample_size(np.zeros(5))

    def test_mismatched_action_dist_raises(self) -> None:
        with pytest.raises(ValueError, match="rows but the feedback has 4"):
            importance_weights(toy_feedback(), np.full((3, 2), 0.5))

    def test_non_negative_clip_required(self) -> None:
        with pytest.raises(ValueError, match="clip_tau must be positive"):
            importance_weights(toy_feedback(), TOY_ACTION_DIST, clip_tau=0.0)


class TestIPSEstimator:
    def test_matches_the_hand_computed_value(self) -> None:
        assert IPSEstimator().estimate(toy_feedback(), TOY_ACTION_DIST) == pytest.approx(TOY_IPS)

    def test_per_sample_values_are_hand_computed(self) -> None:
        np.testing.assert_allclose(
            IPSEstimator().per_sample_values(toy_feedback(), TOY_ACTION_DIST),
            [1.6, 0.0, 2.0, 2.0],
        )

    def test_estimate_is_the_mean_of_per_sample_values(self) -> None:
        estimator = IPSEstimator()
        fb = toy_feedback()
        assert estimator.estimate(fb, TOY_ACTION_DIST) == pytest.approx(
            float(estimator.per_sample_values(fb, TOY_ACTION_DIST).mean())
        )

    def test_clipping_biases_the_estimate_downward(self) -> None:
        # Clipping can only reduce a weight, so the bias direction is not an
        # empirical question -- it is guaranteed.
        clipped = IPSEstimator(clip_tau=1.0).estimate(toy_feedback(), TOY_ACTION_DIST)
        assert clipped == pytest.approx(0.75)
        assert clipped < TOY_IPS

    def test_ess_is_reported_alongside_the_estimate(self) -> None:
        assert IPSEstimator().effective_sample_size(
            toy_feedback(), TOY_ACTION_DIST
        ) == pytest.approx(TOY_ESS)

    def test_name_records_the_clipping_threshold(self) -> None:
        assert IPSEstimator().name == "ips"
        assert IPSEstimator(clip_tau=10.0).name == "ips_clip10"


class TestSNIPSEstimator:
    def test_matches_the_hand_computed_value(self) -> None:
        assert SNIPSEstimator().estimate(toy_feedback(), TOY_ACTION_DIST) == pytest.approx(
            TOY_SNIPS
        )

    def test_per_sample_values_average_to_the_estimate(self) -> None:
        estimator = SNIPSEstimator()
        fb = toy_feedback()
        values = estimator.per_sample_values(fb, TOY_ACTION_DIST)
        assert float(values.mean()) == pytest.approx(TOY_SNIPS)

    def test_is_invariant_to_rescaling_all_weights(self) -> None:
        # Self-normalisation divides out the total weight, which is exactly the
        # fluctuation that dominates plain IPS.
        fb = toy_feedback()
        halved = BanditFeedback(
            context=fb.context,
            action=fb.action,
            reward=fb.reward,
            propensity=fb.propensity * 2.0,
            n_actions=fb.n_actions,
        )
        assert SNIPSEstimator().estimate(halved, TOY_ACTION_DIST) == pytest.approx(
            SNIPSEstimator().estimate(fb, TOY_ACTION_DIST)
        )

    def test_bounded_by_the_reward_range(self) -> None:
        # A weighted average of rewards in [0, 1] must itself lie in [0, 1].
        # Plain IPS carries no such guarantee, and indeed exceeds 1 on this toy.
        assert 0.0 <= SNIPSEstimator().estimate(toy_feedback(), TOY_ACTION_DIST) <= 1.0
        assert IPSEstimator().estimate(toy_feedback(), TOY_ACTION_DIST) > 1.0


class TestRequiredRewardModelPredictions:
    def test_missing_predictions_raise_rather_than_defaulting(self) -> None:
        from odl.ope.base import Estimator

        with pytest.raises(ValueError, match="requires reward_model_preds"):
            Estimator._check_reward_model_preds(toy_feedback(), None)

    def test_wrong_shape_raises(self) -> None:
        from odl.ope.base import Estimator

        with pytest.raises(ValueError, match="must have shape"):
            Estimator._check_reward_model_preds(toy_feedback(), np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# Property-based invariants (spec section 12). These catch most implementation
# errors, because they must hold for every input rather than for one fixture.
# ---------------------------------------------------------------------------


@st.composite
def logged_feedback(draw: st.DrawFn) -> tuple[BanditFeedback, np.ndarray]:
    """Draw feedback logged from a random policy, plus that policy's distribution.

    The returned action distribution *is* the logging distribution, so the
    target equals the logging policy by construction.
    """
    n_actions = draw(st.integers(min_value=2, max_value=6))
    n_rounds = draw(st.integers(min_value=5, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n_rounds, 3))
    # Dirichlet with concentration 1 keeps every action's probability positive,
    # which is the overlap condition IPS requires.
    dist = rng.dirichlet(np.ones(n_actions), size=n_rounds)
    action, propensity = sample_actions(dist, rng)
    reward = rng.binomial(1, 0.3, size=n_rounds).astype(np.float64)

    feedback = BanditFeedback(
        context=context,
        action=action,
        reward=reward,
        propensity=propensity,
        n_actions=n_actions,
    )
    return feedback, dist


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(logged_feedback())
def test_ips_recovers_the_empirical_mean_when_target_equals_logging(
    case: tuple[BanditFeedback, np.ndarray],
) -> None:
    """With ``pi_e == pi_b`` every weight is exactly 1, so IPS is the sample mean.

    This is the single most useful invariant in the project: it holds exactly,
    to floating-point tolerance, and almost any indexing or normalisation error
    breaks it.
    """
    feedback, dist = case
    weights = importance_weights(feedback, dist)
    np.testing.assert_allclose(weights, 1.0, rtol=0, atol=1e-12)
    assert IPSEstimator().estimate(feedback, dist) == pytest.approx(
        float(feedback.reward.mean()), rel=1e-12, abs=1e-12
    )


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(logged_feedback())
def test_snips_also_recovers_the_empirical_mean_when_target_equals_logging(
    case: tuple[BanditFeedback, np.ndarray],
) -> None:
    """With unit weights the self-normalised estimator reduces to the mean too."""
    feedback, dist = case
    assert SNIPSEstimator().estimate(feedback, dist) == pytest.approx(
        float(feedback.reward.mean()), rel=1e-12, abs=1e-12
    )


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(logged_feedback())
def test_ess_equals_n_when_target_equals_logging(
    case: tuple[BanditFeedback, np.ndarray],
) -> None:
    """Unit weights mean no information is lost, so ESS is the full sample size."""
    feedback, dist = case
    assert effective_sample_size(importance_weights(feedback, dist)) == pytest.approx(
        float(feedback.n_rounds)
    )


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(logged_feedback())
def test_ess_never_exceeds_n(case: tuple[BanditFeedback, np.ndarray]) -> None:
    """ESS is bounded above by ``n``, with equality only for equal weights."""
    feedback, _ = case
    rng = np.random.default_rng(0)
    dist = rng.dirichlet(np.ones(feedback.n_actions), size=feedback.n_rounds)
    ess = effective_sample_size(importance_weights(feedback, dist))
    assert 0.0 < ess <= feedback.n_rounds + 1e-9


def test_uniform_target_under_uniform_logging_equals_the_mean_reward() -> None:
    """The stated invariant from the spec, checked at a realistic sample size."""
    rng = np.random.default_rng(0)
    n, n_actions = 5000, 4
    feedback = BanditFeedback(
        context=rng.normal(size=(n, 3)),
        action=rng.integers(0, n_actions, size=n).astype(np.int64),
        reward=rng.binomial(1, 0.25, size=n).astype(np.float64),
        propensity=np.full(n, 1.0 / n_actions),
        n_actions=n_actions,
    )
    dist = np.full((n, n_actions), 1.0 / n_actions)

    assert IPSEstimator().estimate(feedback, dist) == pytest.approx(
        float(feedback.reward.mean()), rel=1e-12
    )
    assert SNIPSEstimator().estimate(feedback, dist) == pytest.approx(
        float(feedback.reward.mean()), rel=1e-12
    )
