"""Contract tests for the core data types (spec sections 5, 12)."""

from __future__ import annotations

import numpy as np
import pytest

from odl.types import (
    BanditFeedback,
    Trajectory,
    trajectories_from_frame,
    trajectories_to_frame,
)

N_ACTIONS = 4


def make_feedback(**overrides: object) -> BanditFeedback:
    rng = np.random.default_rng(0)
    n = 20
    kwargs: dict[str, object] = {
        "context": rng.normal(size=(n, 3)),
        "action": rng.integers(0, N_ACTIONS, size=n).astype(np.int64),
        "reward": rng.binomial(1, 0.3, size=n).astype(np.float64),
        "propensity": np.full(n, 1.0 / N_ACTIONS),
        "n_actions": N_ACTIONS,
    }
    kwargs.update(overrides)
    return BanditFeedback(**kwargs)  # type: ignore[arg-type]


def make_trajectory(horizon: int = 5, d_state: int = 3, done: bool = True) -> Trajectory:
    rng = np.random.default_rng(1)
    return Trajectory(
        states=rng.normal(size=(horizon + 1, d_state)),
        actions=rng.integers(0, N_ACTIONS, size=horizon).astype(np.int64),
        rewards=rng.normal(size=horizon),
        propensities=np.full(horizon, 0.25),
        done=done,
    )


class TestBanditFeedbackValidation:
    def test_valid_feedback_exposes_shape_properties(self) -> None:
        fb = make_feedback()
        assert fb.n_rounds == 20
        assert fb.d_context == 3
        assert fb.has_ground_truth is False

    def test_rejects_zero_propensity(self) -> None:
        propensity = np.full(20, 0.25)
        propensity[3] = 0.0
        with pytest.raises(ValueError, match=r"propensity must lie in \(0, 1\]"):
            make_feedback(propensity=propensity)

    def test_rejects_propensity_above_one(self) -> None:
        propensity = np.full(20, 0.25)
        propensity[0] = 1.5
        with pytest.raises(ValueError, match=r"propensity must lie in \(0, 1\]"):
            make_feedback(propensity=propensity)

    def test_rejects_action_out_of_range(self) -> None:
        action = np.zeros(20, dtype=np.int64)
        action[0] = N_ACTIONS
        with pytest.raises(ValueError, match="action values must lie"):
            make_feedback(action=action)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="reward has 19 rows but context has 20"):
            make_feedback(reward=np.zeros(19))

    def test_rejects_nan(self) -> None:
        reward = np.zeros(20)
        reward[5] = np.nan
        with pytest.raises(ValueError, match="reward contains NaN"):
            make_feedback(reward=reward)

    def test_rejects_wrong_dtype_rather_than_coercing(self) -> None:
        with pytest.raises(ValueError, match="action must have dtype int64"):
            make_feedback(action=np.zeros(20, dtype=np.int32))

    def test_rejects_wrong_shaped_expected_reward(self) -> None:
        with pytest.raises(ValueError, match="expected_reward must have shape"):
            make_feedback(expected_reward=np.zeros((20, N_ACTIONS + 1)))

    def test_is_frozen(self) -> None:
        fb = make_feedback()
        with pytest.raises(AttributeError):
            fb.n_actions = 7  # type: ignore[misc]


class TestGroundTruthGuards:
    def test_require_ground_truth_raises_on_real_data(self) -> None:
        fb = make_feedback()
        with pytest.raises(ValueError, match="requires ground-truth expected_reward"):
            fb.require_ground_truth("true_policy_value")

    def test_require_ground_truth_returns_array_on_synthetic_data(self) -> None:
        expected = np.full((20, N_ACTIONS), 0.5)
        fb = make_feedback(expected_reward=expected)
        assert fb.has_ground_truth is True
        np.testing.assert_array_equal(fb.require_ground_truth("harness"), expected)

    def test_require_no_ground_truth_raises_on_leaked_synthetic_data(self) -> None:
        fb = make_feedback(expected_reward=np.full((20, N_ACTIONS), 0.5))
        with pytest.raises(ValueError, match="critical bug"):
            fb.require_no_ground_truth("obd_loader")

    def test_require_no_ground_truth_passes_on_real_data(self) -> None:
        make_feedback().require_no_ground_truth("obd_loader")


class TestSelect:
    def test_boolean_mask_subsets_all_arrays_consistently(self) -> None:
        fb = make_feedback(expected_reward=np.arange(80, dtype=np.float64).reshape(20, 4))
        mask = np.zeros(20, dtype=bool)
        mask[[1, 4, 9]] = True
        subset = fb.select(mask)

        assert subset.n_rounds == 3
        np.testing.assert_array_equal(subset.action, fb.action[mask])
        np.testing.assert_array_equal(subset.reward, fb.reward[mask])
        assert subset.expected_reward is not None
        np.testing.assert_array_equal(subset.expected_reward, np.array(
            [[4.0, 5.0, 6.0, 7.0], [16.0, 17.0, 18.0, 19.0], [36.0, 37.0, 38.0, 39.0]]
        ))

    def test_integer_index_supports_bootstrap_resampling_with_replacement(self) -> None:
        fb = make_feedback()
        index = np.array([0, 0, 0, 1], dtype=np.int64)
        subset = fb.select(index)
        assert subset.n_rounds == 4
        np.testing.assert_array_equal(subset.action[:3], np.repeat(fb.action[0], 3))

    def test_select_drops_nothing_silently(self) -> None:
        fb = make_feedback()
        subset = fb.select(np.ones(20, dtype=bool))
        assert subset.n_rounds == fb.n_rounds


class TestTrajectory:
    def test_states_must_include_terminal_state(self) -> None:
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="states must have T \\+ 1 = 6 rows"):
            Trajectory(
                states=rng.normal(size=(5, 3)),
                actions=np.zeros(5, dtype=np.int64),
                rewards=np.zeros(5),
                propensities=np.full(5, 0.5),
                done=True,
            )

    def test_discounted_return_matches_hand_computation(self) -> None:
        traj = Trajectory(
            states=np.zeros((4, 2)),
            actions=np.zeros(3, dtype=np.int64),
            rewards=np.array([1.0, 2.0, 4.0]),
            propensities=np.full(3, 0.5),
            done=True,
        )
        # 1 + 0.5 * 2 + 0.25 * 4 = 3.0
        assert traj.discounted_return(0.5) == pytest.approx(3.0)

    def test_discounted_return_rejects_invalid_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must lie"):
            make_trajectory().discounted_return(1.5)


class TestTrajectoryFrameRoundTrip:
    def test_round_trip_is_lossless(self) -> None:
        original = [
            make_trajectory(horizon=5, done=True),
            make_trajectory(horizon=3, done=False),
        ]
        recovered = trajectories_from_frame(trajectories_to_frame(original))

        assert len(recovered) == len(original)
        for before, after in zip(original, recovered, strict=True):
            np.testing.assert_allclose(after.states, before.states)
            np.testing.assert_array_equal(after.actions, before.actions)
            np.testing.assert_allclose(after.rewards, before.rewards)
            np.testing.assert_allclose(after.propensities, before.propensities)
            assert after.done == before.done

    def test_frame_has_one_row_per_transition(self) -> None:
        frame = trajectories_to_frame([make_trajectory(horizon=5), make_trajectory(horizon=3)])
        assert len(frame) == 8
        assert set(frame["episode_id"]) == {0, 1}

    def test_missing_columns_raise(self) -> None:
        frame = trajectories_to_frame([make_trajectory()]).drop(columns=["propensity"])
        with pytest.raises(ValueError, match="missing required columns"):
            trajectories_from_frame(frame)
