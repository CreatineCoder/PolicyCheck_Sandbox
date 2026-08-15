"""Core data contracts (spec section 5).

Every module in this project reads and writes these types. No ad-hoc dicts.

Notation used throughout the codebase and required in docstrings:

    x            context / state features
    a            action, in [0, n_actions)
    r            observed reward
    pi_b(a|x)    logging (behaviour) policy -- generated the data
    pi_e(a|x)    target (evaluation) policy -- the one being assessed
    p_i          logged propensity, = pi_b(a_i|x_i)
    w_i          importance weight, = pi_e(a_i|x_i) / p_i
    V(pi)        policy value, = E[r] under pi -- the estimand
    V_hat        an estimate of V
    r_hat(x, a)  reward-model prediction
    gamma        discount factor (Part 2 only)
    ESS          effective sample size, (sum w)^2 / sum(w^2)

Validation policy: these containers raise on malformed input. There are no
silent coercions, no dropped rows, and no default substitutions anywhere in
this module (spec section 0.4).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BanditFeedback",
    "Trajectory",
    "trajectories_from_frame",
    "trajectories_to_frame",
]


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a data contract is violated.

    Used instead of ``assert`` so that validation survives ``python -O``.
    """
    if not condition:
        raise ValueError(message)


def _check_array(arr: object, name: str, ndim: int, dtype: np.dtype[Any]) -> np.ndarray:
    _require(isinstance(arr, np.ndarray), f"{name} must be a numpy.ndarray, got {type(arr)!r}")
    assert isinstance(arr, np.ndarray)  # narrowing for mypy; _require already raised
    _require(arr.ndim == ndim, f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    _require(
        arr.dtype == dtype,
        f"{name} must have dtype {dtype}, got {arr.dtype}. "
        "Cast explicitly at the boundary rather than relying on coercion here.",
    )
    _require(bool(np.isfinite(arr).all()), f"{name} contains NaN or infinite values")
    return arr


@dataclass(frozen=True)
class BanditFeedback:
    """One-step logged bandit feedback: contexts, actions, rewards, propensities.

    Attributes:
        context: ``(n, d_context)`` float64 context features ``x``.
        action: ``(n,)`` int64 logged actions ``a``, each in ``[0, n_actions)``.
        reward: ``(n,)`` float64 observed rewards ``r``.
        propensity: ``(n,)`` float64 logged propensities ``p_i = pi_b(a_i|x_i)``,
            each in ``(0, 1]``. Zero propensity means the record could not have
            been logged, so it is rejected rather than clipped.
        n_actions: size of the action space.
        action_context: optional ``(n_actions, d_action)`` action features.
        expected_reward: ``(n, n_actions)`` ground-truth ``mu(x, a)``. **Only**
            populated by synthetic environments; must be ``None`` for real data.
            Reading this field while evaluating real data is a critical bug --
            see :meth:`require_no_ground_truth`.
    """

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    propensity: np.ndarray
    n_actions: int
    action_context: np.ndarray | None = None
    expected_reward: np.ndarray | None = None

    def __post_init__(self) -> None:
        f64 = np.dtype(np.float64)
        i64 = np.dtype(np.int64)

        _check_array(self.context, "context", 2, f64)
        _check_array(self.action, "action", 1, i64)
        _check_array(self.reward, "reward", 1, f64)
        _check_array(self.propensity, "propensity", 1, f64)

        _require(
            isinstance(self.n_actions, int) and not isinstance(self.n_actions, bool),
            "n_actions must be an int",
        )
        _require(self.n_actions >= 2, f"n_actions must be >= 2, got {self.n_actions}")

        n = self.context.shape[0]
        _require(n > 0, "BanditFeedback must contain at least one record")
        for name, arr in (
            ("action", self.action),
            ("reward", self.reward),
            ("propensity", self.propensity),
        ):
            _require(
                arr.shape[0] == n,
                f"{name} has {arr.shape[0]} rows but context has {n}",
            )

        _require(
            bool((self.action >= 0).all() and (self.action < self.n_actions).all()),
            f"action values must lie in [0, {self.n_actions})",
        )
        _require(
            bool((self.propensity > 0.0).all() and (self.propensity <= 1.0).all()),
            "propensity must lie in (0, 1]; a zero propensity makes importance "
            "weighting undefined and is never a valid log record",
        )

        if self.action_context is not None:
            _check_array(self.action_context, "action_context", 2, f64)
            _require(
                self.action_context.shape[0] == self.n_actions,
                f"action_context must have {self.n_actions} rows, "
                f"got {self.action_context.shape[0]}",
            )

        if self.expected_reward is not None:
            _check_array(self.expected_reward, "expected_reward", 2, f64)
            _require(
                self.expected_reward.shape == (n, self.n_actions),
                f"expected_reward must have shape {(n, self.n_actions)}, "
                f"got {self.expected_reward.shape}",
            )

    @property
    def n_rounds(self) -> int:
        """Number of logged records ``n``."""
        return int(self.context.shape[0])

    @property
    def d_context(self) -> int:
        """Context dimensionality ``d``."""
        return int(self.context.shape[1])

    @property
    def has_ground_truth(self) -> bool:
        """Whether exact ``mu(x, a)`` is available (synthetic data only)."""
        return self.expected_reward is not None

    def require_ground_truth(self, caller: str) -> np.ndarray:
        """Return ``expected_reward``, raising if this is real (unlabelled) data.

        Ground-truth policy values may only be computed on synthetic logs. This
        accessor is the single structural gate: nothing else in the codebase
        should touch ``expected_reward`` directly.
        """
        if self.expected_reward is None:
            raise ValueError(
                f"{caller} requires ground-truth expected_reward, but this "
                "BanditFeedback has none. True policy value is unobservable on "
                "real logs -- that is the premise of the project."
            )
        return self.expected_reward

    def require_no_ground_truth(self, caller: str) -> None:
        """Assert that this feedback carries no ground truth.

        Called by real-data loaders and by any estimator path that must be
        provably free of ground-truth leakage (spec section 0.4).
        """
        if self.expected_reward is not None:
            raise ValueError(
                f"{caller} expected real logged data, but expected_reward is "
                "populated. Ground-truth leakage into a real-data code path is a "
                "critical bug."
            )

    def select(self, index: np.ndarray) -> BanditFeedback:
        """Return the subset of records selected by a boolean or integer index.

        Used for cross-fitting folds and bootstrap resampling. Ground truth is
        carried along when present so synthetic subsets stay evaluable.
        """
        _require(isinstance(index, np.ndarray), "index must be a numpy.ndarray")
        return replace(
            self,
            context=self.context[index],
            action=self.action[index],
            reward=self.reward[index],
            propensity=self.propensity[index],
            expected_reward=(
                None if self.expected_reward is None else self.expected_reward[index]
            ),
        )


@dataclass(frozen=True)
class Trajectory:
    """One logged episode of sequential decisions (spec section 5.2).

    Attributes:
        states: ``(T + 1, d_state)`` float64 states, including the terminal state.
        actions: ``(T,)`` int64 actions.
        rewards: ``(T,)`` float64 immediate rewards.
        propensities: ``(T,)`` float64 per-step logging probabilities ``pi_b(a_t|s_t)``.
        done: whether the episode ended in a terminal state rather than by
            horizon truncation. Bootstrapping targets in FQI depends on this
            distinction, so it is stored rather than inferred.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    done: bool

    def __post_init__(self) -> None:
        f64 = np.dtype(np.float64)
        i64 = np.dtype(np.int64)

        _check_array(self.states, "states", 2, f64)
        _check_array(self.actions, "actions", 1, i64)
        _check_array(self.rewards, "rewards", 1, f64)
        _check_array(self.propensities, "propensities", 1, f64)
        _require(isinstance(self.done, bool), "done must be a bool")

        horizon = self.actions.shape[0]
        _require(horizon > 0, "Trajectory must contain at least one step")
        _require(
            self.states.shape[0] == horizon + 1,
            f"states must have T + 1 = {horizon + 1} rows, got {self.states.shape[0]}; "
            "the terminal state is part of the contract",
        )
        _require(self.rewards.shape[0] == horizon, "rewards must have length T")
        _require(self.propensities.shape[0] == horizon, "propensities must have length T")
        _require(bool((self.actions >= 0).all()), "actions must be non-negative")
        _require(
            bool((self.propensities > 0.0).all() and (self.propensities <= 1.0).all()),
            "propensities must lie in (0, 1]",
        )

    @property
    def horizon(self) -> int:
        """Number of decisions ``T`` in the episode."""
        return int(self.actions.shape[0])

    def discounted_return(self, gamma: float) -> float:
        """Return ``sum_t gamma^t r_t`` for this episode."""
        _require(0.0 < gamma <= 1.0, f"gamma must lie in (0, 1], got {gamma}")
        discounts = gamma ** np.arange(self.horizon, dtype=np.float64)
        return float(np.dot(discounts, self.rewards))


def trajectories_to_frame(trajectories: list[Trajectory]) -> pd.DataFrame:
    """Flatten trajectories to a long dataframe with an ``episode_id`` column.

    One row per transition. ``next_state_*`` columns carry ``s_{t+1}`` so the
    frame is directly consumable by FQI without a second pass.
    """
    import pandas as pd

    _require(len(trajectories) > 0, "cannot serialise an empty trajectory list")
    d_state = trajectories[0].states.shape[1]

    frames: list[pd.DataFrame] = []
    for episode_id, traj in enumerate(trajectories):
        _require(
            traj.states.shape[1] == d_state,
            f"episode {episode_id} has d_state={traj.states.shape[1]}, expected {d_state}",
        )
        t = np.arange(traj.horizon, dtype=np.int64)
        terminal = np.zeros(traj.horizon, dtype=bool)
        terminal[-1] = traj.done
        block = {
            "episode_id": np.full(traj.horizon, episode_id, dtype=np.int64),
            "t": t,
            "action": traj.actions,
            "reward": traj.rewards,
            "propensity": traj.propensities,
            "done": terminal,
        }
        for j in range(d_state):
            block[f"state_{j}"] = traj.states[:-1, j]
        for j in range(d_state):
            block[f"next_state_{j}"] = traj.states[1:, j]
        frames.append(pd.DataFrame(block))

    return pd.concat(frames, ignore_index=True)


def trajectories_from_frame(frame: pd.DataFrame) -> list[Trajectory]:
    """Inverse of :func:`trajectories_to_frame`.

    Round-trip fidelity is covered by a test; a mismatch here would silently
    corrupt every sequential result downstream.
    """
    required = {"episode_id", "t", "action", "reward", "propensity", "done"}
    missing = required - set(frame.columns)
    _require(not missing, f"frame is missing required columns: {sorted(missing)}")

    state_cols = sorted(
        (c for c in frame.columns if c.startswith("state_")),
        key=lambda c: int(c.removeprefix("state_")),
    )
    next_cols = sorted(
        (c for c in frame.columns if c.startswith("next_state_")),
        key=lambda c: int(c.removeprefix("next_state_")),
    )
    _require(len(state_cols) > 0, "frame has no state_* columns")
    _require(
        len(state_cols) == len(next_cols),
        f"{len(state_cols)} state_* columns but {len(next_cols)} next_state_* columns",
    )

    trajectories: list[Trajectory] = []
    for _, group in frame.groupby("episode_id", sort=True):
        ordered = group.sort_values("t")
        _require(
            bool((ordered["t"].to_numpy() == np.arange(len(ordered))).all()),
            "episode timesteps must be contiguous and start at 0",
        )
        states = np.vstack(
            [
                ordered[state_cols].to_numpy(dtype=np.float64),
                ordered[next_cols].to_numpy(dtype=np.float64)[-1:],
            ]
        )
        trajectories.append(
            Trajectory(
                states=states,
                actions=ordered["action"].to_numpy(dtype=np.int64),
                rewards=ordered["reward"].to_numpy(dtype=np.float64),
                propensities=ordered["propensity"].to_numpy(dtype=np.float64),
                done=bool(ordered["done"].to_numpy()[-1]),
            )
        )
    return trajectories
