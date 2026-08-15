"""Tests for logged-data schema validation (spec section 6.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odl.data.schema import ObdSchema, SchemaError, validate_obd_frame, validate_real_feedback
from tests.test_types import make_feedback

SCHEMA = ObdSchema()


def make_obd_frame(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        "timestamp": np.arange(n),
        "item_id": rng.integers(0, SCHEMA.n_actions, size=n),
        "position": rng.integers(1, 4, size=n),
        "click": rng.binomial(1, 0.05, size=n),
        "propensity_score": np.full(n, 1.0 / SCHEMA.n_actions),
    }
    for col in SCHEMA.user_feature_columns:
        data[col] = np.array([f"hash{i % 3}" for i in range(n)])
    for col in SCHEMA.affinity_columns:
        data[col] = rng.normal(size=n)
    return pd.DataFrame(data)


class TestObdSchemaColumnNaming:
    def test_user_features_use_underscores_and_there_are_four(self) -> None:
        assert SCHEMA.user_feature_columns == [
            "user_feature_0",
            "user_feature_1",
            "user_feature_2",
            "user_feature_3",
        ]

    def test_affinity_columns_use_a_hyphen(self) -> None:
        assert SCHEMA.affinity_columns[0] == "user-item_affinity_0"
        assert len(SCHEMA.affinity_columns) == 80


class TestValidateObdFrame:
    def test_valid_frame_passes(self) -> None:
        validate_obd_frame(make_obd_frame())

    def test_missing_columns_raise(self) -> None:
        frame = make_obd_frame().drop(columns=["user-item_affinity_0"])
        with pytest.raises(SchemaError, match="missing 1 required columns"):
            validate_obd_frame(frame)

    def test_hyphen_underscore_confusion_is_caught_not_silently_ignored(self) -> None:
        frame = make_obd_frame().rename(columns={"user-item_affinity_0": "user_item_affinity_0"})
        with pytest.raises(SchemaError, match="separator convention"):
            validate_obd_frame(frame)

    def test_non_binary_click_raises(self) -> None:
        frame = make_obd_frame()
        frame.loc[0, "click"] = 2
        with pytest.raises(SchemaError, match="click must be binary"):
            validate_obd_frame(frame)

    def test_zero_propensity_raises(self) -> None:
        frame = make_obd_frame()
        frame.loc[0, "propensity_score"] = 0.0
        with pytest.raises(SchemaError, match=r"propensity_score must lie in \(0, 1\]"):
            validate_obd_frame(frame)

    def test_item_id_outside_campaign_range_raises(self) -> None:
        frame = make_obd_frame()
        frame.loc[0, "item_id"] = SCHEMA.n_actions
        with pytest.raises(SchemaError, match="item_id must lie"):
            validate_obd_frame(frame)

    def test_unexpected_position_raises(self) -> None:
        frame = make_obd_frame()
        frame.loc[0, "position"] = 4
        with pytest.raises(SchemaError, match="position contains unexpected values"):
            validate_obd_frame(frame)

    def test_nan_raises(self) -> None:
        frame = make_obd_frame()
        frame.loc[0, "user-item_affinity_3"] = np.nan
        with pytest.raises(SchemaError, match="contains NaNs"):
            validate_obd_frame(frame)

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(SchemaError, match="empty"):
            validate_obd_frame(make_obd_frame(0))


class TestValidateRealFeedback:
    def test_passes_for_real_logs(self) -> None:
        fb = make_feedback()
        assert validate_real_feedback(fb, "obd_loader") is fb

    def test_rejects_ground_truth_leakage(self) -> None:
        fb = make_feedback(expected_reward=np.full((20, 4), 0.5))
        with pytest.raises(ValueError, match="critical bug"):
            validate_real_feedback(fb, "obd_loader")
