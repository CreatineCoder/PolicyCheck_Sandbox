"""Tests for the validation harness (spec sections 7, 12, 13 step 5).

The gate test in :class:`TestPhaseOneGate` is the one the build order says must
pass before any further estimator work begins. If IPS cannot recover the true
policy value under uniform logging -- the easiest possible regime, with perfect
overlap and unit-variance weights -- then the harness itself is wrong, and every
number produced downstream would inherit that error silently.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from odl.envs.synthetic_bandit import SyntheticBanditConfig, SyntheticBanditEnv
from odl.eval.metrics import coverage, mean_ci_width, relative_bias, rmse
from odl.eval.validation_harness import HarnessResult, run_validation
from odl.ope.bootstrap import ConfidenceInterval
from odl.ope.ips import IPSEstimator, SNIPSEstimator
from odl.policies.uniform import UniformPolicy

CONFIG = SyntheticBanditConfig(d_context=5, n_actions=4, beta=1.0, seed=3)


def make_env() -> SyntheticBanditEnv:
    return SyntheticBanditEnv(CONFIG)


class TestMetrics:
    def test_relative_bias_is_signed(self) -> None:
        assert relative_bias([1.1, 1.1], 1.0) == pytest.approx(0.1)
        assert relative_bias([0.9, 0.9], 1.0) == pytest.approx(-0.1)

    def test_rmse_does_not_let_variance_cancel(self) -> None:
        # Symmetric errors give zero bias but non-zero RMSE. That difference is
        # exactly what makes IPS look fine on bias and terrible on RMSE under
        # poor overlap.
        assert relative_bias([0.5, 1.5], 1.0) == pytest.approx(0.0)
        assert rmse([0.5, 1.5], 1.0) == pytest.approx(0.5)

    def test_coverage_counts_containment(self) -> None:
        intervals = [
            ConfidenceInterval(point=1.0, lower=0.9, upper=1.1, alpha=0.05, n_resamples=10),
            ConfidenceInterval(point=1.0, lower=1.05, upper=1.2, alpha=0.05, n_resamples=10),
        ]
        assert coverage(intervals, 1.0) == pytest.approx(0.5)
        assert mean_ci_width(intervals) == pytest.approx(0.175)

    def test_zero_truth_rejected_for_relative_bias(self) -> None:
        with pytest.raises(ValueError, match="undefined when the true policy value is zero"):
            relative_bias([0.1], 0.0)

    def test_empty_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero replications"):
            rmse([], 1.0)
        with pytest.raises(ValueError, match="zero replications"):
            coverage([], 1.0)


class TestHarnessContract:
    def test_duplicate_estimator_names_rejected(self) -> None:
        env = make_env()
        with pytest.raises(ValueError, match="names must be unique"):
            run_validation(
                env=env,
                logging_policy=UniformPolicy(env.n_actions),
                target_policy=UniformPolicy(env.n_actions),
                estimators=[IPSEstimator(), IPSEstimator()],
                n_rounds=100,
                seed=0,
                n_replications=1,
                n_true_value_samples=1000,
                n_bootstrap=10,
            )

    def test_no_estimators_rejected(self) -> None:
        env = make_env()
        with pytest.raises(ValueError, match="at least one estimator"):
            run_validation(
                env=env,
                logging_policy=UniformPolicy(env.n_actions),
                target_policy=UniformPolicy(env.n_actions),
                estimators=[],
                n_rounds=100,
                seed=0,
                n_replications=1,
                n_true_value_samples=1000,
            )

    def test_summary_lookup_by_name(self) -> None:
        result = _small_run()
        assert result.summary_for("ips").estimator == "ips"
        with pytest.raises(KeyError, match="no estimator named"):
            result.summary_for("dr")

    def test_result_shape_matches_the_requested_replications(self) -> None:
        result = _small_run()
        assert len(result.replications) == 3
        assert all(len(rep) == 2 for rep in result.replications)
        assert result.n_replications == 3

    def test_format_table_reports_truth_and_every_estimator(self) -> None:
        table = _small_run().format_table()
        assert "V_true" in table
        assert "coverage" in table
        assert "ips" in table and "snips" in table

    def test_is_deterministic_given_the_seed(self) -> None:
        first, second = _small_run(), _small_run()
        assert first.v_true == second.v_true
        assert first.summaries == second.summaries

    def test_different_seeds_give_different_logs(self) -> None:
        assert _small_run(seed=1).summaries != _small_run(seed=2).summaries


def _small_run(seed: int = 0) -> HarnessResult:
    env = make_env()
    return run_validation(
        env=env,
        logging_policy=UniformPolicy(env.n_actions),
        target_policy=env.make_logging_policy(0.5, name="target"),
        estimators=[IPSEstimator(), SNIPSEstimator()],
        n_rounds=300,
        seed=seed,
        n_replications=3,
        n_true_value_samples=5_000,
        n_bootstrap=50,
    )


@lru_cache(maxsize=1)
def gate_run() -> HarnessResult:
    """The Phase 1 gate configuration, computed once and shared across assertions.

    Uniform logging, so overlap is perfect and IPS is unbiased by theory. Any
    measured bias here is an implementation error, not sampling noise.
    """
    env = make_env()
    return run_validation(
        env=env,
        logging_policy=UniformPolicy(env.n_actions),
        target_policy=env.make_logging_policy(0.5, name="target"),
        estimators=[IPSEstimator(), SNIPSEstimator()],
        n_rounds=4000,
        seed=20260815,
        n_replications=40,
        n_true_value_samples=100_000,
        n_bootstrap=200,
    )


class TestPhaseOneGate:
    """The build-order gate: IPS must recover truth under uniform logging.

    Run at reduced scale so it stays a fast test rather than an experiment. The
    full-scale version with R = 100 replications is E01.
    """

    def test_ips_recovers_the_true_policy_value(self) -> None:
        summary = gate_run().summary_for("ips")
        assert abs(summary.relative_bias) < 0.02, (
            f"IPS relative bias is {summary.relative_bias:+.4f} under uniform logging. "
            "Under perfect overlap IPS is unbiased, so a bias this large means the "
            "estimator, the environment's true value, or the logging propensities are wrong."
        )

    def test_snips_recovers_the_true_policy_value(self) -> None:
        summary = gate_run().summary_for("snips")
        assert abs(summary.relative_bias) < 0.02

    def test_ips_coverage_is_close_to_nominal(self) -> None:
        # Coverage is the sharpest diagnostic: an estimator can be unbiased and
        # still misreport its own uncertainty, and only coverage catches that.
        summary = gate_run().summary_for("ips")
        assert 0.85 <= summary.coverage <= 1.0, (
            f"IPS 95% CI coverage is {summary.coverage:.2%} under uniform logging, "
            "where it should be near nominal. The bootstrap is misreporting uncertainty."
        )

    def test_snips_has_lower_rmse_than_ips(self) -> None:
        # Self-normalisation trades a little bias for variance, and under a
        # non-uniform target that trade should pay.
        result = gate_run()
        assert result.summary_for("snips").rmse < result.summary_for("ips").rmse

    def test_uniform_logging_gives_full_effective_sample_size(self) -> None:
        # ESS well below n under uniform logging would mean the overlap knob is
        # not doing what it claims.
        summary = gate_run().summary_for("ips")
        assert summary.mean_ess > 0.4 * 4000

    def test_ips_recovers_truth_exactly_when_target_equals_logging(self) -> None:
        # The degenerate case, where the answer is known in closed form: every
        # weight is 1, so the estimate is the empirical mean reward.
        env = make_env()
        logging_policy = UniformPolicy(env.n_actions)
        result = run_validation(
            env=env,
            logging_policy=logging_policy,
            target_policy=logging_policy,
            estimators=[IPSEstimator()],
            n_rounds=20_000,
            seed=5,
            n_replications=5,
            n_true_value_samples=200_000,
            n_bootstrap=100,
        )
        summary = result.summary_for("ips")
        assert abs(summary.relative_bias) < 0.01
        assert summary.mean_ess == pytest.approx(20_000.0)
