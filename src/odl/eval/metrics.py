"""Metrics for judging estimators against known truth (spec section 7.2).

These are the numbers the validation harness reports. All of them require
``V_true``, so all of them are available on synthetic data only -- which is
precisely why the harness exists.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from odl.ope.bootstrap import ConfidenceInterval

__all__ = ["coverage", "mean_ci_width", "relative_bias", "rmse"]


def relative_bias(estimates: Sequence[float] | np.ndarray, truth: float) -> float:
    """Return ``(mean(V_hat) - V_true) / V_true`` across replications.

    Signed on purpose: whether an estimator runs high or low is diagnostic.
    Clipped IPS, for instance, is biased downward by construction, and seeing
    the sign confirms the mechanism rather than merely the magnitude.
    """
    if truth == 0.0:
        raise ValueError(
            "relative bias is undefined when the true policy value is zero; "
            "report absolute bias instead"
        )
    values = np.asarray(estimates, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot compute relative bias from zero replications")
    return float((values.mean() - truth) / truth)


def rmse(estimates: Sequence[float] | np.ndarray, truth: float) -> float:
    """Return the root mean squared error of ``V_hat`` about ``V_true``.

    Unlike bias, this does not let variance cancel out. An estimator that is
    unbiased but wildly variable -- IPS under poor overlap -- looks fine by bias
    and terrible by RMSE, and the pair together tell the real story.
    """
    values = np.asarray(estimates, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot compute RMSE from zero replications")
    return float(np.sqrt(np.mean(np.square(values - truth))))


def coverage(intervals: Sequence[ConfidenceInterval], truth: float) -> float:
    """Return the fraction of intervals containing ``V_true``.

    The sharpest diagnostic in the harness. A well-behaved 95% interval should
    contain the truth about 95% of the time; an estimator reporting 60% coverage
    is not merely imprecise, it is misreporting its own uncertainty, which is
    the more dangerous failure because it is invisible on real data.
    """
    if len(intervals) == 0:
        raise ValueError("cannot compute coverage from zero replications")
    return float(np.mean([interval.covers(truth) for interval in intervals]))


def mean_ci_width(intervals: Sequence[ConfidenceInterval]) -> float:
    """Return the mean width of the intervals.

    Read together with :func:`coverage`: an estimator can buy nominal coverage
    by reporting uselessly wide intervals, and only the pair distinguishes an
    honest interval from a vacuous one.
    """
    if len(intervals) == 0:
        raise ValueError("cannot compute mean CI width from zero replications")
    return float(np.mean([interval.width for interval in intervals]))
