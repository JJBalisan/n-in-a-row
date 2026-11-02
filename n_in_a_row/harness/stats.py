"""Statistics for monitoring experiments: AUROC, precision, power, CUSUM.

Standard library only (``statistics.NormalDist`` for the normal quantiles).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import NamedTuple

_NORMAL = statistics.NormalDist()


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Area under the ROC curve, by Mann-Whitney U with ties counted as half.

    It is the chance that a random positive scores above a random negative.

    Args:
        positives: Scores of the cases a detector should flag.
        negatives: Scores of the cases it should pass.

    Returns:
        The AUROC, from 0.0 to 1.0.

    Raises:
        ValueError: If either group is empty.
    """
    if not positives or not negatives:
        raise ValueError("auroc needs at least one positive and one negative")
    pooled = sorted([(v, 1) for v in positives] + [(v, 0) for v in negatives])
    rank_sum = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        mid_rank = (i + j + 1) / 2.0  # ranks i+1..j share their mean
        rank_sum += mid_rank * sum(flag for _, flag in pooled[i:j])
        i = j
    n1, n0 = len(positives), len(negatives)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def precision_at_base_rate(tpr: float, fpr: float, base_rate: float) -> float:
    """Precision of a detector at a given prevalence, from its TPR and FPR.

    At a base rate of 1e-4 even a 1% false-positive rate buries the true
    alarms: ``precision_at_base_rate(0.99, 0.01, 1e-4)`` is under 0.01.

    Args:
        tpr: True-positive rate (recall).
        fpr: False-positive rate.
        base_rate: Share of cases that are positive.

    Returns:
        The share of alarms that are real; 0.0 when nothing is flagged.

    Raises:
        ValueError: If any rate is outside [0, 1].
    """
    for name, p in (("tpr", tpr), ("fpr", fpr), ("base_rate", base_rate)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], not {p}")
    hits = tpr * base_rate
    alarms = hits + fpr * (1.0 - base_rate)
    return hits / alarms if alarms else 0.0


def _z_alpha(alpha: float, two_sided: bool) -> float:
    return _NORMAL.inv_cdf(1.0 - (alpha / 2.0 if two_sided else alpha))


def sample_size(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Samples per group to tell two rates apart (two-proportion z test).

    The normal approximation without continuity correction, e.g. blunder
    rates of an honest agent and a sandbagger.

    Args:
        p1: The rate in one group.
        p2: The rate in the other.
        alpha: False-positive rate of the test.
        power: Chance of detecting a real difference of this size.
        two_sided: Test for a difference either way.

    Returns:
        The smallest whole number of samples per group.

    Raises:
        ValueError: If the rates are equal.
    """
    if p1 == p2:
        raise ValueError("the rates are equal; no sample size detects that")
    p_bar = (p1 + p2) / 2.0
    z_a = _z_alpha(alpha, two_sided)
    z_b = _NORMAL.inv_cdf(power)
    root_n = (
        z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) / abs(p1 - p2)
    return math.ceil(root_n * root_n - 1e-9)


def power(
    n: int,
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of the test :func:`sample_size` sizes, with n per group.

    Args:
        n: Samples per group.
        p1: The rate in one group.
        p2: The rate in the other.
        alpha: False-positive rate of the test.
        two_sided: Test for a difference either way.

    Returns:
        The chance of detecting the difference (the far tail is ignored).
        With both rates 0 or 1 nothing varies, so it is 1.0 or 0.0.
    """
    p_bar = (p1 + p2) / 2.0
    z_a = _z_alpha(alpha, two_sided)
    shift = math.sqrt(n) * abs(p1 - p2) - z_a * math.sqrt(
        2.0 * p_bar * (1.0 - p_bar)
    )
    spread = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    if spread == 0.0:
        return 1.0 if shift > 0.0 else 0.0
    return _NORMAL.cdf(shift / spread)


class CusumResult(NamedTuple):
    """The outcome of a CUSUM run.

    Attributes:
        alarm: Index of the first observation that pushed the statistic
            above the threshold, or None.
        path: The statistic after each observation.
    """

    alarm: int | None
    path: list[float]


def cusum(
    xs: Sequence[float], *, target: float, slack: float, threshold: float
) -> CusumResult:
    """One-sided upper CUSUM: detects a rise in the mean of a stream.

    ``S_t = max(0, S_{t-1} + x_t - target - slack)``, alarming when
    ``S_t > threshold``. For per-move regret, ``target`` is the honest mean,
    ``slack`` about half the shift worth catching, and ``threshold`` sets
    the false-alarm rate.

    Args:
        xs: Observations in order, e.g. one agent's per-move regret.
        target: The in-control mean.
        slack: Allowance subtracted every step.
        threshold: Alarm level.

    Returns:
        The first alarm, if any, and the whole path (computed to the end).
    """
    s = 0.0
    alarm = None
    path = []
    for i, x in enumerate(xs):
        s = max(0.0, s + x - target - slack)
        path.append(s)
        if alarm is None and s > threshold:
            alarm = i
    return CusumResult(alarm, path)
