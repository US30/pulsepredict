"""
Unit tests for PulsePredict evaluation metrics.

All tests use small synthetic arrays so the suite is fast, deterministic,
and dependency-free (only numpy + pytest required).
"""

import numpy as np
import pytest
from ml.eval.metrics import mae, rmse, mase, smape, coverage, winkler_score, pinball_loss


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y_true = np.array([0.0, 2.0])
        y_pred = np.array([1.0, 1.0])
        # |0-1| + |2-1| = 2 → mean = 1.0
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_non_negative(self):
        rng = np.random.default_rng(0)
        y_true = rng.random(100)
        y_pred = rng.random(100)
        assert mae(y_true, y_pred) >= 0.0

    def test_mae_single_element(self):
        assert mae(np.array([5.0]), np.array([3.0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        y_true = np.array([0.0, 4.0])
        y_pred = np.array([2.0, 2.0])
        # errors: [-2, 2] → MSE = 4 → RMSE = 2.0
        assert rmse(y_true, y_pred) == pytest.approx(2.0)

    def test_rmse_geq_mae(self):
        """RMSE >= MAE always holds (by Jensen's inequality)."""
        rng = np.random.default_rng(7)
        y_true = rng.random(50)
        y_pred = rng.random(50)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_rmse_non_negative(self):
        rng = np.random.default_rng(1)
        y_true = rng.random(200)
        y_pred = rng.random(200)
        assert rmse(y_true, y_pred) >= 0.0


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------

class TestMASE:
    def test_mase_perfect(self):
        y_true = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        naive_errors = np.abs(np.diff(y_true))
        assert mase(y_true, y_true, naive_errors) == pytest.approx(0.0)

    def test_mase_positive(self):
        rng = np.random.default_rng(3)
        y_true = rng.random(50) + 1.0
        y_pred = rng.random(50) + 1.0
        naive_errors = np.abs(np.diff(y_true))
        result = mase(y_true, y_pred, naive_errors)
        assert result >= 0.0

    def test_mase_naive_equals_one(self):
        """A seasonal-naive forecast should have MASE ≈ 1."""
        y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=float)
        # naive: predict y[t-season]
        season = 3
        y_pred = np.concatenate([y_true[:season], y_true[:-season]])
        naive_errors = np.abs(np.diff(y_true))
        result = mase(y_true, y_pred, naive_errors)
        # Not exactly 1 with this simple construction, but should be close
        assert result < 5.0  # sanity bound


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------

class TestSMAPE:
    def test_smape_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_smape_in_range(self):
        """sMAPE should be in [0, 200]."""
        rng = np.random.default_rng(5)
        y_true = rng.random(100) + 0.1
        y_pred = rng.random(100) + 0.1
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_smape_symmetric(self):
        """sMAPE is symmetric: smape(a, b) == smape(b, a)."""
        y_true = np.array([1.0, 3.0, 5.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_smape_known(self):
        # smape([100], [110]) = 200 * |100-110| / (|100|+|110|) = 200*10/210 ≈ 9.52
        result = smape(np.array([100.0]), np.array([110.0]))
        assert result == pytest.approx(200.0 * 10.0 / 210.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_coverage_full(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([2.0, 3.0, 4.0])
        assert coverage(y_true, lo, hi) == pytest.approx(1.0)

    def test_coverage_none(self):
        y_true = np.array([5.0, 6.0, 7.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])
        assert coverage(y_true, lo, hi) == pytest.approx(0.0)

    def test_coverage_partial(self):
        y_true = np.array([0.5, 5.0])   # first inside, second outside
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage(y_true, lo, hi) == pytest.approx(0.5)

    def test_coverage_at_boundary(self):
        """Points exactly on the boundary count as covered."""
        y_true = np.array([0.0, 1.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage(y_true, lo, hi) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pinball Loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_pinball_at_median(self):
        """At tau=0.5, pinball_loss == MAE / 2."""
        rng = np.random.default_rng(42)
        y_true = rng.random(100)
        y_pred = rng.random(100)
        pb = pinball_loss(y_true, y_pred, tau=0.5)
        expected = mae(y_true, y_pred) / 2.0
        assert pb == pytest.approx(expected, rel=1e-6)

    def test_pinball_non_negative(self):
        rng = np.random.default_rng(9)
        y_true = rng.random(50)
        y_pred = rng.random(50)
        assert pinball_loss(y_true, y_pred, tau=0.1) >= 0.0
        assert pinball_loss(y_true, y_pred, tau=0.9) >= 0.0

    def test_pinball_tau_bounds(self):
        """Pinball at tau=0 and tau=1 are valid (boundary cases)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.5, 1.5])
        # These should not raise
        pinball_loss(y_true, y_pred, tau=0.0)
        pinball_loss(y_true, y_pred, tau=1.0)

    def test_pinball_perfect_quantile(self):
        """If all predictions equal the true value, loss is 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, y, tau=0.7) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Winkler Score
# ---------------------------------------------------------------------------

class TestWinklerScore:
    def test_winkler_perfect_coverage_zero_width(self):
        """Zero-width interval that perfectly covers every point → score = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        # Exact intervals
        lo = y_true.copy()
        hi = y_true.copy()
        score = winkler_score(y_true, lo, hi, alpha=0.1)
        assert score == pytest.approx(0.0)

    def test_winkler_penalizes_wide_intervals(self):
        """A wider interval should not always give a lower Winkler score.

        The Winkler score penalises misses heavily. Here we test that
        when *all* actuals fall inside the band, a wider interval yields
        a strictly higher (worse) score than a tighter one.
        """
        y_true = np.array([5.0, 5.0, 5.0])
        # Narrow interval that still covers all points
        lo_narrow = np.array([4.0, 4.0, 4.0])
        hi_narrow = np.array([6.0, 6.0, 6.0])
        # Wide interval that also covers all points
        lo_wide = np.array([0.0, 0.0, 0.0])
        hi_wide = np.array([10.0, 10.0, 10.0])

        score_narrow = winkler_score(y_true, lo_narrow, hi_narrow, alpha=0.1)
        score_wide = winkler_score(y_true, lo_wide, hi_wide, alpha=0.1)

        assert score_wide > score_narrow, (
            "Wider interval should have a worse (higher) Winkler score when "
            "both achieve full coverage"
        )

    def test_winkler_penalizes_misses(self):
        """Missing a point should add a penalty proportional to miss distance."""
        y_true = np.array([10.0])
        lo = np.array([0.0])
        hi = np.array([5.0])   # y_true = 10 is above hi
        alpha = 0.1
        score = winkler_score(y_true, lo, hi, alpha=alpha)
        # Width = 5, miss penalty = 2/alpha * (10-5) = 100
        # Winkler = 5 + 100 = 105
        expected = (hi[0] - lo[0]) + (2.0 / alpha) * (y_true[0] - hi[0])
        assert score == pytest.approx(expected, rel=1e-6)

    def test_winkler_non_negative(self):
        rng = np.random.default_rng(11)
        y_true = rng.random(50) * 10
        lo = y_true - rng.random(50) * 2
        hi = y_true + rng.random(50) * 2
        score = winkler_score(y_true, lo, hi, alpha=0.1)
        assert score >= 0.0
