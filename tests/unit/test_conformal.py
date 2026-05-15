"""
Unit tests for split conformal and adaptive conformal predictors.

All tests use synthetic Gaussian residuals so they are fast and CPU-only.
"""

import numpy as np
import pytest
from ml.conformal.split_conformal import SplitConformalPredictor, SplitConformalConfig
from ml.conformal.adaptive_conformal import AdaptiveConformalPredictor, AdaptiveConformalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_residuals(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc=0.0, scale=1.0, size=n))


# ---------------------------------------------------------------------------
# SplitConformalPredictor
# ---------------------------------------------------------------------------

class TestSplitConformal:
    """Tests for SplitConformalPredictor."""

    @pytest.fixture()
    def predictor(self):
        cfg = SplitConformalConfig(alpha=0.1)
        return SplitConformalPredictor(cfg)

    @pytest.fixture()
    def calibrated_predictor(self, predictor):
        cal_residuals = _gaussian_residuals(500, seed=1)
        predictor.calibrate(cal_residuals)
        return predictor

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------

    def test_calibrate_sets_q_hat(self, calibrated_predictor):
        """After calibration, q_hat should be a positive finite float."""
        q_hat = calibrated_predictor.q_hat
        assert q_hat is not None, "q_hat should be set after calibrate()"
        assert isinstance(q_hat, float)
        assert q_hat > 0.0, "q_hat must be positive"
        assert np.isfinite(q_hat), "q_hat must be finite"

    def test_calibrate_idempotent(self, predictor):
        """Calibrating twice with the same data should give the same q_hat."""
        residuals = _gaussian_residuals(300, seed=2)
        predictor.calibrate(residuals)
        q_hat_first = predictor.q_hat
        predictor.calibrate(residuals)
        q_hat_second = predictor.q_hat
        assert q_hat_first == pytest.approx(q_hat_second)

    def test_calibrate_larger_alpha_smaller_q(self, predictor):
        """Higher alpha (less conservative) → smaller q_hat quantile."""
        residuals = _gaussian_residuals(500, seed=3)

        cfg_small = SplitConformalConfig(alpha=0.05)
        cfg_large = SplitConformalConfig(alpha=0.30)
        p_small = SplitConformalPredictor(cfg_small)
        p_large = SplitConformalPredictor(cfg_large)
        p_small.calibrate(residuals)
        p_large.calibrate(residuals)

        assert p_small.q_hat > p_large.q_hat, (
            "Smaller alpha (90% → 95% PI) should produce a larger q_hat"
        )

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def test_predict_returns_bounds(self, calibrated_predictor):
        """predict(y_hat) should return (lo, hi) with hi > lo."""
        y_hat = 10.0
        lo, hi = calibrated_predictor.predict(y_hat)
        assert hi > lo, "Upper bound must exceed lower bound"

    def test_predict_symmetric_around_y_hat(self, calibrated_predictor):
        """Split conformal PI should be symmetric: y_hat ± q_hat."""
        y_hat = 5.0
        lo, hi = calibrated_predictor.predict(y_hat)
        assert lo == pytest.approx(y_hat - calibrated_predictor.q_hat)
        assert hi == pytest.approx(y_hat + calibrated_predictor.q_hat)

    def test_predict_batch(self, calibrated_predictor):
        """predict() on an array should return arrays of matching shape."""
        y_hat = np.array([1.0, 2.0, 3.0])
        lo, hi = calibrated_predictor.predict(y_hat)
        assert lo.shape == y_hat.shape
        assert hi.shape == y_hat.shape
        assert np.all(hi > lo)

    def test_predict_raises_before_calibrate(self):
        """predict() before calibrate() should raise RuntimeError."""
        cfg = SplitConformalConfig(alpha=0.1)
        predictor = SplitConformalPredictor(cfg)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(1.0)

    # ------------------------------------------------------------------
    # coverage
    # ------------------------------------------------------------------

    def test_coverage_achieves_target(self):
        """On N(0,1) residuals, empirical coverage should be ≈ 90% ± 5%.

        We use a large calibration set (n=2000) and large test set (n=1000)
        so the estimate is tight enough for the ± 5% tolerance.
        """
        rng = np.random.default_rng(42)
        alpha = 0.1  # target coverage = 90%

        # Calibrate on 2000 points
        cal_residuals = np.abs(rng.normal(0, 1, size=2000))
        cfg = SplitConformalConfig(alpha=alpha)
        predictor = SplitConformalPredictor(cfg)
        predictor.calibrate(cal_residuals)

        # Generate test set
        y_true = rng.normal(0, 1, size=1000)
        y_hat = np.zeros_like(y_true)   # point forecast = 0 (mean)
        lo, hi = predictor.predict(y_hat)

        empirical_coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
        target = 1.0 - alpha  # 0.90

        assert abs(empirical_coverage - target) <= 0.05, (
            f"Coverage {empirical_coverage:.3f} deviates from target {target} "
            f"by more than 5%"
        )


# ---------------------------------------------------------------------------
# AdaptiveConformalPredictor
# ---------------------------------------------------------------------------

class TestAdaptiveConformal:
    """Tests for AdaptiveConformalPredictor (online, distribution-shift-aware)."""

    @pytest.fixture()
    def predictor(self):
        cfg = AdaptiveConformalConfig(alpha_init=0.1, gamma=0.005)
        return AdaptiveConformalPredictor(cfg)

    # ------------------------------------------------------------------
    # update() mechanics
    # ------------------------------------------------------------------

    def test_update_decreases_alpha_on_miss(self, predictor):
        """When covered=False, alpha_t should increase so future PI is wider."""
        alpha_before = predictor.alpha_t
        predictor.update(covered=False)
        alpha_after = predictor.alpha_t
        assert alpha_after > alpha_before, (
            "alpha_t should increase (making PI larger) after a miss"
        )

    def test_update_increases_alpha_on_hit(self, predictor):
        """When covered=True, alpha_t should decrease slightly (tighter PI)."""
        # First push alpha up so we have room to decrease it
        for _ in range(5):
            predictor.update(covered=False)
        alpha_before = predictor.alpha_t
        predictor.update(covered=True)
        alpha_after = predictor.alpha_t
        assert alpha_after < alpha_before, (
            "alpha_t should decrease (tightening PI) after a hit"
        )

    def test_alpha_stays_bounded(self, predictor):
        """After 100 random updates, alpha_t must stay in [0, 0.5]."""
        rng = np.random.default_rng(77)
        for _ in range(100):
            covered = bool(rng.integers(0, 2))
            predictor.update(covered=covered)

        assert 0.0 <= predictor.alpha_t <= 0.5, (
            f"alpha_t={predictor.alpha_t} is outside [0, 0.5]"
        )

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def test_predict_after_calibrate(self, predictor):
        """After calibration, predict should return (lo, hi) with hi > lo."""
        cal_residuals = _gaussian_residuals(200, seed=10)
        predictor.calibrate(cal_residuals)
        lo, hi = predictor.predict(y_hat=5.0)
        assert hi > lo

    def test_predict_returns_finite_values(self, predictor):
        cal_residuals = _gaussian_residuals(200, seed=20)
        predictor.calibrate(cal_residuals)
        lo, hi = predictor.predict(y_hat=0.0)
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    # ------------------------------------------------------------------
    # online simulation
    # ------------------------------------------------------------------

    def test_long_run_coverage_stable(self):
        """Over 500 online steps with stationary data, cumulative coverage
        should settle near the nominal 90%."""
        rng = np.random.default_rng(123)
        cfg = AdaptiveConformalConfig(alpha_init=0.1, gamma=0.005)
        predictor = AdaptiveConformalPredictor(cfg)

        # Warm up with 200 calibration residuals
        cal_res = np.abs(rng.normal(0, 1, size=200))
        predictor.calibrate(cal_res)

        hits = 0
        n_steps = 500
        for _ in range(n_steps):
            y_true = rng.normal(0, 1)
            y_hat = 0.0
            lo, hi = predictor.predict(y_hat)
            covered = lo <= y_true <= hi
            predictor.update(covered=covered)
            if covered:
                hits += 1

        empirical_coverage = hits / n_steps
        assert abs(empirical_coverage - 0.9) <= 0.08, (
            f"Long-run coverage {empirical_coverage:.3f} too far from 90%"
        )
