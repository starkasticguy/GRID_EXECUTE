"""
Tests for core indicator modules: KAMA, ER, ATR, Z-Score, Regime Detection.

Uses synthetic data to validate calculations without lookahead bias.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.kama import calculate_er, calculate_kama, detect_regime, REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND
from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility


def make_uptrend(n=500, start=100.0, step=0.1, noise=0.5):
    """Generate uptrending price series."""
    np.random.seed(42)
    trend = np.linspace(start, start + step * n, n)
    noise_arr = np.random.normal(0, noise, n)
    return trend + noise_arr


def make_sideways(n=500, center=100.0, noise=2.0):
    """Generate sideways (mean-reverting) price series."""
    np.random.seed(42)
    return center + np.random.normal(0, noise, n)


def make_downtrend(n=500, start=100.0, step=0.1, noise=0.5):
    """Generate downtrending price series."""
    np.random.seed(42)
    trend = np.linspace(start, start - step * n, n)
    noise_arr = np.random.normal(0, noise, n)
    return np.maximum(trend + noise_arr, 1.0)


def make_ohlc(close):
    """Generate synthetic OHLC from close prices."""
    n = len(close)
    np.random.seed(42)
    spread = np.abs(close) * 0.005
    high = close + np.abs(np.random.normal(0, 1, n)) * spread
    low = close - np.abs(np.random.normal(0, 1, n)) * spread
    low = np.maximum(low, 0.01)
    open_p = close + np.random.normal(0, 0.3, n) * spread
    return open_p, high, low, close


class TestER:
    def test_uptrend_high_er(self):
        """Strong uptrend should have ER close to 1."""
        close = make_uptrend(500, noise=0.01)
        er = calculate_er(close, 10)
        # Last portion should have high ER
        assert np.mean(er[100:]) > 0.5

    def test_sideways_low_er(self):
        """Sideways market should have low ER."""
        close = make_sideways(500, noise=2.0)
        er = calculate_er(close, 10)
        assert np.mean(er[100:]) < 0.5

    def test_er_range(self):
        """ER should be in [0, 1]."""
        close = make_uptrend(500)
        er = calculate_er(close, 10)
        assert np.all(er >= 0)
        assert np.all(er <= 1.0 + 1e-9)

    def test_er_warmup(self):
        """First `period` values should be 0."""
        close = make_uptrend(100)
        er = calculate_er(close, 10)
        assert np.all(er[:10] == 0)


class TestKAMA:
    def test_kama_tracks_trend(self):
        """KAMA should follow price in trend."""
        close = make_uptrend(500, noise=0.01)
        er = calculate_er(close, 10)
        kama = calculate_kama(close, er)
        # KAMA should be close to price in trend
        assert np.corrcoef(close[100:], kama[100:])[0, 1] > 0.99

    def test_kama_smooth_in_noise(self):
        """KAMA should be smoother than price in sideways."""
        close = make_sideways(500, noise=3.0)
        er = calculate_er(close, 10)
        kama = calculate_kama(close, er)
        # KAMA std should be less than close std
        assert np.std(kama[100:]) < np.std(close[100:])


class TestATR:
    def test_atr_positive(self):
        """ATR should always be positive."""
        close = make_uptrend(200)
        _, high, low, close = make_ohlc(close)
        atr = calculate_atr(high, low, close, 14)
        assert np.all(atr >= 0)

    def test_atr_higher_in_volatile(self):
        """ATR should be higher in volatile market."""
        close_calm = make_sideways(200, noise=0.5)
        close_vol = make_sideways(200, noise=5.0)
        _, h1, l1, c1 = make_ohlc(close_calm)
        _, h2, l2, c2 = make_ohlc(close_vol)
        atr1 = calculate_atr(h1, l1, c1, 14)
        atr2 = calculate_atr(h2, l2, c2, 14)
        assert np.mean(atr2[50:]) > np.mean(atr1[50:])


class TestZScore:
    def test_z_score_crash(self):
        """Z-Score should be strongly negative during crash."""
        # Start normal, then crash
        close = np.concatenate([
            make_sideways(200, center=100, noise=1.0),
            np.linspace(100, 60, 50),  # Sharp crash
        ])
        _, high, low, close = make_ohlc(close)
        atr = calculate_atr(high, low, close, 14)
        z = calculate_z_score(close, atr, 20)
        # During crash, Z should be very negative
        assert np.min(z[200:]) < -2.0

    def test_z_score_normal(self):
        """Z-Score should be near 0 in sideways market."""
        close = make_sideways(300, noise=1.0)
        _, high, low, close = make_ohlc(close)
        atr = calculate_atr(high, low, close, 14)
        z = calculate_z_score(close, atr, 20)
        assert abs(np.mean(z[50:])) < 2.0


class TestRegimeDetection:
    def test_uptrend_detected(self):
        """Uptrend regime should be detected in trending data."""
        close = make_uptrend(500, step=0.3, noise=0.1)
        _, high, low, close = make_ohlc(close)
        er = calculate_er(close, 10)
        kama = calculate_kama(close, er)
        atr = calculate_atr(high, low, close, 14)
        regime = detect_regime(kama, er, atr, 0.15, 0.5)
        # Should have mostly UPTREND in the middle
        uptrend_pct = np.mean(regime[100:400] > 0)
        assert uptrend_pct > 0.3  # At least 30% uptrend detection

    def test_noise_detected_in_sideways(self):
        """Sideways market should be classified as NOISE."""
        close = make_sideways(500, noise=2.0)
        _, high, low, close = make_ohlc(close)
        er = calculate_er(close, 10)
        kama = calculate_kama(close, er)
        atr = calculate_atr(high, low, close, 14)
        regime = detect_regime(kama, er, atr, 0.15, 0.5)
        noise_pct = np.mean(regime[100:] == REGIME_NOISE)
        assert noise_pct > 0.4  # At least 40% noise

    def test_downtrend_detected(self):
        """Downtrend regime should be detected."""
        close = make_downtrend(500, step=0.3, noise=0.1)
        _, high, low, close = make_ohlc(close)
        er = calculate_er(close, 10)
        kama = calculate_kama(close, er)
        atr = calculate_atr(high, low, close, 14)
        regime = detect_regime(kama, er, atr, 0.15, 0.5)
        downtrend_pct = np.mean(regime[100:400] < 0)
        assert downtrend_pct > 0.3


class TestRollingVolatility:
    def test_vol_positive(self):
        """Volatility should be non-negative."""
        close = make_uptrend(200)
        vol = calculate_rolling_volatility(close, 14)
        assert np.all(vol >= 0)

    def test_vol_warmup(self):
        """First `period` values should be 0."""
        close = make_uptrend(100)
        vol = calculate_rolling_volatility(close, 14)
        assert np.all(vol[:14] == 0)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
