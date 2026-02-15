"""
KAMA (Kaufman Adaptive Moving Average) + Efficiency Ratio + Regime FSM.

Pure NumPy implementation (Numba-ready structure — add @njit later).
"""
import numpy as np


# ─── Regime Constants ──────────────────────────────────────────
REGIME_NOISE = 0
REGIME_UPTREND = 1
REGIME_DOWNTREND = -1
REGIME_BREAKOUT_UP = 2
REGIME_BREAKOUT_DOWN = -2

REGIME_NAMES = {
    REGIME_NOISE: "NOISE",
    REGIME_UPTREND: "UPTREND",
    REGIME_DOWNTREND: "DOWNTREND",
    REGIME_BREAKOUT_UP: "BREAKOUT_UP",
    REGIME_BREAKOUT_DOWN: "BREAKOUT_DOWN",
}


def calculate_er(close_arr: np.ndarray, period: int) -> np.ndarray:
    """
    Efficiency Ratio: ER = |Net Change| / Total Volatility.

    ER ~ 1.0 → trending (straight-line price movement)
    ER ~ 0.0 → noisy (random walk, lots of chop)

    Args:
        close_arr: Close prices (float64)
        period: Lookback window (default 10 = 150 min on 15m candles)
    Returns:
        er_arr: Array same length as close_arr, first `period` values are 0.
    """
    n = len(close_arr)
    er_arr = np.zeros(n, dtype=np.float64)

    for i in range(period, n):
        net_change = abs(close_arr[i] - close_arr[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close_arr[j] - close_arr[j - 1])

        if volatility > 1e-12:
            er_arr[i] = net_change / volatility
        # else stays 0.0

    return er_arr


def calculate_kama(close_arr: np.ndarray, er_arr: np.ndarray,
                   fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average.

    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA[t] = KAMA[t-1] + SC * (Price[t] - KAMA[t-1])

    Squaring SC suppresses noise response, amplifies trend response.
    """
    n = len(close_arr)
    kama_arr = np.zeros(n, dtype=np.float64)

    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc_diff = fast_sc - slow_sc

    kama_arr[0] = close_arr[0]

    for i in range(1, n):
        sc = (er_arr[i] * sc_diff + slow_sc) ** 2
        kama_arr[i] = kama_arr[i - 1] + sc * (close_arr[i] - kama_arr[i - 1])

    return kama_arr


def detect_regime(kama_arr: np.ndarray, er_arr: np.ndarray, atr_arr: np.ndarray,
                  threshold: float = 0.15,
                  er_trend_thresh: float = 0.5,
                  breakout_mult: float = 2.0) -> np.ndarray:
    """
    Finite State Machine regime detector.

    Normalized KAMA slope = (KAMA[t] - KAMA[t-1]) / ATR[t]

    Regimes:
      NOISE (0):          |slope| < threshold OR ER < er_trend_thresh
      UPTREND (1):        slope > threshold AND ER > er_trend_thresh
      DOWNTREND (-1):     slope < -threshold AND ER > er_trend_thresh
      BREAKOUT_UP (2):    slope > breakout_mult * threshold
      BREAKOUT_DOWN (-2): slope < -breakout_mult * threshold
    """
    n = len(kama_arr)
    regime_arr = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        if atr_arr[i] < 1e-12:
            regime_arr[i] = regime_arr[i - 1]  # carry forward
            continue

        slope = (kama_arr[i] - kama_arr[i - 1]) / atr_arr[i]
        er = er_arr[i]

        # Breakout first (strongest signal)
        if abs(slope) > breakout_mult * threshold:
            regime_arr[i] = REGIME_BREAKOUT_UP if slope > 0 else REGIME_BREAKOUT_DOWN
        elif er > er_trend_thresh:
            if slope > threshold:
                regime_arr[i] = REGIME_UPTREND
            elif slope < -threshold:
                regime_arr[i] = REGIME_DOWNTREND
            else:
                regime_arr[i] = REGIME_NOISE
        else:
            regime_arr[i] = REGIME_NOISE

    return regime_arr
