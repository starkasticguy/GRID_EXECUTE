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


def apply_regime_hysteresis(regime_arr: np.ndarray, min_bars: int = 3) -> np.ndarray:
    """
    Debounce regime transitions: a regime change only takes effect after
    it has persisted for `min_bars` consecutive bars.

    This prevents single-bar regime flips from causing grid regeneration
    on every minor KAMA slope fluctuation — a major source of churn on
    15m crypto data.

    Algorithm:
      - Track `candidate` regime and how long it has been seen
      - Only commit to `current` regime once candidate persists min_bars bars
      - Breakout regimes (|regime| == 2) use min_bars=1 (urgent, no delay)

    Args:
        regime_arr: Raw regime array from detect_regime()
        min_bars:   Bars required to confirm a regime change (default 3 = 45 min)

    Returns:
        smoothed_regime: Same dtype as input, transitions are delayed/debounced
    """
    n = len(regime_arr)
    smoothed = np.zeros(n, dtype=np.int8)

    if n == 0:
        return smoothed

    current = int(regime_arr[0])
    candidate = current
    candidate_count = 1

    smoothed[0] = current

    for i in range(1, n):
        raw = int(regime_arr[i])
        is_breakout = abs(raw) == 2  # breakout always immediate

        if raw == current:
            # Already in this regime — reset candidate
            candidate = current
            candidate_count = 1
        elif raw == candidate:
            # Still seeing the same candidate
            candidate_count += 1
            threshold = 1 if is_breakout else min_bars
            if candidate_count >= threshold:
                current = candidate
                candidate_count = 1
        else:
            # New candidate regime
            candidate = raw
            candidate_count = 1
            # Breakout takes effect immediately
            if is_breakout:
                current = raw
                candidate_count = 1

        smoothed[i] = current

    return smoothed


def resample_ohlcv(close_arr: np.ndarray, high_arr: np.ndarray,
                   low_arr: np.ndarray, mult: int = 4) -> tuple:
    """
    Downsample 15m OHLCV arrays to a higher timeframe (e.g., 1H with mult=4).

    Each output bar aggregates `mult` consecutive 15m bars:
      - close: last bar's close
      - high:  max of highs
      - low:   min of lows

    Returns arrays of length n (same as input), where each value is the
    most-recent completed higher-timeframe bar, expanded back to 15m resolution.
    This allows direct index alignment with the 15m bar loop.

    Args:
        close_arr: 15m close prices
        high_arr:  15m high prices
        low_arr:   15m low prices
        mult:      Bars per higher-timeframe candle (4 = 1H from 15m)

    Returns:
        (htf_close, htf_high, htf_low): Arrays same length as input,
        each value is the last completed HTF bar expanded to 15m indices.
    """
    n = len(close_arr)
    htf_close = np.zeros(n, dtype=np.float64)
    htf_high = np.zeros(n, dtype=np.float64)
    htf_low = np.full(n, np.inf, dtype=np.float64)

    last_htf_close = close_arr[0]
    last_htf_high = high_arr[0]
    last_htf_low = low_arr[0]

    # Running aggregates within current HTF bar
    run_high = high_arr[0]
    run_low = low_arr[0]

    for i in range(n):
        bar_in_htf = i % mult

        if bar_in_htf == 0 and i > 0:
            # New HTF bar started — commit the completed HTF bar
            last_htf_close = close_arr[i - 1]
            last_htf_high = run_high
            last_htf_low = run_low
            # Reset running aggregates
            run_high = high_arr[i]
            run_low = low_arr[i]
        else:
            # Update running aggregates
            if high_arr[i] > run_high:
                run_high = high_arr[i]
            if low_arr[i] < run_low:
                run_low = low_arr[i]

        htf_close[i] = last_htf_close
        htf_high[i] = last_htf_high
        htf_low[i] = last_htf_low

    # Replace infs (first HTF bar not yet completed)
    htf_low[htf_low == np.inf] = low_arr[0]

    return htf_close, htf_high, htf_low
