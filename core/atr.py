"""
ATR (Average True Range) + Z-Score for Circuit Breaker.

Pure NumPy implementation.
"""
import numpy as np


def calculate_atr(high_arr: np.ndarray, low_arr: np.ndarray,
                  close_arr: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range with Wilder's EMA smoothing.

    TR = max(H-L, |H - Prev_Close|, |L - Prev_Close|)
    ATR = EMA(TR, alpha = 2/(period+1))
    """
    n = len(close_arr)
    atr_arr = np.zeros(n, dtype=np.float64)
    tr_arr = np.zeros(n, dtype=np.float64)

    # First TR is just high-low
    tr_arr[0] = high_arr[0] - low_arr[0]

    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close_arr[i - 1])
        lc = abs(low_arr[i] - close_arr[i - 1])
        tr_arr[i] = max(hl, hc, lc)

    # Wilder's smoothing (EMA)
    alpha = 2.0 / (period + 1)
    atr_arr[0] = tr_arr[0]
    for i in range(1, n):
        atr_arr[i] = alpha * tr_arr[i] + (1.0 - alpha) * atr_arr[i - 1]

    return atr_arr


def calculate_z_score(close_arr: np.ndarray, atr_arr: np.ndarray,
                      period: int = 20) -> np.ndarray:
    """
    Z-Score = (Close - SMA(period)) / ATR

    Used for Circuit Breaker:
      Z < -3.0 → crash → HALT
      Z > -1.0 → normal → RESUME
    """
    n = len(close_arr)
    z_score = np.zeros(n, dtype=np.float64)

    # Rolling SMA
    current_sum = 0.0
    for i in range(n):
        current_sum += close_arr[i]
        if i >= period:
            current_sum -= close_arr[i - period]

        count = min(i + 1, period)
        sma = current_sum / count

        if atr_arr[i] > 1e-9:
            z_score[i] = (close_arr[i] - sma) / atr_arr[i]

    return z_score


def calculate_rolling_volatility(close_arr: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Rolling standard deviation of log returns.

    Used for Avellaneda-Stoikov sigma parameter.
    Returns sigma (annualized on 15m: × sqrt(96*365) but we keep raw 15m σ
    for the model since we use T=1.0 rolling horizon).
    """
    n = len(close_arr)
    vol = np.zeros(n, dtype=np.float64)

    for i in range(period, n):
        # Log returns over window
        log_rets = np.zeros(period)
        for j in range(period):
            idx = i - period + 1 + j
            if close_arr[idx - 1] > 1e-12:
                log_rets[j] = np.log(close_arr[idx] / close_arr[idx - 1])

        vol[i] = np.std(log_rets)

    return vol
