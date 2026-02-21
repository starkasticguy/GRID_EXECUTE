"""
ADX (Average Directional Index) — Crypto-Native Implementation.

ADX measures trend STRENGTH (not direction). Used as a veto filter:
- ADX > 25: trend has real institutional conviction
- ADX < 25: high-vol chop, fakeout, or whale manipulation → stay in NOISE

Pure NumPy. No lookahead.
"""
import numpy as np


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """
    Compute ADX from OHLC data.

    Steps:
      1. True Range (TR)
      2. Directional Movement (+DM, -DM)
      3. Smoothed ATR, +DI, -DI (Wilder smoothing)
      4. DX = |+DI - -DI| / (+DI + -DI)
      5. ADX = Wilder smoothed DX

    Args:
        high, low, close: Price arrays (float64)
        period: Lookback window (default 14 = Wilder's original)
    Returns:
        adx: Array same length as input. First `2*period` values are 0.
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)

    if n < period * 2 + 1:
        return adx

    # Step 1 & 2: TR, +DM, -DM
    tr  = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)  # +DM
    ndm = np.zeros(n, dtype=np.float64)  # -DM

    for i in range(1, n):
        hl  = high[i]  - low[i]
        hpc = abs(high[i]  - close[i - 1])
        lpc = abs(low[i]   - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        ndm[i] = down if (down > up and down > 0)  else 0.0

    # Step 3: Wilder smoothing (RMA) — same as Wilder's ATR
    atr_w  = np.zeros(n, dtype=np.float64)
    pdm_w  = np.zeros(n, dtype=np.float64)
    ndm_w  = np.zeros(n, dtype=np.float64)

    # Seed with simple sum of first `period` bars
    atr_w[period]  = tr[1:period + 1].sum()
    pdm_w[period]  = pdm[1:period + 1].sum()
    ndm_w[period]  = ndm[1:period + 1].sum()

    for i in range(period + 1, n):
        atr_w[i]  = atr_w[i - 1]  - atr_w[i - 1]  / period + tr[i]
        pdm_w[i]  = pdm_w[i - 1]  - pdm_w[i - 1]  / period + pdm[i]
        ndm_w[i]  = ndm_w[i - 1]  - ndm_w[i - 1]  / period + ndm[i]

    # Step 4: DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr_w[i] < 1e-12:
            continue
        pdi = 100.0 * pdm_w[i] / atr_w[i]
        ndi = 100.0 * ndm_w[i] / atr_w[i]
        denom = pdi + ndi
        if denom < 1e-12:
            continue
        dx[i] = 100.0 * abs(pdi - ndi) / denom

    # Step 5: ADX = Wilder smoothed DX
    start = period * 2
    if start >= n:
        return adx
    adx[start] = dx[period:start].mean()
    for i in range(start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx
