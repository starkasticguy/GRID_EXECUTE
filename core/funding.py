"""
Funding Rate Module.

Simulates funding rate dynamics for perpetual futures backtesting.
Since historical funding rates are hard to get for backtests, we
use a synthetic model based on price deviation from a reference (SMA).

Funding rate logic:
  - Positive rate: Longs pay shorts (price premium to spot)
  - Negative rate: Shorts pay longs (price discount to spot)
  - Typical range: -0.03% to +0.03% per 8h interval
  - 15m candles → ~32 candles per funding interval

In backtest: Apply funding every 32 bars (8h) or use historical
rates if provided.
"""
import numpy as np


# Funding is typically applied every 8 hours (32 × 15m candles)
FUNDING_INTERVAL_BARS = 32


def generate_synthetic_funding_rates(close_arr: np.ndarray,
                                     sma_period: int = 96,
                                     base_rate: float = 0.0001,
                                     sensitivity: float = 0.5) -> np.ndarray:
    """
    Generate synthetic funding rates based on price deviation from SMA.

    When price > SMA → positive funding (longs pay)
    When price < SMA → negative funding (shorts pay)

    This approximates real funding rate dynamics without needing
    historical funding data.

    Args:
        close_arr: Close prices
        sma_period: SMA window (96 = 1 day of 15m candles)
        base_rate: Base funding rate magnitude
        sensitivity: How much deviation amplifies the rate
    Returns:
        funding_rates: Per-bar funding rate array (applied every FUNDING_INTERVAL_BARS)
    """
    n = len(close_arr)
    rates = np.zeros(n, dtype=np.float64)

    # Rolling SMA
    cumsum = 0.0
    for i in range(n):
        cumsum += close_arr[i]
        if i >= sma_period:
            cumsum -= close_arr[i - sma_period]

        count = min(i + 1, sma_period)
        sma = cumsum / count

        if sma > 1e-12:
            # Deviation as percentage
            dev = (close_arr[i] - sma) / sma
            # Clamp to realistic range [-0.03%, +0.03%]
            rate = base_rate + sensitivity * dev * base_rate
            rates[i] = max(-0.0003, min(0.0003, rate))

    return rates


def is_funding_interval(bar_index: int,
                        interval: int = FUNDING_INTERVAL_BARS) -> bool:
    """Check if current bar is a funding settlement bar."""
    return bar_index > 0 and bar_index % interval == 0


def apply_funding(position_size: float, mark_price: float,
                  funding_rate: float, side: int) -> float:
    """
    Calculate funding payment for a position.

    Returns PnL impact (negative = pay, positive = receive).

    Long + positive rate → pay (negative PnL)
    Long + negative rate → receive (positive PnL)
    Short + positive rate → receive (positive PnL)
    Short + negative rate → pay (negative PnL)
    """
    if abs(position_size) < 1e-12:
        return 0.0

    notional = abs(position_size) * mark_price
    fee = notional * funding_rate

    if side == 1:  # Long
        return -fee
    else:  # Short
        return fee


def should_bias_for_funding(regime: int, funding_rate: float,
                            threshold: float = 0.0003) -> tuple:
    """
    In NOISE regime, bias grid toward receiving funding.

    Returns:
        (long_bias, short_bias) — multipliers for grid spacing.
        Values < 1.0 = tighter (more aggressive), > 1.0 = wider.

    Logic:
      Positive funding → Short grid closer (receives funding)
      Negative funding → Long grid closer (receives funding)
    """
    # Only bias in NOISE regime (0)
    if regime != 0:
        return 1.0, 1.0

    if abs(funding_rate) < threshold * 0.1:  # negligible
        return 1.0, 1.0

    if funding_rate > threshold:
        # Positive funding — favor shorts (they receive)
        return 1.2, 0.8  # Widen long grid, tighten short grid
    elif funding_rate < -threshold:
        # Negative funding — favor longs (they receive)
        return 0.8, 1.2
    else:
        return 1.0, 1.0
