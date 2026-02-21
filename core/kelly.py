"""
Quarter-Kelly Position Sizing — Crypto-Adapted.

Standard Kelly assumes independent bets with known odds.
Crypto trades are auto-correlated (losing streaks = regime shift, not bad luck).
Quarter-Kelly (25% of full Kelly) prevents over-sizing after lucky streaks.

Trades are filtered by current regime so the win-rate/payoff estimates
reflect performance *within the same market type*, not across all regimes mixed.
"""
import numpy as np


def compute_kelly_fraction(
    recent_trades: list,
    current_regime: int,
    kelly_fraction: float = 0.25,
    min_trades: int = 20,
    max_fraction: float = 1.0,
) -> float:
    """
    Compute fractional Kelly bet size from recent trade history.

    Args:
        recent_trades : list of dicts with keys 'pnl' and 'regime'
        current_regime: int regime label to filter by
        kelly_fraction: scale factor (0.25 = Quarter-Kelly)
        min_trades    : minimum regime-filtered trades before Kelly takes effect
        max_fraction  : cap on returned fraction (default 1.0 = use configured max)

    Returns:
        multiplier in (0.0, 1.0] applied to base order_pct.
        Returns 1.0 (no adjustment) when there's insufficient data.
    """
    # Filter to same-regime trades only
    regime_trades = [t for t in recent_trades if t.get('regime') == current_regime]

    if len(regime_trades) < min_trades:
        return 1.0  # Not enough data — use base order_pct unchanged

    wins  = [t['pnl'] for t in regime_trades if t['pnl'] > 0]
    losses= [abs(t['pnl']) for t in regime_trades if t['pnl'] <= 0]

    if not wins or not losses:
        return 1.0

    win_rate   = len(wins) / len(regime_trades)
    avg_win    = float(np.mean(wins))
    avg_loss   = float(np.mean(losses))

    if avg_loss < 1e-12:
        return 1.0

    payoff = avg_win / avg_loss

    # Full Kelly: f = win_rate - (1 - win_rate) / payoff
    full_kelly = win_rate - (1.0 - win_rate) / payoff

    if full_kelly <= 0:
        # Edge is negative in this regime — shrink to minimum
        return 0.25

    # Apply fraction (Quarter-Kelly by default)
    frac = kelly_fraction * full_kelly
    return float(min(max(frac, 0.1), max_fraction))
