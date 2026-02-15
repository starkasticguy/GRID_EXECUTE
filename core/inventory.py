"""
Avellaneda-Stoikov Inventory Management Model (Price-Normalized).

For grid trading at 15m resolution, the A-S model provides two key functions:

1. Reservation Price (r): Shifts the grid anchor based on inventory risk.
   r = s * (1 - q * γ * σ² * T)
   - Long inventory (q > 0) → r < s → sells closer, buys further
   - Short inventory (q < 0) → r > s → buys closer, sells further

2. Volatility Spread Adjustment: Widens grid spacing during high volatility.
   δ_vol = γ * σ² * T * s   (price-scaled volatility component)

The fill-probability term (2/γ)ln(1+γ/κ) from the original HFT formula
is excluded because ATR-based spacing already serves that role for 15m grids.

Pure NumPy implementation.
"""
import numpy as np
import math


def calculate_reservation_price(mid_price: float, inventory_q: float,
                                gamma: float, volatility: float,
                                time_horizon: float = 1.0) -> float:
    """
    Avellaneda-Stoikov reservation price (price-scaled).

    r = s * (1 - q * γ * σ² * T)

    All terms are dimensionless; multiply by s for absolute price.

    Args:
        mid_price: Current mid/reference price (s)
        inventory_q: Normalized inventory [-1, 1] where positive = long bias
        gamma: Risk aversion parameter (0.1 – 2.0)
        volatility: σ (raw 15m standard deviation of log returns)
        time_horizon: Rolling horizon T (1.0 for perpetuals)
    """
    skew_frac = inventory_q * gamma * (volatility ** 2) * time_horizon
    return mid_price * (1.0 - skew_frac)


def calculate_optimal_spread(volatility: float, gamma: float,
                             kappa: float = 1.5,
                             time_horizon: float = 1.0,
                             ref_price: float = 1.0) -> float:
    """
    Volatility-based spread component from A-S model (price-scaled).

    δ = γ * σ² * T * ref_price

    This is the pure volatility risk component. The fill-probability term
    (2/γ)ln(1+γ/κ) is omitted because:
    - For 15m grid trading, ATR-based spacing handles minimum distance
    - The fill-probability term is designed for HFT order-book dynamics
    - With γ=0.5, κ=1.5 it produces a 115% spread which is unreachable

    Higher volatility or gamma → wider spread → fewer fills but safer.
    """
    if gamma < 1e-9:
        return 0.0
    return gamma * (volatility ** 2) * time_horizon * ref_price


def get_skewed_grid_params(mid_price: float, inventory_q: float,
                           gamma: float, volatility: float,
                           kappa: float = 1.5,
                           min_spacing: float = 0.0,
                           atr_spacing: float = 0.0) -> tuple:
    """
    Full Avellaneda-Stoikov grid parameter computation.

    1. Calculate reservation price r (shifted from mid by inventory risk)
    2. Calculate volatility-based spread δ
    3. Use max(δ, atr_spacing, min_spacing) as final spacing
    4. Grid centers on r, not on mid

    For grid trading, ATR spacing typically dominates as the base distance.
    The A-S model contributes:
    - Inventory skew via the reservation price shift
    - Volatility widening when σ spikes

    Returns:
        (buy_spacing, sell_spacing, anchor_price_r)
    """
    r = calculate_reservation_price(mid_price, inventory_q, gamma, volatility)
    delta = calculate_optimal_spread(volatility, gamma, kappa, ref_price=mid_price)

    # Enforce minimum spacing (from ATR-based or absolute floor)
    final_spacing = max(delta, min_spacing, atr_spacing)

    # In pure A-S the skew comes from r shifting, spacing is symmetric around r
    return final_spacing, final_spacing, r


def normalize_inventory(position_size: float, max_inventory: float) -> float:
    """
    Normalize inventory to [-1, 1] range for A-S model.

    Args:
        position_size: Current position (positive = long, negative = short)
        max_inventory: Maximum allowed inventory per side
    Returns:
        q: Normalized inventory clamped to [-1, 1]
    """
    if max_inventory < 1e-9:
        return 0.0
    q = position_size / max_inventory
    return max(-1.0, min(1.0, q))
