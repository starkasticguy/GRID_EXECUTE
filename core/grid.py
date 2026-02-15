"""
Grid level generation + dynamic spacing.

Pure NumPy implementation.
"""
import numpy as np


def generate_grid_levels(anchor_price: float,
                         buy_spacing: float,
                         sell_spacing: float,
                         grid_levels_count: int,
                         min_price: float = 0.0) -> tuple:
    """
    Generate arithmetic grid levels centered on anchor price.

    Buy levels:  anchor - spacing * (1, 2, ..., N)
    Sell levels: anchor + spacing * (1, 2, ..., N)

    Returns:
        (buy_levels, sell_levels) — both descending/ascending from anchor
    """
    buy_levels = np.zeros(grid_levels_count, dtype=np.float64)
    sell_levels = np.zeros(grid_levels_count, dtype=np.float64)

    for i in range(grid_levels_count):
        b = anchor_price - buy_spacing * (i + 1)
        buy_levels[i] = max(b, min_price)
        sell_levels[i] = anchor_price + sell_spacing * (i + 1)

    return buy_levels, sell_levels


def calculate_order_qty(capital: float, price: float, order_pct: float,
                        leverage: float = 1.0) -> float:
    """
    Order size in base asset terms.

    Qty = (Capital * OrderPct * Leverage) / Price
    """
    if price <= 1e-9:
        return 0.0
    return (capital * order_pct * leverage) / price


def calculate_dynamic_spacing(atr: float, spacing_k: float,
                              spacing_floor: float = 0.005,
                              ref_price: float = 0.0) -> float:
    """
    Dynamic grid spacing = max(k * ATR, floor * price).

    Volatility expands → grid widens → covers more range
    Volatility contracts → grid tightens → captures smaller moves
    """
    atr_spacing = spacing_k * atr
    floor_spacing = spacing_floor * ref_price if ref_price > 0 else 0.0
    return max(atr_spacing, floor_spacing)
