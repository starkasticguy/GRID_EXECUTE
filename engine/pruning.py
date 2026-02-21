"""
5-Method Pruning ("Gardener" Module).

Actively closes stale or toxic inventory to prevent capital lock-up.

Methods (in priority order):
  1. Deviance:       Position > 3σ from KAMA → immediate close (toxic)
  2. Oldest Trade:   Held > 24 hours → time-stop (stale)
  3. Gap:            Price-grid gap too large → grid invalidated
  4. Funding Cost:   Accumulated funding > 50% of grid profit potential
  5. Profit Offset:  Use accumulated profit to subsidize closing worst loser
"""
from engine.types import (
    PositionTracker,
    LABEL_PRUNE_OLDEST, LABEL_PRUNE_DEVIANCE,
    LABEL_PRUNE_GAP, LABEL_PRUNE_FUNDING, LABEL_PRUNE_OFFSET,
)


def check_deviance_prune(position: PositionTracker, current_price: float,
                         kama_price: float, atr: float,
                         sigma_mult: float = 3.0) -> int:
    """
    Method 2: Deviance Pruning.

    If price has deviated > sigma_mult * ATR from KAMA, the position
    is likely toxic (trend has moved decisively against it).

    Long:  prune if current_price < kama - sigma_mult * atr
    Short: prune if current_price > kama + sigma_mult * atr

    Returns fill index to prune (-1 if none).
    """
    if not position.fills:
        return -1

    if position.side == 1:  # Long
        if current_price < kama_price - sigma_mult * atr:
            # Find the worst fill (bought highest)
            worst_idx = max(range(len(position.fills)),
                           key=lambda i: position.fills[i]['price'])
            return worst_idx
    else:  # Short
        if current_price > kama_price + sigma_mult * atr:
            # Find the worst fill (sold lowest)
            worst_idx = min(range(len(position.fills)),
                           key=lambda i: position.fills[i]['price'])
            return worst_idx

    return -1


def check_oldest_prune(position: PositionTracker, current_time: float,
                       max_age_seconds: float = 86400.0,
                       current_price: float = None) -> int:
    """
    Method 1: Oldest Trade Pruning.

    Close positions held longer than max_age (default 24h).
    Prevents capital being locked in stale fills that aren't moving.

    Skips fills that are currently in profit — let them reach their TP naturally.

    Returns fill index to prune (-1 if none).
    """
    if not position.fills:
        return -1

    oldest_idx = -1
    oldest_age = 0.0

    for i, fill in enumerate(position.fills):
        age = current_time - fill['timestamp']
        if age > max_age_seconds and age > oldest_age:
            # Skip profitable fills — let them reach TP naturally
            if current_price is not None:
                if position.side == 1:
                    fill_pnl = (current_price - fill['price']) * fill['qty']
                else:
                    fill_pnl = (fill['price'] - current_price) * fill['qty']
                if fill_pnl > 0:
                    continue
            oldest_age = age
            oldest_idx = i

    return oldest_idx


def check_gap_prune(position: PositionTracker, current_price: float,
                    grid_anchor: float, grid_spacing: float,
                    gap_mult: float = 3.0) -> int:
    """
    Method 3: Gap Pruning.

    If the gap between current price and the nearest grid level
    exceeds gap_mult * grid_spacing, the grid is effectively broken.
    Close the most distant fill.

    Skips fills that are currently in profit — let them reach their TP naturally.

    Returns fill index to prune (-1 if none).
    """
    if not position.fills:
        return -1

    gap_threshold = gap_mult * grid_spacing

    # Check if any fill is too far from current price
    worst_idx = -1
    worst_gap = 0.0

    for i, fill in enumerate(position.fills):
        gap = abs(current_price - fill['price'])
        if gap > gap_threshold and gap > worst_gap:
            # Skip profitable fills — gap prune is for broken grid, not harvest
            if position.side == 1:
                fill_pnl = (current_price - fill['price']) * fill['qty']
            else:
                fill_pnl = (fill['price'] - current_price) * fill['qty']
            if fill_pnl > 0:
                continue
            worst_gap = gap
            worst_idx = i

    return worst_idx


def check_funding_prune(position: PositionTracker,
                        grid_profit_potential: float,
                        cost_ratio: float = 0.5) -> int:
    """
    Method 5: Funding Cost Pruning.

    Close fills whose accumulated funding cost exceeds
    cost_ratio × grid_profit_potential.

    Grid profit potential = typical grid spacing profit per fill.

    Returns fill index to prune (-1 if none).
    """
    if not position.fills or grid_profit_potential < 1e-12:
        return -1

    threshold = cost_ratio * grid_profit_potential

    worst_idx = -1
    worst_cost = 0.0

    for i, fill in enumerate(position.fills):
        if fill['funding_cost'] > threshold and fill['funding_cost'] > worst_cost:
            worst_cost = fill['funding_cost']
            worst_idx = i

    return worst_idx


def check_profit_offset_prune(position: PositionTracker,
                              current_price: float,
                              accumulated_profit: float,
                              min_profit_buffer: float = 0.0,
                              offset_ratio: float = 1.5) -> int:
    """
    Method 4: Profit Offset Pruning.

    Use accumulated realized profit as a buffer to close the worst
    losing fill at reduced net cost.

    Only triggers if accumulated_profit > worst_loss * offset_ratio.
    Higher offset_ratio = less aggressive (default 1.5 means profit must
    cover 150% of the loss before subsidizing a close).

    Returns fill index to prune (-1 if none).
    """
    if not position.fills:
        return -1

    if accumulated_profit <= min_profit_buffer:
        return -1

    # Find the worst underwater fill
    worst_idx = -1
    worst_loss = 0.0

    for i, fill in enumerate(position.fills):
        if position.side == 1:
            loss = (fill['price'] - current_price) * fill['qty']
        else:
            loss = (current_price - fill['price']) * fill['qty']

        # loss > 0 means the fill is underwater
        if loss > worst_loss:
            worst_loss = loss
            worst_idx = i

    # Only prune if profit buffer covers offset_ratio × loss
    if worst_idx >= 0 and worst_loss > 0 and accumulated_profit > worst_loss * offset_ratio:
        return worst_idx

    return -1



def run_pruning_cycle(position: PositionTracker, current_price: float,
                      current_time: float, kama_price: float, atr: float,
                      grid_anchor: float, grid_spacing: float,
                      grid_profit_potential: float,
                      accumulated_profit: float,
                      config: dict,
                      is_on_cooldown: bool = False) -> tuple:
    """
    Run all 5 pruning methods in priority order.

    Args:
        is_on_cooldown: If True, only PRUNE_DEVIANCE (toxic) fires.
                        All other prune types are suppressed to prevent
                        cascading prunes in volatile markets.

    Returns:
        (fill_index, prune_label) or (-1, None) if no prune needed.

    Priority: Deviance > Oldest > Gap > Funding > Profit Offset
    (Toxic first, then stale, then structural)
    """
    sigma_mult = config.get('deviance_sigma', 3.0)
    max_age = config.get('max_position_age_hours', 24) * 3600.0 * 1000.0  # ms (timestamps are in ms)
    gap_mult = config.get('gap_prune_mult', 3.0)
    funding_ratio = config.get('funding_cost_ratio', 0.5)
    offset_ratio = config.get('offset_prune_ratio', 1.5)

    # 1. Deviance (most urgent — toxic position) — ALWAYS fires, ignores cooldown
    idx = check_deviance_prune(position, current_price, kama_price, atr, sigma_mult)
    if idx >= 0:
        return idx, LABEL_PRUNE_DEVIANCE

    # All other prune types are suppressed during cooldown
    if is_on_cooldown:
        return -1, None

    # 2. Oldest (stale capital) — skip profitable fills
    idx = check_oldest_prune(position, current_time, max_age, current_price)
    if idx >= 0:
        return idx, LABEL_PRUNE_OLDEST

    # 3. Gap (grid broken)
    idx = check_gap_prune(position, current_price, grid_anchor, grid_spacing, gap_mult)
    if idx >= 0:
        return idx, LABEL_PRUNE_GAP

    # 4. Funding cost (bleeding to counterparty)
    idx = check_funding_prune(position, grid_profit_potential, funding_ratio)
    if idx >= 0:
        return idx, LABEL_PRUNE_FUNDING

    # 5. Profit offset (subsidized close)
    idx = check_profit_offset_prune(position, current_price, accumulated_profit,
                                    offset_ratio=offset_ratio)
    if idx >= 0:
        return idx, LABEL_PRUNE_OFFSET

    return -1, None

