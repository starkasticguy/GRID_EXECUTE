"""
Data structures for GridStrategyV4 Engine.

Hedge Mode: Two independent position trackers (Long + Short).
Opening a long does NOT close a short.
"""

# ─── Order Status Constants ────────────────────────────────────
STATUS_PENDING = 0
STATUS_OPEN = 1
STATUS_FILLED = 2
STATUS_CANCELLED = 3

# ─── Side Constants ────────────────────────────────────────────
SIDE_BUY = 1
SIDE_SELL = -1

# ─── Direction (Grid Type) Constants ───────────────────────────
DIR_LONG = 1     # Long grid order
DIR_SHORT = -1   # Short grid order

# ─── Trade Label Constants ─────────────────────────────────────
LABEL_BUY_OPEN_LONG = "BUY_OPEN_LONG"
LABEL_SELL_CLOSE_LONG = "SELL_CLOSE_LONG"
LABEL_SELL_OPEN_SHORT = "SELL_OPEN_SHORT"
LABEL_BUY_CLOSE_SHORT = "BUY_CLOSE_SHORT"
LABEL_STOP_LONG = "STOP_LONG"
LABEL_STOP_SHORT = "STOP_SHORT"
LABEL_PRUNE_OLDEST = "PRUNE_OLDEST"
LABEL_PRUNE_DEVIANCE = "PRUNE_DEVIANCE"
LABEL_PRUNE_GAP = "PRUNE_GAP"
LABEL_PRUNE_FUNDING = "PRUNE_FUNDING"
LABEL_PRUNE_OFFSET = "PRUNE_OFFSET"
LABEL_PRUNE_VAR_WARNING = "PRUNE_VAR_WARNING"
LABEL_CIRCUIT_BREAKER = "CIRCUIT_BREAKER_HALT"
LABEL_LIQUIDATION = "LIQUIDATION"


def create_order(order_id: int, side: int, price: float, qty: float,
                 timestamp: float, grid_level: int, direction: int,
                 reduce_only: bool = False) -> dict:
    """Create a grid order dict."""
    return {
        'id': order_id,
        'side': side,           # 1=Buy, -1=Sell
        'price': price,
        'qty': qty,
        'status': STATUS_OPEN,
        'timestamp': timestamp,
        'entry_price': price,
        'grid_level': grid_level,
        'direction': direction,  # 1=Long grid, -1=Short grid
        'reduce_only': reduce_only,
    }


class PositionTracker:
    """
    Tracks a single side of a hedge-mode position.

    Each side (Long / Short) is fully independent:
      - Own size, avg entry, PnL, fills list
      - Fills list used by pruning to close specific positions
      - FIFO or targeted close supported
    """

    def __init__(self, side: int):
        self.side = side              # 1=Long, -1=Short
        self.size = 0.0               # Absolute position size (always >= 0)
        self.avg_entry = 0.0          # Weighted average entry price
        self.realized_pnl = 0.0       # Cumulative realized PnL
        self.unrealized_pnl = 0.0     # Current unrealized PnL
        self.funding_pnl = 0.0        # Cumulative funding payments
        self.num_fills = 0            # Number of open fills
        self.fills = []               # Individual fill dicts for pruning

    def add_fill(self, price: float, qty: float, timestamp: float,
                 fee: float = 0.0):
        """Add a new fill (opening trade) to this position."""
        if qty <= 0:
            return

        # Weighted avg entry
        old_notional = self.avg_entry * self.size
        new_notional = price * qty
        self.size += qty
        if self.size > 1e-12:
            self.avg_entry = (old_notional + new_notional) / self.size
        else:
            self.avg_entry = price

        self.realized_pnl -= fee
        self.num_fills += 1

        self.fills.append({
            'price': price,
            'qty': qty,
            'timestamp': timestamp,
            'funding_cost': 0.0,
        })

    def close_fill(self, close_price: float, qty: float,
                   fee: float = 0.0) -> float:
        """
        Close (part of) position FIFO.
        Returns realized PnL from this close.
        """
        qty = min(qty, self.size)
        if qty < 1e-12:
            return 0.0

        if self.side == 1:  # Long
            pnl = (close_price - self.avg_entry) * qty
        else:  # Short
            pnl = (self.avg_entry - close_price) * qty

        self.realized_pnl += pnl - fee
        self.size -= qty

        if self.size < 1e-12:
            self.size = 0.0
            self.avg_entry = 0.0
            self.fills.clear()
            self.num_fills = 0
        else:
            self._remove_fills_fifo(qty)

        return pnl - fee

    def close_specific_fill(self, fill_index: int, close_price: float,
                            fee: float = 0.0) -> float:
        """Close a specific fill by index (for pruning). Returns PnL."""
        if fill_index >= len(self.fills):
            return 0.0

        fill = self.fills[fill_index]
        qty = fill['qty']

        if self.side == 1:
            pnl = (close_price - fill['price']) * qty
        else:
            pnl = (fill['price'] - close_price) * qty

        self.realized_pnl += pnl - fee
        self.size -= qty
        self.num_fills -= 1
        self.fills.pop(fill_index)

        if self.size < 1e-12:
            self.size = 0.0
            self.avg_entry = 0.0
            self.fills.clear()
            self.num_fills = 0
        else:
            self._recalculate_avg_entry()

        return pnl - fee

    def close_all(self, close_price: float, fee: float = 0.0) -> float:
        """Close entire position. Returns total realized PnL."""
        return self.close_fill(close_price, self.size, fee)

    def update_unrealized(self, current_price: float):
        """Refresh unrealized PnL at current market price."""
        if self.size < 1e-12:
            self.unrealized_pnl = 0.0
            return
        if self.side == 1:
            self.unrealized_pnl = (current_price - self.avg_entry) * self.size
        else:
            self.unrealized_pnl = (self.avg_entry - current_price) * self.size

    def add_funding(self, funding_pnl: float):
        """Apply funding payment."""
        self.funding_pnl += funding_pnl
        self.realized_pnl += funding_pnl
        # Distribute cost to fills for funding-based pruning
        if self.fills and abs(funding_pnl) > 1e-12:
            per_fill = funding_pnl / len(self.fills)
            for f in self.fills:
                f['funding_cost'] += abs(per_fill)

    def _remove_fills_fifo(self, qty_to_remove: float):
        """Remove fills FIFO."""
        remaining = qty_to_remove
        while remaining > 1e-12 and self.fills:
            fill = self.fills[0]
            if fill['qty'] <= remaining + 1e-12:
                remaining -= fill['qty']
                self.fills.pop(0)
                self.num_fills -= 1
            else:
                fill['qty'] -= remaining
                remaining = 0.0

    def _recalculate_avg_entry(self):
        """Recalculate weighted avg entry from remaining fills."""
        if not self.fills:
            self.avg_entry = 0.0
            return
        total_notional = sum(f['price'] * f['qty'] for f in self.fills)
        total_qty = sum(f['qty'] for f in self.fills)
        self.avg_entry = total_notional / total_qty if total_qty > 1e-12 else 0.0

    @property
    def is_open(self) -> bool:
        return self.size > 1e-12

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
