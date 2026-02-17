"""
Virtual Order Book + Fill Detection for Backtesting.

Conservative fill model:
  - Limit Buy fills if candle Low <= Order Price
  - Limit Sell fills if candle High >= Order Price

No partial fills, no slippage model in matching (slippage applied at execution).
"""
from engine.types import STATUS_OPEN, STATUS_FILLED, STATUS_CANCELLED
import random


class OrderBook:
    """
    In-memory order book for backtest simulation.

    Maintains separate lists for long-grid and short-grid orders.
    """

    def __init__(self, max_orders: int = 500):
        self.max_orders = max_orders
        self.orders = []       # All active (OPEN) orders
        self.next_id = 0

    def add_order(self, side: int, price: float, qty: float,
                  timestamp: float, grid_level: int, direction: int,
                  reduce_only: bool = False) -> dict:
        """
        Place a new limit order.

        Args:
            side: 1=Buy, -1=Sell
            direction: 1=Long grid, -1=Short grid
        Returns:
            Order dict (or None if book is full)
        """
        if len(self.orders) >= self.max_orders:
            return None

        order = {
            'id': self.next_id,
            'side': side,
            'price': price,
            'qty': qty,
            'status': STATUS_OPEN,
            'timestamp': timestamp,
            'grid_level': grid_level,
            'direction': direction,
            'reduce_only': reduce_only,
        }
        self.next_id += 1
        self.orders.append(order)
        return order

    def check_fills(self, high: float, low: float, fill_prob: float = 1.0) -> list:
        """
        Check which open orders would fill given the candle's H/L range.

        Conservative assumption:
          - Buy limit fills if Low <= order price
          - Sell limit fills if High >= order price
        
        Args:
            high: Candle High
            low: Candle Low
            fill_prob: Probability [0.0, 1.0] of filling if price touches.
                       Simulates liquidity/queue position.

        Returns list of filled order dicts.
        """
        filled = []
        remaining = []

        for order in self.orders:
            if order['status'] != STATUS_OPEN:
                remaining.append(order)
                continue

            is_fill = False
            if order['side'] == 1:  # Buy
                if low <= order['price']:
                    if random.random() < fill_prob:
                        is_fill = True
            elif order['side'] == -1:  # Sell
                if high >= order['price']:
                    if random.random() < fill_prob:
                        is_fill = True

            if is_fill:
                order['status'] = STATUS_FILLED
                filled.append(order)
            else:
                remaining.append(order)

        self.orders = remaining
        return filled

    def cancel_all(self, direction: int = None, side: int = None):
        """
        Cancel all orders, optionally filtered by direction and/or side.

        direction: 1=Long grid only, -1=Short grid only, None=all
        side: 1=Buys only, -1=Sells only, None=all
        """
        remaining = []
        for order in self.orders:
            cancel = True
            if direction is not None and order['direction'] != direction:
                cancel = False
            if side is not None and order['side'] != side:
                cancel = False
            if cancel:
                order['status'] = STATUS_CANCELLED
            else:
                remaining.append(order)
        self.orders = remaining

    def cancel_by_level(self, direction: int, grid_level: int):
        """Cancel orders at a specific grid level."""
        self.orders = [
            o for o in self.orders
            if not (o['direction'] == direction and o['grid_level'] == grid_level)
        ]

    @property
    def open_count(self) -> int:
        return len(self.orders)

    @property
    def long_buy_count(self) -> int:
        return sum(1 for o in self.orders if o['direction'] == 1 and o['side'] == 1)

    @property
    def short_sell_count(self) -> int:
        return sum(1 for o in self.orders if o['direction'] == -1 and o['side'] == -1)
