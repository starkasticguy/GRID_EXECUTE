"""
BinanceExecutor — Exchange API interaction layer.

Wraps ccxt for Binance USDM Perpetual Futures in hedge mode.
Handles order placement, position queries, balance checks,
leverage setting, and funding rate retrieval.

All methods include retry logic with exponential backoff.
Dry-run mode logs orders without sending them to the exchange.
"""
import time
import logging
from typing import Optional

import ccxt

logger = logging.getLogger('executor')


class BinanceExecutor:

    def __init__(self, api_key: str, api_secret: str, config: dict,
                 dry_run: bool = False, testnet: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.testnet = testnet
        self._max_retries = config.get('max_retry_attempts', 3)
        self._retry_delay = config.get('retry_delay_seconds', 2)
        self._dry_run_counter = 0

        opts = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'hedgeMode': True,
            },
        }
        self.exchange = ccxt.binance(opts)
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._markets_loaded = False

    # ─── Connection ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Load markets and verify/enable hedge mode."""
        try:
            self.exchange.load_markets()
            self._markets_loaded = True
            logger.info("Markets loaded successfully")

            # Skip authenticated calls in dry-run mode (no valid API keys required)
            if self.dry_run:
                logger.info("Dry-run mode: skipping hedge mode verification")
                return True

            # Verify hedge mode (requires valid API keys)
            try:
                resp = self.exchange.fapiPrivateGetPositionSideDual()
                is_hedge = resp.get('dualSidePosition', False)
                if not is_hedge:
                    logger.info("Enabling hedge mode (dual side position)...")
                    self.exchange.fapiPrivatePostPositionSideDual(
                        params={'dualSidePosition': 'true'})
                    logger.info("Hedge mode enabled")
                else:
                    logger.info("Hedge mode already enabled")
            except Exception as e:
                logger.warning(f"Could not verify hedge mode: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (applies to both sides)."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")
            return False

    # ─── Balance & Positions ─────────────────────────────────────

    def get_balance(self) -> dict:
        """Fetch current USDT balance."""
        def _fetch():
            bal = self.exchange.fetch_balance()
            return {
                'total': float(bal.get('total', {}).get('USDT', 0)),
                'free': float(bal.get('free', {}).get('USDT', 0)),
                'used': float(bal.get('used', {}).get('USDT', 0)),
            }
        result = self._retry(_fetch)
        return result if result else {'total': 0.0, 'free': 0.0, 'used': 0.0}

    def get_positions(self, symbol: str) -> dict:
        """Fetch current positions for a symbol in hedge mode."""
        def _fetch():
            positions = self.exchange.fetch_positions([symbol])
            result = {
                'long': {'size': 0.0, 'avg_entry': 0.0, 'unrealized_pnl': 0.0,
                         'notional': 0.0, 'leverage': 1},
                'short': {'size': 0.0, 'avg_entry': 0.0, 'unrealized_pnl': 0.0,
                          'notional': 0.0, 'leverage': 1},
            }
            for p in positions:
                side_key = 'long' if p.get('side') == 'long' else 'short'
                contracts = abs(float(p.get('contracts', 0) or 0))
                contract_size = float(p.get('contractSize', 1) or 1)
                size = contracts * contract_size
                result[side_key] = {
                    'size': size,
                    'avg_entry': float(p.get('entryPrice', 0) or 0),
                    'unrealized_pnl': float(p.get('unrealizedPnl', 0) or 0),
                    'notional': float(p.get('notional', 0) or 0),
                    'leverage': int(float(p.get('leverage', 1) or 1)),
                }
            return result
        result = self._retry(_fetch)
        if result is None:
            return {
                'long': {'size': 0.0, 'avg_entry': 0.0, 'unrealized_pnl': 0.0,
                         'notional': 0.0, 'leverage': 1},
                'short': {'size': 0.0, 'avg_entry': 0.0, 'unrealized_pnl': 0.0,
                          'notional': 0.0, 'leverage': 1},
            }
        return result

    def get_open_orders(self, symbol: str) -> list:
        """Fetch all open orders for a symbol."""
        def _fetch():
            orders = self.exchange.fetch_open_orders(symbol)
            return [{
                'id': str(o['id']),
                'side': o['side'],
                'price': float(o.get('price', 0) or 0),
                'amount': float(o.get('amount', 0) or 0),
                'filled': float(o.get('filled', 0) or 0),
                'status': o.get('status', 'open'),
                'positionSide': o.get('info', {}).get('positionSide', ''),
                'reduceOnly': o.get('info', {}).get('reduceOnly', 'false') == 'true',
                'timestamp': int(o.get('timestamp', 0) or 0),
                'clientOrderId': o.get('clientOrderId', ''),
            } for o in orders]
        result = self._retry(_fetch)
        return result if result else []

    # ─── Order Placement ─────────────────────────────────────────

    def place_limit_order(self, symbol: str, side: str, amount: float,
                          price: float, position_side: str,
                          reduce_only: bool = False,
                          client_order_id: str = None) -> Optional[dict]:
        """Place a limit order in hedge mode."""
        # Precision
        amount = self._amount_precision(symbol, amount)
        price = self._price_precision(symbol, price)

        if amount <= 0 or price <= 0:
            logger.warning(f"Invalid order params: amount={amount}, price={price}")
            return None

        # Min notional check
        min_cost = self._min_cost(symbol)
        if amount * price < min_cost:
            logger.warning(f"Order notional {amount * price:.2f} below min {min_cost}")
            return None

        if self.dry_run:
            return self._dry_run_order(symbol, 'limit', side, amount, price,
                                       position_side, reduce_only, client_order_id)

        def _place():
            params = {'positionSide': position_side}
            # Note: reduceOnly is NOT sent in hedge mode — positionSide
            # already determines open vs close, and Binance rejects it.
            if client_order_id:
                params['clientOrderId'] = client_order_id
            order = self.exchange.create_order(
                symbol, 'limit', side, amount, price, params)
            logger.info(
                f"LIMIT {side.upper()} {amount} {symbol} @ {price} "
                f"[{position_side}] {'ReduceOnly' if reduce_only else 'Entry'} "
                f"→ id={order['id']}")
            return order
        return self._retry(_place)

    def place_market_order(self, symbol: str, side: str, amount: float,
                           position_side: str,
                           reduce_only: bool = False) -> Optional[dict]:
        """Place a market order (for stops, prunes, emergency closes)."""
        amount = self._amount_precision(symbol, amount)
        if amount <= 0:
            logger.warning(f"Invalid market order amount: {amount}")
            return None

        if self.dry_run:
            return self._dry_run_order(symbol, 'market', side, amount, 0,
                                       position_side, reduce_only)

        def _place():
            params = {'positionSide': position_side}
            # Note: reduceOnly is NOT sent in hedge mode — positionSide
            # already determines open vs close, and Binance rejects it.
            order = self.exchange.create_order(
                symbol, 'market', side, amount, None, params)
            logger.info(
                f"MARKET {side.upper()} {amount} {symbol} "
                f"[{position_side}] {'ReduceOnly' if reduce_only else 'Entry'} "
                f"→ id={order['id']}")
            return order
        return self._retry(_place)

    def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count cancelled."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would cancel all orders for {symbol}")
            return 0

        try:
            self.exchange.cancel_all_orders(symbol)
            orders = self.get_open_orders(symbol)
            remaining = len(orders)
            logger.info(f"Cancelled all orders for {symbol} ({remaining} remaining)")
            return 0 if remaining == 0 else remaining
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return -1

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a specific order by ID."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would cancel order {order_id}")
            return True

        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found (already filled/cancelled)")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    # ─── Market Data ─────────────────────────────────────────────

    def get_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        try:
            info = self.exchange.fetch_funding_rate(symbol)
            rate = float(info.get('fundingRate', 0) or 0)
            return rate
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate: {e}")
            return 0.0

    def get_latest_candles(self, symbol: str, timeframe: str,
                           limit: int = 200) -> list:
        """Fetch the most recent N candles as list of [ts, O, H, L, C, V]."""
        def _fetch():
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        result = self._retry(_fetch)
        return result if result else []

    def get_ticker(self, symbol: str) -> dict:
        """Fetch current ticker."""
        try:
            t = self.exchange.fetch_ticker(symbol)
            return {
                'last': float(t.get('last', 0) or 0),
                'bid': float(t.get('bid', 0) or 0),
                'ask': float(t.get('ask', 0) or 0),
                'timestamp': int(t.get('timestamp', 0) or 0),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch ticker: {e}")
            return {'last': 0.0, 'bid': 0.0, 'ask': 0.0, 'timestamp': 0}

    def get_server_time(self) -> int:
        """Fetch exchange server time in milliseconds."""
        try:
            return self.exchange.milliseconds()
        except Exception:
            return int(time.time() * 1000)

    # ─── Precision Helpers ───────────────────────────────────────

    def _amount_precision(self, symbol: str, amount: float) -> float:
        """Round amount to exchange precision."""
        if not self._markets_loaded:
            return amount
        try:
            return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            return amount

    def _price_precision(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        if not self._markets_loaded:
            return price
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            return price

    def _min_cost(self, symbol: str) -> float:
        """Get minimum order cost (notional) for symbol."""
        if not self._markets_loaded:
            return 5.0
        try:
            market = self.exchange.market(symbol)
            return float(market.get('limits', {}).get('cost', {}).get('min', 5.0) or 5.0)
        except Exception:
            return 5.0

    # ─── Retry Logic ─────────────────────────────────────────────

    def _retry(self, func, max_retries: int = None):
        """Retry wrapper with exponential backoff."""
        retries = max_retries or self._max_retries
        for attempt in range(retries + 1):
            try:
                return func()
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                if attempt < retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(f"Network error (attempt {attempt + 1}): {e}. "
                                   f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Network error after {retries} retries: {e}")
                    return None
            except ccxt.RateLimitExceeded as e:
                if attempt < retries:
                    delay = 10 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {retries} retries: {e}")
                    return None
            except ccxt.InsufficientFunds as e:
                logger.error(f"Insufficient funds: {e}")
                return None
            except ccxt.InvalidOrder as e:
                logger.error(f"Invalid order: {e}")
                return None
            except ccxt.ExchangeError as e:
                if attempt < 1:  # retry once for generic exchange errors
                    logger.warning(f"Exchange error: {e}. Retrying once...")
                    time.sleep(2)
                else:
                    logger.error(f"Exchange error: {e}")
                    return None
        return None

    # ─── Dry-Run Helpers ─────────────────────────────────────────

    def _dry_run_order(self, symbol, order_type, side, amount, price,
                       position_side, reduce_only, client_order_id=None):
        """Generate a synthetic order for dry-run mode."""
        self._dry_run_counter += 1
        order_id = f"DRY_{self._dry_run_counter}_{int(time.time() * 1000)}"
        logger.info(
            f"[DRY-RUN] {order_type.upper()} {side.upper()} {amount} {symbol} "
            f"{'@ ' + str(price) if price else ''} "
            f"[{position_side}] {'ReduceOnly' if reduce_only else 'Entry'}")
        return {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'timestamp': int(time.time() * 1000),
            'info': {
                'positionSide': position_side,
                'reduceOnly': str(reduce_only).lower(),
            },
        }
