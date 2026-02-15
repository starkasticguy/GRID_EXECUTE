"""
StateManager — Persistence layer for live trading state.

Saves/loads complete runner state to enable crash recovery.
Uses atomic writes (write to temp file, then rename) to prevent corruption.
Trade logs are persisted to CSV and JSON lines files.
"""
import json
import csv
import os
import time
import logging
import glob as globmod
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger('state')


class StateManager:

    TRADE_CSV_HEADERS = [
        'timestamp', 'datetime', 'symbol', 'price', 'qty', 'label',
        'regime', 'pnl', 'exchange_order_id', 'fill_price', 'fee',
        'wallet_balance_after', 'equity_after', 'pos_long_size', 'pos_short_size',
    ]

    def __init__(self, state_dir: str, symbol: str):
        self.state_dir = state_dir
        self.symbol = symbol
        os.makedirs(state_dir, exist_ok=True)

        self.state_file = os.path.join(state_dir, f'{symbol}_state.json')
        self.trade_csv = os.path.join(state_dir, f'{symbol}_trades.csv')
        self.trade_jsonl = os.path.join(state_dir, f'{symbol}_trades.jsonl')

        # Initialize CSV with headers if needed
        if not os.path.exists(self.trade_csv):
            with open(self.trade_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.TRADE_CSV_HEADERS)
                writer.writeheader()

        logger.info(f"StateManager initialized: {state_dir}/{symbol}")

    def save(self, state: dict):
        """Save complete state to JSON file (atomic write)."""
        state['_saved_at'] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.state_file + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_path, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def load(self) -> Optional[dict]:
        """Load state from JSON file. Returns None if not found or corrupt."""
        if not os.path.exists(self.state_file):
            logger.info("No saved state found — starting fresh")
            return None
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"State loaded from {self.state_file} "
                        f"(saved at {state.get('_saved_at', 'unknown')})")
            return state
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"State file corrupted: {e}")
            corrupt_path = f"{self.state_file}.corrupt.{int(time.time())}"
            os.rename(self.state_file, corrupt_path)
            logger.info(f"Corrupted state moved to {corrupt_path}")
            return None

    def save_trade(self, trade: dict):
        """Append a single trade to CSV and JSON lines files."""
        # CSV append
        try:
            row = {h: trade.get(h, '') for h in self.TRADE_CSV_HEADERS}
            with open(self.trade_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.TRADE_CSV_HEADERS)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write trade CSV: {e}")

        # JSON lines append
        try:
            with open(self.trade_jsonl, 'a') as f:
                f.write(json.dumps(trade, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write trade JSONL: {e}")

    def load_trades(self) -> list:
        """Load all trades from JSON lines file."""
        if not os.path.exists(self.trade_jsonl):
            return []
        trades = []
        try:
            with open(self.trade_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trades.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
        return trades

    def save_snapshot(self, state: dict, label: str = ''):
        """Save a timestamped snapshot."""
        ts = int(time.time())
        suffix = f"_{label}" if label else ""
        snap_path = os.path.join(
            self.state_dir, f'{self.symbol}_snapshot_{ts}{suffix}.json')
        try:
            with open(snap_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Snapshot saved: {snap_path}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def cleanup_old_snapshots(self, keep_last: int = 10):
        """Remove old snapshots, keeping only the most recent N."""
        pattern = os.path.join(self.state_dir, f'{self.symbol}_snapshot_*.json')
        files = sorted(globmod.glob(pattern))
        if len(files) > keep_last:
            for old in files[:-keep_last]:
                try:
                    os.remove(old)
                except OSError:
                    pass
