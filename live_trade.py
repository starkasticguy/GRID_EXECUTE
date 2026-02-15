#!/usr/bin/env python3
"""
GridStrategyV4 Live Trading CLI.

Usage:
  python3 live_trade.py --coin BTC --capital 10000 --leverage 20
  python3 live_trade.py --coin ETH --dry-run
  python3 live_trade.py --coin BTC --resume
  python3 live_trade.py --coin SOL --testnet --dry-run --log-level DEBUG

Environment Variables:
  BINANCE_API_KEY      Your Binance API key
  BINANCE_API_SECRET   Your Binance API secret
"""
import argparse
import os
import sys
import signal
import logging

from dotenv import load_dotenv

from config import STRATEGY_PARAMS, LIVE_CONFIG
from live.executor import BinanceExecutor
from live.runner import LiveRunner
from live.state import StateManager
from live.logger import TradeLogger
from live.monitor import HealthMonitor


def main():
    parser = argparse.ArgumentParser(
        description="GridStrategyV4 Live Trader — Binance USDM Perpetual Futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run with BTC (no real orders)
  python3 live_trade.py --coin BTC --dry-run

  # Live trade ETH with $5000 capital at 10x leverage
  python3 live_trade.py --coin ETH --capital 5000 --leverage 10

  # Resume from saved state
  python3 live_trade.py --coin BTC --resume

  # Testnet trading
  python3 live_trade.py --coin BTC --testnet --capital 1000

  # Long-only mode
  python3 live_trade.py --coin SOL --no-short --capital 3000
""")
    parser.add_argument("--coin", required=True,
                        help="Coin to trade (e.g. BTC, ETH, SOL)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Capital allocation in USDT (default: full wallet balance)")
    parser.add_argument("--leverage", type=int, default=None,
                        help="Leverage multiplier (default: from config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate orders without sending to exchange")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state file")
    parser.add_argument("--testnet", action="store_true",
                        help="Use Binance Futures testnet")
    parser.add_argument("--no-short", action="store_true",
                        help="Disable short grid (long-only mode)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # ─── Logging Setup ─────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # ─── API Keys ──────────────────────────────────────────────
    load_dotenv()  # Load .env file into environment (must be before os.environ.get)
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')

    if not api_key or not api_secret:
        if not args.dry_run:
            print("ERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
            print()
            print("  export BINANCE_API_KEY='your_api_key_here'")
            print("  export BINANCE_API_SECRET='your_api_secret_here'")
            print()
            print("Or use --dry-run for simulation mode.")
            sys.exit(1)
        else:
            print("WARNING: No API keys set. Running in dry-run mode with limited data access.")
            api_key = api_key or 'dry_run_key'
            api_secret = api_secret or 'dry_run_secret'

    # ─── Build Configs ─────────────────────────────────────────
    strategy_config = STRATEGY_PARAMS.copy()
    live_config = LIVE_CONFIG.copy()

    if args.capital:
        strategy_config['initial_capital'] = args.capital
    if args.leverage:
        strategy_config['leverage'] = float(args.leverage)
    if args.no_short:
        strategy_config['allow_short'] = False
    live_config['dry_run'] = args.dry_run
    live_config['log_level'] = args.log_level

    symbol = f"{args.coin}/USDT:USDT"
    safe_symbol = f"{args.coin}USDT"

    # ─── Create Directories ────────────────────────────────────
    os.makedirs(live_config['state_dir'], exist_ok=True)
    os.makedirs(live_config['log_dir'], exist_ok=True)

    # ─── Initialize Components ─────────────────────────────────
    executor = BinanceExecutor(
        api_key, api_secret, live_config,
        dry_run=args.dry_run, testnet=args.testnet)

    state_mgr = StateManager(live_config['state_dir'], safe_symbol)
    trade_logger = TradeLogger(
        live_config['log_dir'], safe_symbol, args.log_level)

    # ─── Connect to Exchange ───────────────────────────────────
    mode_str = 'DRY-RUN' if args.dry_run else ('TESTNET' if args.testnet else 'LIVE')
    print(f"\nConnecting to Binance {mode_str}...")

    if not executor.connect():
        print("ERROR: Failed to connect to exchange.")
        print("Check your API credentials and network connection.")
        sys.exit(1)

    # Set leverage
    lev = int(strategy_config['leverage'])
    executor.set_leverage(symbol, lev)

    # Get initial balance
    balance = executor.get_balance()
    initial_equity = args.capital or balance.get('free', 0)

    if initial_equity < 1 and not args.dry_run:
        print(f"ERROR: Wallet balance too low: ${initial_equity:.2f}")
        sys.exit(1)

    # Override initial capital with actual balance if not specified
    if not args.capital:
        strategy_config['initial_capital'] = initial_equity

    health_monitor = HealthMonitor(live_config, initial_equity)

    # ─── Create Runner ─────────────────────────────────────────
    runner = LiveRunner(
        executor=executor,
        symbol=symbol,
        strategy_config=strategy_config,
        live_config=live_config,
        state_manager=state_mgr,
        trade_logger=trade_logger,
        health_monitor=health_monitor,
    )

    # ─── Signal Handlers ───────────────────────────────────────
    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\nReceived {sig_name}, initiating graceful shutdown...")
        runner.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ─── Banner ────────────────────────────────────────────────
    border = '=' * 62
    print(f"\n{border}")
    print(f"  GridStrategyV4 LIVE TRADER | {mode_str}")
    print(f"{border}")
    print(f"  Symbol:     {symbol}")
    print(f"  Capital:    ${initial_equity:,.2f}")
    print(f"  Leverage:   {lev}x")
    print(f"  Short:      {'ON' if strategy_config['allow_short'] else 'OFF'}")
    print(f"  Grid Lvls:  {strategy_config['grid_levels']} per side")
    print(f"  Timeframe:  15m")
    print(f"  Resume:     {args.resume}")
    print(f"  State Dir:  {live_config['state_dir']}")
    print(f"  Log Dir:    {live_config['log_dir']}")
    print(f"{border}\n")

    # ─── Initialize and Run ────────────────────────────────────
    if not runner.initialize(resume=args.resume):
        print("ERROR: Failed to initialize runner. Check logs for details.")
        sys.exit(1)

    print("Starting live trading loop... (Ctrl+C to stop)\n")
    runner.run()

    print("\nLive trader stopped.")


if __name__ == "__main__":
    main()
