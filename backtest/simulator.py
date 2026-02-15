"""
Backtest Simulator.

Thin orchestration layer that:
  1. Loads data via data_fetcher
  2. Instantiates GridStrategyV4
  3. Runs the backtest
  4. Returns results for analysis

The actual backtest kernel lives in engine/strategy.py.
"""
import pandas as pd
import numpy as np
from config import STRATEGY_PARAMS, BACKTEST_CONFIG
from engine.strategy import GridStrategyV4
from backtest.data_fetcher import fetch_data


class BacktestSimulator:
    """Run a full backtest for one or more coins."""

    def __init__(self, config: dict = None, backtest_config: dict = None):
        self.config = config or STRATEGY_PARAMS.copy()
        self.bt_config = backtest_config or BACKTEST_CONFIG.copy()
        self.results = {}

    def run_single(self, coin: str, start: str = None, end: str = None) -> dict:
        """
        Run backtest for a single coin.

        Returns dict with equity_curve, trades, metrics.
        """
        start = start or self.bt_config['start_date']
        end = end or self.bt_config.get('end_date')
        tf = self.bt_config.get('timeframe', '15m')

        # Fetch data
        df = fetch_data(coin, tf, start, end)
        if df is None or len(df) < 100:
            print(f"  Insufficient data for {coin}")
            return None

        # Run strategy
        strat = GridStrategyV4(self.config)
        result = strat.run(df)

        self.results[coin] = result
        return result

    def run_multi(self, coins: list = None, start: str = None,
                  end: str = None) -> dict:
        """
        Run backtest for multiple coins.

        Returns dict of {coin: result}.
        """
        coins = coins or self.bt_config['coins']
        start = start or self.bt_config['start_date']
        end = end or self.bt_config.get('end_date')

        all_results = {}
        for coin in coins:
            # Normalize symbol
            if '/' not in coin:
                coin = f"{coin}/USDT"
            print(f"\n{'='*50}")
            print(f"  Running {coin}...")
            print(f"{'='*50}")
            result = self.run_single(coin, start, end)
            if result:
                all_results[coin] = result

        # Portfolio aggregate
        if all_results:
            all_results['_portfolio'] = self._compute_portfolio(all_results)

        return all_results

    def _compute_portfolio(self, results: dict) -> dict:
        """Aggregate metrics across all coins (equal weight)."""
        n_coins = len(results)
        if n_coins == 0:
            return {}

        metrics_keys = ['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio',
                        'sortino_ratio', 'calmar_ratio', 'win_rate_pct',
                        'profit_factor', 'total_trades']

        portfolio_metrics = {}
        for key in metrics_keys:
            values = [r['metrics'].get(key, 0) for r in results.values()
                      if 'metrics' in r]
            portfolio_metrics[key] = round(sum(values) / max(len(values), 1), 2)

        # Combined final capital (sum of all)
        portfolio_metrics['final_capital'] = round(
            sum(r['metrics'].get('final_capital', 0) for r in results.values()
                if 'metrics' in r), 2)

        return {'metrics': portfolio_metrics}
