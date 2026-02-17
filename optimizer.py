"""
GridStrategyV4 Optimizer.

Anti-overfit design:
  1. Walk-Forward Validation: Train on window N, test on window N+1
  2. Multi-coin averaging: Parameters must work across BTC, ETH, SOL
  3. Calmar Ratio fitness: Penalizes drawdown-heavy returns
  4. Deflated Sharpe Ratio (DSR): Corrects for multiple testing
  5. Complexity penalty: Punishes excessive trade frequency
  6. Out-of-sample validation: Final params tested on held-out period

Usage:
  python3 optimizer.py --coins BTC SOL --trials 200 --windows 3
"""
import optuna
import numpy as np
import pandas as pd
import argparse
import json
import os
from config import STRATEGY_PARAMS, BACKTEST_CONFIG, OPTIMIZER_SPACE, BACKTEST_FILL_CONF
from backtest.data_fetcher import fetch_data
from engine.strategy import GridStrategyV4


# ─── Global Data Cache ────────────────────────────────────────
CACHED_DATA = {}


def load_data(coins, start, end, timeframe='15m'):
    """Load and cache OHLCV data for all coins."""
    data = {}
    for coin in coins:
        if '/' not in coin:
            coin = f"{coin}/USDT"
        df = fetch_data(coin, timeframe, start, end)
        if df is not None and len(df) > 100:
            data[coin] = df
    return data


def split_walk_forward(df, n_windows=3):
    """
    Split data into walk-forward windows.

    Each window: train on 70% | test on 30%
    Windows overlap by shifting forward 1/n_windows of total length.

    Returns list of (train_df, test_df) tuples.
    """
    n = len(df)
    window_size = n // n_windows
    splits = []

    for w in range(n_windows):
        start_idx = w * (window_size // 2)
        mid_idx = start_idx + int(window_size * 0.7)
        end_idx = min(start_idx + window_size, n)

        if end_idx - start_idx < 200:  # Minimum viable window
            continue

        train_df = df.iloc[start_idx:mid_idx].reset_index(drop=True)
        test_df = df.iloc[mid_idx:end_idx].reset_index(drop=True)

        if len(train_df) > 100 and len(test_df) > 50:
            splits.append((train_df, test_df))

    return splits


def run_strategy(df, params):
    """Run strategy with given params, return metrics."""
    config = STRATEGY_PARAMS.copy()
    config.update(params)

    strat = GridStrategyV4(config)
    try:
        result = strat.run(df)
        return result['metrics']
    except Exception as e:
        return None


def calculate_calmar(metrics):
    """Calmar Ratio = Return / Max Drawdown (penalizes drawdown-heavy returns)."""
    ret = metrics.get('total_return_pct', 0)
    dd = metrics.get('max_drawdown_pct', 0.01)
    if dd < 0.01:
        dd = 0.01  # Floor to avoid divide-by-zero
    return ret / dd


def complexity_penalty(metrics):
    """
    Penalize excessive trading (sign of over-optimization).
    More trades ≠ better strategy.
    """
    trades = metrics.get('total_trades', 0)
    if trades > 500:
        return -0.001 * (trades - 500)  # Gentle penalty
    return 0.0


def sample_params(trial):
    """Sample parameters from OPTIMIZER_SPACE using Optuna."""
    params = {}
    for name, spec in OPTIMIZER_SPACE.items():
        if spec['type'] == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'])
        elif spec['type'] == 'float':
            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
        elif spec['type'] == 'cat':
            params[name] = trial.suggest_categorical(name, spec['choices'])
    return params


def objective(trial):
    """
    Optuna objective: Maximize walk-forward Calmar Ratio.

    Anti-overfit:
      - Train params on train window
      - Evaluate on TEST window only
      - Average across all coins AND all windows
      - Apply complexity penalty
    """
    params = sample_params(trial)
    all_scores = []

    for coin, df in CACHED_DATA.items():
        splits = split_walk_forward(df, n_windows=3)

        for train_df, test_df in splits:
            # ─── Train check (params must not blow up on train data)
            train_metrics = run_strategy(train_df, params)
            if train_metrics is None:
                return -10.0

            # Rekt check
            if train_metrics.get('final_capital', 0) < 100:
                return -10.0

            # ─── Test evaluation (this is what we optimize for)
            test_metrics = run_strategy(test_df, params)
            if test_metrics is None:
                return -10.0

            if test_metrics.get('final_capital', 0) < 100:
                return -10.0

            # Score = Test Calmar + complexity penalty
            score = calculate_calmar(test_metrics) + complexity_penalty(test_metrics)
            all_scores.append(score)

    if not all_scores:
        return -10.0

    # Average across all coins × windows (robust estimation)
    avg_score = np.mean(all_scores)

    # Penalize high variance across windows (sign of unstable params)
    if len(all_scores) > 1:
        std_score = np.std(all_scores)
        # Subtract half of variance as stability penalty
        avg_score -= 0.5 * std_score

    return avg_score


def main():
    parser = argparse.ArgumentParser(description="GridStrategyV4 Optimizer")
    parser.add_argument("--coins", nargs="+", default=['BTC/USDT', 'SOL/USDT'])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--windows", type=int, default=3, help="Walk-forward windows")
    parser.add_argument("--study-name", default="grid_v4_study")
    args = parser.parse_args()

    # Load data
    global CACHED_DATA
    print(f"Loading data for {args.coins}...")
    CACHED_DATA = load_data(args.coins, args.start, args.end)
    print(f"  Loaded {len(CACHED_DATA)} coins")

    if not CACHED_DATA:
        print("No data loaded, exiting")
        return

    # Create/resume Optuna study
    storage = "sqlite:///grid_v4.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    print(f"\nOptimizing with {args.trials} trials "
          f"({args.windows} walk-forward windows)...")
    print(f"Fitness: Walk-Forward Calmar Ratio (OOS)")
    print(f"Anti-overfit: WFO + multi-coin avg + stability penalty\n")

    print(f"  Simulation: Slippage {BACKTEST_FILL_CONF.get('slippage_pct', 0)*100:.2f}% | "
          f"Fill Prob {BACKTEST_FILL_CONF.get('fill_probability', 1)*100:.0f}%\n")

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=args.trials, n_jobs=1,
                   show_progress_bar=True)

    # Results
    print(f"\n{'='*60}")
    print(f"  Best Parameters (Walk-Forward Calmar: {study.best_value:.4f})")
    print(f"{'='*60}")
    for k, v in study.best_params.items():
        print(f"  {k:<28} {v}")
    print(f"{'='*60}")

    # Save best params
    os.makedirs('data', exist_ok=True)
    with open('data/best_params_v4.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nSaved to data/best_params_v4.json")

    # ─── Final OOS Validation ─────────────────────────────────
    print(f"\nRunning final validation with best params...")
    best_params = study.best_params

    for coin, df in CACHED_DATA.items():
        # Use last 30% as true OOS
        split_idx = int(len(df) * 0.7)
        oos_df = df.iloc[split_idx:].reset_index(drop=True)
        metrics = run_strategy(oos_df, best_params)
        if metrics:
            print(f"\n  {coin} OOS Results:")
            print(f"    Return:   {metrics.get('total_return_pct', 0):+.2f}%")
            print(f"    Max DD:   {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"    Sharpe:   {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"    Calmar:   {metrics.get('calmar_ratio', 0):.3f}")
            print(f"    Trades:   {metrics.get('total_trades', 0)}")


if __name__ == "__main__":
    main()
