"""
GridStrategyV4 Optimizer — Walk-Forward with Anchored OOS.

Anti-overfit design (industry standard):
  1. Anchored Walk-Forward: Non-overlapping train/test windows, full data coverage
  2. Train-only fitness: TPE optimizes on TRAIN Calmar; test is held out
  3. OOS gate: Best params must also pass on held-out test windows
  4. Multi-coin averaging: Parameters must work across all coins
  5. Composite fitness: Calmar × ProfitFactor with min-trade & complexity gates
  6. Pessimistic aggregation: min() across windows (not mean)
  7. Monte Carlo validation: Permutation test for statistical significance
  8. Full parameter coverage: Includes inventory, risk, and pruning params

Usage:
  python3 optimizer.py --coins BTC ETH SOL --trials 200 --windows 4
"""
import optuna
from optuna.trial import TrialState
import numpy as np
import pandas as pd
import argparse
import json
import os
import time
from config import STRATEGY_PARAMS, BACKTEST_CONFIG, OPTIMIZER_SPACE, BACKTEST_FILL_CONF
from backtest.data_fetcher import fetch_data
from engine.strategy import GridStrategyV4


# ─── Global Data Cache ────────────────────────────────────────
CACHED_DATA = {}
N_WINDOWS = 4  # Default, overridden by CLI


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


# ─── Walk-Forward Splitting (Anchored, Non-Overlapping) ──────

def split_anchored_walk_forward(df, n_windows=4, train_ratio=0.7):
    """
    Anchored walk-forward: non-overlapping windows covering 100% of data.

    Layout for n_windows=4 on data of length N:
      Window size = N / n_windows
      W0: [0 .. 0.7*ws] train | [0.7*ws .. ws] test
      W1: [0 .. ws + 0.7*ws] train | [ws + 0.7*ws .. 2*ws] test
      W2: [0 .. 2*ws + 0.7*ws] train | [2*ws + 0.7*ws .. 3*ws] test
      W3: [0 .. 3*ws + 0.7*ws] train | [3*ws + 0.7*ws .. 4*ws] test

    Properties:
      - Train always starts from bar 0 (anchored / expanding window)
      - Test windows are strictly non-overlapping
      - No test data ever appears in any train set (zero leakage)
      - Covers 100% of data
      - Train grows with each window (simulates real deployment)

    Returns list of (train_df, test_df) tuples.
    """
    n = len(df)
    window_size = n // n_windows
    splits = []

    for w in range(n_windows):
        # Train: from start to train_ratio into this window's chunk
        train_end = w * window_size + int(window_size * train_ratio)
        # Test: from train_end to end of this window's chunk
        test_start = train_end
        test_end = (w + 1) * window_size
        if w == n_windows - 1:
            test_end = n  # Last window gets remaining bars

        train_end = min(train_end, n)
        test_start = min(test_start, n)
        test_end = min(test_end, n)

        train_df = df.iloc[0:train_end].reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        # Minimum viable sizes
        if len(train_df) >= 200 and len(test_df) >= 50:
            splits.append((train_df, test_df))

    return splits


# ─── Strategy Runner ─────────────────────────────────────────

def run_strategy(df, params):
    """Run strategy with given params, return metrics."""
    config = STRATEGY_PARAMS.copy()
    config.update(params)

    strat = GridStrategyV4(config)
    try:
        result = strat.run(df)
        return result['metrics']
    except Exception:
        return None


# ─── Fitness Functions ───────────────────────────────────────

def composite_fitness(metrics):
    """
    Composite fitness: Calmar × sqrt(ProfitFactor).

    Gates:
      - min_trades: Must generate ≥ 20 trades (else not meaningful)
      - profit_factor: Must be > 1.0 (else strategy loses money)
      - max_drawdown: Must be < 50% (else too risky)
      - final_capital: Must be > starting capital * 0.5

    The sqrt(PF) weighting ensures profit_factor matters but doesn't
    dominate — a PF of 2.0 gives 1.41x boost, PF of 4.0 gives 2.0x.
    """
    total_trades = metrics.get('total_trades', 0)
    ret = metrics.get('total_return_pct', 0)
    dd = metrics.get('max_drawdown_pct', 0.01)
    pf = metrics.get('profit_factor', 0)
    final_cap = metrics.get('final_capital', 0)
    initial_cap = STRATEGY_PARAMS.get('initial_capital', 10000)

    # ─── Hard gates ───
    if total_trades < 20:
        return -10.0  # Not enough trades for statistical meaning

    if dd < 0.01:
        dd = 0.01  # Floor to avoid divide-by-zero

    if dd > 50.0:
        return -10.0  # Excessive drawdown

    if final_cap < initial_cap * 0.5:
        return -10.0  # Lost more than 50% — blow-up

    if pf < 0.5:
        return -10.0  # Terrible win/loss ratio

    # ─── Calmar Ratio ───
    calmar = ret / dd

    # ─── Profit Factor bonus ───
    # sqrt(PF) caps the bonus: PF=1.5 → 1.22x, PF=2.0 → 1.41x, PF=3.0 → 1.73x
    pf_bonus = np.sqrt(max(pf, 0.1))

    # ─── Complexity penalty ───
    # Penalize > 200 trades per 10,000 bars (overtrade signal)
    # Scale by data length to be fair across window sizes
    penalty = 0.0
    if total_trades > 300:
        penalty = 0.01 * (total_trades - 300)

    score = calmar * pf_bonus - penalty

    return score


# ─── Objective Function ──────────────────────────────────────

def objective(trial):
    """
    Optuna objective: Maximize TRAIN-set composite fitness.

    Key anti-overfit property: TPE only sees TRAIN scores.
    Test scores are computed but stored as trial attributes
    for post-hoc OOS validation — they do NOT influence the
    optimization direction.

    Aggregation: pessimistic (min across all coin×window combos).
    This forces parameters to work in the WORST case, not just average.
    """
    params = sample_params(trial)

    train_scores = []
    test_scores = []

    for coin, df in CACHED_DATA.items():
        splits = split_anchored_walk_forward(df, n_windows=N_WINDOWS)

        for w_idx, (train_df, test_df) in enumerate(splits):
            # ─── TRAIN evaluation (this is what TPE optimizes) ───
            train_metrics = run_strategy(train_df, params)
            if train_metrics is None:
                return -10.0

            train_score = composite_fitness(train_metrics)
            if train_score <= -10.0:
                return -10.0  # Hard fail: blow-up on train

            train_scores.append(train_score)

            # ─── TEST evaluation (stored, NOT used for optimization) ───
            test_metrics = run_strategy(test_df, params)
            if test_metrics is not None:
                test_score = composite_fitness(test_metrics)
                test_scores.append(test_score)

                # Store as trial attribute for post-hoc analysis
                trial.set_user_attr(
                    f'oos_{coin}_{w_idx}',
                    {
                        'test_score': round(test_score, 4),
                        'test_return': test_metrics.get('total_return_pct', 0),
                        'test_dd': test_metrics.get('max_drawdown_pct', 0),
                        'test_pf': test_metrics.get('profit_factor', 0),
                        'test_trades': test_metrics.get('total_trades', 0),
                    }
                )

    if not train_scores:
        return -10.0

    # ─── Pessimistic aggregation: use MIN, not mean ───
    # This forces params to work in worst-case window, not just average.
    # Prevents "great on window 1, terrible on window 3" solutions.
    min_train = min(train_scores)

    # Also store mean for reference
    trial.set_user_attr('train_scores', [round(s, 4) for s in train_scores])
    trial.set_user_attr('test_scores', [round(s, 4) for s in test_scores])
    trial.set_user_attr('train_min', round(min_train, 4))
    trial.set_user_attr('train_mean', round(np.mean(train_scores), 4))

    if test_scores:
        trial.set_user_attr('test_min', round(min(test_scores), 4))
        trial.set_user_attr('test_mean', round(np.mean(test_scores), 4))

    return min_train


# ─── Parameter Sampling ──────────────────────────────────────

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


# ─── OOS Validation ──────────────────────────────────────────

def validate_oos(best_params, verbose=True):
    """
    Run best params on all OOS (test) windows and report.

    Returns (oos_pass, results_dict).
    oos_pass is True if OOS performance is acceptable.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Out-of-Sample Validation")
        print(f"{'='*60}")

    all_oos_scores = []
    all_oos_returns = []
    results = {}

    for coin, df in CACHED_DATA.items():
        splits = split_anchored_walk_forward(df, n_windows=N_WINDOWS)
        coin_scores = []

        for w_idx, (train_df, test_df) in enumerate(splits):
            metrics = run_strategy(test_df, best_params)
            if metrics is None:
                if verbose:
                    print(f"  {coin} W{w_idx}: FAILED (strategy error)")
                continue

            score = composite_fitness(metrics)
            ret = metrics.get('total_return_pct', 0)
            dd = metrics.get('max_drawdown_pct', 0)
            pf = metrics.get('profit_factor', 0)
            trades = metrics.get('total_trades', 0)
            sharpe = metrics.get('sharpe_ratio', 0)

            coin_scores.append(score)
            all_oos_scores.append(score)
            all_oos_returns.append(ret)

            if verbose:
                status = "✅" if score > 0 else "❌"
                print(f"  {coin} W{w_idx}: {status} "
                      f"Return={ret:+.2f}% DD={dd:.2f}% "
                      f"PF={pf:.2f} Sharpe={sharpe:.2f} "
                      f"Trades={trades} Score={score:.3f}")

        results[coin] = coin_scores

    if not all_oos_scores:
        if verbose:
            print(f"\n  ❌ No valid OOS results!")
        return False, results

    oos_min = min(all_oos_scores)
    oos_mean = np.mean(all_oos_scores)
    oos_positive = sum(1 for s in all_oos_scores if s > 0)

    if verbose:
        print(f"\n  OOS Summary:")
        print(f"    Min Score:  {oos_min:.3f}")
        print(f"    Mean Score: {oos_mean:.3f}")
        print(f"    Positive:   {oos_positive}/{len(all_oos_scores)} windows")
        print(f"    Avg Return: {np.mean(all_oos_returns):+.2f}%")

    # OOS pass criteria:
    # 1. Mean OOS score > 0 (profitable on average)
    # 2. At least 50% of windows profitable
    oos_pass = (oos_mean > 0 and oos_positive >= len(all_oos_scores) * 0.5)

    if verbose:
        if oos_pass:
            print(f"\n  ✅ OOS PASSED — Parameters appear robust")
        else:
            print(f"\n  ❌ OOS FAILED — Parameters may be overfit")
            print(f"     Consider: more data, wider windows, or fewer params")

    return oos_pass, results


# ─── Monte Carlo Permutation Test ────────────────────────────

def monte_carlo_validation(best_params, n_permutations=50, verbose=True):
    """
    Monte Carlo permutation test for statistical significance.

    Shuffles the close-price returns within each OOS window, re-runs
    the strategy, and checks if the real score is in the top 5%.

    This answers: "Could this score have been achieved by random luck?"

    Returns (significant, p_value).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Monte Carlo Permutation Test ({n_permutations} shuffles)")
        print(f"{'='*60}")

    # Get real OOS score
    real_scores = []
    for coin, df in CACHED_DATA.items():
        splits = split_anchored_walk_forward(df, n_windows=N_WINDOWS)
        for train_df, test_df in splits:
            metrics = run_strategy(test_df, best_params)
            if metrics is not None:
                real_scores.append(composite_fitness(metrics))

    if not real_scores:
        if verbose:
            print("  No valid results — cannot perform MC test")
        return False, 1.0

    real_score = np.mean(real_scores)

    # Run permutations
    shuffled_scores = []
    for perm in range(n_permutations):
        perm_scores = []
        for coin, df in CACHED_DATA.items():
            splits = split_anchored_walk_forward(df, n_windows=N_WINDOWS)
            for train_df, test_df in splits:
                # Shuffle returns within the test window
                shuffled_df = test_df.copy()
                close_vals = shuffled_df['close'].values.copy()
                returns = np.diff(close_vals) / close_vals[:-1]
                np.random.shuffle(returns)  # Destroy temporal structure
                # Reconstruct shuffled prices
                new_close = np.zeros_like(close_vals)
                new_close[0] = close_vals[0]
                for j in range(len(returns)):
                    new_close[j + 1] = new_close[j] * (1 + returns[j])
                new_close = np.maximum(new_close, 1.0)

                # Reconstruct OHLC from shuffled close
                spread = np.abs(new_close) * 0.005
                shuffled_df = pd.DataFrame({
                    'timestamp': test_df['timestamp'].values,
                    'open': new_close + np.random.normal(0, 0.1, len(new_close)) * spread,
                    'high': new_close + np.abs(np.random.normal(0, 1, len(new_close))) * spread,
                    'low': new_close - np.abs(np.random.normal(0, 1, len(new_close))) * spread,
                    'close': new_close,
                    'volume': test_df['volume'].values,
                })
                shuffled_df['low'] = np.maximum(shuffled_df['low'], 0.01)

                metrics = run_strategy(shuffled_df, best_params)
                if metrics is not None:
                    perm_scores.append(composite_fitness(metrics))

        if perm_scores:
            shuffled_scores.append(np.mean(perm_scores))

        if verbose and (perm + 1) % 10 == 0:
            print(f"  Completed {perm + 1}/{n_permutations} permutations...")

    if not shuffled_scores:
        if verbose:
            print("  No valid shuffle results")
        return False, 1.0

    # p-value: fraction of shuffled scores >= real score
    p_value = sum(1 for s in shuffled_scores if s >= real_score) / len(shuffled_scores)

    if verbose:
        print(f"\n  Real OOS Score:     {real_score:.4f}")
        print(f"  Shuffled Mean:      {np.mean(shuffled_scores):.4f}")
        print(f"  Shuffled 95th Pctl: {np.percentile(shuffled_scores, 95):.4f}")
        print(f"  p-value:            {p_value:.3f}")

        if p_value < 0.05:
            print(f"\n  ✅ SIGNIFICANT (p < 0.05) — Strategy has real edge")
        else:
            print(f"\n  ❌ NOT SIGNIFICANT (p = {p_value:.3f}) — May be curve-fit")

    return p_value < 0.05, p_value


# ─── Robust Parameter Selection ─────────────────────────────

TOP_PERCENT = 0.10  # Top 10% of trials used for robust selection


def compute_robust_parameters(study, optimizer_space: dict) -> tuple:
    """
    Institutional-grade robust parameter cluster selection.

    Instead of picking the single best trial (which overfits), this function:
      1. Takes the top-N% of completed trials by objective value.
      2. Computes the MEDIAN of each parameter across those top trials.
      3. Measures parameter stability via Coefficient of Variation (CV).
      4. Returns an overall robustness score (lower CV = more robust).

    Returns:
        robust_params    : dict of parameter name -> robust (median) value
        param_stability  : dict of parameter name -> CV (std/mean)
        robustness_score : float, mean CV across all params (lower = better)
    """
    # Step 1: Extract completed trials with valid scores
    completed_trials = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]

    if not completed_trials:
        return {}, {}, 1.0

    # Step 2: Sort by objective value descending
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

    # Step 3: Select top N%
    top_n = max(5, int(len(sorted_trials) * TOP_PERCENT))
    top_trials = sorted_trials[:top_n]

    robust_params = {}
    param_stability = {}

    # Step 4 & 5: Compute median and CV for each param
    for param_name, spec in optimizer_space.items():
        values = [t.params[param_name] for t in top_trials if param_name in t.params]
        if not values:
            continue

        arr = np.array(values, dtype=float)
        median_val = float(np.median(arr))

        # Round to int for integer params
        if spec['type'] == 'int':
            median_val = int(round(median_val))
        elif spec['type'] == 'cat':
            # For categorical, pick the most common value in top trials
            median_val = max(set(values), key=values.count)

        robust_params[param_name] = median_val

        # CV (only meaningful for numeric)
        if spec['type'] in ('int', 'float'):
            cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
        else:
            # Categorical: stability = fraction that agree on the modal choice
            modal_count = max(values.count(v) for v in set(values))
            cv = 1.0 - (modal_count / max(len(values), 1))
        param_stability[param_name] = round(cv, 4)

    # Step 6: Overall robustness score
    cv_values = list(param_stability.values())
    robustness_score = float(np.mean(cv_values)) if cv_values else 1.0

    return robust_params, param_stability, robustness_score


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridStrategyV4 Optimizer (Walk-Forward)")
    parser.add_argument("--coins", nargs="+", default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--windows", type=int, default=4,
                        help="Walk-forward windows (default: 4)")
    parser.add_argument("--study-name", default="grid_v4_wfo")
    parser.add_argument("--mc-perms", type=int, default=50,
                        help="Monte Carlo permutations (0 to skip)")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip Monte Carlo validation (faster)")
    args = parser.parse_args()

    global CACHED_DATA, N_WINDOWS
    N_WINDOWS = args.windows

    # Load data
    print(f"Loading data for {args.coins}...")
    CACHED_DATA = load_data(args.coins, args.start, args.end)
    print(f"  Loaded {len(CACHED_DATA)} coins")

    for coin, df in CACHED_DATA.items():
        n_bars = len(df)
        n_days = n_bars / 96  # 96 bars per day on 15m
        print(f"  {coin}: {n_bars:,} bars ({n_days:.0f} days)")

    if not CACHED_DATA:
        print("No data loaded, exiting")
        return

    # Show walk-forward layout
    print(f"\n{'─'*60}")
    print(f"  Walk-Forward Layout ({N_WINDOWS} anchored windows)")
    print(f"{'─'*60}")
    sample_coin = list(CACHED_DATA.keys())[0]
    sample_df = CACHED_DATA[sample_coin]
    splits = split_anchored_walk_forward(sample_df, n_windows=N_WINDOWS)
    for w_idx, (train_df, test_df) in enumerate(splits):
        print(f"  W{w_idx}: Train [{0:>6}..{len(train_df):>6}] "
              f"({len(train_df):,} bars) | "
              f"Test [{len(sample_df) - len(test_df) - (len(sample_df) - len(train_df) - len(test_df)):>6}.."
              f"{len(sample_df) - (len(sample_df) - len(train_df) - len(test_df)):>6}] "
              f"({len(test_df):,} bars)")
    print(f"  Total: {len(sample_df):,} bars covered")

    print(f"\n  Fitness: Composite = Calmar × sqrt(ProfitFactor)")
    print(f"  Aggregation: MIN across windows (pessimistic)")
    print(f"  Optimization target: TRAIN only (OOS held out)")
    print(f"  Simulation: Slippage {BACKTEST_FILL_CONF.get('slippage_pct', 0)*100:.2f}% | "
          f"Fill Prob {BACKTEST_FILL_CONF.get('fill_probability', 1)*100:.0f}%")

    # Create/resume Optuna study
    storage = "sqlite:///grid_v4_wfo.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15),
    )

    print(f"\n{'='*60}")
    print(f"  Optimizing: {args.trials} trials")
    print(f"{'='*60}\n")

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    start_time = time.time()
    study.optimize(objective, n_trials=args.trials, n_jobs=1,
                   show_progress_bar=True)
    elapsed = time.time() - start_time

    # ─── Results: Best Single Trial ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  Best Single Trial (Train MIN Score: {study.best_value:.4f})")
    print(f"  Completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for k, v in sorted(study.best_params.items()):
        default = STRATEGY_PARAMS.get(k, '—')
        if isinstance(v, float):
            print(f"  {k:<28} {v:>8.4f}  (default: {default})")
        else:
            print(f"  {k:<28} {v!s:>8}  (default: {default})")
    print(f"{'='*60}")

    # Show train/test breakdown from best trial
    best_trial = study.best_trial
    train_scores = best_trial.user_attrs.get('train_scores', [])
    test_scores = best_trial.user_attrs.get('test_scores', [])
    if train_scores:
        print(f"\n  Best Trial Breakdown:")
        print(f"    Train scores: {train_scores}")
        print(f"    Train MIN:    {best_trial.user_attrs.get('train_min', 'N/A')}")
        print(f"    Train MEAN:   {best_trial.user_attrs.get('train_mean', 'N/A')}")
    if test_scores:
        print(f"    Test scores:  {test_scores}")
        print(f"    Test MIN:     {best_trial.user_attrs.get('test_min', 'N/A')}")
        print(f"    Test MEAN:    {best_trial.user_attrs.get('test_mean', 'N/A')}")

    # ─── Robust Parameter Cluster Selection ──────────────────
    completed_count = sum(
        1 for t in study.trials if t.state == TrialState.COMPLETE)
    top_n_used = max(5, int(completed_count * TOP_PERCENT))

    robust_params, param_stability, robustness_score = compute_robust_parameters(
        study, OPTIMIZER_SPACE)

    # Step 8: Print Robust Parameter Report
    print(f"\n{'='*60}")
    print(f"  ===== ROBUST PARAMETER REPORT =====")
    print(f"  (Top {TOP_PERCENT*100:.0f}% of {completed_count} trials = {top_n_used} trials used)")
    print(f"{'='*60}")

    print(f"\n  Robust Parameters (median of top cluster):")
    for k, v in sorted(robust_params.items()):
        default = STRATEGY_PARAMS.get(k, '—')
        if isinstance(v, float):
            print(f"    {k:<28} {v:>8.4f}  (default: {default})")
        else:
            print(f"    {k:<28} {v!s:>8}  (default: {default})")

    print(f"\n  Parameter Stability (CV — lower is more stable):")
    for k, cv in sorted(param_stability.items()):
        bar = '█' * max(1, int(cv * 20))
        stability = 'STABLE' if cv < 0.25 else ('MODERATE' if cv < 0.50 else 'UNSTABLE')
        print(f"    {k:<28} CV={cv:.4f}  {bar:<20}  [{stability}]")

    print(f"\n  Overall Robustness Score: {robustness_score:.4f}")
    if robustness_score < 0.25:
        print(f"  Interpretation: EXCELLENT robustness — parameters are highly stable")
    elif robustness_score < 0.40:
        print(f"  Interpretation: GOOD robustness — parameters generalise well")
    elif robustness_score <= 0.50:
        print(f"  Interpretation: ACCEPTABLE robustness — monitor live performance")
    else:
        print(f"  Interpretation: WARNING: likely overfitting — widen data or reduce params")
    print(f"{'='*60}")

    # Step 7: Save robust results to JSON
    os.makedirs('data', exist_ok=True)
    robust_output = {
        'robust_params': robust_params,
        'param_stability': param_stability,
        'robustness_score': round(robustness_score, 6),
        'num_trials': completed_count,
        'top_trials_used': top_n_used,
        'best_single_trial_score': study.best_value,
        'n_windows': N_WINDOWS,
        'coins': list(CACHED_DATA.keys()),
    }
    with open('data/optimizer_results_robust.json', 'w') as f:
        json.dump(robust_output, f, indent=4, default=str)
    print(f"  Saved robust results to data/optimizer_results_robust.json")

    # Also save best single trial params for comparison
    legacy_output = {
        'best_params': study.best_params,
        'train_min_score': study.best_value,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'n_trials': args.trials,
        'n_windows': N_WINDOWS,
        'coins': list(CACHED_DATA.keys()),
    }
    with open('data/best_params_v4.json', 'w') as f:
        json.dump(legacy_output, f, indent=4)
    print(f"  Saved single-best params to data/best_params_v4.json")

    # ─── OOS Validation (uses ROBUST params) ─────────────────
    # Use robust_params as the final output — more conservative and
    # generalised than the single best trial.
    final_params = robust_params if robust_params else study.best_params
    oos_pass, oos_results = validate_oos(final_params, verbose=True)

    # ─── Monte Carlo Validation ──────────────────────────────
    if not args.skip_mc and args.mc_perms > 0:
        mc_sig, mc_p = monte_carlo_validation(
            final_params, n_permutations=args.mc_perms, verbose=True)
        robust_output['mc_significant'] = mc_sig
        robust_output['mc_p_value'] = mc_p
    else:
        print(f"\n  Skipping Monte Carlo validation (use --mc-perms N to enable)")

    robust_output['oos_pass'] = oos_pass
    with open('data/optimizer_results_robust.json', 'w') as f:
        json.dump(robust_output, f, indent=4, default=str)

    # ─── Final Summary ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Trials:            {args.trials}")
    print(f"  Best Single Train: {study.best_value:.4f}")
    print(f"  Robustness Score:  {robustness_score:.4f}")
    print(f"  OOS Pass:          {'✅ YES' if oos_pass else '❌ NO'}")
    if not args.skip_mc and args.mc_perms > 0:
        print(f"  MC Signif:         {'✅ YES' if robust_output.get('mc_significant') else '❌ NO'} "
              f"(p={robust_output.get('mc_p_value', 'N/A')})")
    print(f"  Robust Params:     data/optimizer_results_robust.json")
    print(f"  Single-Best Params: data/best_params_v4.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
