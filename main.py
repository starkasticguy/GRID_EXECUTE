"""
GridStrategyV4 CLI Entry Point.

Usage:
  python3 main.py --coins BTC ETH SOL --start 2024-01-01
  python3 main.py --coins BTC --start 2023-01-01 --end 2024-12-31 --capital 5000
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

from config import STRATEGY_PARAMS, BACKTEST_CONFIG
from backtest.data_fetcher import fetch_data
from engine.strategy import GridStrategyV4
from core.kama import REGIME_NAMES


# ─── TRADE LOG: PAIR ENTRIES WITH EXITS ────────────────────────────

def build_paired_trade_log(trades: list) -> pd.DataFrame:
    """
    Build a paired trade log that matches entries with exits.

    For each entry (BUY_OPEN_LONG / SELL_OPEN_SHORT), finds the
    corresponding exit (SELL_CLOSE_LONG / BUY_CLOSE_SHORT / STOP / PRUNE)
    and computes per-trade metrics.

    Returns a DataFrame with one row per round-trip trade.
    """
    if not trades:
        return pd.DataFrame()

    # Classify trades
    EXIT_LONG_LABELS = {'SELL_CLOSE_LONG', 'STOP_LONG', 'LIQUIDATION',
                        'PRUNE_DEVIANCE', 'PRUNE_OLDEST', 'PRUNE_GAP',
                        'PRUNE_FUNDING', 'PRUNE_OFFSET'}
    EXIT_SHORT_LABELS = {'BUY_CLOSE_SHORT', 'STOP_SHORT', 'LIQUIDATION',
                         'PRUNE_DEVIANCE', 'PRUNE_OLDEST', 'PRUNE_GAP',
                         'PRUNE_FUNDING', 'PRUNE_OFFSET'}

    # FIFO queues for unmatched entries
    long_entries = []
    short_entries = []

    paired = []

    for t in trades:
        label = t['label']

        if label == 'BUY_OPEN_LONG':
            long_entries.append(t)
        elif label == 'SELL_OPEN_SHORT':
            short_entries.append(t)
        elif label in EXIT_LONG_LABELS and long_entries:
            entry = long_entries.pop(0)  # FIFO
            paired.append(_make_pair(entry, t, 'LONG'))
        elif label in EXIT_SHORT_LABELS and short_entries:
            entry = short_entries.pop(0)  # FIFO
            paired.append(_make_pair(entry, t, 'SHORT'))
        elif label == 'CIRCUIT_BREAKER_HALT':
            continue  # Not a trade, just an event

    if not paired:
        return pd.DataFrame()

    df = pd.DataFrame(paired)

    # Compute additional metrics on the full log
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['trade_number'] = range(1, len(df) + 1)

    # Running win rate
    df['is_win'] = (df['pnl'] > 0).astype(int)
    df['running_win_rate'] = df['is_win'].expanding().mean() * 100

    # Running profit factor
    cum_gross_profit = df['pnl'].clip(lower=0).cumsum()
    cum_gross_loss = df['pnl'].clip(upper=0).abs().cumsum()
    df['running_profit_factor'] = cum_gross_profit / cum_gross_loss.replace(0, np.nan)
    df['running_profit_factor'] = df['running_profit_factor'].fillna(0)

    return df


def _make_pair(entry: dict, exit_t: dict, direction: str) -> dict:
    """Create a paired round-trip trade record."""
    entry_price = entry['price']
    exit_price = exit_t['price']
    qty = entry['qty']
    pnl = exit_t.get('pnl', 0.0)

    if direction == 'LONG':
        return_pct = ((exit_price - entry_price) / entry_price * 100
                      if entry_price > 1e-12 else 0.0)
    else:  # SHORT
        return_pct = ((entry_price - exit_price) / entry_price * 100
                      if entry_price > 1e-12 else 0.0)

    entry_ts = entry['timestamp']
    exit_ts = exit_t['timestamp']
    hold_ms = exit_ts - entry_ts
    hold_hours = hold_ms / (1000 * 3600) if hold_ms > 0 else 0

    # Risk/Reward: use |pnl| relative to position notional
    notional = abs(qty * entry_price)
    r_multiple = pnl / max(notional * 0.01, 1e-12)  # PnL per 1% of notional

    return {
        'direction': direction,
        'entry_time': pd.to_datetime(entry_ts, unit='ms'),
        'exit_time': pd.to_datetime(exit_ts, unit='ms'),
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'qty': round(qty, 8),
        'notional': round(notional, 2),
        'pnl': round(pnl, 4),
        'return_pct': round(return_pct, 4),
        'r_multiple': round(r_multiple, 4),
        'hold_hours': round(hold_hours, 2),
        'entry_label': entry['label'],
        'exit_label': exit_t['label'],
        'entry_regime': entry['regime'],
        'exit_regime': exit_t['regime'],
        'entry_bar': entry['bar'],
        'exit_bar': exit_t['bar'],
    }


def save_trade_log(coin: str, paired_df: pd.DataFrame, raw_trades: list,
                   metrics: dict, save_dir: str):
    """Save both paired trade log and raw trade log to CSV."""
    if paired_df.empty and not raw_trades:
        return

    os.makedirs(save_dir, exist_ok=True)
    safe_coin = coin.replace('/', '')

    # 1. Paired trade log (the main deliverable)
    if not paired_df.empty:
        path = os.path.join(save_dir, f'{safe_coin}_trade_log.csv')
        paired_df.to_csv(path, index=False)
        print(f"  Trade log saved to {path}")

    # 2. Raw trades (all events including CB halts, for debugging)
    if raw_trades:
        raw_df = pd.DataFrame(raw_trades)
        raw_df['datetime'] = pd.to_datetime(raw_df['timestamp'], unit='ms')
        path_raw = os.path.join(save_dir, f'{safe_coin}_raw_trades.csv')
        raw_df.to_csv(path_raw, index=False)
        print(f"  Raw trades saved to {path_raw}")

    # 3. Performance summary CSV
    summary_path = os.path.join(save_dir, f'{safe_coin}_summary.csv')
    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")


# ─── CONSOLE OUTPUT ────────────────────────────────────────────────

def print_metrics(coin: str, metrics: dict):
    """Print formatted metrics table."""
    print(f"\n{'='*62}")
    print(f"  {coin} -- Performance Summary")
    print(f"{'='*62}")

    # Section 1: Performance
    print(f"  --- Performance ---")
    perf_rows = [
        ("Total Return",   f"{metrics.get('total_return_pct', 0):+.2f}%"),
        ("Buy & Hold",     f"{metrics.get('buy_hold_return_pct', 0):+.2f}%"),
        ("Max Drawdown",   f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Final Capital",  f"${metrics.get('final_capital', 0):,.2f}"),
    ]
    for label, value in perf_rows:
        print(f"  {label:<20} {value:>12}")

    # Section 2: Ratios
    print(f"\n  --- Risk-Adjusted Ratios ---")
    ratio_rows = [
        ("Sharpe Ratio",   f"{metrics.get('sharpe_ratio', 0):.3f}"),
        ("Sortino Ratio",  f"{metrics.get('sortino_ratio', 0):.3f}"),
        ("Calmar Ratio",   f"{metrics.get('calmar_ratio', 0):.3f}"),
        ("Win Rate",       f"{metrics.get('win_rate_pct', 0):.1f}%"),
        ("Profit Factor",  f"{metrics.get('profit_factor', 0):.3f}"),
    ]
    for label, value in ratio_rows:
        print(f"  {label:<20} {value:>12}")

    # Section 3: Trading Activity
    print(f"\n  --- Trading Activity ---")
    trade_rows = [
        ("Total Trades",   f"{metrics.get('total_trades', 0)}"),
        ("Longs Opened",   f"{metrics.get('longs_opened', 0)}"),
        ("Longs Closed",   f"{metrics.get('longs_closed', 0)}"),
        ("Shorts Opened",  f"{metrics.get('shorts_opened', 0)}"),
        ("Shorts Closed",  f"{metrics.get('shorts_closed', 0)}"),
        ("Gross Profit",   f"${metrics.get('gross_profit', 0):,.2f}"),
        ("Gross Loss",     f"${metrics.get('gross_loss', 0):,.2f}"),
    ]
    for label, value in trade_rows:
        print(f"  {label:<20} {value:>12}")

    # Section 4: Risk Events
    print(f"\n  --- Risk Events ---")
    risk_rows = [
        ("Stops (L/S)",    f"{metrics.get('stops_long', 0)} / {metrics.get('stops_short', 0)}"),
        ("Pruned",         f"{metrics.get('prune_count', 0)}"),
        ("CB Triggers",    f"{metrics.get('circuit_breaker_triggers', 0)}"),
        ("Trail Shifts",   f"{metrics.get('trailing_shifts', 0)}"),
        ("VaR Blocks",     f"{metrics.get('var_blocks', 0)}"),
        ("Liquidations",   f"{metrics.get('liquidations', 0)}"),
        ("Funding PnL",    f"${metrics.get('funding_pnl', 0):+.2f}"),
    ]
    for label, value in risk_rows:
        print(f"  {label:<20} {value:>12}")

    # Prune breakdown
    prune_types = metrics.get('prune_types', {})
    if prune_types:
        print(f"\n  --- Prune Breakdown ---")
        for ptype, count in prune_types.items():
            short_name = ptype.replace('PRUNE_', '')
            print(f"    {short_name:<23} {count:>5}")

    print(f"{'='*62}")


def print_trade_log_summary(paired_df: pd.DataFrame):
    """Print a summary of the paired trade log."""
    if paired_df.empty:
        return

    n = len(paired_df)
    longs = paired_df[paired_df['direction'] == 'LONG']
    shorts = paired_df[paired_df['direction'] == 'SHORT']
    winners = paired_df[paired_df['pnl'] > 0]
    losers = paired_df[paired_df['pnl'] < 0]

    print(f"\n  --- Trade Log Summary ---")
    print(f"  {'Round-Trip Trades':<20} {n:>12}")
    print(f"  {'  Long Trades':<20} {len(longs):>12}")
    print(f"  {'  Short Trades':<20} {len(shorts):>12}")
    print(f"  {'Winners':<20} {len(winners):>12}")
    print(f"  {'Losers':<20} {len(losers):>12}")

    if len(winners) > 0:
        avg_win = winners['pnl'].mean()
        best_trade = winners['pnl'].max()
        print(f"  {'Avg Win':<20} {'$' + f'{avg_win:.2f}':>12}")
        print(f"  {'Best Trade':<20} {'$' + f'{best_trade:.2f}':>12}")
    if len(losers) > 0:
        avg_loss = losers['pnl'].mean()
        worst_trade = losers['pnl'].min()
        print(f"  {'Avg Loss':<20} {'$' + f'{avg_loss:.2f}':>12}")
        print(f"  {'Worst Trade':<20} {'$' + f'{worst_trade:.2f}':>12}")

    print(f"  {'Avg Hold Time':<20} {paired_df['hold_hours'].mean():>10.1f}h")
    print(f"  {'Avg Return':<20} {paired_df['return_pct'].mean():>+10.4f}%")
    print(f"  {'Avg R-Multiple':<20} {paired_df['r_multiple'].mean():>+10.4f}")

    # Exit type breakdown
    exit_counts = paired_df['exit_label'].value_counts()
    if len(exit_counts) > 0:
        print(f"\n  --- Exit Reasons ---")
        for label, count in exit_counts.items():
            pct = count / n * 100
            short_label = label.replace('SELL_CLOSE_', '').replace('BUY_CLOSE_', '')
            print(f"    {short_label:<23} {count:>5} ({pct:>5.1f}%)")


# ─── CHARTS ────────────────────────────────────────────────────────

def plot_results(coin: str, result: dict, df: pd.DataFrame,
                 paired_df: pd.DataFrame, save_dir: str):
    """Generate publication-quality charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)
    safe_coin = coin.replace('/', '')

    ec = result['equity_curve']
    trades = result['trades']
    regime_log = result['regime_log']
    metrics = result['metrics']
    closes = df['close'].values.astype(np.float64)
    dates = pd.to_datetime(df['timestamp'].values.astype(np.float64), unit='ms')

    # ─── Style setup ────────────────────────────────────────────
    BG_COLOR = '#0e1117'
    GRID_COLOR = '#1e2530'
    TEXT_COLOR = '#e0e0e0'
    EQUITY_COLOR = '#00d4aa'
    BH_COLOR = '#ff6b6b'
    PRICE_COLOR = '#a0a0a0'
    DD_COLOR = '#ff4444'

    REGIME_COLORS = {
        0: '#2a2e35',    # NOISE - dark gray
        1: '#0a3d1a',    # UPTREND - dark green
        -1: '#3d0a0a',   # DOWNTREND - dark red
        2: '#0a3d3d',    # BREAKOUT_UP - dark teal
        -2: '#3d1a0a',   # BREAKOUT_DOWN - dark orange
    }

    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'axes.edgecolor': '#333',
        'axes.labelcolor': TEXT_COLOR,
        'text.color': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.4,
        'font.size': 10,
    })

    # ===========================================================
    # CHART 1: Equity Curve + Drawdown + Regime
    # ===========================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [4, 1.2, 0.8]},
                             sharex=True)
    fig.suptitle(f'{coin} -- GridStrategyV4 Backtest',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

    # -- Panel 1: Equity Curve --
    ax1 = axes[0]

    # Regime background shading
    _draw_regime_shading(ax1, dates, regime_log, REGIME_COLORS)

    # Buy & Hold line
    bh_equity = ec[0] * (closes / closes[0])
    ax1.plot(dates, bh_equity, color=BH_COLOR, linewidth=1.0, alpha=0.6,
             label=f'Buy & Hold ({metrics.get("buy_hold_return_pct", 0):+.1f}%)')

    # Strategy equity
    ax1.plot(dates, ec, color=EQUITY_COLOR, linewidth=1.5,
             label=f'Grid V4 ({metrics.get("total_return_pct", 0):+.1f}%)')

    # Initial capital line
    ax1.axhline(y=ec[0], color='#555', linestyle='--', alpha=0.5, linewidth=0.8)

    ax1.set_ylabel('Equity ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Legend with metrics
    legend = ax1.legend(loc='upper left', fontsize=9, framealpha=0.7,
                        facecolor='#1a1f2e', edgecolor='#333')
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # Metrics annotation box
    metrics_text = (
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Sortino: {metrics.get('sortino_ratio', 0):.2f}  |  "
        f"Calmar: {metrics.get('calmar_ratio', 0):.2f}\n"
        f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%  |  "
        f"PF: {metrics.get('profit_factor', 0):.2f}  |  "
        f"Trades: {metrics.get('total_trades', 0)}  |  "
        f"Max DD: {metrics.get('max_drawdown_pct', 0):.1f}%"
    )
    ax1.text(0.99, 0.02, metrics_text, transform=ax1.transAxes,
             fontsize=8, color='#999', ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1f2e',
                       edgecolor='#333', alpha=0.85))

    # -- Panel 2: Drawdown --
    ax2 = axes[1]
    peak = np.maximum.accumulate(ec)
    drawdown = (peak - ec) / peak * 100
    ax2.fill_between(dates, drawdown, color=DD_COLOR, alpha=0.4)
    ax2.plot(dates, drawdown, color=DD_COLOR, linewidth=0.8, alpha=0.7)
    ax2.set_ylabel('DD %', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    max_dd_val = max(drawdown) if len(drawdown) > 0 else 1
    ax2.set_ylim(max_dd_val * 1.2, 0)

    # -- Panel 3: Regime strip --
    ax3 = axes[2]
    _draw_regime_strip(ax3, dates, regime_log, REGIME_COLORS)
    ax3.set_ylabel('Regime', fontsize=9)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_yticks([])

    # Regime legend
    regime_handles = []
    regime_labels_list = []
    for code in sorted(REGIME_COLORS.keys()):
        if np.any(regime_log == code):
            name = REGIME_NAMES.get(code, str(code))
            regime_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=REGIME_COLORS[code], ec='none', alpha=0.6))
            regime_labels_list.append(name)
    if regime_handles:
        ax3.legend(regime_handles, regime_labels_list,
                   loc='center left', bbox_to_anchor=(1.01, 0.5),
                   fontsize=7, framealpha=0.7, facecolor='#1a1f2e',
                   edgecolor='#333', labelcolor=TEXT_COLOR)

    fig.subplots_adjust(right=0.88)
    plt.savefig(os.path.join(save_dir, f'{safe_coin}_equity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ===========================================================
    # CHART 2: Trade Map on Price
    # ===========================================================
    if trades:
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle(f'{coin} -- Trade Map',
                     fontsize=14, fontweight='bold', color=TEXT_COLOR)

        # Regime shading
        _draw_regime_shading(ax, dates, regime_log, REGIME_COLORS)

        # Price line
        ax.plot(dates, closes, color=PRICE_COLOR, linewidth=0.8, alpha=0.7,
                label='Price')

        # Classify and plot trades
        trade_groups = {
            'BUY_OPEN_LONG':    {'color': '#00e676', 'marker': '^', 'size': 30, 'zorder': 5},
            'SELL_CLOSE_LONG':  {'color': '#448aff', 'marker': 'v', 'size': 30, 'zorder': 5},
            'SELL_OPEN_SHORT':  {'color': '#ff5252', 'marker': 'v', 'size': 30, 'zorder': 5},
            'BUY_CLOSE_SHORT':  {'color': '#ffab40', 'marker': '^', 'size': 30, 'zorder': 5},
            'STOP_LONG':        {'color': '#ff1744', 'marker': 'X', 'size': 50, 'zorder': 6},
            'STOP_SHORT':       {'color': '#ff1744', 'marker': 'X', 'size': 50, 'zorder': 6},
            'PRUNE':            {'color': '#e040fb', 'marker': 'D', 'size': 25, 'zorder': 5},
            'CIRCUIT_BREAKER':  {'color': '#ffff00', 'marker': 's', 'size': 40, 'zorder': 7},
            'LIQUIDATION':      {'color': '#ffffff', 'marker': '*', 'size': 80, 'zorder': 8},
        }

        # Collect by group for batch scatter
        group_data = {k: {'x': [], 'y': []} for k in trade_groups}

        for t in trades:
            bar = t['bar']
            if bar >= len(dates):
                continue
            label = t['label']

            group_key = None
            if label in trade_groups:
                group_key = label
            elif 'PRUNE' in label:
                group_key = 'PRUNE'
            elif 'CIRCUIT_BREAKER' in label:
                group_key = 'CIRCUIT_BREAKER'
            elif 'STOP' in label:
                group_key = 'STOP_LONG' if 'LONG' in label else 'STOP_SHORT'

            if group_key and group_key in group_data:
                group_data[group_key]['x'].append(dates[bar])
                group_data[group_key]['y'].append(t['price'])

        for key, data_pts in group_data.items():
            if not data_pts['x']:
                continue
            style = trade_groups[key]
            short_name = key.replace('_', ' ').title()
            ax.scatter(data_pts['x'], data_pts['y'],
                      c=style['color'], marker=style['marker'],
                      s=style['size'], alpha=0.8, edgecolors='none',
                      zorder=style['zorder'], label=short_name)

        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Legend
        legend = ax.legend(loc='upper left', fontsize=8, framealpha=0.7,
                          facecolor='#1a1f2e', edgecolor='#333', ncol=2)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        plt.savefig(os.path.join(save_dir, f'{safe_coin}_trades.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ===========================================================
    # CHART 3: Performance Dashboard
    # ===========================================================
    if not paired_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{coin} -- Performance Dashboard',
                     fontsize=14, fontweight='bold', color=TEXT_COLOR)

        # -- Panel A: Cumulative PnL --
        ax_a = axes[0, 0]
        cum_pnl = paired_df['cumulative_pnl'].values
        trade_nums = paired_df['trade_number'].values
        ax_a.fill_between(trade_nums, cum_pnl, 0,
                         where=cum_pnl >= 0, color=EQUITY_COLOR, alpha=0.3)
        ax_a.fill_between(trade_nums, cum_pnl, 0,
                         where=cum_pnl < 0, color=DD_COLOR, alpha=0.3)
        ax_a.plot(trade_nums, cum_pnl, color=EQUITY_COLOR, linewidth=1.2)
        ax_a.axhline(y=0, color='#555', linestyle='--', linewidth=0.8)
        ax_a.set_title('Cumulative PnL by Trade', fontsize=11, color=TEXT_COLOR)
        ax_a.set_xlabel('Trade #', fontsize=9)
        ax_a.set_ylabel('PnL ($)', fontsize=9)
        ax_a.grid(True, alpha=0.3)

        # -- Panel B: PnL Distribution --
        ax_b = axes[0, 1]
        pnls = paired_df['pnl'].values
        pnl_wins = pnls[pnls > 0]
        pnl_losses = pnls[pnls < 0]
        if len(pnls) > 0:
            bins = min(50, max(10, len(pnls) // 5))
            if len(pnl_wins) > 0:
                ax_b.hist(pnl_wins, bins=bins, color=EQUITY_COLOR, alpha=0.7,
                         label=f'Wins ({len(pnl_wins)})', edgecolor='none')
            if len(pnl_losses) > 0:
                ax_b.hist(pnl_losses, bins=bins, color=DD_COLOR, alpha=0.7,
                         label=f'Losses ({len(pnl_losses)})', edgecolor='none')
            ax_b.axvline(x=np.mean(pnls), color='#ffff00', linestyle='--',
                        linewidth=1.0, label=f'Mean: ${np.mean(pnls):.2f}')
        ax_b.set_title('PnL Distribution', fontsize=11, color=TEXT_COLOR)
        ax_b.set_xlabel('PnL ($)', fontsize=9)
        ax_b.set_ylabel('Count', fontsize=9)
        ax_b.grid(True, alpha=0.3)
        legend = ax_b.legend(fontsize=8, framealpha=0.7,
                            facecolor='#1a1f2e', edgecolor='#333')
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        # -- Panel C: Return % Distribution --
        ax_c = axes[1, 0]
        returns = paired_df['return_pct'].values
        if len(returns) > 0:
            bins = min(50, max(10, len(returns) // 5))
            ax_c.hist(returns, bins=bins, color='#7c4dff', alpha=0.7,
                     edgecolor='none')
            ax_c.axvline(x=np.mean(returns), color='#ffff00', linestyle='--',
                        linewidth=1.0, label=f'Mean: {np.mean(returns):.3f}%')
            ax_c.axvline(x=0, color='#555', linestyle='-', linewidth=0.8)
        ax_c.set_title('Return % per Trade', fontsize=11, color=TEXT_COLOR)
        ax_c.set_xlabel('Return %', fontsize=9)
        ax_c.set_ylabel('Count', fontsize=9)
        ax_c.grid(True, alpha=0.3)
        legend = ax_c.legend(fontsize=8, framealpha=0.7,
                            facecolor='#1a1f2e', edgecolor='#333')
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        # -- Panel D: Hold Time vs Return --
        ax_d = axes[1, 1]
        hold = paired_df['hold_hours'].values
        ret = paired_df['return_pct'].values
        colors_scatter = [EQUITY_COLOR if r >= 0 else DD_COLOR for r in ret]
        ax_d.scatter(hold, ret, c=colors_scatter, s=15, alpha=0.6, edgecolors='none')
        ax_d.axhline(y=0, color='#555', linestyle='--', linewidth=0.8)
        ax_d.set_title('Hold Time vs Return', fontsize=11, color=TEXT_COLOR)
        ax_d.set_xlabel('Hold Time (hours)', fontsize=9)
        ax_d.set_ylabel('Return %', fontsize=9)
        ax_d.grid(True, alpha=0.3)

        # Add quadrant labels
        ax_d.text(0.02, 0.98, 'Quick Wins', transform=ax_d.transAxes,
                 fontsize=7, color=EQUITY_COLOR, alpha=0.5, va='top')
        ax_d.text(0.98, 0.98, 'Slow Wins', transform=ax_d.transAxes,
                 fontsize=7, color=EQUITY_COLOR, alpha=0.5, va='top', ha='right')
        ax_d.text(0.02, 0.02, 'Quick Losses', transform=ax_d.transAxes,
                 fontsize=7, color=DD_COLOR, alpha=0.5, va='bottom')
        ax_d.text(0.98, 0.02, 'Slow Losses', transform=ax_d.transAxes,
                 fontsize=7, color=DD_COLOR, alpha=0.5, va='bottom', ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{safe_coin}_dashboard.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Charts saved to {save_dir}/")

    # Reset style
    plt.rcParams.update(plt.rcParamsDefault)


def _draw_regime_shading(ax, dates, regime_log, colors):
    """Draw regime-colored background shading on an axis."""
    n = len(dates)
    if n < 2:
        return

    i = 0
    while i < n:
        regime = regime_log[i]
        j = i
        while j < n and regime_log[j] == regime:
            j += 1
        color = colors.get(regime, '#2a2e35')
        ax.axvspan(dates[i], dates[min(j, n - 1)],
                  color=color, alpha=0.3, zorder=0)
        i = j


def _draw_regime_strip(ax, dates, regime_log, colors):
    """Draw a thin regime color strip."""
    n = len(dates)
    if n < 2:
        return

    i = 0
    while i < n:
        regime = regime_log[i]
        j = i
        while j < n and regime_log[j] == regime:
            j += 1
        color = colors.get(regime, '#2a2e35')
        name = REGIME_NAMES.get(regime, '?')
        ax.axvspan(dates[i], dates[min(j, n - 1)],
                  color=color, alpha=0.6, zorder=0)
        # Label if segment is wide enough
        mid_idx = (i + j) // 2
        if j - i > n * 0.03 and mid_idx < n:
            ax.text(dates[mid_idx], 0.5, name, ha='center', va='center',
                   fontsize=6, color='#999', alpha=0.8,
                   transform=ax.get_xaxis_transform())
        i = j


# ─── MAIN ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridStrategyV4 Backtest Runner")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Coins to test (e.g. BTC ETH SOL)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None, help="Initial capital")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-short", action="store_true", help="Disable short grid")
    args = parser.parse_args()

    # Build config
    config = STRATEGY_PARAMS.copy()
    if args.capital:
        config['initial_capital'] = args.capital
    if args.no_short:
        config['allow_short'] = False

    coins = args.coins or [c.replace('/USDT', '') for c in BACKTEST_CONFIG['coins']]
    start = args.start or BACKTEST_CONFIG['start_date']
    end = args.end or BACKTEST_CONFIG.get('end_date')

    print(f"\n{'='*62}")
    print(f"  GridStrategyV4 | Hedge Mode | KAMA Regime | 15m Candles")
    print(f"  Capital: ${config['initial_capital']:,.0f} | "
          f"Leverage: {config['leverage']}x | "
          f"Short: {'ON' if config['allow_short'] else 'OFF'}")
    print(f"  Coins: {coins} | Period: {start} -> {end or 'now'}")
    print(f"{'='*62}")

    for raw_coin in coins:
        coin = raw_coin if '/' in raw_coin else f"{raw_coin}/USDT"

        print(f"\nFetching {coin}...")
        df = fetch_data(coin, BACKTEST_CONFIG['timeframe'], start, end)

        if df is None or len(df) < 100:
            print(f"  Insufficient data for {coin}, skipping")
            continue

        print(f"  {len(df)} candles loaded "
              f"({pd.to_datetime(df['timestamp'].iloc[0], unit='ms').strftime('%Y-%m-%d')} -> "
              f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms').strftime('%Y-%m-%d')})")

        # Run strategy
        strat = GridStrategyV4(config)
        result = strat.run(df)

        # Build paired trade log
        paired_df = build_paired_trade_log(result['trades'])

        # Print metrics
        print_metrics(coin, result['metrics'])

        # Print trade log summary
        print_trade_log_summary(paired_df)

        # Save trade logs + summary
        save_trade_log(coin, paired_df, result['trades'],
                       result['metrics'], 'data/output')

        # Generate charts
        if not args.no_plot:
            plot_results(coin, result, df, paired_df, 'data/charts')


if __name__ == "__main__":
    main()
