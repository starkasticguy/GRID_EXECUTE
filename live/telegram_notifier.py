"""
live/telegram_notifier.py — Telegram alert integration.

Sends bot status messages to a Telegram chat using the Bot API.
Uses only stdlib (urllib) — no extra dependencies.

Configure via environment variables:
    TELEGRAM_BOT_TOKEN  — from BotFather
    TELEGRAM_CHAT_ID    — your personal chat ID

Failures are silently swallowed so a Telegram outage never
interrupts the trading bot.
"""
import os
import json
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
_DEFAULT_CHAT  = os.getenv('TELEGRAM_CHAT_ID', '')


class TelegramNotifier:
    """Fire-and-forget Telegram message sender.

    Sends messages in a background thread so the main loop
    is never blocked by network latency.
    """

    def __init__(self, token: str = '', chat_id: str = ''):
        self.token   = token   or _DEFAULT_TOKEN
        self.chat_id = chat_id or _DEFAULT_CHAT
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.warning(
                'TelegramNotifier: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID '
                'not set — notifications disabled.')

    # ─── Public API ──────────────────────────────────────────

    def send(self, text: str, silent: bool = False):
        """Send a plain-text message (non-blocking)."""
        if not self.enabled:
            return
        threading.Thread(
            target=self._post,
            args=(text, 'HTML', silent),
            daemon=True,
        ).start()

    def send_now(self, text: str, silent: bool = False):
        """Send synchronously (blocks until sent or timeout)."""
        if not self.enabled:
            return
        self._post(text, 'HTML', silent)

    # ─── Internal ────────────────────────────────────────────

    def _post(self, text: str, parse_mode: str = 'HTML', silent: bool = False):
        url = f'https://api.telegram.org/bot{self.token}/sendMessage'
        payload = json.dumps({
            'chat_id':                  self.chat_id,
            'text':                     text,
            'parse_mode':               parse_mode,
            'disable_notification':     silent,
            'disable_web_page_preview': True,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status != 200:
                    logger.debug(f'Telegram HTTP {resp.status}')
        except urllib.error.URLError as e:
            logger.debug(f'Telegram send failed: {e}')
        except Exception as e:
            logger.debug(f'Telegram unexpected error: {e}')


# ─── Module-level singleton ───────────────────────────────────────
# Populated by live_trade.py / runner after reading config.
_notifier: TelegramNotifier | None = None


def init_notifier(token: str = '', chat_id: str = '') -> TelegramNotifier:
    """Create and register the module-level singleton."""
    global _notifier
    _notifier = TelegramNotifier(token=token, chat_id=chat_id)
    return _notifier


def tg_send(text: str, silent: bool = False):
    """Send via the singleton (no-op if not initialized)."""
    if _notifier:
        _notifier.send(text, silent=silent)


# ─── Message helpers ──────────────────────────────────────────────

def fmt_start(symbol: str, capital: float, leverage: float, mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return (
        f'🟢 <b>GRID BOT STARTED</b>\n'
        f'Symbol:   <code>{symbol}</code>\n'
        f'Capital:  <code>${capital:,.2f}</code>\n'
        f'Leverage: <code>{leverage}x</code>\n'
        f'Mode:     <code>{mode}</code>\n'
        f'Time:     <code>{ts}</code>'
    )


def fmt_stop(symbol: str, reason: str = 'graceful') -> str:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return (
        f'🔴 <b>GRID BOT STOPPED</b>\n'
        f'Symbol: <code>{symbol}</code>\n'
        f'Reason: <code>{reason}</code>\n'
        f'Time:   <code>{ts}</code>'
    )


def fmt_trade(label: str, symbol: str, price: float, qty: float,
              pnl: float, regime: str, equity: float) -> str:
    emoji = {
        'BUY_OPEN_LONG':    '📈 BUY OPEN',
        'SELL_CLOSE_LONG':  '✅ SELL CLOSE',
        'SELL_OPEN_SHORT':  '📉 SELL OPEN',
        'BUY_CLOSE_SHORT':  '✅ BUY CLOSE',
        'STOP_LONG':        '🛑 STOP LONG',
        'STOP_SHORT':       '🛑 STOP SHORT',
        'PRUNE_DEVIANCE':   '✂️ PRUNE DEVIANCE',
        'PRUNE_GAP':        '✂️ PRUNE GAP',
        'PRUNE_OLDEST':     '✂️ PRUNE OLDEST',
        'PRUNE_FUNDING':    '✂️ PRUNE FUNDING',
        'PRUNE_OFFSET':     '✂️ PRUNE OFFSET',
        'CIRCUIT_BREAKER':  '⚡ CIRCUIT BREAKER',
        'LIQUIDATION':      '💀 LIQUIDATION',
    }.get(label, f'🔔 {label}')

    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
    return (
        f'{emoji}\n'
        f'Symbol:  <code>{symbol}</code>\n'
        f'Price:   <code>${price:,.2f}</code>  qty=<code>{qty:.4f}</code>\n'
        f'PnL:     <code>{pnl_str}</code>  equity=<code>${equity:,.2f}</code>\n'
        f'Regime:  <code>{regime}</code>'
    )


def fmt_bar_summary(symbol: str, price: float, regime: str,
                    equity: float, pos_long_str: str,
                    pos_short_str: str, regen: bool) -> str:
    regen_tag = '🔄 regen' if regen else '📌 no regen'
    return (
        f'📊 <b>15m Bar</b> — <code>{symbol}</code>\n'
        f'Price:  <code>${price:,.2f}</code>  {regen_tag}\n'
        f'Regime: <code>{regime}</code>\n'
        f'Long:   <code>{pos_long_str}</code>\n'
        f'Short:  <code>{pos_short_str}</code>\n'
        f'Equity: <code>${equity:,.2f}</code>'
    )


def fmt_equity_warning(symbol: str, equity: float,
                       peak: float, drawdown_pct: float) -> str:
    return (
        f'⚠️ <b>EQUITY WARNING</b> — <code>{symbol}</code>\n'
        f'Equity:   <code>${equity:,.2f}</code>\n'
        f'Peak:     <code>${peak:,.2f}</code>\n'
        f'Drawdown: <code>{drawdown_pct:.1f}%</code>'
    )
