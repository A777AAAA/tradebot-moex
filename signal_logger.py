"""
signal_logger.py — Paper Trading лог сигналов
Логирует каждый BUY/SELL сигнал, через 6ч проверяет результат.
Хранит данные в SQLite: /app/data/signals.db
"""

import os
import sqlite3
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH     = os.path.join(os.path.dirname(__file__), "data", "signals.db")
CHECK_HOURS = 6   # через сколько часов проверяем результат
WIN_PCT     = 1.0 # порог % для WIN
LOSS_PCT    = -1.0


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          INTEGER NOT NULL,
            symbol      TEXT    NOT NULL,
            signal      TEXT    NOT NULL,
            price_open  REAL    NOT NULL,
            confidence  REAL,
            p_buy       REAL,
            p_sell      REAL,
            p_hold      REAL,
            price_close REAL,
            ts_close    INTEGER,
            pnl_pct     REAL,
            result      TEXT
        )
    """)
    conn.commit()
    return conn


def log_signal(symbol: str, signal: str, price: float,
               confidence: float = 0.0, p_buy: float = 0.0,
               p_sell: float = 0.0, p_hold: float = 0.0):
    """Записывает новый сигнал в базу."""
    try:
        conn = _get_conn()
        ts   = int(time.time())
        conn.execute(
            "INSERT INTO signals (ts, symbol, signal, price_open, confidence, p_buy, p_sell, p_hold) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, signal, price, confidence, p_buy, p_sell, p_hold)
        )
        conn.commit()
        conn.close()
        logger.info(f"[signal_logger] Записан {signal} {symbol} @ {price:.4f}")
    except Exception as e:
        logger.error(f"[signal_logger] log_signal: {e}")


def _get_current_price(symbol: str) -> float:
    """Получает текущую цену инструмента."""
    try:
        # Пробуем через live_signal
        import importlib
        ls = importlib.import_module("live_signal")
        if hasattr(ls, "get_live_signal"):
            data = ls.get_live_signal()
            if data and "price" in data:
                return float(data["price"])
        # Для MOEX — через moex_client
        if hasattr(ls, "get_all_signals"):
            signals = ls.get_all_signals([symbol])
            if signals:
                return float(signals[0]["price"])
    except Exception as e:
        logger.warning(f"[signal_logger] get_price {symbol}: {e}")
    return 0.0


def check_pending_signals() -> list:
    """
    Проверяет сигналы старше CHECK_HOURS часов без результата.
    Возвращает список закрытых записей.
    """
    closed = []
    try:
        conn     = _get_conn()
        deadline = int(time.time()) - CHECK_HOURS * 3600
        rows     = conn.execute(
            "SELECT id, symbol, signal, price_open FROM signals "
            "WHERE result IS NULL AND ts < ?",
            (deadline,)
        ).fetchall()

        for row_id, symbol, signal, price_open in rows:
            price_close = _get_current_price(symbol)
            if price_close <= 0:
                continue

            ts_close = int(time.time())
            if signal == "BUY":
                pnl_pct = (price_close - price_open) / (price_open + 1e-9) * 100
            else:  # SELL
                pnl_pct = (price_open - price_close) / (price_open + 1e-9) * 100

            if pnl_pct >= WIN_PCT:
                result = "WIN"
            elif pnl_pct <= LOSS_PCT:
                result = "LOSS"
            else:
                result = "NEUTRAL"

            conn.execute(
                "UPDATE signals SET price_close=?, ts_close=?, pnl_pct=?, result=? WHERE id=?",
                (price_close, ts_close, pnl_pct, result, row_id)
            )
            conn.commit()

            closed.append({
                "symbol":      symbol,
                "signal":      signal,
                "price_open":  price_open,
                "price_close": price_close,
                "pnl_pct":     pnl_pct,
                "result":      result,
            })
            logger.info(f"[signal_logger] Закрыт {signal} {symbol}: {pnl_pct:+.2f}% → {result}")

        conn.close()
    except Exception as e:
        logger.error(f"[signal_logger] check_pending: {e}")
    return closed


def get_signal_stats(days: int = 7) -> dict:
    """Возвращает статистику сигналов за последние N дней."""
    try:
        conn     = _get_conn()
        since    = int(time.time()) - days * 86400
        rows     = conn.execute(
            "SELECT signal, result, pnl_pct FROM signals "
            "WHERE result IS NOT NULL AND ts > ?",
            (since,)
        ).fetchall()
        conn.close()

        if not rows:
            return {"total": 0, "days": days}

        total  = len(rows)
        wins   = sum(1 for _, r, _ in rows if r == "WIN")
        losses = sum(1 for _, r, _ in rows if r == "LOSS")
        pnls   = [p for _, _, p in rows if p is not None]

        return {
            "days":      days,
            "total":     total,
            "wins":      wins,
            "losses":    losses,
            "neutral":   total - wins - losses,
            "winrate":   round(wins / total * 100, 1) if total else 0,
            "avg_pnl":   round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "total_pnl": round(sum(pnls), 2) if pnls else 0,
        }
    except Exception as e:
        logger.error(f"[signal_logger] get_stats: {e}")
        return {"total": 0, "days": days, "error": str(e)}


def format_signal_stats_message(stats: dict) -> str:
    """Форматирует статистику для Telegram."""
    if stats.get("total", 0) == 0:
        return f"📝 <b>Signal Logger ({stats.get('days',7)}д):</b> сигналов пока нет"

    return (
        f"📝 <b>Signal Logger за {stats['days']}д:</b>\n"
        f"   Всего сигналов: <b>{stats['total']}</b>\n"
        f"   ✅ WIN:    <b>{stats['wins']}</b>\n"
        f"   ❌ LOSS:   <b>{stats['losses']}</b>\n"
        f"   ➖ NEUTRAL: <b>{stats['neutral']}</b>\n"
        f"   🎯 Win Rate: <b>{stats['winrate']}%</b>\n"
        f"   💰 Avg P&L:  <b>{stats['avg_pnl']:+.2f}%</b>\n"
        f"   📊 Total P&L: <b>{stats['total_pnl']:+.2f}%</b>"
    )
