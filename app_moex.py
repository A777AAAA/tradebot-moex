"""
app.py — TradeBot MOEX v8.3
Изменения vs v2.0:
  - retrainer_loop читает RETRAIN_INTERVAL_HRS из config (был жёсткий 24ч)
  - signal_logger: paper trading лог каждого сигнала + авто-проверка через 6ч
  - get_config() — динамическая перезагрузка config без рестарта
"""

import threading
import time
import logging
import os
import json
import importlib
from flask import Flask

import requests as _req

from config import (
    SYMBOLS_ALL, SYMBOL_NAMES, SYMBOL_BOARD,
    MIN_CONFIDENCE, STRONG_SIGNAL,
    SIGNAL_INTERVAL_MINUTES, validate_config,
    INITIAL_BALANCE,
)
from live_signal   import get_all_signals
from auto_trainer  import train_model
from paper_trader  import (
    open_trade, monitor_trades,
    get_stats, format_stats_message,
)
from moex_client   import is_trading_session, get_minutes_to_session

try:
    from signal_logger import (
        log_signal, check_pending_signals,
        get_signal_stats, format_signal_stats_message
    )
    SIGNAL_LOGGER_OK = True
except ImportError:
    SIGNAL_LOGGER_OK = False

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_config(param, default):
    try:
        import config as _cfg
        importlib.reload(_cfg)
        return getattr(_cfg, param, default)
    except Exception:
        return default


# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
def send_message(text: str):
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        _req.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
    except Exception as e:
        logger.error(f"[TG] {e}")


# ═══════════════════════════════════════════
# HEALTHCHECK
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot MOEX v8.3 — signal_logger + retrain 6h"}, 200

@health_app.route("/")
def index():
    try:
        stats = get_stats()
        return {
            "bot":              "TradeBot MOEX v8.3 — signal_logger + retrain 6h",
            "symbols":          SYMBOLS_ALL,
            "balance_rub":      stats["balance"],
            "growth_pct":       stats["growth_pct"],
            "open_positions":   stats["n_open_positions"],
            "drawdown_pct":     stats["current_drawdown"],
            "trading_session":  is_trading_session(),
            "retrain_interval": get_config("RETRAIN_INTERVAL_HRS", 6),
        }, 200
    except Exception:
        return {"status": "initializing"}, 200


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═══════════════════════════════════════════
# ФОРМАТИРОВАНИЕ СООБЩЕНИЙ
# ═══════════════════════════════════════════

def _format_buy_message(sig: dict, trade: dict) -> str:
    ticker     = sig["ticker"]
    name       = sig["name"]
    price      = sig["price"]
    confidence = sig["confidence"]
    p_buy      = sig["p_buy"]
    hurst      = sig["hurst"]
    regime     = sig["regime"]
    adx        = sig["adx"]
    drawdown_20 = sig.get("drawdown_20", 0)
    drawdown_60 = sig.get("drawdown_60", 0)
    consec_down = sig.get("consec_down", 0)
    vol_surge   = sig.get("vol_surge", 1.0)
    models_used = sig.get("models_used", "XGB")
    imoex_ch    = sig.get("imoex_change", 0.0)
    p_hold      = sig.get("p_hold", 0.0)

    is_avg  = trade.get("is_averaging", False)
    avg_p   = trade.get("avg_price", price)
    entries = trade.get("entry_id", 1)

    strength = "🔥 STRONG" if confidence >= STRONG_SIGNAL else "📶 NORMAL"
    hurst_l  = "Тренд" if hurst > 0.6 else ("Mean-Rev" if hurst < 0.4 else "Нейтр.")
    avg_str  = f"\n📊 Ср.цена:        <b>{avg_p:.2f} ₽</b> (вход {entries}/{3})" if is_avg else ""
    hold_str = f"\n⏸ p(HOLD):        <b>{p_hold:.1%}</b>" if p_hold > 0 else ""

    return (
        f"{'📉 УСРЕДНЕНИЕ' if is_avg else '🟢 ПОКУПКА'} {strength}\n\n"
        f"📈 <b>{name} ({ticker})</b>\n"
        f"💵 Цена:          <b>{price:.2f} ₽</b>{avg_str}\n"
        f"🎯 p(BUY):        <b>{p_buy:.1%}</b> | conf={confidence:.1%}"
        f"{hold_str}\n"
        f"📉 Просадка 20д:  <b>{drawdown_20:.1f}%</b>\n"
        f"📉 Просадка 60д:  <b>{drawdown_60:.1f}%</b>\n"
        f"🕯 Красных баров: <b>{consec_down}</b>\n"
        f"💹 Объём:         <b>×{vol_surge:.1f}</b> от среднего\n"
        f"🌊 Hurst:         <b>{hurst:.3f} ({hurst_l})</b>\n"
        f"📊 Режим:         <b>{regime}</b> | ADX={adx:.1f}\n"
        f"📊 IMOEX:         <b>{imoex_ch:+.2f}%</b>\n"
        f"🤖 Модели:        <b>{models_used}</b>\n"
        f"💼 Размер:        <b>{trade.get('amount_rub', 0):,.0f} ₽ ({trade.get('kelly_pct',0):.1%})</b>\n"
        f"🛑 SL:            <b>{trade.get('sl',0):.2f} ₽</b>"
    )


def _format_sell_event_message(event: dict) -> str:
    symbol    = event.get("symbol", "")
    name      = SYMBOL_NAMES.get(symbol, symbol)
    level_pct = event.get("level_pct", 0)
    price     = event.get("price", 0)
    avg_price = event.get("avg_price", 0)
    pnl_pct   = event.get("pnl_pct", 0)
    pnl_rub   = event.get("pnl_rub", 0)
    sell_amt  = event.get("sell_amount", 0)

    return (
        f"✅ ЧАСТИЧНАЯ ПРОДАЖА +{level_pct*100:.0f}%\n\n"
        f"📈 <b>{name} ({symbol})</b>\n"
        f"💵 Цена продажи:  <b>{price:.2f} ₽</b>\n"
        f"📊 Ср.цена входа: <b>{avg_price:.2f} ₽</b>\n"
        f"💰 P&L:           <b>{pnl_pct:+.2f}% ({pnl_rub:+.0f} ₽)</b>\n"
        f"📤 Продано на:    <b>{sell_amt:,.0f} ₽</b>"
    )


def _format_sl_message(event: dict) -> str:
    symbol    = event.get("symbol", "")
    name      = SYMBOL_NAMES.get(symbol, symbol)
    entry_id  = event.get("entry_id", 1)
    price_o   = event.get("price_open", 0)
    price_c   = event.get("price_close", 0)
    pnl_pct   = event.get("pnl_pct", 0)
    pnl_rub   = event.get("pnl_rub", 0)
    reason    = event.get("reason", "SL")

    return (
        f"❌ СТОП-ЛОСС — вход #{entry_id}\n\n"
        f"📈 <b>{name} ({symbol})</b>\n"
        f"🔵 Вход:  <b>{price_o:.2f} ₽</b>\n"
        f"🔴 Выход: <b>{price_c:.2f} ₽</b>\n"
        f"💸 P&L:   <b>{pnl_pct:+.2f}% ({pnl_rub:+.0f} ₽)</b>\n"
        f"🔒 Причина: <b>{reason}</b>"
    )


# ═══════════════════════════════════════════
# ПРОВЕРКА ИСХОДОВ СИГНАЛОВ (signal_logger)
# ═══════════════════════════════════════════
def signal_checker_loop():
    """Каждые 30 минут проверяет результаты сигналов из signal_logger."""
    if not SIGNAL_LOGGER_OK:
        logger.warning("[SignalChecker] signal_logger не найден, пропускаем")
        return

    time.sleep(300)
    logger.info("🔍 Signal checker MOEX запущен")

    while True:
        try:
            closed = check_pending_signals()
            for r in closed:
                ticker = r.get("symbol", "?")
                name   = SYMBOL_NAMES.get(ticker, ticker)
                emoji  = "✅" if r["result"] == "WIN" else ("❌" if r["result"] == "LOSS" else "➖")
                send_message(
                    f"{emoji} <b>Сигнал проверен — {r['result']}</b>\n\n"
                    f"📈 <b>{name} ({ticker})</b>\n"
                    f"📊 Сигнал: <b>{r['signal']}</b>\n"
                    f"🔵 Цена входа:  <b>{r['price_open']:.2f} ₽</b>\n"
                    f"🔴 Цена через 6ч: <b>{r['price_close']:.2f} ₽</b>\n"
                    f"💰 Изменение:   <b>{r['pnl_pct']:+.2f}%</b>"
                )
        except Exception as e:
            logger.error(f"[SignalChecker] {e}")

        time.sleep(30 * 60)


# ═══════════════════════════════════════════
# ТОРГОВЫЙ ЦИКЛ
# ═══════════════════════════════════════════
def trading_loop():
    logger.info("🚀 Торговый цикл MOEX v8.3 запущен")
    logger.info(f"   Инструменты: {', '.join(SYMBOLS_ALL)}")

    while True:
        try:
            # Ожидание торговой сессии
            if not is_trading_session():
                mins = get_minutes_to_session()
                logger.info(f"⏰ Вне сессии MOEX. До открытия: {mins} мин.")
                sleep_secs = min(mins * 60, 3600) if mins > 0 else 3600
                time.sleep(sleep_secs)
                continue

            # ── Мониторинг открытых позиций ──
            events = monitor_trades(SYMBOLS_ALL)
            for event in events:
                evt_type = event.get("type")
                if evt_type == "PARTIAL_SELL":
                    send_message(_format_sell_event_message(event))
                elif evt_type == "SL_CLOSE":
                    send_message(_format_sl_message(event))

            # ── Получаем сигналы по всем инструментам ──
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(get_all_signals, SYMBOLS_ALL)
                try:
                    all_signals = _fut.result(timeout=120)
                except _cf.TimeoutError:
                    logger.error("❌ Watchdog: get_all_signals завис >120s, пропускаем цикл")
                    time.sleep(60)
                    continue

            # Логируем все сигналы
            for sig in all_signals:
                s = sig["signal"]
                emoji = "🟢" if s == "BUY" else ("🔴" if s == "SELL" else "⏸")
                logger.info(
                    f"{emoji} {sig['ticker']:12s} | {s:4s} conf={sig['confidence']:.1%} | "
                    f"p_buy={sig['p_buy']:.1%} p_sell={sig['p_sell']:.1%} | "
                    f"DD20={sig.get('drawdown_20',0):.1f}% | "
                    f"ADX={sig['adx']:.1f} | Regime={sig['regime']}"
                )
                # Логируем в signal_logger
                if SIGNAL_LOGGER_OK and s in ("BUY", "SELL"):
                    try:
                        log_signal(
                            symbol=sig["ticker"],
                            signal=s,
                            price=sig["price"],
                            confidence=sig["confidence"],
                            p_buy=sig["p_buy"],
                            p_sell=sig["p_sell"],
                            p_hold=sig.get("p_hold", 0.0),
                        )
                    except Exception as _le:
                        logger.warning(f"[signal_logger] {_le}")

            # ── Обрабатываем BUY-сигналы ──
            buy_signals = [s for s in all_signals if s["signal"] == "BUY" and s["confidence"] >= MIN_CONFIDENCE]
            for sig in buy_signals:
                ticker = sig["ticker"]
                try:
                    trade = open_trade(
                        signal     = "BUY",
                        price      = sig["price"],
                        confidence = sig["confidence"],
                        symbol     = ticker,
                        atr        = sig["atr"],
                        extra_info = {
                            "p_buy":       sig["p_buy"],
                            "p_sell":      sig["p_sell"],
                            "models_used": sig["models_used"],
                            "hurst":       sig["hurst"],
                            "regime":      sig["regime"],
                            "adx":         sig["adx"],
                            "drawdown_20": sig.get("drawdown_20", 0),
                            "drawdown_60": sig.get("drawdown_60", 0),
                        }
                    )
                    if trade:
                        send_message(_format_buy_message(sig, trade))
                except Exception as e:
                    logger.error(f"[Trading] {ticker} BUY ошибка: {e}")

            # ── SELL-сигналы логируем ──
            sell_signals = [s for s in all_signals if s["signal"] == "SELL" and s["confidence"] >= MIN_CONFIDENCE]
            for sig in sell_signals:
                logger.info(
                    f"🔴 SELL-сигнал: {sig['ticker']} conf={sig['confidence']:.1%} | "
                    f"Продажа через monitor_trades при +4/7/12%"
                )

            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка торгового цикла: {e}", exc_info=True)
            time.sleep(60)


# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ — ЧИТАЕТ RETRAIN_INTERVAL_HRS
# ═══════════════════════════════════════════
def retrainer_loop():
    time.sleep(120)
    interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
    logger.info(f"🧠 Retrainer MOEX v8.3 запущен (интервал: {interval_hrs}ч)")

    while True:
        try:
            result = train_model()
            ok     = result.get("tickers_ok", 0)
            total  = result.get("tickers_total", 0)
            avg_bp = result.get("avg_buy_precision", 0)
            avg_sh = result.get("avg_wf_sharpe_buy", 0)
            avg_kl = result.get("avg_kelly", 0)

            lines = []
            for ticker, r in result.get("results", {}).items():
                name = SYMBOL_NAMES.get(ticker, ticker)
                if r.get("success"):
                    lines.append(
                        f"  ✅ {name}: BUY={r['avg_buy_precision']:.1%} "
                        f"WF={r['wf_buy_sharpe']:.2f} K={r['kelly_fraction']:.1%}"
                    )
                else:
                    lines.append(f"  ❌ {name}: {r.get('error','?')}")

            interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
            send_message(
                f"🧠 <b>Переобучение MOEX завершено!</b>\n\n"
                f"✅ Обучено: <b>{ok}/{total}</b> инструментов\n"
                f"📊 Avg BUY precision: <b>{avg_bp:.1%}</b>\n"
                f"📐 Avg WF Sharpe:     <b>{avg_sh:.2f}</b>\n"
                f"💰 Avg Kelly:         <b>{avg_kl:.1%}</b>\n\n"
                f"<b>Детали:</b>\n" + "\n".join(lines) +
                f"\n\n⏱ Следующее через: <b>{interval_hrs}ч</b>"
            )
        except Exception as e:
            logger.error(f"[Retrainer] {e}", exc_info=True)
            time.sleep(2 * 3600)
            continue

        interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
        time.sleep(interval_hrs * 3600)


# ═══════════════════════════════════════════
# ЕЖЕДНЕВНЫЙ ОТЧЁТ
# ═══════════════════════════════════════════
def stats_loop():
    time.sleep(300)
    while True:
        try:
            stats = get_stats()
            msg   = format_stats_message(stats)

            # Добавляем статистику сигналов если есть
            if SIGNAL_LOGGER_OK:
                try:
                    sig_stats = get_signal_stats(days=7)
                    msg += "\n\n" + format_signal_stats_message(sig_stats)
                except Exception:
                    pass

            send_message(msg)
        except Exception as e:
            logger.error(f"[Stats] {e}")
        time.sleep(24 * 3600)


# ═══════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        logger.critical(f"❌ Не заданы переменные: {errors}")
        exit(1)

    os.makedirs("models", exist_ok=True)

    n_models = sum(
        1 for ticker in SYMBOLS_ALL
        for mtype in ["buy_xgb", "sell_xgb"]
        if os.path.exists(f"models/{ticker}_{mtype}.pkl")
    )

    interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
    logger.info(f"✅ Конфиг OK | Моделей: {n_models}/{len(SYMBOLS_ALL)*2} | retrain={interval_hrs}ч | Запускаем...")

    threading.Thread(target=run_health_server,    daemon=True).start()
    threading.Thread(target=retrainer_loop,       daemon=True).start()
    threading.Thread(target=trading_loop,         daemon=True).start()
    threading.Thread(target=stats_loop,           daemon=True).start()
    threading.Thread(target=signal_checker_loop,  daemon=True).start()

    send_message(
        "🤖 <b>TradeBot MOEX v8.3 запущен!</b>\n\n"
        f"📊 <b>Инструменты:</b>\n"
        + "".join(f"  • {SYMBOL_NAMES.get(t,t)} ({t})\n" for t in SYMBOLS_ALL) +
        f"\n🕐 Сессия MOEX: <b>10:00–18:50 МСК</b>\n"
        f"⏱ Интервал:    <b>{SIGNAL_INTERVAL_MINUTES} мин</b>\n"
        f"🎯 Уверенность: <b>≥ {MIN_CONFIDENCE:.0%}</b>\n"
        f"🔄 Retrain:     <b>каждые {interval_hrs}ч</b>\n\n"
        f"📐 <b>Стратегия:</b>\n"
        f"  🟢 BUY:  покупка/усреднение на дне (до 3 входов)\n"
        f"  🎯 SELL: частичная продажа +4% / +7% / +12%\n"
        f"  💰 HOLD: держим для дивидендов/купонов\n"
        f"  🛡 SL:   стоп-лосс по каждому входу независимо\n"
        f"  🚫 Дивзащита: блок продажи за 7 дней до отсечки\n\n"
        f"🔬 <b>ML-архитектура v8.3:</b>\n"
        f"  XGB+LGBM+CatBoost+RidgeCV+Calibrated\n"
        f"  Triple Barrier + HOLD класс + p_hold\n"
        f"  SMOTE, Optuna, WF, Kelly\n"
        f"  📊 Моделей загружено: <b>{n_models}/{len(SYMBOLS_ALL)*2}</b>\n"
        f"  📝 Signal logger: <b>{'✅ Активен' if SIGNAL_LOGGER_OK else '⚠️ Не найден'}</b>"
    )

    while True:
        time.sleep(60)
