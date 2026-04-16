"""
live_signal.py MOEX v2.0
Генерирует сигналы для всех инструментов портфеля.

Для каждого тикера:
  - BUY-сигнал: когда покупать / усреднять (дно, разворот вверх)
  - SELL-сигнал: когда продавать частями (разворот вниз)
  - Все фильтры: ADX, 4H MTF, IMOEX, сессия MOEX, Regime-Switching

Использует отдельную модель для каждого тикера.
"""

import json
import joblib
import logging
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    FEATURE_COLS, FEATURE_COLS_LEGACY,
    MIN_CONFIDENCE, CONFIDENCE_PERCENTILE,
    MTF_ENABLED, IMOEX_FILTER_ENABLED, IMOEX_CHANGE_THRESH,
    REGIME_FILTER_ENABLED, REGIME_ADX_THRESHOLD,
    SYMBOLS_ALL, SYMBOL_NAMES,
    model_path, features_path,
)
from moex_client import (
    get_candles_multi, candles_to_df, resample_to_4h,
    get_imoex, is_trading_session,
)
from auto_trainer import calc_indicators_1h, calc_indicators_4h

logger = logging.getLogger(__name__)

_confidence_history: dict = {}   # {ticker: [conf1, conf2, ...]}
_HISTORY_MAX = 48

# Кэш данных (не перезагружаем каждую минуту)
_data_cache: dict = {}   # {ticker: {"df": df, "ts": timestamp}}
_CACHE_TTL = 55 * 60     # 55 минут (чуть меньше таймфрейма)


# ═══════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛЕЙ
# ═══════════════════════════════════════════

def _load_models_for_ticker(ticker: str) -> dict:
    """Загружает все доступные модели для конкретного тикера."""
    models = {}
    for mtype in ["buy_xgb", "buy_lgbm", "sell_xgb", "sell_lgbm",
                  "stack_buy", "stack_sell", "calib_buy", "calib_sell"]:
        p = model_path(ticker, mtype)
        if os.path.exists(p):
            try:
                models[mtype] = joblib.load(p)
            except Exception as e:
                logger.warning(f"[Signal] {ticker} {mtype}: {e}")
    return models


def _apply_stacking(models: dict, p_xgb: float, p_lgbm: float, direction: str) -> float:
    key = f"stack_{direction}"
    if key not in models:
        return (p_xgb + p_lgbm) / 2.0
    try:
        b  = models[key]
        Xs = np.array([[p_xgb, p_lgbm, (p_xgb+p_lgbm)/2, p_xgb-p_lgbm]])
        return float(b["model"].predict_proba(b["scaler"].transform(Xs))[0][1])
    except Exception:
        return (p_xgb + p_lgbm) / 2.0


def _get_calibrated_prob(models: dict, X: np.ndarray, direction: str, p_raw: float) -> float:
    key = f"calib_{direction}"
    if key not in models:
        return p_raw
    try:
        return float(models[key].predict_proba(X)[0][1])
    except Exception:
        return p_raw


# ═══════════════════════════════════════════
# REGIME-SWITCHING
# ═══════════════════════════════════════════

def get_regime_adjusted_threshold(hurst, adx, atr_ratio, base_threshold) -> tuple:
    if hurst > 0.62 and adx > 25:
        regime, mult, note = "TREND",    0.93, f"TREND H={hurst:.3f}"
    elif hurst < 0.40 and adx < 22:
        regime, mult, note = "MEAN_REV", 0.96, f"MEAN_REV H={hurst:.3f}"
    elif 0.43 <= hurst <= 0.57:
        regime, mult, note = "RANDOM",   1.10, f"RANDOM H={hurst:.3f}"
    elif atr_ratio > 1.8:
        regime, mult, note = "VOLATILE", 1.06, f"VOLATILE ATR={atr_ratio:.2f}"
    else:
        regime, mult, note = "NEUTRAL",  1.02, f"NEUTRAL H={hurst:.3f}"
    return regime, round(min(max(base_threshold * mult, 0.50), 0.90), 4), note


def _percentile_filter(ticker: str, confidence: float) -> bool:
    hist = _confidence_history.setdefault(ticker, [])
    hist.append(confidence)
    if len(hist) > _HISTORY_MAX:
        _confidence_history[ticker] = hist[-_HISTORY_MAX:]
    if len(hist) < 10:
        return True
    return confidence >= np.percentile(hist, CONFIDENCE_PERCENTILE)


# ═══════════════════════════════════════════
# IMOEX МАКРО-ФИЛЬТР
# ═══════════════════════════════════════════

_imoex_cache = {"change": 0.0, "ts": 0.0}
_IMOEX_TTL   = 600  # 10 минут

def get_imoex_change() -> float:
    now = time.time()
    if now - _imoex_cache["ts"] < _IMOEX_TTL:
        return _imoex_cache["change"]
    try:
        df = get_imoex(limit=50)
        if not df.empty and len(df) >= 2:
            ch = float((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100)
            _imoex_cache.update({"change": ch, "ts": now})
            return ch
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════
# ЗАГРУЗКА И КЭШИРОВАНИЕ ДАННЫХ
# ═══════════════════════════════════════════

def _get_data_for_ticker(ticker: str) -> tuple:
    """Возвращает (df1h_feats, df4h_feats) с кэшированием."""
    now = time.time()
    cache = _data_cache.get(ticker, {})
    if cache and now - cache.get("ts", 0) < _CACHE_TTL:
        return cache.get("df1h"), cache.get("df4h")

    try:
        raw = get_candles_multi(ticker, interval="60", total=700)
        df_raw = candles_to_df(raw)
        if df_raw.empty or len(df_raw) < 100:
            logger.warning(f"[Signal] {ticker}: мало данных ({len(df_raw)} баров)")
            return None, None

        df1h = calc_indicators_1h(df_raw)
        if len(df1h) < 50:
            return None, None

        df4h_raw = resample_to_4h(df_raw)
        df4h     = calc_indicators_4h(df4h_raw) if len(df4h_raw) >= 30 else None

        _data_cache[ticker] = {"df1h": df1h, "df4h": df4h, "ts": now}
        return df1h, df4h

    except Exception as e:
        logger.error(f"[Signal] {ticker}: ошибка загрузки данных: {e}")
        return None, None


# ═══════════════════════════════════════════
# СИГНАЛ ДЛЯ ОДНОГО ТИКЕРА
# ═══════════════════════════════════════════

def get_signal_for_ticker(ticker: str) -> dict | None:
    """
    Генерирует BUY/SELL/HOLD сигнал для одного инструмента.
    BUY  = покупать/усреднять (дно, разворот вверх)
    SELL = продавать частями (разворот вниз, хай)
    """
    start_ts = time.time()

    # 1. Данные
    df1h, df4h = _get_data_for_ticker(ticker)
    if df1h is None:
        return None

    # 2. Модели
    models = _load_models_for_ticker(ticker)
    if "buy_xgb" not in models and "buy_lgbm" not in models:
        logger.warning(f"[Signal] {ticker}: нет моделей, пропускаем")
        return None

    # 3. Признаки
    try:
        feat_p = features_path(ticker)
        if os.path.exists(feat_p):
            with open(feat_p) as f:
                feature_cols = json.load(f)
        else:
            feature_cols = FEATURE_COLS

        last_1h = df1h.iloc[-1].copy()

        # Добавляем 4H признаки
        if df4h is not None and not df4h.empty:
            cols_4h   = [c for c in df4h.columns if "_4h" in c]
            last_4h_r = df4h[cols_4h].iloc[-1]
            for col in cols_4h:
                if col in last_4h_r.index:
                    last_1h[col] = last_4h_r[col]

        avail = [c for c in feature_cols if c in last_1h.index]
        if len(avail) < 10:
            avail = [c for c in FEATURE_COLS_LEGACY if c in last_1h.index]

        X = np.nan_to_num(np.array([[float(last_1h.get(c, 0.0)) for c in avail]]))

    except Exception as e:
        logger.error(f"[Signal] {ticker}: ошибка признаков: {e}")
        return None

    # 4. Inference
    try:
        p_buy_xgb   = float(models["buy_xgb"].predict_proba(X)[0][1])  if "buy_xgb"   in models else 0.5
        p_sell_xgb  = float(models["sell_xgb"].predict_proba(X)[0][1]) if "sell_xgb"  in models else 0.5
        p_buy_lgbm  = float(models["buy_lgbm"].predict_proba(X)[0][1]) if "buy_lgbm"  in models else p_buy_xgb
        p_sell_lgbm = float(models["sell_lgbm"].predict_proba(X)[0][1])if "sell_lgbm" in models else p_sell_xgb

        p_buy_cal  = _get_calibrated_prob(models, X, "buy",  p_buy_xgb)
        p_sell_cal = _get_calibrated_prob(models, X, "sell", p_sell_xgb)

        p_buy  = p_buy_cal  if "calib_buy"  in models else _apply_stacking(models, p_buy_xgb,  p_buy_lgbm,  "buy")
        p_sell = p_sell_cal if "calib_sell" in models else _apply_stacking(models, p_sell_xgb, p_sell_lgbm, "sell")

        models_used = "+".join(filter(None, [
            "XGB"   if "buy_xgb"   in models else "",
            "LGBM"  if "buy_lgbm"  in models else "",
            "CALIB" if "calib_buy" in models else "",
            "STACK" if "stack_buy" in models and "calib_buy" not in models else "",
        ]))
    except Exception as e:
        logger.error(f"[Signal] {ticker}: inference ошибка: {e}")
        return None

    # 5. Regime-Switching
    adx_1h    = float(last_1h.get("ADX",      25.0))
    atr_ratio = float(last_1h.get("ATR_ratio", 1.0))
    rsi_14    = float(last_1h.get("RSI_14",   50.0))
    hurst     = float(last_1h.get("Hurst",     0.5))

    regime, eff_threshold, regime_note = get_regime_adjusted_threshold(hurst, adx_1h, atr_ratio, MIN_CONFIDENCE)

    # 6. Первичный сигнал
    if p_buy >= eff_threshold and p_sell >= eff_threshold:
        signal, confidence = ("BUY", p_buy) if p_buy >= p_sell else ("SELL", p_sell)
    elif p_buy >= eff_threshold:
        signal, confidence = "BUY",  p_buy
    elif p_sell >= eff_threshold:
        signal, confidence = "SELL", p_sell
    else:
        signal, confidence = "HOLD", max(p_buy, p_sell)

    # 7. Фильтры
    filter_log = []

    if signal != "HOLD" and not _percentile_filter(ticker, confidence):
        filter_log.append("PERCENTILE_LOW")
        signal = "HOLD"

    if REGIME_FILTER_ENABLED and signal != "HOLD" and adx_1h < REGIME_ADX_THRESHOLD:
        filter_log.append(f"ADX={adx_1h:.1f}")
        signal = "HOLD"

    # 4H MTF
    mtf_confirmed = True
    if MTF_ENABLED and df4h is not None and signal != "HOLD":
        last_4h      = df4h.iloc[-1]
        rsi_4h       = float(last_4h.get("RSI_14_4h",  50.0))
        ema_ratio_4h = float(last_4h.get("EMA_ratio_4h", 1.0))
        if signal == "BUY":
            mtf_ok = ema_ratio_4h > 0.993 and rsi_4h > 35
        else:
            mtf_ok = ema_ratio_4h < 1.007 and rsi_4h < 65
        if not mtf_ok:
            filter_log.append(f"4H_RSI={rsi_4h:.0f}")
            signal        = "HOLD"
            mtf_confirmed = False

    # IMOEX фильтр (только для BUY)
    imoex_change  = 0.0
    imoex_blocked = False
    if IMOEX_FILTER_ENABLED and signal == "BUY":
        imoex_change = get_imoex_change()
        if imoex_change < IMOEX_CHANGE_THRESH * 100:
            filter_log.append(f"IMOEX={imoex_change:+.2f}%")
            signal        = "HOLD"
            imoex_blocked = True

    # 8. Итог
    cur_price   = float(last_1h["Close"])
    current_atr = float(last_1h.get("ATR",       0.0))
    change_24h  = float(last_1h.get("Return_24h", 0.0))
    volume      = float(last_1h.get("Volume",     0.0))
    elapsed     = round(time.time() - start_ts, 2)

    # Buy-the-Dip контекст
    drawdown_20 = float(last_1h.get("Drawdown_from_high_20", 0.0))
    drawdown_60 = float(last_1h.get("Drawdown_from_high_60", 0.0))
    consec_down = float(last_1h.get("Consec_down_bars",      0.0))
    vol_surge   = float(last_1h.get("Vol_surge",             1.0))

    logger.info(
        f"[Signal] {ticker} → {signal} | "
        f"p_buy={p_buy:.1%} p_sell={p_sell:.1%} | "
        f"Hurst={hurst:.3f} Regime={regime} thresh={eff_threshold:.1%} | "
        f"ADX={adx_1h:.1f} 4H={'✅' if mtf_confirmed else '❌'} | "
        f"DD20={drawdown_20:.1f}% DD60={drawdown_60:.1f}% | "
        f"IMOEX={imoex_change:+.2f}% | filters={filter_log} | {elapsed}s"
    )

    return {
        "ticker":        ticker,
        "name":          SYMBOL_NAMES.get(ticker, ticker),
        "signal":        signal,
        "confidence":    round(confidence, 4),
        "p_buy":         round(p_buy,      4),
        "p_sell":        round(p_sell,     4),
        "p_buy_cal":     round(p_buy_cal,  4),
        "p_sell_cal":    round(p_sell_cal, 4),
        "models_used":   models_used,
        "hurst":         round(hurst, 3),
        "regime":        regime,
        "regime_note":   regime_note,
        "eff_threshold": round(eff_threshold, 4),
        "price":         cur_price,
        "atr":           current_atr,
        "change_24h":    change_24h,
        "volume":        volume,
        "adx":           round(adx_1h, 2),
        "rsi14":         round(rsi_14,  2),
        "mtf_confirmed": mtf_confirmed,
        "imoex_change":  imoex_change,
        "imoex_blocked": imoex_blocked,
        "filter_log":    filter_log,
        # Buy-the-Dip контекст
        "drawdown_20":   round(drawdown_20, 2),
        "drawdown_60":   round(drawdown_60, 2),
        "consec_down":   int(consec_down),
        "vol_surge":     round(vol_surge, 2),
        "inference_ms":  int(elapsed * 1000),
        "timestamp":     datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════
# СИГНАЛЫ ДЛЯ ВСЕГО ПОРТФЕЛЯ
# ═══════════════════════════════════════════

def get_all_signals(tickers: list = None) -> list:
    """
    Генерирует сигналы для всех инструментов.
    Возвращает список только тех, у кого signal != HOLD,
    отсортированный по confidence (лучшие первые).
    """
    if tickers is None:
        tickers = SYMBOLS_ALL

    if not is_trading_session():
        from moex_client import get_minutes_to_session
        mins = get_minutes_to_session()
        logger.info(f"[Signal] ⏰ Вне сессии MOEX. До открытия: {mins} мин.")
        return []

    results = []
    for ticker in tickers:
        try:
            sig = get_signal_for_ticker(ticker)
            if sig:
                results.append(sig)
            time.sleep(0.5)  # небольшая пауза между тикерами
        except Exception as e:
            logger.error(f"[Signal] {ticker}: {e}")

    # Возвращаем все сигналы (HOLD тоже — для логирования)
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# Обратная совместимость с app.py
def get_live_signal(ticker: str = None) -> dict | None:
    if ticker:
        if not is_trading_session():
            return None
        return get_signal_for_ticker(ticker)
    # Если тикер не указан — возвращаем лучший сигнал по всему портфелю
    signals = get_all_signals()
    action_signals = [s for s in signals if s["signal"] in ("BUY", "SELL")]
    return action_signals[0] if action_signals else None
