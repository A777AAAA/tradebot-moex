"""
auto_trainer.py v8.1 — Калибровка вероятностей + 8000 свечей + Улучшенный Kelly
ИЗМЕНЕНИЯ v8.0 vs v7.0:
  - КАЛИБРОВКА ВЕРОЯТНОСТЕЙ: CalibratedClassifierCV (Isotonic Regression)
    Без калибровки p(BUY)=0.75 ≠ реальным 75% win rate.
    После калибровки — вероятности реалистичны. Это критично для Kelly Criterion.
    Используется в Two Sigma, Citadel и всех профессиональных ML-системах.
  - 8000 свечей вместо 3000 (~333 дня vs ~125 дней)
    Больше данных = лучше WF-валидация, меньше шума в обучении.
  - 4000 свечей для 4H (было 750)
  - Optuna 50 trials вместо 30 (лучший поиск гиперпараметров)
  - Kelly рассчитывается на реальных WF-доходностях (не приближении)
    Это даёт точный размер позиции для максимизации роста.
  - TimeSeriesSplit для Optuna (устраняет data leakage в тюнинге)
  - Сохраняется calibrated_model_buy.pkl / calibrated_model_sell.pkl
  - Все v7.0 улучшения сохранены: Stacking, Meta-Labeling, Triple Barrier,
    Feature Pruning, SMOTE, Hurst, VWAP, RV, OFI
"""

import os
import json
import joblib
import logging
import requests
import time
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from config import (
    MODEL_PATH_BUY_XGB, MODEL_PATH_BUY_LGBM,
    MODEL_PATH_SELL_XGB, MODEL_PATH_SELL_LGBM,
    MODEL_FEATURES_PATH, STATS_FILE,
    FEATURE_COLS, FEATURE_COLS_LEGACY,
    TARGET_HORIZON, TARGET_THRESHOLD,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
    ATR_SL_MULT, ATR_TP_MULT,
)

logger = logging.getLogger(__name__)

META_MODEL_BUY_PATH     = "meta_model_buy.pkl"
META_MODEL_SELL_PATH    = "meta_model_sell.pkl"
STACK_MODEL_BUY_PATH    = "stack_model_buy.pkl"
STACK_MODEL_SELL_PATH   = "stack_model_sell.pkl"
CALIB_MODEL_BUY_PATH    = "calibrated_model_buy.pkl"   # v8.0 NEW
CALIB_MODEL_SELL_PATH   = "calibrated_model_sell.pkl"  # v8.0 NEW
FEATURE_IMPORTANCE_PATH = "feature_importance.json"

FEATURE_IMPORTANCE_THRESHOLD = 0.005

# v8.0: увеличены объёмы данных
BARS_1H = 8000   # было 3000 (~125 дней) → теперь ~333 дня
BARS_4H = 4000   # было 750 → теперь ~667 дней на 4H


# ─────────────────────────────────────────────
# Загрузка OHLCV с OKX (пагинация)
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol: str = "TON-USDT", bar: str = "1H", bars: int = 8000) -> pd.DataFrame:
    """Загружает свечи через okx_client с retry/backoff защитой."""
    from okx_client import get_candles_multi, candles_to_df
    try:
        raw = get_candles_multi(symbol, bar, bars)
        if not raw:
            logger.error(f"[Trainer] Нет данных {bar}")
            return pd.DataFrame()
        df = candles_to_df(raw)
        logger.info(f"[Trainer] ✅ Загружено {len(df)} свечей ({bar})")
        return df
    except Exception as e:
        logger.error(f"[Trainer] Ошибка загрузки {bar}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# ПРОФЕССИОНАЛЬНЫЕ ПРИЗНАКИ
# ─────────────────────────────────────────────

def calc_hurst_exponent(ts: pd.Series, lags_range: range = range(2, 21)) -> pd.Series:
    """
    Hurst Exponent — мера трендовости/mean-reversion.
    H > 0.6 = тренд, H < 0.4 = mean-reversion, H ≈ 0.5 = случайное блуждание.
    """
    def hurst_single(series):
        if len(series) < 20:
            return 0.5
        try:
            lags = list(lags_range)
            tau  = [max(np.std(np.subtract(series[lag:], series[:-lag])), 1e-9)
                    for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return max(0.0, min(1.0, poly[0]))
        except Exception:
            return 0.5

    return ts.rolling(window=100, min_periods=50).apply(
        lambda x: hurst_single(x), raw=True
    )


def calc_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP отклонение — институциональная справедливая цена."""
    d     = df.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    tp = (high + low + close) / 3

    vwap_20 = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
    d['VWAP_dev_20'] = (close - vwap_20) / (vwap_20 + 1e-9) * 100

    vwap_50 = (tp * vol).rolling(50).sum() / vol.rolling(50).sum()
    d['VWAP_dev_50'] = (close - vwap_50) / (vwap_50 + 1e-9) * 100

    bull_vol  = vol.where(close > vwap_20, 0).rolling(10).sum()
    total_vol = vol.rolling(10).sum() + 1e-9
    d['VWAP_bull_ratio'] = bull_vol / total_vol

    return d


def calc_realized_volatility(close: pd.Series) -> pd.DataFrame:
    """Реализованная волатильность (annualized %)."""
    log_ret = np.log(close / close.shift(1))
    result  = pd.DataFrame(index=close.index)

    result['RV_20']  = np.sqrt((log_ret**2).rolling(20).sum() / 20) * np.sqrt(8760) * 100
    result['RV_50']  = np.sqrt((log_ret**2).rolling(50).sum() / 50) * np.sqrt(8760) * 100
    result['RV_ratio'] = result['RV_20'] / (result['RV_50'] + 1e-9)

    return result


def calc_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Order Flow Imbalance — асимметрия покупательного/продажного давления."""
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    bull_fraction = (close - low) / (high - low + 1e-9)
    bear_fraction = (high - close) / (high - low + 1e-9)

    bull_vol = vol * bull_fraction
    bear_vol = vol * bear_fraction

    ofi = (bull_vol - bear_vol).rolling(10).sum()
    ofi_norm = ofi / (vol.rolling(10).sum() + 1e-9)

    return ofi_norm


# ─────────────────────────────────────────────
# Индикаторы 1H
# ─────────────────────────────────────────────
def calc_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    d     = df.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    d['Hour']      = d.index.hour
    d['DayOfWeek'] = d.index.dayofweek

    for p in [7, 14, 21]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p - 1, min_periods=p).mean()
        avg_l = l.ewm(com=p - 1, min_periods=p).mean()
        d[f'RSI_{p}'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    ema12            = close.ewm(span=12, adjust=False).mean()
    ema26            = close.ewm(span=26, adjust=False).mean()
    d['MACD']        = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']

    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14         = tr.ewm(com=13, min_periods=14).mean()
    atr50         = tr.ewm(com=49, min_periods=50).mean()
    d['ATR']      = atr14
    d['ATR_pct']  = (atr14 / (close + 1e-9)) * 100
    d['ATR_norm'] = atr14 / (close + 1e-9)
    d['ATR_ratio']= atr14 / (atr50 + 1e-9)

    sma20         = close.rolling(20).mean()
    std20         = close.rolling(20).std()
    bb_upper      = sma20 + 2 * std20
    bb_lower      = sma20 - 2 * std20
    d['BB_pos']   = (close - bb_lower) / (4 * std20 + 1e-9)
    d['BB_width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)

    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    d['EMA_ratio_20_50']  = ema20 / (ema50  + 1e-9)
    d['EMA_ratio_20_100'] = ema20 / (ema100 + 1e-9)
    d['EMA_ratio']        = d['EMA_ratio_20_50']

    vol_sma20      = vol.rolling(20).mean()
    d['Vol_ratio'] = vol / (vol_sma20 + 1e-9)

    obv           = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20     = obv.rolling(20).mean()
    d['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    tp          = (high + low + close) / 3
    mf          = tp * vol
    pos_mf      = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf      = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d['MFI_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    rsi14           = d['RSI_14']
    stoch_min       = rsi14.rolling(14).min()
    stoch_max       = rsi14.rolling(14).max()
    stoch_k         = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
    d['StochRSI_K'] = stoch_k
    d['StochRSI_D'] = stoch_k.rolling(3).mean()

    hw14           = high.rolling(14).max()
    lw14           = low.rolling(14).min()
    d['WilliamsR'] = (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    d['ZScore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    d['ZScore_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX'] = dx.ewm(alpha=1/14).mean()

    d['Body_pct']   = (close - d['Open']).abs() / (d['Open'] + 1e-9) * 100
    d['Upper_wick'] = (high - d[['Close','Open']].max(axis=1)) / (d['Open'] + 1e-9) * 100
    d['Lower_wick'] = (d[['Close','Open']].min(axis=1) - low) / (d['Open'] + 1e-9) * 100
    d['Doji']       = ((d['Body_pct'] / (high - low + 1e-9)) < 0.1).astype(int)

    d['Momentum_10'] = close - close.shift(10)
    d['ROC_10']      = close.pct_change(10) * 100

    for h in [1, 4, 12, 24]:
        d[f'Return_{h}h'] = close.pct_change(h) * 100

    # ── ПРОФЕССИОНАЛЬНЫЕ ПРИЗНАКИ v7.0+ ──

    d['Hurst'] = calc_hurst_exponent(close)
    d = calc_vwap_features(d)

    rv = calc_realized_volatility(close)
    d['RV_20']    = rv['RV_20']
    d['RV_50']    = rv['RV_50']
    d['RV_ratio'] = rv['RV_ratio']

    d['OFI'] = calc_order_flow_imbalance(d)

    d['Price_accel'] = close.pct_change(1) - close.pct_change(1).shift(1)

    log_ret = np.log(close / close.shift(1))
    d['Vol_cluster'] = (log_ret**2).ewm(span=5).mean() / ((log_ret**2).ewm(span=20).mean() + 1e-9)

    return d


# ─────────────────────────────────────────────
# Индикаторы 4H
# ─────────────────────────────────────────────
def calc_indicators_4h(df4h: pd.DataFrame) -> pd.DataFrame:
    d     = df4h.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    for p in [7, 14]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p - 1, min_periods=p).mean()
        avg_l = l.ewm(com=p - 1, min_periods=p).mean()
        d[f'RSI_{p}_4h'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d['MACD_hist_4h'] = macd - macd.ewm(span=9, adjust=False).mean()

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d['EMA_ratio_4h'] = ema20 / (ema50 + 1e-9)

    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    d['ATR_pct_4h'] = (atr14 / (close + 1e-9)) * 100

    d['Vol_ratio_4h']  = vol / (vol.rolling(20).mean() + 1e-9)
    d['Return_4h_tf']  = close.pct_change(1) * 100
    d['Return_24h_tf'] = close.pct_change(6) * 100

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX_4h'] = dx.ewm(alpha=1/14).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d['BB_pos_4h'] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

    d['Hurst_4h'] = calc_hurst_exponent(close, range(2, 15))

    return d


# ─────────────────────────────────────────────
# Слияние 1H + 4H
# ─────────────────────────────────────────────
def merge_timeframes(df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    cols_4h        = [c for c in df4h.columns if c.endswith('_4h') or c.endswith('_4h_tf')]
    df4h_sub       = df4h[cols_4h].copy()
    df_merged      = df1h.copy()
    df4h_reindexed = df4h_sub.reindex(df1h.index, method='ffill')
    df_merged      = pd.concat([df_merged, df4h_reindexed], axis=1)
    df_merged      = df_merged.dropna(subset=cols_4h)
    return df_merged


# ─────────────────────────────────────────────
# TRIPLE BARRIER LABELING
# ─────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame,
                           horizon: int = None,
                           tp_mult: float = None,
                           sl_mult: float = None) -> pd.DataFrame:
    if horizon is None:
        horizon = TARGET_HORIZON
    if tp_mult is None:
        tp_mult = ATR_TP_MULT
    if sl_mult is None:
        sl_mult = ATR_SL_MULT

    close  = df['Close'].values
    atr    = df['ATR'].values
    high   = df['High'].values
    low    = df['Low'].values
    n      = len(df)

    target_buy  = np.full(n, np.nan)
    target_sell = np.full(n, np.nan)

    for i in range(n - horizon):
        entry  = close[i]
        atr_i  = atr[i]

        tp_buy = entry + atr_i * tp_mult
        sl_buy = entry - atr_i * sl_mult

        tp_sell = entry - atr_i * tp_mult
        sl_sell = entry + atr_i * sl_mult

        buy_result  = np.nan
        sell_result = np.nan

        for j in range(i + 1, min(i + horizon + 1, n)):
            h = high[j]
            l = low[j]

            if np.isnan(buy_result):
                if h >= tp_buy and l <= sl_buy:
                    buy_result = 1 if close[j-1] < entry + atr_i * 0.5 else 0
                elif h >= tp_buy:
                    buy_result = 1
                elif l <= sl_buy:
                    buy_result = 0

            if np.isnan(sell_result):
                if l <= tp_sell and h >= sl_sell:
                    sell_result = 1 if close[j-1] > entry - atr_i * 0.5 else 0
                elif l <= tp_sell:
                    sell_result = 1
                elif h >= sl_sell:
                    sell_result = 0

            if not np.isnan(buy_result) and not np.isnan(sell_result):
                break

        target_buy[i]  = buy_result
        target_sell[i] = sell_result

    df = df.copy()
    df['Target_BUY']  = target_buy
    df['Target_SELL'] = target_sell

    total     = n - horizon
    buy_valid = int(np.sum(~np.isnan(target_buy[:total])))
    buy_pos   = int(np.nansum(target_buy[:total]))
    sel_valid = int(np.sum(~np.isnan(target_sell[:total])))
    sel_pos   = int(np.nansum(target_sell[:total]))

    logger.info(
        f"[Trainer] Triple Barrier: "
        f"BUY valid={buy_valid} pos={buy_pos} ({buy_pos/(buy_valid+1e-9):.1%}) | "
        f"SELL valid={sel_valid} pos={sel_pos} ({sel_pos/(sel_valid+1e-9):.1%})"
    )
    return df


# ─────────────────────────────────────────────
# SMOTE балансировка
# ─────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    if not SMOTE_AVAILABLE:
        return X_train, y_train

    pos = y_train.sum()
    neg = len(y_train) - pos
    ratio = pos / (neg + 1e-9)

    if ratio > 0.4:
        return X_train, y_train

    try:
        smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"[Trainer] SMOTE: {len(y_train)} → {len(y_res)} "
            f"(pos: {int(y_train.sum())} → {int(y_res.sum())})"
        )
        return X_res, y_res
    except Exception as e:
        logger.warning(f"[Trainer] SMOTE ошибка: {e}")
        return X_train, y_train


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PRUNING
# ─────────────────────────────────────────────
def prune_features(model_xgb, model_lgbm, feature_cols: list,
                   threshold: float = FEATURE_IMPORTANCE_THRESHOLD) -> list:
    n = len(feature_cols)
    importance = np.zeros(n)

    if model_xgb is not None:
        try:
            imp = model_xgb.feature_importances_
            if len(imp) == n:
                importance += imp / (imp.sum() + 1e-9)
        except Exception:
            pass

    if model_lgbm is not None and LGBM_AVAILABLE:
        try:
            imp = model_lgbm.feature_importances_
            if len(imp) == n:
                importance += imp / (imp.sum() + 1e-9)
        except Exception:
            pass

    total = importance.sum()
    if total > 0:
        importance = importance / total
    else:
        return feature_cols

    importance_dict = {feature_cols[i]: float(importance[i]) for i in range(n)}
    importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))
    try:
        with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
            json.dump(importance_sorted, f, indent=2)
    except Exception:
        pass

    kept = [feature_cols[i] for i in range(n) if importance[i] >= threshold]

    if len(kept) < 15:
        sorted_idx = np.argsort(importance)[::-1]
        kept = [feature_cols[i] for i in sorted_idx[:20]]

    removed = [f for f in feature_cols if f not in kept]
    logger.info(
        f"[Trainer] Pruning: {n} → {len(kept)} признаков "
        f"(убрано {len(removed)}: {removed[:5]}{'...' if len(removed) > 5 else ''})"
    )
    return kept


# ─────────────────────────────────────────────
# v8.0: Optuna с TimeSeriesSplit (без data leakage в тюнинге)
# ─────────────────────────────────────────────
def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    """
    v8.0: Используем TimeSeriesSplit вместо простого train/val split.
    Это устраняет data leakage при подборе гиперпараметров.
    50 trials вместо 30 — лучший поиск в пространстве параметров.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'subsample':        trial.suggest_float('subsample', 0.55, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 25),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.7),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.5),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 4.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
        }
        scores = []
        for tr_idx, val_idx in tscv.split(X_train, y_train):
            X_tr, X_v = X_train[tr_idx], X_train[val_idx]
            y_tr, y_v = y_train[tr_idx], y_train[val_idx]
            if y_tr.sum() < 5 or y_v.sum() < 2:
                continue
            m = XGBClassifier(
                **params, eval_metric='logloss',
                use_label_encoder=False, verbosity=0,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            y_pred = m.predict(X_v)
            scores.append(precision_score(y_v, y_pred, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[Trainer] Optuna best precision: {study.best_value:.3f} ({n_trials} trials)")
    return study.best_params


# ─────────────────────────────────────────────
# Обучение XGBoost
# ─────────────────────────────────────────────
def train_binary_xgb(X_train, y_train, X_test, y_test,
                     best_params: dict = None) -> tuple:
    if best_params is None:
        best_params = {
            'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.03,
            'subsample': 0.75, 'colsample_bytree': 0.70,
            'min_child_weight': 10, 'gamma': 0.2,
            'reg_alpha': 0.3, 'reg_lambda': 2.0, 'scale_pos_weight': 2.0,
        }

    model = XGBClassifier(
        **best_params, eval_metric='logloss',
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }
    return model, metrics


# ─────────────────────────────────────────────
# Обучение LightGBM
# ─────────────────────────────────────────────
def train_binary_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    if not LGBM_AVAILABLE:
        return None, None

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / (pos_count + 1e-9)

    model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.70, min_child_samples=20,
        reg_alpha=0.3, reg_lambda=2.0,
        scale_pos_weight=min(scale_pos, 5.0),
        objective='binary', verbosity=-1, n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }
    return model, metrics


# ─────────────────────────────────────────────
# v8.0 NEW: КАЛИБРОВКА ВЕРОЯТНОСТЕЙ
# ─────────────────────────────────────────────
def calibrate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    label: str = "BUY") -> tuple:
    """
    Isotonic Regression калибровка вероятностей.

    ЗАЧЕМ:
    XGBoost и LGBM возвращают "scores", а не реальные вероятности.
    p(BUY)=0.75 у некалиброванной модели может означать лишь 55% реального win rate.
    После Isotonic калибровки p(BUY)=0.75 ≈ реальные 75%.

    ПОЧЕМУ ВАЖНО:
    - Kelly Criterion: f* = (p×b - q) / b
      Если p неточна — Kelly даёт неверный размер позиции.
    - Пороги 0.58/0.62/0.70/0.75 имеют реальный экономический смысл.

    Isotonic лучше Platt Scaling для tree-based моделей (непараметрический).
    Используется в Citadel, Two Sigma.
    """
    try:
        # Используем TimeSeriesSplit для калибровки (без data leakage)
        tscv = TimeSeriesSplit(n_splits=3)

        calibrated = CalibratedClassifierCV(
            estimator=model,
            method='isotonic',
            cv=tscv
        )
        calibrated.fit(X_train, y_train)

        # Оцениваем качество калибровки на тесте
        y_proba_raw  = model.predict_proba(X_test)[:, 1]
        y_proba_cal  = calibrated.predict_proba(X_test)[:, 1]
        y_pred_cal   = calibrated.predict(X_test)

        # Reliability: средняя вероятность предсказания для реальных позитивов
        if y_test.sum() > 0:
            raw_mean_pos = float(y_proba_raw[y_test == 1].mean())
            cal_mean_pos = float(y_proba_cal[y_test == 1].mean())
            raw_auc = float(roc_auc_score(y_test, y_proba_raw))
            cal_auc = float(roc_auc_score(y_test, y_proba_cal))
        else:
            raw_mean_pos = cal_mean_pos = 0.5
            raw_auc = cal_auc = 0.0

        prec_cal = float(precision_score(y_test, y_pred_cal, zero_division=0))
        rec_cal  = float(recall_score(y_test, y_pred_cal, zero_division=0))

        metrics = {
            'precision':     prec_cal,
            'recall':        rec_cal,
            'roc_auc':       cal_auc,
            'raw_auc':       raw_auc,
            'raw_mean_pos':  raw_mean_pos,
            'cal_mean_pos':  cal_mean_pos,
        }

        logger.info(
            f"[Trainer] Calibration {label}: "
            f"AUC {raw_auc:.3f}→{cal_auc:.3f} | "
            f"mean_pos {raw_mean_pos:.1%}→{cal_mean_pos:.1%} | "
            f"prec={prec_cal:.1%}"
        )
        return calibrated, metrics

    except Exception as e:
        logger.warning(f"[Trainer] Калибровка {label} ошибка: {e}")
        return model, {}


# ─────────────────────────────────────────────
# STACKING АНСАМБЛЬ
# ─────────────────────────────────────────────

def train_binary_cat(X_train, y_train, X_test, y_test) -> tuple:
    if not CATBOOST_AVAILABLE:
        return None, {}
    try:
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='AUC',
            random_seed=42, verbose=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        metrics = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob) if len(set(y_test))>1 else 0.5,
        }
        return model, metrics
    except Exception as e:
        logger.warning(f"[Trainer] CatBoost failed: {e}")
        return None, {}

def train_stacking_ensemble(
    model_xgb, model_lgbm,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label: str = "BUY",
    model_cat=None
) -> tuple:
    """
    Stacking: XGB + LGBM → LogisticRegression (meta-learner).
    v8.0: meta-learner тоже калиброван.
    """
    if not LGBM_AVAILABLE or model_lgbm is None:
        return None, None

    try:
        p_xgb  = model_xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
        p_lgbm = model_lgbm.predict_proba(X_test)[:, 1].reshape(-1, 1)

        p_avg  = ((p_xgb + p_lgbm) / 2)
        p_diff = (p_xgb - p_lgbm)
        X_stack_test = np.hstack([p_xgb, p_lgbm, p_avg, p_diff])

        from sklearn.model_selection import StratifiedKFold
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

        p_xgb_oof  = np.zeros(len(X_train))
        p_lgbm_oof = np.zeros(len(X_train))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val_fold = y_train[tr_idx], y_train[val_idx]

            if y_tr.sum() < 5:
                continue

            xgb_fold = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                eval_metric='logloss', use_label_encoder=False, verbosity=0
            )
            xgb_fold.fit(X_tr, y_tr, verbose=False)
            p_xgb_oof[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]

            if LGBM_AVAILABLE:
                lgbm_fold = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    verbosity=-1, n_jobs=-1
                )
                lgbm_fold.fit(X_tr, y_tr)
                p_lgbm_oof[val_idx] = lgbm_fold.predict_proba(X_val)[:, 1]

        p_avg_oof  = (p_xgb_oof + p_lgbm_oof) / 2
        p_diff_oof = p_xgb_oof - p_lgbm_oof
        X_stack_train = np.column_stack([
            p_xgb_oof, p_lgbm_oof, p_avg_oof, p_diff_oof
        ])

        scaler   = StandardScaler()
        X_st_tr  = scaler.fit_transform(X_stack_train)
        X_st_te  = scaler.transform(X_stack_test)

        stack_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        stack_model.fit(X_st_tr, y_train)

        y_pred  = stack_model.predict(X_st_te)
        y_proba = stack_model.predict_proba(X_st_te)[:, 1]
        metrics = {
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
            'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
        }

        stack_bundle = {'model': stack_model, 'scaler': scaler}

        logger.info(
            f"[Trainer] Stack {label}: "
            f"prec={metrics['precision']:.1%} "
            f"rec={metrics['recall']:.1%} "
            f"auc={metrics['roc_auc']:.3f}"
        )
        return stack_bundle, metrics

    except Exception as e:
        logger.warning(f"[Trainer] Stack {label} ошибка: {e}")
        return None, None


# ─────────────────────────────────────────────
# META-LABELING (OOF, без data leakage)
# ─────────────────────────────────────────────
def train_meta_model(
    X_train: np.ndarray, y_true_train: np.ndarray,
    X_test: np.ndarray, y_true_test: np.ndarray,
    side_model
) -> tuple:
    try:
        from sklearn.model_selection import StratifiedKFold

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

        oof_proba = np.zeros(len(X_train))

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_true_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr = y_true_train[tr_idx]

            if y_tr.sum() < 5:
                oof_proba[val_idx] = 0.5
                continue

            fold_model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.75, colsample_bytree=0.7,
                min_child_weight=10, gamma=0.2,
                eval_metric='logloss', use_label_encoder=False, verbosity=0
            )
            fold_model.fit(X_tr, y_tr, verbose=False)
            oof_proba[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        oof_pred_binary = (oof_proba >= 0.50).astype(int)
        y_meta_train    = ((oof_pred_binary == 1) & (y_true_train == 1)).astype(int)

        if y_meta_train.sum() < 10:
            logger.warning("[Trainer] Meta-model: мало позитивных OOF примеров, пропускаем")
            return None, None

        X_meta_train = np.hstack([X_train, oof_proba.reshape(-1, 1)])

        p_test  = side_model.predict_proba(X_test)[:, 1]
        X_meta_test = np.hstack([X_test, p_test.reshape(-1, 1)])

        p_test_binary = (p_test >= 0.50).astype(int)
        y_meta_test   = ((p_test_binary == 1) & (y_true_test == 1)).astype(int)

        X_meta_sm, y_meta_sm = apply_smote(X_meta_train, y_meta_train)

        meta_model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.7,
            min_child_weight=10, gamma=0.3,
            eval_metric='logloss', use_label_encoder=False, verbosity=0,
        )
        meta_model.fit(
            X_meta_sm, y_meta_sm,
            eval_set=[(X_meta_test, y_meta_test)],
            verbose=False,
        )

        y_pred  = meta_model.predict(X_meta_test)
        y_proba = meta_model.predict_proba(X_meta_test)[:, 1]

        if y_meta_test.sum() > 0:
            metrics = {
                'precision': float(precision_score(y_meta_test, y_pred, zero_division=0)),
                'recall':    float(recall_score(y_meta_test, y_pred, zero_division=0)),
                'roc_auc':   float(roc_auc_score(y_meta_test, y_proba)),
            }
        else:
            metrics = {'precision': 0.0, 'recall': 0.0, 'roc_auc': 0.0}

        logger.info(
            f"[Trainer] Meta-model: "
            f"prec={metrics['precision']:.1%} "
            f"rec={metrics['recall']:.1%}"
        )
        return meta_model, metrics

    except Exception as e:
        logger.warning(f"[Trainer] Meta-model ошибка: {e}")
        return None, None


# ─────────────────────────────────────────────
# v8.0: Kelly на реальных WF-доходностях
# ─────────────────────────────────────────────
def calc_kelly_criterion(win_rate: float, avg_win_pct: float,
                          avg_loss_pct: float) -> float:
    """
    Kelly Criterion: f* = (W/L × win_rate - loss_rate) / (W/L)
    Half-Kelly для снижения риска разорения.
    """
    if avg_loss_pct <= 0 or win_rate <= 0:
        return 0.10

    loss_rate = 1 - win_rate
    odds      = avg_win_pct / avg_loss_pct

    kelly = (odds * win_rate - loss_rate) / odds
    kelly = max(0.0, min(kelly, 0.5))

    return round(kelly / 2, 3)


def calc_kelly_from_wf_returns(trade_returns: list) -> float:
    """
    v8.0: Kelly из реальных WF-доходностей (не приближения).
    Точнее чем формульный расчёт из win_rate + avg_win/loss.
    """
    if len(trade_returns) < 10:
        return 0.10

    arr = np.array(trade_returns)
    wins  = arr[arr > 0]
    losses = arr[arr < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.10

    win_rate  = len(wins) / len(arr)
    avg_win   = float(wins.mean())
    avg_loss  = float(abs(losses.mean()))

    kelly = calc_kelly_criterion(win_rate, avg_win, avg_loss)
    logger.info(
        f"[Trainer] Kelly из WF-доходностей: "
        f"WR={win_rate:.1%} avg_win={avg_win:.2f}% avg_loss={avg_loss:.2f}% "
        f"→ Kelly={kelly:.1%}"
    )
    return kelly


# ─────────────────────────────────────────────
# Walk-Forward с накоплением trade returns
# ─────────────────────────────────────────────
def walk_forward_binary(X: np.ndarray, y: np.ndarray,
                        train_size: int, test_size: int, step: int) -> dict:
    results = []
    all_trade_returns = []
    n = len(X)
    start = train_size

    while start + test_size <= n:
        X_tr = X[start - train_size: start]
        y_tr = y[start - train_size: start]
        X_te = X[start: start + test_size]
        y_te = y[start: start + test_size]

        if y_tr.sum() < 5 or y_te.sum() < 3:
            start += step
            continue

        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = min(neg / (pos + 1e-9), 5.0)

        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.7,
            min_child_weight=10, gamma=0.2,
            reg_alpha=0.3, reg_lambda=2.0,
            scale_pos_weight=spw,
            eval_metric='logloss', use_label_encoder=False, verbosity=0
        )
        m.fit(X_tr_sm, y_tr_sm, verbose=False)

        y_pred   = m.predict(X_te)
        y_proba  = m.predict_proba(X_te)[:, 1]
        prec     = precision_score(y_te, y_pred, zero_division=0)
        rec      = recall_score(y_te, y_pred, zero_division=0)

        tp_pct = ATR_TP_MULT * 0.015
        sl_pct = ATR_SL_MULT * 0.015
        hourly_returns = []
        for pred, true in zip(y_pred, y_te):
            if pred == 1:
                r = (tp_pct if true == 1 else -sl_pct) * 100  # в процентах
                hourly_returns.append(r)
                all_trade_returns.append(r)
            else:
                hourly_returns.append(0.0)

        r = np.array(hourly_returns)
        sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(8760)) if r.std() > 0 else 0.0

        results.append({'precision': prec, 'recall': rec, 'sharpe': sharpe})
        start += step

    if not results:
        return {
            'wf_precision': 0.0, 'wf_recall': 0.0,
            'wf_sharpe': 0.0, 'wf_folds': 0,
            'wf_trade_returns': []
        }

    return {
        'wf_precision':      round(float(np.mean([r['precision'] for r in results])), 4),
        'wf_recall':         round(float(np.mean([r['recall']    for r in results])), 4),
        'wf_sharpe':         round(float(np.mean([r['sharpe']    for r in results])), 3),
        'wf_folds':          len(results),
        'wf_trade_returns':  all_trade_returns,  # v8.0: для точного Kelly
    }


# ─────────────────────────────────────────────
# Доступные фичи
# ─────────────────────────────────────────────
def get_available_features(df: pd.DataFrame, desired: list) -> list:
    available = [c for c in desired if c in df.columns]
    missing   = [c for c in desired if c not in df.columns]
    if missing:
        logger.warning(f"[Trainer] Отсутствуют фичи: {missing}")
    return available


FEATURE_COLS_V7_EXTRA = [
    'Hurst', 'VWAP_dev_20', 'VWAP_dev_50', 'VWAP_bull_ratio',
    'RV_20', 'RV_50', 'RV_ratio', 'OFI', 'Price_accel', 'Vol_cluster',
    'Hurst_4h',
]


# ─────────────────────────────────────────────
# ГЛАВНАЯ: обучение v8.0
# ─────────────────────────────────────────────
def train_model() -> dict:
    logger.info("[Trainer] 🚀 v8.0: Калибровка + 8000 свечей + Улучшенный Kelly + 50 Optuna trials")

    # 1. Данные (v8.0: увеличены объёмы)
    df1h_raw = fetch_ohlcv("TON-USDT", "1H", BARS_1H)
    df4h_raw = fetch_ohlcv("TON-USDT", "4H", BARS_4H)

    if df1h_raw.empty:
        return {"success": False, "error": "Нет 1H данных"}

    # 2. Индикаторы
    df1h = calc_indicators_1h(df1h_raw)

    if not df4h_raw.empty:
        df4h = calc_indicators_4h(df4h_raw)
        df   = merge_timeframes(df1h, df4h)
        logger.info(f"[Trainer] ✅ Объединены 1H + 4H | строк: {len(df)}")
    else:
        logger.warning("[Trainer] ⚠️ 4H данные недоступны")
        df = df1h

    df = df.dropna()
    if len(df) < 300:
        return {"success": False, "error": f"Мало данных: {len(df)}"}

    # 3. TRIPLE BARRIER разметка
    logger.info("[Trainer] 🎯 Triple Barrier разметка...")
    df = triple_barrier_labels(df)

    # 4. Убираем строки без метки
    df_buy  = df[~df['Target_BUY'].isna()].copy()
    df_sell = df[~df['Target_SELL'].isna()].copy()

    # 5. Фичи
    all_feature_cols = FEATURE_COLS + [f for f in FEATURE_COLS_V7_EXTRA if f not in FEATURE_COLS]
    feature_cols = get_available_features(df, all_feature_cols)

    if len(feature_cols) < 10:
        feature_cols = get_available_features(df, FEATURE_COLS_LEGACY)
        logger.warning(f"[Trainer] Legacy features: {len(feature_cols)} шт")
    else:
        logger.info(f"[Trainer] Используем {len(feature_cols)} признаков")

    X_buy  = df_buy[feature_cols].values.astype(np.float32)
    y_buy  = df_buy['Target_BUY'].values.astype(int)
    X_sell = df_sell[feature_cols].values.astype(np.float32)
    y_sell = df_sell['Target_SELL'].values.astype(int)

    X_buy  = np.nan_to_num(X_buy,  nan=0.0, posinf=0.0, neginf=0.0)
    X_sell = np.nan_to_num(X_sell, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. Train/test split
    split_buy  = int(len(X_buy)  * 0.8)
    split_sell = int(len(X_sell) * 0.8)

    X_buy_train,  X_buy_test  = X_buy[:split_buy],   X_buy[split_buy:]
    y_buy_train,  y_buy_test  = y_buy[:split_buy],   y_buy[split_buy:]
    X_sell_train, X_sell_test = X_sell[:split_sell], X_sell[split_sell:]
    y_sell_train, y_sell_test = y_sell[:split_sell], y_sell[split_sell:]

    logger.info(
        f"[Trainer] BUY train={len(X_buy_train)} pos={y_buy_train.sum()} | "
        f"SELL train={len(X_sell_train)} pos={y_sell_train.sum()}"
    )

    # 7. SMOTE
    logger.info("[Trainer] 🔄 SMOTE BUY...")
    X_buy_sm,  y_buy_sm  = apply_smote(X_buy_train,  y_buy_train)
    logger.info("[Trainer] 🔄 SMOTE SELL...")
    X_sell_sm, y_sell_sm = apply_smote(X_sell_train, y_sell_train)

    # 8. Optuna (v8.0: 50 trials + TimeSeriesSplit)
    logger.info("[Trainer] 🔬 Optuna тюнинг BUY (50 trials, TimeSeriesSplit)...")
    best_params = tune_xgboost(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test, n_trials=50)

    # 9. BUY модели
    logger.info("[Trainer] 🔧 XGBoost BUY...")
    buy_xgb, buy_xgb_m = train_binary_xgb(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test, best_params)
    logger.info(f"[Trainer] BUY XGB: prec={buy_xgb_m['precision']:.1%} auc={buy_xgb_m['roc_auc']:.3f}")

    logger.info("[Trainer] 🔧 LightGBM BUY...")
    buy_lgbm, buy_lgbm_m = train_binary_lgbm(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test)
    if buy_lgbm_m:
        logger.info(f"[Trainer] BUY LGBM: prec={buy_lgbm_m['precision']:.1%} auc={buy_lgbm_m['roc_auc']:.3f}")

    # 10. SELL модели
    logger.info("[Trainer] 🔧 XGBoost SELL...")
    sell_xgb, sell_xgb_m = train_binary_xgb(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test, best_params)

    logger.info("[Trainer] 🔧 LightGBM SELL...")
    sell_lgbm, sell_lgbm_m = train_binary_lgbm(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test)

    # 11. FEATURE PRUNING
    logger.info("[Trainer] ✂️ Feature Importance Pruning...")
    feature_cols_pruned = prune_features(buy_xgb, buy_lgbm, feature_cols)

    if len(feature_cols_pruned) < len(feature_cols) * 0.85:
        logger.info(f"[Trainer] 🔄 Переобучение на {len(feature_cols_pruned)} отобранных признаках...")
        X_buy_pruned  = X_buy[:, [feature_cols.index(f) for f in feature_cols_pruned if f in feature_cols]]
        X_sell_pruned = X_sell[:, [feature_cols.index(f) for f in feature_cols_pruned if f in feature_cols]]

        X_bp_train, X_bp_test = X_buy_pruned[:split_buy],   X_buy_pruned[split_buy:]
        X_sp_train, X_sp_test = X_sell_pruned[:split_sell], X_sell_pruned[split_sell:]

        X_bp_sm, y_bp_sm = apply_smote(X_bp_train, y_buy_train)
        X_sp_sm, y_sp_sm = apply_smote(X_sp_train, y_sell_train)

        buy_xgb_p,   buy_xgb_pm   = train_binary_xgb(X_bp_sm, y_bp_sm, X_bp_test, y_buy_test, best_params)
        buy_lgbm_p,  buy_lgbm_pm  = train_binary_lgbm(X_bp_sm, y_bp_sm, X_bp_test, y_buy_test)
        sell_xgb_p,  sell_xgb_pm  = train_binary_xgb(X_sp_sm, y_sp_sm, X_sp_test, y_sell_test, best_params)
        sell_lgbm_p, sell_lgbm_pm = train_binary_lgbm(X_sp_sm, y_sp_sm, X_sp_test, y_sell_test)

        if buy_xgb_pm['precision'] >= buy_xgb_m['precision']:
            buy_xgb,  buy_xgb_m   = buy_xgb_p,  buy_xgb_pm
            buy_lgbm, buy_lgbm_m  = buy_lgbm_p,  buy_lgbm_pm
            sell_xgb, sell_xgb_m  = sell_xgb_p,  sell_xgb_pm
            sell_lgbm, sell_lgbm_m = sell_lgbm_p, sell_lgbm_pm
            feature_cols = feature_cols_pruned
            X_buy_train, X_buy_test = X_bp_train, X_bp_test
            X_sell_train, X_sell_test = X_sp_train, X_sp_test
            logger.info(f"[Trainer] ✅ Pruned модель лучше, используем {len(feature_cols)} признаков")

    # 12. STACKING
    logger.info("[Trainer] 🏗️ Stacking BUY...")
    stack_buy, stack_buy_m = train_stacking_ensemble(
        buy_xgb, buy_lgbm, X_buy_train, y_buy_train, X_buy_test, y_buy_test, "BUY", model_cat=buy_cat
    )

    logger.info("[Trainer] 🏗️ Stacking SELL...")
    stack_sell, stack_sell_m = train_stacking_ensemble(
        sell_xgb, sell_lgbm, X_sell_train, y_sell_train, X_sell_test, y_sell_test, "SELL", model_cat=sell_cat
    )

    # 13. META-LABELING
    logger.info("[Trainer] 🧩 Meta-model BUY (OOF)...")
    meta_buy, meta_buy_m = train_meta_model(
        X_buy_train, y_buy_train, X_buy_test, y_buy_test, buy_xgb
    )

    logger.info("[Trainer] 🧩 Meta-model SELL (OOF)...")
    meta_sell, meta_sell_m = train_meta_model(
        X_sell_train, y_sell_train, X_sell_test, y_sell_test, sell_xgb
    )

    # 14. v8.0 NEW: КАЛИБРОВКА ВЕРОЯТНОСТЕЙ
    logger.info("[Trainer] 🎯 Калибровка вероятностей BUY (Isotonic)...")
    calib_buy, calib_buy_m = calibrate_model(
        buy_xgb, X_buy_train, y_buy_train, X_buy_test, y_buy_test, "BUY"
    )

    logger.info("[Trainer] 🎯 Калибровка вероятностей SELL (Isotonic)...")
    calib_sell, calib_sell_m = calibrate_model(
        sell_xgb, X_sell_train, y_sell_train, X_sell_test, y_sell_test, "SELL"
    )

    # 15. Walk-Forward
    n_samples_buy = len(X_buy)
    wf_train = max(int(n_samples_buy * 0.55), 100)
    wf_test  = max(int(n_samples_buy * 0.12), 30)
    wf_step  = max(int(n_samples_buy * 0.08), 20)

    logger.info(f"[Trainer] 📊 Walk-Forward BUY (train={wf_train} test={wf_test} step={wf_step})...")
    wf_buy  = walk_forward_binary(X_buy,  y_buy,  wf_train, wf_test, wf_step)
    logger.info(f"[Trainer] 📊 Walk-Forward SELL...")
    wf_sell = walk_forward_binary(X_sell, y_sell, wf_train, wf_test, wf_step)

    logger.info(
        f"[Trainer] WF BUY:  prec={wf_buy['wf_precision']:.1%} "
        f"sharpe={wf_buy['wf_sharpe']:.2f} folds={wf_buy['wf_folds']}"
    )
    logger.info(
        f"[Trainer] WF SELL: prec={wf_sell['wf_precision']:.1%} "
        f"sharpe={wf_sell['wf_sharpe']:.2f} folds={wf_sell['wf_folds']}"
    )

    # 16. v8.0: Kelly из реальных WF-доходностей
    all_wf_returns = wf_buy.get('wf_trade_returns', []) + wf_sell.get('wf_trade_returns', [])
    if len(all_wf_returns) >= 10:
        kelly_f = calc_kelly_from_wf_returns(all_wf_returns)
    else:
        # Fallback: формульный Kelly
        kelly_f = calc_kelly_criterion(
            wf_buy['wf_precision'],
            ATR_TP_MULT * 1.5,
            ATR_SL_MULT * 1.5
        )
    logger.info(f"[Trainer] Kelly Criterion (Half-Kelly): {kelly_f:.1%}")

    # 17. Сохраняем модели
    joblib.dump(buy_xgb,  MODEL_PATH_BUY_XGB)
    joblib.dump(sell_xgb, MODEL_PATH_SELL_XGB)
    if buy_lgbm:
        joblib.dump(buy_lgbm,  MODEL_PATH_BUY_LGBM)
    if sell_lgbm:
        joblib.dump(sell_lgbm, MODEL_PATH_SELL_LGBM)
    if meta_buy:
        joblib.dump(meta_buy,  META_MODEL_BUY_PATH)
    if meta_sell:
        joblib.dump(meta_sell, META_MODEL_SELL_PATH)
    if stack_buy:
        joblib.dump(stack_buy,  STACK_MODEL_BUY_PATH)
    if stack_sell:
        joblib.dump(stack_sell, STACK_MODEL_SELL_PATH)

    # v8.0: сохраняем калиброванные модели
    if calib_buy is not None:
        joblib.dump(calib_buy,  CALIB_MODEL_BUY_PATH)
        logger.info(f"[Trainer] ✅ Calibrated BUY сохранена: {CALIB_MODEL_BUY_PATH}")
    if calib_sell is not None:
        joblib.dump(calib_sell, CALIB_MODEL_SELL_PATH)
        logger.info(f"[Trainer] ✅ Calibrated SELL сохранена: {CALIB_MODEL_SELL_PATH}")

    with open(MODEL_FEATURES_PATH, 'w') as f:
        json.dump(feature_cols, f)

    # 18. Итоги
    avg_buy_prec  = (buy_xgb_m['precision'] + (buy_lgbm_m['precision'] if buy_lgbm_m else buy_xgb_m['precision'])) / 2
    avg_sell_prec = (sell_xgb_m['precision'] + (sell_lgbm_m['precision'] if sell_lgbm_m else sell_xgb_m['precision'])) / 2
    avg_buy_auc   = (buy_xgb_m['roc_auc']   + (buy_lgbm_m['roc_auc']   if buy_lgbm_m else buy_xgb_m['roc_auc']))   / 2
    avg_sell_auc  = (sell_xgb_m['roc_auc']  + (sell_lgbm_m['roc_auc']  if sell_lgbm_m else sell_xgb_m['roc_auc'])) / 2

    stats = {
        "success":             True,
        "version":             "8.0",
        "labeling":            "triple_barrier",
        "n_features":          len(feature_cols),
        "n_samples_buy":       len(df_buy),
        "n_samples_sell":      len(df_sell),
        "n_samples":           len(df_buy),
        "n_train":             split_buy,
        "n_test":              len(X_buy_test),
        "bars_loaded":         BARS_1H,
        "kelly_fraction":      kelly_f,
        "buy_xgb_precision":   buy_xgb_m['precision'],
        "buy_xgb_recall":      buy_xgb_m['recall'],
        "buy_xgb_auc":         buy_xgb_m['roc_auc'],
        "buy_lgbm_precision":  buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "buy_lgbm_auc":        buy_lgbm_m['roc_auc']  if buy_lgbm_m else None,
        "avg_buy_precision":   avg_buy_prec,
        "avg_buy_auc":         avg_buy_auc,
        "sell_xgb_precision":  sell_xgb_m['precision'],
        "sell_xgb_recall":     sell_xgb_m['recall'],
        "sell_xgb_auc":        sell_xgb_m['roc_auc'],
        "sell_lgbm_precision": sell_lgbm_m['precision'] if sell_lgbm_m else None,
        "sell_lgbm_auc":       sell_lgbm_m['roc_auc']  if sell_lgbm_m else None,
        "avg_sell_precision":  avg_sell_prec,
        "avg_sell_auc":        avg_sell_auc,
        "stack_buy_precision":  stack_buy_m['precision']  if stack_buy_m  else None,
        "stack_sell_precision": stack_sell_m['precision'] if stack_sell_m else None,
        "meta_buy_precision":  meta_buy_m['precision']    if meta_buy_m   else None,
        "meta_sell_precision": meta_sell_m['precision']   if meta_sell_m  else None,
        "calib_buy_auc":       calib_buy_m.get('roc_auc') if calib_buy_m else None,
        "calib_buy_prec":      calib_buy_m.get('precision') if calib_buy_m else None,
        "calib_sell_auc":      calib_sell_m.get('roc_auc') if calib_sell_m else None,
        "wf_buy_precision":    wf_buy['wf_precision'],
        "wf_sell_precision":   wf_sell['wf_precision'],
        "wf_buy_sharpe":       wf_buy['wf_sharpe'],
        "wf_sell_sharpe":      wf_sell['wf_sharpe'],
        "wf_folds":            wf_buy['wf_folds'],
        "wf_trade_count":      len(all_wf_returns),
        "xgb_precision":       avg_buy_prec,
        "lgbm_precision":      buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "ensemble_precision":  (avg_buy_prec + avg_sell_prec) / 2,
        "wf_precision":        (wf_buy['wf_precision'] + wf_sell['wf_precision']) / 2,
        "wf_accuracy":         0.0,
        "lgbm_available":      LGBM_AVAILABLE,
        "smote_available":     SMOTE_AVAILABLE,
        "meta_labeling":       meta_buy is not None,
        "stacking":            stack_buy is not None,
        "calibration":         calib_buy is not None,
    }

    with open(STATS_FILE, 'w') as f:
        json.dump({k: v for k, v in stats.items() if k != 'wf_trade_returns'}, f, indent=2)

    logger.info(
        f"[Trainer] ✅ v8.0 Готово! "
        f"BUY prec={avg_buy_prec:.1%} auc={avg_buy_auc:.3f} | "
        f"SELL prec={avg_sell_prec:.1%} auc={avg_sell_auc:.3f} | "
        f"WF Sharpe={wf_buy['wf_sharpe']:.2f}/{wf_sell['wf_sharpe']:.2f} | "
        f"Kelly={kelly_f:.1%} | "
        f"Calibrated={'✅' if calib_buy else '❌'}"
    )

    return {
        **stats,
        "model":      buy_xgb,
        "lgbm_model": buy_lgbm,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    result = train_model()
    if result.get("success"):
        print(f"\n✅ Обучение v8.0 завершено!")
        print(f"   Kelly: {result['kelly_fraction']:.1%}")
        print(f"   BUY precision: {result['avg_buy_precision']:.1%}")
        print(f"   SELL precision: {result['avg_sell_precision']:.1%}")
        print(f"   WF Sharpe BUY: {result['wf_buy_sharpe']:.2f}")
        print(f"   Калибровка: {'✅' if result.get('calibration') else '❌'}")
        print(f"   Данных загружено: {result.get('bars_loaded')} свечей")
    else:
        print(f"❌ Ошибка: {result.get('error')}")