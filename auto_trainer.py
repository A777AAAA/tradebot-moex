"""
auto_trainer.py MOEX v2.0
Отдельная ML-модель на каждый инструмент из SYMBOLS_ALL.

Почему отдельные модели:
  - SBERP падает на санкциях, TRNFP — на нефти, MOEX — на ставке ЦБ
  - Разные ATR: TRNFP ≈ 500₽, SBERP ≈ 3₽ — нельзя смешивать
  - Buy-the-Dip признаки (просадка от хая) специфичны для каждой бумаги
  - Точность: +2-4% precision vs универсальная модель

Стратегия: Triple Barrier + Buy-the-Dip признаки + WF + Optuna + Calib + Stacking
"""

import os
import json
import joblib
import logging
import time
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
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

from config import (
    SYMBOLS_ALL, FEATURE_COLS, FEATURE_COLS_LEGACY,
    TARGET_HORIZON, TARGET_THRESHOLD,
    ATR_SL_MULT, ATR_TP_MULT,
    STATS_FILE, model_path, features_path, stats_path,
)
from moex_client import get_candles_multi, candles_to_df, resample_to_4h

logger = logging.getLogger(__name__)

FEATURE_IMPORTANCE_THRESHOLD = 0.005
BARS_1H = 5000   # ~555 торговых дней на 1H (9 ч/день)


# ─────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str, bars: int = BARS_1H) -> pd.DataFrame:
    try:
        raw = get_candles_multi(ticker, interval="60", total=bars)
        if not raw:
            logger.error(f"[Trainer] {ticker}: нет данных")
            return pd.DataFrame()
        df = candles_to_df(raw)
        logger.info(f"[Trainer] {ticker}: загружено {len(df)} свечей")
        return df
    except Exception as e:
        logger.error(f"[Trainer] {ticker} ошибка загрузки: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Buy-the-Dip специфичные признаки
# ─────────────────────────────────────────────
def calc_dip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки специально для стратегии Buy-the-Dip.
    Помогают модели находить локальные дна.
    """
    d     = df.copy()
    close = d["Close"]
    vol   = d["Volume"]

    # Просадка от 20-дневного (≈180 баров 1H) максимума
    high_20 = close.rolling(180).max()
    d["Drawdown_from_high_20"] = (close - high_20) / (high_20 + 1e-9) * 100

    # Просадка от 60-дневного (≈540 баров 1H) максимума
    high_60 = close.rolling(540).max()
    d["Drawdown_from_high_60"] = (close - high_60) / (high_60 + 1e-9) * 100

    # Баров с 20-дневного хая
    def bars_since_high(series, window=180):
        result = np.zeros(len(series))
        for i in range(window, len(series)):
            window_slice = series.values[max(0, i-window):i+1]
            result[i] = window - np.argmax(window_slice[::-1])
        return pd.Series(result, index=series.index)

    d["Days_since_high_20"] = bars_since_high(close, 180)

    # Последовательных красных баров
    is_down = (close < close.shift(1)).astype(int)
    consec  = is_down.groupby((is_down == 0).cumsum()).cumsum()
    d["Consec_down_bars"] = consec

    # Всплеск объёма (признак дна/разворота)
    vol_avg20        = vol.rolling(20).mean()
    d["Vol_surge"]   = vol / (vol_avg20 + 1e-9)

    return d


# ─────────────────────────────────────────────
# Индикаторы 1H
# ─────────────────────────────────────────────
def calc_hurst_exponent(ts: pd.Series, lags_range=range(2, 21)) -> pd.Series:
    def hurst_single(series):
        if len(series) < 20:
            return 0.5
        try:
            lags = list(lags_range)
            tau  = [max(np.std(np.subtract(series[lag:], series[:-lag])), 1e-9) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(max(0.0, min(1.0, poly[0])))
        except Exception:
            return 0.5
    return ts.rolling(100, min_periods=50).apply(lambda x: hurst_single(x), raw=True)


def calc_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    d     = df.copy()
    close = d["Close"]
    high  = d["High"]
    low   = d["Low"]
    vol   = d["Volume"]

    d["Hour"]      = d.index.hour
    d["DayOfWeek"] = d.index.dayofweek

    for p in [7, 14, 21]:
        diff = close.diff()
        g    = diff.clip(lower=0)
        l    = -diff.clip(upper=0)
        ag   = g.ewm(com=p-1, min_periods=p).mean()
        al   = l.ewm(com=p-1, min_periods=p).mean()
        d[f"RSI_{p}"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    d["MACD"]      = ema12 - ema26
    d["MACD_signal"]= d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    tr    = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    atr50 = tr.ewm(com=49, min_periods=50).mean()
    d["ATR"]      = atr14
    d["ATR_pct"]  = atr14 / (close + 1e-9) * 100
    d["ATR_norm"] = atr14 / (close + 1e-9)
    d["ATR_ratio"]= atr14 / (atr50 + 1e-9)

    sma20       = close.rolling(20).mean()
    std20       = close.rolling(20).std()
    d["BB_pos"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)
    d["BB_width"]= ((sma20+2*std20) - (sma20-2*std20)) / (sma20 + 1e-9)

    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    d["EMA_ratio_20_50"]  = ema20 / (ema50  + 1e-9)
    d["EMA_ratio_20_100"] = ema20 / (ema100 + 1e-9)

    d["Vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-9)

    obv         = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    d["OBV_norm"]= (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

    tp_mfi    = (high + low + close) / 3
    mf        = tp_mfi * vol
    pos_mf    = mf.where(tp_mfi > tp_mfi.shift(1), 0).rolling(14).sum()
    neg_mf    = mf.where(tp_mfi < tp_mfi.shift(1), 0).rolling(14).sum()
    d["MFI_14"]= 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    rsi14      = d["RSI_14"]
    sk_min     = rsi14.rolling(14).min()
    sk_max     = rsi14.rolling(14).max()
    stoch_k    = (rsi14 - sk_min) / (sk_max - sk_min + 1e-9) * 100
    d["StochRSI_K"] = stoch_k
    d["StochRSI_D"] = stoch_k.rolling(3).mean()

    hw14         = high.rolling(14).max()
    lw14         = low.rolling(14).min()
    d["WilliamsR"]= (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    d["ZScore_20"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    d["ZScore_50"] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    up   = high.diff(); down = -low.diff()
    pdm  = up.where((up > down) & (up > 0), 0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d["ADX"] = dx.ewm(alpha=1/14).mean()

    d["Body_pct"]   = (close - d["Open"]).abs() / (d["Open"] + 1e-9) * 100
    d["Upper_wick"] = (high - d[["Close","Open"]].max(axis=1)) / (d["Open"] + 1e-9) * 100
    d["Lower_wick"] = (d[["Close","Open"]].min(axis=1) - low) / (d["Open"] + 1e-9) * 100
    d["Doji"]       = ((d["Body_pct"] / (high - low + 1e-9)) < 0.1).astype(int)

    d["Momentum_10"] = close - close.shift(10)
    d["ROC_10"]      = close.pct_change(10) * 100

    for h in [1, 4, 12, 24]:
        d[f"Return_{h}h"] = close.pct_change(h) * 100

    # Профессиональные признаки
    d["Hurst"] = calc_hurst_exponent(close)

    tp_s = (high + low + close) / 3
    vwap20 = (tp_s * vol).rolling(20).sum() / vol.rolling(20).sum()
    vwap50 = (tp_s * vol).rolling(50).sum() / vol.rolling(50).sum()
    d["VWAP_dev_20"]     = (close - vwap20) / (vwap20 + 1e-9) * 100
    d["VWAP_dev_50"]     = (close - vwap50) / (vwap50 + 1e-9) * 100
    d["VWAP_bull_ratio"] = vol.where(close > vwap20, 0).rolling(10).sum() / (vol.rolling(10).sum() + 1e-9)

    log_ret    = np.log(close / close.shift(1))
    d["RV_20"] = np.sqrt((log_ret**2).rolling(20).sum() / 20 * 2268) * 100
    d["RV_50"] = np.sqrt((log_ret**2).rolling(50).sum() / 50 * 2268) * 100
    d["RV_ratio"] = d["RV_20"] / (d["RV_50"] + 1e-9)

    bf = (close - low) / (high - low + 1e-9)
    bv = (high - close) / (high - low + 1e-9)
    d["OFI"] = (bf * vol - bv * vol).rolling(10).sum() / (vol.rolling(10).sum() + 1e-9)

    ret1 = close.pct_change(1)
    d["Price_accel"] = ret1 - ret1.shift(1)
    d["Vol_cluster"] = (log_ret**2).ewm(span=5).mean() / ((log_ret**2).ewm(span=20).mean() + 1e-9)

    # Buy-the-Dip признаки
    d = calc_dip_features(d)

    return d


# ─────────────────────────────────────────────
# Индикаторы 4H
# ─────────────────────────────────────────────
def calc_indicators_4h(df4h: pd.DataFrame) -> pd.DataFrame:
    d     = df4h.copy()
    close = d["Close"]
    high  = d["High"]
    low   = d["Low"]
    vol   = d["Volume"]

    for p in [7, 14]:
        diff = close.diff(); g = diff.clip(lower=0); l = -diff.clip(upper=0)
        ag = g.ewm(com=p-1, min_periods=p).mean(); al = l.ewm(com=p-1, min_periods=p).mean()
        d[f"RSI_{p}_4h"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["MACD_hist_4h"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["EMA_ratio_4h"] = close.ewm(span=20).mean() / (close.ewm(span=50).mean() + 1e-9)

    tr    = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    d["ATR_pct_4h"]    = atr14 / (close + 1e-9) * 100
    d["Vol_ratio_4h"]  = vol / (vol.rolling(20).mean() + 1e-9)
    d["Return_4h_tf"]  = close.pct_change(1) * 100
    d["Return_24h_tf"] = close.pct_change(6) * 100

    up = high.diff(); down = -low.diff()
    pdm = up.where((up > down) & (up > 0), 0)
    mdm = down.where((down > up) & (down > 0), 0)
    pdi = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d["ADX_4h"] = dx.ewm(alpha=1/14).mean()

    sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    d["BB_pos_4h"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

    d["Hurst_4h"] = calc_hurst_exponent(close, range(2, 15))
    return d


def merge_timeframes(df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    cols_4h = [c for c in df4h.columns if c.endswith("_4h") or c.endswith("_4h_tf")]
    df4h_r  = df4h[cols_4h].reindex(df1h.index, method="ffill")
    merged  = pd.concat([df1h, df4h_r], axis=1).dropna(subset=cols_4h)
    return merged


# ─────────────────────────────────────────────
# Triple Barrier Labeling
# ─────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame, horizon=None, tp_mult=None, sl_mult=None) -> pd.DataFrame:
    if horizon is None: horizon = TARGET_HORIZON
    if tp_mult is None: tp_mult = ATR_TP_MULT
    if sl_mult is None: sl_mult = ATR_SL_MULT

    close = df["Close"].values; atr = df["ATR"].values
    high  = df["High"].values;  low = df["Low"].values
    n     = len(df)

    target_buy  = np.full(n, np.nan)
    target_sell = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close[i]; atr_i = atr[i]
        tp_b  = entry + atr_i * tp_mult; sl_b = entry - atr_i * sl_mult
        tp_s  = entry - atr_i * tp_mult; sl_s = entry + atr_i * sl_mult
        rb = rs = np.nan

        for j in range(i+1, min(i+horizon+1, n)):
            h = high[j]; l = low[j]
            if np.isnan(rb):
                if h >= tp_b and l <= sl_b:
                    rb = 1 if close[j-1] < entry + atr_i*0.5 else 0
                elif h >= tp_b: rb = 1
                elif l <= sl_b: rb = 0
            if np.isnan(rs):
                if l <= tp_s and h >= sl_s:
                    rs = 1 if close[j-1] > entry - atr_i*0.5 else 0
                elif l <= tp_s: rs = 1
                elif h >= sl_s: rs = 0
            if not np.isnan(rb) and not np.isnan(rs):
                break

        target_buy[i]  = rb
        target_sell[i] = rs

    df = df.copy()
    df["Target_BUY"]  = target_buy
    df["Target_SELL"] = target_sell

    total   = n - horizon
    bp = int(np.nansum(target_buy[:total]));  bv = int(np.sum(~np.isnan(target_buy[:total])))
    sp = int(np.nansum(target_sell[:total])); sv = int(np.sum(~np.isnan(target_sell[:total])))
    logger.info(
        f"[Trainer] Triple Barrier: BUY pos={bp}/{bv} ({bp/(bv+1e-9):.1%}) | "
        f"SELL pos={sp}/{sv} ({sp/(sv+1e-9):.1%})"
    )
    return df


# ─────────────────────────────────────────────
# ML утилиты (SMOTE, Optuna, XGB, LGBM, Calib, Stack, Meta, WF, Kelly)
# ─────────────────────────────────────────────
def apply_smote(X, y):
    if not SMOTE_AVAILABLE: return X, y
    if y.sum() / (len(y) + 1e-9) > 0.4: return X, y
    try:
        sm = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)
        return sm.fit_resample(X, y)
    except Exception as e:
        logger.warning(f"SMOTE: {e}"); return X, y


def tune_xgboost(X_train, y_train, n_trials=50) -> dict:
    tscv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 150, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample":        trial.suggest_float("subsample", 0.55, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 25),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.7),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 4.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
        }
        scores = []
        for tr_i, val_i in tscv.split(X_train, y_train):
            Xtr, Xv = X_train[tr_i], X_train[val_i]
            ytr, yv = y_train[tr_i], y_train[val_i]
            if ytr.sum() < 5 or yv.sum() < 2: continue
            m = XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False, verbosity=0)
            m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            scores.append(precision_score(yv, m.predict(Xv), zero_division=0))
        return float(np.mean(scores)) if scores else 0.0
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[Trainer] Optuna best: {study.best_value:.3f}")
    return study.best_params


def train_xgb(X_tr, y_tr, X_te, y_te, params=None) -> tuple:
    if params is None:
        params = {"n_estimators":400,"max_depth":4,"learning_rate":0.03,
                  "subsample":0.75,"colsample_bytree":0.70,"min_child_weight":10,
                  "gamma":0.2,"reg_alpha":0.3,"reg_lambda":2.0,"scale_pos_weight":2.0}
    m = XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    yp = m.predict(X_te); ypr = m.predict_proba(X_te)[:,1]
    return m, {
        "precision": float(precision_score(y_te, yp, zero_division=0)),
        "recall":    float(recall_score(y_te, yp, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_te, ypr)) if y_te.sum() > 0 else 0.0,
    }


def train_lgbm(X_tr, y_tr, X_te, y_te) -> tuple:
    if not LGBM_AVAILABLE: return None, None
    scale = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-9)
    m = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                             subsample=0.75, colsample_bytree=0.70, min_child_samples=20,
                             reg_alpha=0.3, reg_lambda=2.0, scale_pos_weight=min(scale,5.0),
                             objective="binary", verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    yp = m.predict(X_te); ypr = m.predict_proba(X_te)[:,1]
    return m, {
        "precision": float(precision_score(y_te, yp, zero_division=0)),
        "recall":    float(recall_score(y_te, yp, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_te, ypr)) if y_te.sum() > 0 else 0.0,
    }


def calibrate_model(model, X_tr, y_tr, X_te, y_te, label="BUY") -> tuple:
    try:
        tscv = TimeSeriesSplit(n_splits=3)
        cal  = CalibratedClassifierCV(estimator=model, method="isotonic", cv=tscv)
        cal.fit(X_tr, y_tr)
        ypr_raw = model.predict_proba(X_te)[:,1]
        ypr_cal = cal.predict_proba(X_te)[:,1]
        yp_cal  = cal.predict(X_te)
        raw_auc = float(roc_auc_score(y_te, ypr_raw)) if y_te.sum() > 0 else 0.0
        cal_auc = float(roc_auc_score(y_te, ypr_cal)) if y_te.sum() > 0 else 0.0
        logger.info(f"[Trainer] Calib {label}: AUC {raw_auc:.3f}→{cal_auc:.3f}")
        return cal, {"precision": float(precision_score(y_te, yp_cal, zero_division=0)),
                     "roc_auc": cal_auc}
    except Exception as e:
        logger.warning(f"[Trainer] Calib {label}: {e}")
        return model, {}


def train_stacking(m_xgb, m_lgbm, X_tr, y_tr, X_te, y_te, label="BUY") -> tuple:
    if not LGBM_AVAILABLE or m_lgbm is None: return None, None
    try:
        from sklearn.model_selection import StratifiedKFold
        p_xgb = m_xgb.predict_proba(X_te)[:,1].reshape(-1,1)
        p_lgb = m_lgbm.predict_proba(X_te)[:,1].reshape(-1,1)
        X_st  = np.hstack([p_xgb, p_lgb, (p_xgb+p_lgb)/2, p_xgb-p_lgb])

        skf = StratifiedKFold(n_splits=5, shuffle=False)
        oof_xgb = np.zeros(len(X_tr)); oof_lgb = np.zeros(len(X_tr))
        for tri, vli in skf.split(X_tr, y_tr):
            if y_tr[tri].sum() < 5: continue
            mx = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                eval_metric="logloss", use_label_encoder=False, verbosity=0)
            mx.fit(X_tr[tri], y_tr[tri], verbose=False)
            oof_xgb[vli] = mx.predict_proba(X_tr[vli])[:,1]
            if LGBM_AVAILABLE:
                ml = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         verbosity=-1, n_jobs=-1)
                ml.fit(X_tr[tri], y_tr[tri])
                oof_lgb[vli] = ml.predict_proba(X_tr[vli])[:,1]

        X_str = np.column_stack([oof_xgb, oof_lgb, (oof_xgb+oof_lgb)/2, oof_xgb-oof_lgb])
        scaler = StandardScaler()
        sm = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        sm.fit(scaler.fit_transform(X_str), y_tr)
        yp = sm.predict(scaler.transform(X_st))
        ypr = sm.predict_proba(scaler.transform(X_st))[:,1]
        logger.info(f"[Trainer] Stack {label}: prec={precision_score(y_te, yp, zero_division=0):.1%}")
        return {"model": sm, "scaler": scaler}, {
            "precision": float(precision_score(y_te, yp, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_te, ypr)) if y_te.sum() > 0 else 0.0,
        }
    except Exception as e:
        logger.warning(f"[Trainer] Stack {label}: {e}"); return None, None


def walk_forward(X, y, train_size, test_size, step) -> dict:
    results = []; wf_returns = []; n = len(X); start = train_size
    while start + test_size <= n:
        Xtr = X[start-train_size:start]; ytr = y[start-train_size:start]
        Xte = X[start:start+test_size];  yte = y[start:start+test_size]
        if ytr.sum() < 5 or yte.sum() < 2: start += step; continue
        try:
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               eval_metric="logloss", use_label_encoder=False, verbosity=0)
            m.fit(Xtr, ytr, verbose=False)
            yp = m.predict(Xte)
            results.append(float(precision_score(yte, yp, zero_division=0)))
            for pred, actual in zip(yp, yte):
                if pred == 1:
                    ret = (ATR_TP_MULT*1.5 if actual==1 else -ATR_SL_MULT*1.5) * 0.01
                    wf_returns.append(ret)
        except Exception: pass
        start += step

    wf_prec   = float(np.mean(results)) if results else 0.0
    wf_sharpe = 0.0
    if len(wf_returns) >= 3:
        arr = np.array(wf_returns)
        wf_sharpe = float((arr.mean() / (arr.std() + 1e-9)) * np.sqrt(2268/6))
    return {"wf_precision": wf_prec, "wf_sharpe": wf_sharpe,
            "wf_folds": len(results), "wf_trade_returns": wf_returns}


def calc_kelly(wf_returns: list) -> float:
    if len(wf_returns) < 10: return 0.05
    arr  = np.array(wf_returns)
    wins = arr[arr > 0]; loss = arr[arr < 0]
    if len(wins) == 0 or len(loss) == 0: return 0.05
    wr = len(wins) / len(arr)
    b  = float(wins.mean()) / float(abs(loss.mean()) + 1e-9)
    q  = 1 - wr
    f  = (wr * b - q) / (b + 1e-9)
    return float(max(0.01, min(f * 0.5, 0.25)))


def prune_features(m_xgb, m_lgbm, cols: list, threshold=FEATURE_IMPORTANCE_THRESHOLD) -> list:
    n = len(cols); imp = np.zeros(n)
    for m in [m_xgb, m_lgbm]:
        if m is None: continue
        try:
            fi = m.feature_importances_
            if len(fi) == n: imp += fi / (fi.sum() + 1e-9)
        except Exception: pass
    total = imp.sum()
    if total == 0: return cols
    imp /= total
    kept = [cols[i] for i in range(n) if imp[i] >= threshold]
    if len(kept) < 15: kept = [cols[i] for i in np.argsort(imp)[::-1][:20]]
    logger.info(f"[Trainer] Pruning: {n}→{len(kept)} признаков")
    return kept


# ─────────────────────────────────────────────
# ОБУЧЕНИЕ ОДНОГО ТИКЕРА
# ─────────────────────────────────────────────
def train_ticker(ticker: str) -> dict:
    """Полный цикл обучения для одного инструмента."""
    logger.info(f"\n{'='*50}\n[Trainer] 🚀 Обучение: {ticker}\n{'='*50}")

    # 1. Данные
    df_raw = fetch_ohlcv(ticker, BARS_1H)
    if df_raw.empty or len(df_raw) < 300:
        return {"success": False, "ticker": ticker, "error": "Мало данных"}

    # 2. Индикаторы
    df4h_raw = resample_to_4h(df_raw)
    df1h     = calc_indicators_1h(df_raw)
    df4h     = calc_indicators_4h(df4h_raw) if len(df4h_raw) >= 50 else pd.DataFrame()

    if len(df1h) < 200:
        return {"success": False, "ticker": ticker, "error": "Мало баров после индикаторов"}

    # 3. Слияние
    if not df4h.empty:
        df = merge_timeframes(df1h, df4h)
        if len(df) < 150: df = df1h
    else:
        df = df1h

    # 4. Разметка
    df = triple_barrier_labels(df)

    # 5. Признаки
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(feat_cols) < 10:
        feat_cols = [c for c in FEATURE_COLS_LEGACY if c in df.columns]
    logger.info(f"[Trainer] {ticker}: {len(feat_cols)} признаков")

    # 6. Выборки
    df_b = df.dropna(subset=["Target_BUY"]  + feat_cols)
    df_s = df.dropna(subset=["Target_SELL"] + feat_cols)
    if len(df_b) < 100 or len(df_s) < 100:
        return {"success": False, "ticker": ticker,
                "error": f"Мало примеров: BUY={len(df_b)} SELL={len(df_s)}"}

    Xb = np.nan_to_num(df_b[feat_cols].values.astype(float))
    yb = df_b["Target_BUY"].values.astype(int)
    Xs = np.nan_to_num(df_s[feat_cols].values.astype(float))
    ys = df_s["Target_SELL"].values.astype(int)

    # 7. Split 80/20
    sp_b = int(len(Xb)*0.80); sp_s = int(len(Xs)*0.80)
    Xb_tr, Xb_te = Xb[:sp_b], Xb[sp_b:]
    yb_tr, yb_te = yb[:sp_b], yb[sp_b:]
    Xs_tr, Xs_te = Xs[:sp_s], Xs[sp_s:]
    ys_tr, ys_te = ys[:sp_s], ys[sp_s:]

    if yb_te.sum() < 3 or ys_te.sum() < 3:
        return {"success": False, "ticker": ticker, "error": "Мало позитивов в тест-сете"}

    # 8. SMOTE
    Xb_sm, yb_sm = apply_smote(Xb_tr, yb_tr)
    Xs_sm, ys_sm = apply_smote(Xs_tr, ys_tr)

    # 9. Optuna
    logger.info(f"[Trainer] {ticker}: Optuna 50 trials...")
    best_params = tune_xgboost(Xb_sm, yb_sm, n_trials=50)

    # 10. Обучение XGB + LGBM
    b_xgb, b_xgb_m  = train_xgb(Xb_sm, yb_sm, Xb_te, yb_te, best_params)
    b_lgbm,b_lgbm_m = train_lgbm(Xb_sm, yb_sm, Xb_te, yb_te)
    s_xgb, s_xgb_m  = train_xgb(Xs_sm, ys_sm, Xs_te, ys_te, best_params)
    s_lgbm,s_lgbm_m = train_lgbm(Xs_sm, ys_sm, Xs_te, ys_te)

    # 11. Pruning
    feat_p = prune_features(b_xgb, b_lgbm, feat_cols)
    if len(feat_p) < len(feat_cols):
        idx = [feat_cols.index(c) for c in feat_p]
        Xb_tr2, Xb_te2 = Xb_tr[:,idx], Xb_te[:,idx]
        Xs_tr2, Xs_te2 = Xs_tr[:,idx], Xs_te[:,idx]
        Xb_sm2, yb_sm2 = apply_smote(Xb_tr2, yb_tr)
        Xs_sm2, ys_sm2 = apply_smote(Xs_tr2, ys_tr)
        bx2, bxm2  = train_xgb(Xb_sm2, yb_sm2, Xb_te2, yb_te, best_params)
        bl2, blm2  = train_lgbm(Xb_sm2, yb_sm2, Xb_te2, yb_te)
        sx2, sxm2  = train_xgb(Xs_sm2, ys_sm2, Xs_te2, ys_te, best_params)
        sl2, slm2  = train_lgbm(Xs_sm2, ys_sm2, Xs_te2, ys_te)
        if bxm2["precision"] >= b_xgb_m["precision"]:
            b_xgb, b_xgb_m, b_lgbm, b_lgbm_m = bx2, bxm2, bl2, blm2
            s_xgb, s_xgb_m, s_lgbm, s_lgbm_m = sx2, sxm2, sl2, slm2
            feat_cols = feat_p
            Xb_tr, Xb_te = Xb_tr2, Xb_te2
            Xs_tr, Xs_te = Xs_tr2, Xs_te2
            Xb_sm, yb_sm = Xb_sm2, yb_sm2
            Xs_sm, ys_sm = Xs_sm2, ys_sm2
            logger.info(f"[Trainer] {ticker}: Pruned лучше → {len(feat_cols)} признаков")

    # 12. Stacking
    stack_b, stack_bm = train_stacking(b_xgb, b_lgbm, Xb_tr, yb_tr, Xb_te, yb_te, "BUY")
    stack_s, stack_sm = train_stacking(s_xgb, s_lgbm, Xs_tr, ys_tr, Xs_te, ys_te, "SELL")

    # 13. Калибровка
    calib_b, calib_bm = calibrate_model(b_xgb, Xb_tr, yb_tr, Xb_te, yb_te, "BUY")
    calib_s, calib_sm = calibrate_model(s_xgb, Xs_tr, ys_tr, Xs_te, ys_te, "SELL")

    # 14. Walk-Forward
    wf_b = walk_forward(Xb, yb, max(int(len(Xb)*0.55),100), max(int(len(Xb)*0.12),30), max(int(len(Xb)*0.08),20))
    wf_s = walk_forward(Xs, ys, max(int(len(Xs)*0.55),100), max(int(len(Xs)*0.12),30), max(int(len(Xs)*0.08),20))
    logger.info(f"[Trainer] {ticker} WF: BUY prec={wf_b['wf_precision']:.1%} sharpe={wf_b['wf_sharpe']:.2f}")
    logger.info(f"[Trainer] {ticker} WF: SELL prec={wf_s['wf_precision']:.1%} sharpe={wf_s['wf_sharpe']:.2f}")

    # 15. Kelly
    all_ret = wf_b.get("wf_trade_returns",[]) + wf_s.get("wf_trade_returns",[])
    kelly_f = calc_kelly(all_ret)
    logger.info(f"[Trainer] {ticker} Kelly: {kelly_f:.1%}")

    # 16. Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(b_xgb,  model_path(ticker, "buy_xgb"))
    joblib.dump(s_xgb,  model_path(ticker, "sell_xgb"))
    if b_lgbm:  joblib.dump(b_lgbm,  model_path(ticker, "buy_lgbm"))
    if s_lgbm:  joblib.dump(s_lgbm,  model_path(ticker, "sell_lgbm"))
    if stack_b: joblib.dump(stack_b, model_path(ticker, "stack_buy"))
    if stack_s: joblib.dump(stack_s, model_path(ticker, "stack_sell"))
    if calib_b: joblib.dump(calib_b, model_path(ticker, "calib_buy"))
    if calib_s: joblib.dump(calib_s, model_path(ticker, "calib_sell"))

    with open(features_path(ticker), "w") as f:
        json.dump(feat_cols, f)

    avg_bp = (b_xgb_m["precision"] + (b_lgbm_m["precision"] if b_lgbm_m else b_xgb_m["precision"])) / 2
    avg_sp = (s_xgb_m["precision"] + (s_lgbm_m["precision"] if s_lgbm_m else s_xgb_m["precision"])) / 2

    result = {
        "success":            True,
        "ticker":             ticker,
        "version":            "moex_2.0",
        "labeling":           "triple_barrier",
        "n_features":         len(feat_cols),
        "n_samples_buy":      len(df_b),
        "n_samples_sell":     len(df_s),
        "kelly_fraction":     kelly_f,
        "avg_buy_precision":  avg_bp,
        "avg_buy_auc":        b_xgb_m["roc_auc"],
        "avg_sell_precision": avg_sp,
        "stack_buy_prec":     stack_bm["precision"] if stack_bm else None,
        "calib_buy_auc":      calib_bm.get("roc_auc") if calib_bm else None,
        "wf_buy_precision":   wf_b["wf_precision"],
        "wf_sell_precision":  wf_s["wf_precision"],
        "wf_buy_sharpe":      wf_b["wf_sharpe"],
        "wf_sell_sharpe":     wf_s["wf_sharpe"],
        "lgbm_available":     LGBM_AVAILABLE,
        "smote_available":    SMOTE_AVAILABLE,
        "stacking":           stack_b is not None,
        "calibration":        calib_b is not None,
    }

    with open(stats_path(ticker), "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"[Trainer] ✅ {ticker} готово! "
        f"BUY prec={avg_bp:.1%} | SELL prec={avg_sp:.1%} | "
        f"WF Sharpe={wf_b['wf_sharpe']:.2f} | Kelly={kelly_f:.1%}"
    )
    return result


# ─────────────────────────────────────────────
# ОБУЧЕНИЕ ВСЕХ ТИКЕРОВ
# ─────────────────────────────────────────────
def train_model(tickers: list = None) -> dict:
    """Обучает модели для всех инструментов последовательно."""
    if tickers is None:
        tickers = SYMBOLS_ALL

    all_results = {}
    for ticker in tickers:
        try:
            result = train_ticker(ticker)
            all_results[ticker] = result
            time.sleep(2)  # пауза между тикерами — уважаем ISS API
        except Exception as e:
            logger.error(f"[Trainer] {ticker} критическая ошибка: {e}", exc_info=True)
            all_results[ticker] = {"success": False, "ticker": ticker, "error": str(e)}

    # Сводная статистика
    successful = [r for r in all_results.values() if r.get("success")]
    summary = {
        "success":       len(successful) > 0,
        "tickers_total": len(tickers),
        "tickers_ok":    len(successful),
        "tickers_fail":  len(tickers) - len(successful),
        "results":       all_results,
    }
    if successful:
        summary["avg_buy_precision"]  = round(np.mean([r["avg_buy_precision"] for r in successful]), 4)
        summary["avg_wf_sharpe_buy"]  = round(np.mean([r["wf_buy_sharpe"] for r in successful]), 3)
        summary["avg_kelly"]          = round(np.mean([r["kelly_fraction"] for r in successful]), 4)

    with open(STATS_FILE, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "results"}, f, indent=2)

    logger.info(
        f"\n[Trainer] ✅ Обучение завершено: {len(successful)}/{len(tickers)} тикеров | "
        f"avg BUY prec={summary.get('avg_buy_precision',0):.1%} | "
        f"avg WF Sharpe={summary.get('avg_wf_sharpe_buy',0):.2f}"
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    result = train_model()
    print(f"\n✅ Готово: {result['tickers_ok']}/{result['tickers_total']} тикеров")
    print(f"   Avg BUY precision: {result.get('avg_buy_precision',0):.1%}")
    print(f"   Avg WF Sharpe:     {result.get('avg_wf_sharpe_buy',0):.2f}")
    print(f"   Avg Kelly:         {result.get('avg_kelly',0):.1%}")
