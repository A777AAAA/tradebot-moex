"""
config.py — TradeBot MOEX v2.0
Стратегия: Buy the Dip + Sell the Rally + Дивиденды/Купоны
Архитектура: отдельная ML-модель на каждый инструмент (7 моделей)

Инструменты:
  Акции: SBERP, TATNP, TRNFP, LKOH, FIVE, MOEX
  ОФЗ:   SU26238RMFS4

Логика:
  - BUY:  покупаем на просадках/дне, усредняем (до 3 входов на бумагу)
  - SELL: продаём частями при +4/7/12% и сигнале разворота вниз
  - Держим для дивидендов/купонов — не продаём за 7 дней до отсечки
"""

import os

# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════
# TINKOFF INVEST API (реальная торговля — будущее)
# ═══════════════════════════════════════════
TINKOFF_TOKEN      = os.getenv("TINKOFF_TOKEN", "")
TINKOFF_ACCOUNT_ID = os.getenv("TINKOFF_ACCOUNT_ID", "")

# ═══════════════════════════════════════════
# ИНСТРУМЕНТЫ MOEX
# ═══════════════════════════════════════════
SYMBOLS_STOCKS = [
    "SBERP",   # Сбербанк ап
    "TATNP",   # Татнефть ап
    "TRNFP",   # Транснефть ап
    "LKOH",    # Лукойл
    'X5',      # X5 RetailGroup
    "MOEX",    # Московская биржа
]

SYMBOLS_OFZ = [
    "SU26238RMFS4",   # ОФЗ 26238 — длинная (купон 7.1%)
]

SYMBOLS_ALL = SYMBOLS_STOCKS + SYMBOLS_OFZ

# Главный символ (legacy совместимость)
SYMBOL = "SBERP"

# board/market для ISS API
SYMBOL_BOARD = {
    "SBERP":          ("TQBR", "shares"),
    "TATNP":          ("TQBR", "shares"),
    "TRNFP":          ("TQBR", "shares"),
    "LKOH":           ("TQBR", "shares"),
    'X5':             ("TQBR", "shares"),
    "MOEX":           ("TQBR", "shares"),
    "SU26238RMFS4":   ("TQOB", "bonds"),
}

# Названия для Telegram
SYMBOL_NAMES = {
    "SBERP":          "Сбербанк ап",
    "TATNP":          "Татнефть ап",
    "TRNFP":          "Транснефть ап",
    "LKOH":           "Лукойл",
    'X5':             "X5 RetailGroup",
    "MOEX":           "Мосбиржа",
    "SU26238RMFS4":   "ОФЗ 26238",
}

# Минимальный лот на MOEX
SYMBOL_LOT = {
    "SBERP":          10,
    "TATNP":          1,
    "TRNFP":          1,
    "LKOH":           1,
    'X5':             1,
    "MOEX":           10,
    "SU26238RMFS4":   1,
}

TIMEFRAME    = "60"   # 1H в минутах для ISS API
TIMEFRAME_4H = "60"   # 4H получаем ресэмплингом из 1H

# ═══════════════════════════════════════════
# ТОРГОВАЯ СЕССИЯ MOEX
# ═══════════════════════════════════════════
MOEX_TZ              = "Europe/Moscow"
MOEX_SESSION_START   = "10:00"
MOEX_SESSION_END     = "18:50"
MOEX_EVENING_SESSION = False
MOEX_EVENING_START   = "19:05"
MOEX_EVENING_END     = "23:50"

# ═══════════════════════════════════════════
# СТРАТЕГИЯ: BUY THE DIP + SELL THE RALLY
# ═══════════════════════════════════════════

# --- Усреднение позиции (несколько входов в одну бумагу) ---
MAX_ENTRIES_PER_SYMBOL   = 3      # максимум 3 докупки одной бумаги
AVERAGING_DIP_PCT        = 0.030  # докупаем если цена упала ещё на 3% от последней покупки
AVERAGING_CONFIDENCE_MIN = 0.50   # минимальная уверенность для докупки

# --- Частичные продажи (Sell the Rally) ---
# Уровни фиксации прибыли от средней цены входа
SELL_LEVELS = [
    {"pct": 0.04, "sell_ratio": 0.30},   # +4%  → продаём 30% позиции
    {"pct": 0.07, "sell_ratio": 0.30},   # +7%  → продаём ещё 30%
    {"pct": 0.12, "sell_ratio": 0.30},   # +12% → продаём ещё 30%
    # оставшиеся ~10% держим вечно (дивиденды)
]
SELL_KEEP_FOREVER_RATIO = 0.10   # 10% позиции никогда не продаём через бота
SELL_REQUIRE_SIGNAL     = True   # продаём только при SELL-сигнале модели

# --- Дивидендный/купонный фильтр ---
DIVIDEND_PROTECTION_DAYS = 7     # за N дней до отсечки — блокируем продажи
DIVIDEND_CALENDAR_FILE   = "dividend_calendar.json"

# ═══════════════════════════════════════════
# РИСК-МЕНЕДЖМЕНТ
# ═══════════════════════════════════════════
STOP_LOSS_PCT   = 0.025   # 2.5% (шире — долгосрочная стратегия)
TAKE_PROFIT_PCT = 0.050   # 5.0% минимальный TP

ATR_SL_MULT  = 2.0
ATR_TP_MULT  = 4.0
SL_FLOOR_PCT = 0.015
SL_CAP_PCT   = 0.060

TRAILING_ENABLED        = True
TRAILING_ACTIVATION_PCT = 0.020
TRAILING_DISTANCE_PCT   = 0.010
BREAKEVEN_ACTIVATION    = 0.015

# Максимальная экспозиция на один тикер (% от баланса)
MAX_EXPOSURE_PER_SYMBOL = 0.30   # не более 30% баланса в одной бумаге
MAX_TOTAL_EXPOSURE      = 0.80   # не более 80% суммарно

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════
MIN_CONFIDENCE          = 0.54
STRONG_SIGNAL           = 0.67
SIGNAL_INTERVAL_MINUTES = 60

CONFIDENCE_PERCENTILE   = 50

MTF_ENABLED              = True
IMOEX_FILTER_ENABLED     = True
IMOEX_CHANGE_THRESH      = -0.015   # блок BUY если IMOEX < -1.5% за 1H

REGIME_FILTER_ENABLED    = True
REGIME_ADX_THRESHOLD     = 15.0

# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ
# ═══════════════════════════════════════════
RETRAIN_INTERVAL_HRS = 24
RETRAIN_DAY          = os.getenv("RETRAIN_DAY", "sunday")
RETRAIN_HOUR         = int(os.getenv("RETRAIN_HOUR", "3"))
MIN_NEW_SAMPLES      = 50

WF_TRAIN_DAYS = 90
WF_TEST_DAYS  = 14
WF_STEP_DAYS  = 7

# ═══════════════════════════════════════════
# ML — ОТДЕЛЬНАЯ МОДЕЛЬ НА КАЖДЫЙ ТИКЕР
# ═══════════════════════════════════════════
import os as _os

def _safe(ticker: str) -> str:
    return ticker.replace("/", "_").replace("-", "_")

def model_path(ticker: str, mtype: str) -> str:
    """Пример: models/SBERP_buy_xgb.pkl"""
    _os.makedirs("models", exist_ok=True)
    return f"models/{_safe(ticker)}_{mtype}.pkl"

def features_path(ticker: str) -> str:
    _os.makedirs("models", exist_ok=True)
    return f"models/{_safe(ticker)}_features.json"

def stats_path(ticker: str) -> str:
    _os.makedirs("models", exist_ok=True)
    return f"models/{_safe(ticker)}_stats.json"

# Legacy пути (для совместимости со старым кодом)
MODEL_PATH_BUY_XGB   = "models/SBERP_buy_xgb.pkl"
MODEL_PATH_BUY_LGBM  = "models/SBERP_buy_lgbm.pkl"
MODEL_PATH_SELL_XGB  = "models/SBERP_sell_xgb.pkl"
MODEL_PATH_SELL_LGBM = "models/SBERP_sell_lgbm.pkl"
MODEL_FEATURES_PATH  = "models/SBERP_features.json"
STATS_FILE           = "training_stats.json"

# ═══════════════════════════════════════════
# ПРИЗНАКИ
# ═══════════════════════════════════════════
FEATURE_COLS_1H = [
    'RSI_14', 'RSI_7', 'RSI_21',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ATR_norm', 'ATR_ratio',
    'ADX', 'BB_pos', 'BB_width',
    'EMA_ratio_20_50', 'EMA_ratio_20_100',
    'Vol_ratio', 'OBV_norm', 'MFI_14',
    'Body_pct', 'Upper_wick', 'Lower_wick', 'Doji',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'StochRSI_K', 'StochRSI_D',
    'ZScore_20', 'ZScore_50',
    'WilliamsR',
    'Hour', 'DayOfWeek',
    'Momentum_10', 'ROC_10',
    # Специфичные для Buy-the-Dip стратегии:
    'Drawdown_from_high_20',  # просадка от 20-дневного максимума (%)
    'Drawdown_from_high_60',  # просадка от 60-дневного максимума (%)
    'Days_since_high_20',     # баров с 20-дневного хая (чем больше — тем глубже дно)
    'Consec_down_bars',       # последовательных красных баров подряд
    'Vol_surge',              # объём / средний объём (всплеск = признак дна/разворота)
]

FEATURE_COLS_4H = [
    'RSI_14_4h', 'RSI_7_4h',
    'MACD_hist_4h',
    'EMA_ratio_4h',
    'ATR_pct_4h',
    'Vol_ratio_4h',
    'Return_4h_tf', 'Return_24h_tf',
    'ADX_4h',
    'BB_pos_4h',
]

FEATURE_COLS_V7 = [
    'Hurst',
    'Hurst_4h',
    'VWAP_dev_20',
    'VWAP_dev_50',
    'VWAP_bull_ratio',
    'RV_20',
    'RV_50',
    'RV_ratio',
    'OFI',
    'Price_accel',
    'Vol_cluster',
]

FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H + FEATURE_COLS_V7

FEATURE_COLS_LEGACY = [
    'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX', 'BB_pos', 'EMA_ratio_20_50', 'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'Hour'
]

# ═══════════════════════════════════════════
# ТАРГЕТ
# ═══════════════════════════════════════════
TARGET_HORIZON   = 6
TARGET_THRESHOLD = 0.010   # 1% движение вверх = цель для BUY

# ═══════════════════════════════════════════
# PAPER TRADING
# ═══════════════════════════════════════════
INITIAL_BALANCE = 300_000.0   # 300 000 ₽


def validate_config() -> list:
    required = {
        "TELEGRAM_TOKEN":   TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    return [key for key, val in required.items() if not val]
