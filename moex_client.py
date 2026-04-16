"""
moex_client.py — Клиент ISS MOEX REST API v2.0
Публичный API — авторизация не нужна.
https://iss.moex.com/iss/reference/

Поддерживает: акции (TQBR), ОФЗ (TQOB), индекс IMOEX, дивидендный календарь.
"""

import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import (
    SYMBOL, TIMEFRAME, MOEX_TZ, SYMBOL_BOARD,
    MOEX_SESSION_START, MOEX_SESSION_END,
    MOEX_EVENING_SESSION, MOEX_EVENING_START, MOEX_EVENING_END,
)

logger  = logging.getLogger(__name__)
ISS_BASE = "https://iss.moex.com/iss"
MSK_TZ   = ZoneInfo(MOEX_TZ)

TIMEFRAME_MAP = {"1": 1, "10": 10, "60": 60, "1440": 24, "D": 24, "1H": 60, "4H": 60}


def _get(url: str, params: dict = None, retries: int = 4) -> list:
    """GET запрос к ISS с retry + exponential backoff."""
    if params is None:
        params = {}
    params.setdefault("iss.meta", "off")
    params.setdefault("iss.json", "extended")

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code in (429, 503):
                wait = 2 ** attempt * 5
                logger.warning(f"[MOEX] Rate limit {r.status_code} — ждём {wait}с")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                logger.error(f"[MOEX] HTTP {r.status_code}: {url}")
                return []
            return r.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[MOEX] Timeout (попытка {attempt+1})")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"[MOEX] Ошибка: {e}")
            time.sleep(2 ** attempt)

    logger.error(f"[MOEX] Все {retries} попытки неудачны: {url}")
    return []


def _parse_candles(data) -> list:
    """Парсинг ISS extended JSON → список [ts, open, high, low, close, volume]."""
    try:
        for block in data:
            if isinstance(block, dict) and "candles" in block:
                result = []
                for c in block["candles"]:
                    result.append([
                        c.get("begin"), c.get("open"), c.get("high"),
                        c.get("low"),  c.get("close"), c.get("volume"),
                    ])
                return result
    except Exception as e:
        logger.error(f"[MOEX] _parse_candles: {e}")
    return []


def get_candles_multi(
    ticker:   str = None,
    interval: str = None,
    total:    int = 2000,
    board:    str = None,
    market:   str = None,
) -> list:
    """
    Загружает total свечей через несколько запросов.
    Автоматически определяет board/market из SYMBOL_BOARD если не указан.
    """
    if ticker is None:
        ticker = SYMBOL
    if interval is None:
        interval = TIMEFRAME

    if board is None or market is None:
        b, m = SYMBOL_BOARD.get(ticker, ("TQBR", "shares"))
        board  = board  or b
        market = market or m

    interval_code = TIMEFRAME_MAP.get(str(interval), 60)
    hours_per_day = 9 if interval_code == 60 else max(1, 540 / interval_code)
    days_needed   = int(total / hours_per_day) + 15

    date_till = datetime.now(MSK_TZ)
    date_from = date_till - timedelta(days=days_needed)

    url       = f"{ISS_BASE}/engines/stock/markets/{market}/boards/{board}/securities/{ticker}/candles.json"
    all_data  = []
    start_idx = 0
    max_req   = (total // 500) + 5

    for _ in range(max_req):
        params = {
            "interval":  interval_code,
            "limit":     500,
            "start":     start_idx,
            "from":      date_from.strftime("%Y-%m-%d"),
            "till":      date_till.strftime("%Y-%m-%d %H:%M:%S"),
            "iss.meta":  "off",
            "iss.json":  "extended",
        }
        data  = _get(url, params)
        batch = _parse_candles(data)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        start_idx += len(batch)
        time.sleep(0.4)

    logger.info(f"[MOEX] {ticker}: загружено {len(all_data)} свечей ({interval_code}мин)")
    return all_data[:total]


def candles_to_df(data: list) -> pd.DataFrame:
    """Raw ISS список → DataFrame с UTC-индексом."""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df = df.dropna(subset=["Close"])
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(MSK_TZ).dt.tz_convert("UTC")
    df.set_index("ts", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[df["Close"] > 0]
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Ресэмплирует 1H DataFrame в 4H (MOEX не отдаёт 4H напрямую)."""
    if df_1h.empty:
        return pd.DataFrame()
    return df_1h.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def get_ticker(ticker: str = None, board: str = None, market: str = None) -> dict:
    """Текущая цена инструмента."""
    if ticker is None:
        ticker = SYMBOL
    if board is None or market is None:
        b, m = SYMBOL_BOARD.get(ticker, ("TQBR", "shares"))
        board  = board  or b
        market = market or m

    url  = f"{ISS_BASE}/engines/stock/markets/{market}/boards/{board}/securities/{ticker}.json"
    data = _get(url, {"securities.columns": "SECID,PREVPRICE", "marketdata.columns": "LAST,OPEN,HIGH,LOW,VOLTODAY,LASTTOPREVPRICE"})
    try:
        for block in data:
            if isinstance(block, dict) and "marketdata" in block:
                md = block["marketdata"]
                if md:
                    row = md[0]
                    return {
                        "ticker":     ticker,
                        "last":       float(row.get("LAST") or 0),
                        "open":       float(row.get("OPEN") or 0),
                        "high":       float(row.get("HIGH") or 0),
                        "low":        float(row.get("LOW") or 0),
                        "volume":     float(row.get("VOLTODAY") or 0),
                        "change_pct": float(row.get("LASTTOPREVPRICE") or 0),
                    }
    except Exception as e:
        logger.error(f"[MOEX] get_ticker {ticker}: {e}")
    return {}


def get_current_price(ticker: str = None, board: str = None, market: str = None) -> float:
    """Быстрое получение последней цены."""
    info = get_ticker(ticker, board, market)
    price = info.get("last", 0.0)
    if price == 0:
        # Fallback: берём из последней свечи
        raw = get_candles_multi(ticker, interval="60", total=2, board=board, market=market)
        if raw:
            df = candles_to_df(raw)
            if not df.empty:
                price = float(df["Close"].iloc[-1])
    return price


def get_imoex(limit: int = 20) -> pd.DataFrame:
    """Данные индекса IMOEX для макро-фильтра."""
    try:
        url  = f"{ISS_BASE}/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json"
        data = _get(url, {"interval": 60, "limit": limit})
        raw  = _parse_candles(data)
        if raw:
            return candles_to_df(raw)
    except Exception as e:
        logger.error(f"[MOEX] get_imoex: {e}")
    return pd.DataFrame()


def get_dividend_calendar() -> dict:
    """
    Загружает даты ближайших дивидендных отсечек с MOEX ISS.
    Возвращает dict: {ticker: [date1, date2, ...]}
    Используется для блокировки продаж перед отсечкой.
    """
    calendar = {}
    try:
        url  = f"{ISS_BASE}/statistics/engines/stock/markets/shares/dividends.json"
        data = _get(url, {"limit": 100})
        for block in data:
            if isinstance(block, dict) and "dividends" in block:
                for row in block["dividends"]:
                    ticker    = row.get("secid", "")
                    registry_date = row.get("registryclosedate", "")
                    if ticker and registry_date:
                        calendar.setdefault(ticker, []).append(registry_date)
    except Exception as e:
        logger.warning(f"[MOEX] dividend_calendar: {e}")
    return calendar


# ═══════════════════════════════════════════
# ФИЛЬТР ТОРГОВОЙ СЕССИИ
# ═══════════════════════════════════════════

def is_trading_session() -> bool:
    """True если сейчас идёт торговая сессия MOEX."""
    now = datetime.now(MSK_TZ)
    if now.weekday() >= 5:
        return False

    def _t(ts: str):
        h, m = map(int, ts.split(":"))
        return now.replace(hour=h, minute=m, second=0, microsecond=0)

    if _t(MOEX_SESSION_START) <= now <= _t(MOEX_SESSION_END):
        return True
    if MOEX_EVENING_SESSION and _t(MOEX_EVENING_START) <= now <= _t(MOEX_EVENING_END):
        return True
    return False


def get_minutes_to_session() -> int:
    """Минут до следующего открытия сессии."""
    now = datetime.now(MSK_TZ)
    days_ahead = {5: 2, 6: 1}.get(now.weekday(), 0)
    target     = (now + timedelta(days=days_ahead)).date() if days_ahead else now.date()
    h, m       = map(int, MOEX_SESSION_START.split(":"))
    open_time  = datetime(target.year, target.month, target.day, h, m, tzinfo=MSK_TZ)
    if open_time > now:
        return int((open_time - now).total_seconds() / 60)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print(f"Торговая сессия: {is_trading_session()}")
    for tkr in ["SBERP", "LKOH", "MOEX"]:
        p = get_current_price(tkr)
        print(f"{tkr}: {p:.2f} ₽")
