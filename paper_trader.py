"""
paper_trader.py MOEX v2.0
Стратегия: Buy the Dip + Sell the Rally + Дивиденды/Купоны

Ключевые отличия от v1.0:
  - Несколько одновременных позиций (по одной на каждый тикер)
  - Усреднение: до 3 входов в одну бумагу (докупки на просадках)
  - Средняя цена входа отслеживается по позиции
  - Частичные продажи на уровнях +4% / +7% / +12%
  - Продажа только при SELL-сигнале модели (разворот вниз)
  - Дивидендный фильтр: блок продажи за 7 дней до отсечки
  - Экспозиционные лимиты: max 30% на тикер, max 80% суммарно
  - Валюта ₽, Kelly на каждую позицию независимо
"""

import json
import os
import logging
from datetime import datetime, timezone, date, timedelta

from config import (
    STOP_LOSS_PCT, ATR_SL_MULT, SL_FLOOR_PCT, SL_CAP_PCT,
    TRAILING_ENABLED, TRAILING_ACTIVATION_PCT,
    TRAILING_DISTANCE_PCT, BREAKEVEN_ACTIVATION,
    STRONG_SIGNAL, SYMBOL, SYMBOLS_ALL, SYMBOL_BOARD, SYMBOL_NAMES,
    MAX_ENTRIES_PER_SYMBOL, AVERAGING_DIP_PCT, AVERAGING_CONFIDENCE_MIN,
    SELL_LEVELS, SELL_KEEP_FOREVER_RATIO, SELL_REQUIRE_SIGNAL,
    DIVIDEND_PROTECTION_DAYS, DIVIDEND_CALENDAR_FILE,
    MAX_EXPOSURE_PER_SYMBOL, MAX_TOTAL_EXPOSURE,
    INITIAL_BALANCE, stats_path,
)
from moex_client import get_current_price, is_trading_session

PORTFOLIO_FILE = "portfolio.json"    # все открытые позиции
BALANCE_FILE   = "paper_balance.json"
TRADES_FILE    = "paper_trades.json"  # история закрытых сделок

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_balance() -> dict:
    if not os.path.exists(BALANCE_FILE):
        data = {
            "balance":            INITIAL_BALANCE,
            "peak_balance":       INITIAL_BALANCE,
            "total_pnl":          0.0,
            "trades_closed":      0,
            "wins":               0,
            "losses":             0,
            "consecutive_losses": 0,
            "created_at":         _now(),
            "currency":           "RUB",
        }
        _save_json(BALANCE_FILE, data)
        return data
    data = _load_json(BALANCE_FILE)
    data.setdefault("consecutive_losses", 0)
    data.setdefault("peak_balance", max(data.get("balance", INITIAL_BALANCE), INITIAL_BALANCE))
    return data


def load_portfolio() -> dict:
    """
    Возвращает dict {ticker: position_dict}.
    position_dict содержит все активные входы (entries) по этому тикеру.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    return _load_json(PORTFOLIO_FILE)


def load_trades() -> list:
    if not os.path.exists(TRADES_FILE):
        return []
    return _load_json(TRADES_FILE)


def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_balance(data: dict):
    _save_json(BALANCE_FILE, data)


def save_portfolio(data: dict):
    _save_json(PORTFOLIO_FILE, data)


def save_trades(data: list):
    _save_json(TRADES_FILE, data)


# ═══════════════════════════════════════════
# ДИВИДЕНДНЫЙ ФИЛЬТР
# ═══════════════════════════════════════════

def _load_dividend_calendar() -> dict:
    """
    Загружает локальный календарь дат отсечек.
    Формат файла: {"SBERP": ["2025-05-20", "2024-12-15"], ...}
    Обновляйте вручную или запускайте moex_client.get_dividend_calendar().
    """
    if os.path.exists(DIVIDEND_CALENDAR_FILE):
        try:
            return _load_json(DIVIDEND_CALENDAR_FILE)
        except Exception:
            pass
    return {}


def is_dividend_protected(ticker: str) -> bool:
    """
    True если нельзя продавать (близко к дивидендной отсечке).
    Блокируем продажу за DIVIDEND_PROTECTION_DAYS дней до отсечки.
    """
    calendar = _load_dividend_calendar()
    dates    = calendar.get(ticker, [])
    today    = date.today()
    for d_str in dates:
        try:
            cutoff = date.fromisoformat(d_str)
            days_until = (cutoff - today).days
            if 0 <= days_until <= DIVIDEND_PROTECTION_DAYS:
                logger.info(f"[Paper] 💰 {ticker}: отсечка {d_str} через {days_until} дн. — продажа заблокирована")
                return True
        except Exception:
            pass
    return False


# ═══════════════════════════════════════════
# KELLY + ЭКСПОЗИЦИОННЫЙ КОНТРОЛЬ
# ═══════════════════════════════════════════

def get_kelly_pct(ticker: str, confidence: float, consecutive_losses: int = 0,
                   is_averaging: bool = False) -> float:
    """
    Читает Kelly fraction из файла статистики конкретного тикера.
    При усреднении — уменьшаем размер (докупки меньше первого входа).
    """
    kelly_f = 0.0
    try:
        spath = stats_path(ticker)
        if os.path.exists(spath):
            with open(spath) as f:
                stats = json.load(f)
            kelly_f = float(stats.get("kelly_fraction", 0.0))
    except Exception:
        pass

    if kelly_f < 0.03:
        kelly_f = 0.08  # дефолт если модель ещё не обучена

    # Масштаб по уверенности
    if confidence >= STRONG_SIGNAL:
        pct = kelly_f * 1.2
    elif confidence >= 0.60:
        pct = kelly_f * 1.0
    else:
        pct = kelly_f * 0.75

    # При усреднении (докупке) — берём 60% от обычного размера
    if is_averaging:
        pct *= 0.60

    # Штраф за серию убытков
    if consecutive_losses >= 2:
        penalty = max(0.40, 1.0 - consecutive_losses * 0.15)
        pct    *= penalty

    return round(max(0.03, min(pct, 0.25)), 3)


def check_exposure_limits(portfolio: dict, balance: float,
                           ticker: str, new_amount: float) -> tuple:
    """
    Проверяет лимиты экспозиции перед открытием позиции.
    Возвращает (can_open: bool, reason: str)
    """
    # Текущая экспозиция по тикеру
    pos = portfolio.get(ticker)
    ticker_exposure = sum(e["amount_rub"] for e in pos["entries"]) if pos else 0.0

    # Суммарная экспозиция портфеля
    total_exposure = sum(
        sum(e["amount_rub"] for e in p["entries"])
        for p in portfolio.values()
    )

    if ticker_exposure + new_amount > balance * MAX_EXPOSURE_PER_SYMBOL:
        return False, f"Лимит на {ticker}: {ticker_exposure:.0f}+{new_amount:.0f} > {balance*MAX_EXPOSURE_PER_SYMBOL:.0f} ₽"

    if total_exposure + new_amount > balance * MAX_TOTAL_EXPOSURE:
        return False, f"Общий лимит портфеля: {total_exposure:.0f}+{new_amount:.0f} > {balance*MAX_TOTAL_EXPOSURE:.0f} ₽"

    return True, ""


def check_drawdown_guard(balance_data: dict) -> tuple:
    balance      = balance_data.get("balance", INITIAL_BALANCE)
    peak_balance = balance_data.get("peak_balance", balance)
    if balance > peak_balance:
        balance_data["peak_balance"] = balance
        peak_balance = balance
    dd_pct    = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0.0
    is_blocked = dd_pct >= 20.0
    return is_blocked, round(dd_pct, 2)


# ═══════════════════════════════════════════
# ОТКРЫТИЕ ПОЗИЦИИ / УСРЕДНЕНИЕ
# ═══════════════════════════════════════════

def open_trade(signal: str, price: float, confidence: float,
               symbol: str = None, atr: float = 0.0,
               extra_info: dict = None) -> dict | None:
    """
    Открывает новую позицию или докупает (усредняет) существующую.

    Логика:
    - BUY: первый вход или усреднение (если цена упала на AVERAGING_DIP_PCT от последней покупки)
    - SELL: проверяем уровни продажи по открытым позициям (это обрабатывается в monitor_trades)
    """
    if symbol is None:
        symbol = SYMBOL

    if not is_trading_session():
        logger.info(f"[Paper] ⏰ Вне сессии MOEX — {symbol} пропускаем")
        return None

    if signal not in ("BUY",):
        # SELL обрабатывается в monitor_trades при проверке уровней
        return None

    balance_data = load_balance()
    portfolio    = load_portfolio()

    is_blocked, dd_pct = check_drawdown_guard(balance_data)
    if is_blocked:
        logger.warning(f"[Paper] 🚫 DrawDown Guard {dd_pct:.1f}% — блок открытия {symbol}")
        return None

    cl = balance_data.get("consecutive_losses", 0)

    # ── Проверяем есть ли уже позиция по этому тикеру ──
    pos          = portfolio.get(symbol)
    is_averaging = False

    if pos is not None:
        entries = pos.get("entries", [])
        n_entries = len(entries)

        if n_entries >= MAX_ENTRIES_PER_SYMBOL:
            logger.info(f"[Paper] {symbol}: уже {n_entries} входов (макс {MAX_ENTRIES_PER_SYMBOL})")
            return None

        # Проверяем условие усреднения: цена должна упасть на AVERAGING_DIP_PCT
        last_price = entries[-1]["price"]
        dip_needed = last_price * (1 - AVERAGING_DIP_PCT)
        if price > dip_needed:
            logger.info(
                f"[Paper] {symbol}: усреднение ещё не нужно. "
                f"Цена {price:.2f} > {dip_needed:.2f} (нужно -3% от {last_price:.2f})"
            )
            return None

        if confidence < AVERAGING_CONFIDENCE_MIN:
            logger.info(f"[Paper] {symbol}: уверенность {confidence:.1%} < мин {AVERAGING_CONFIDENCE_MIN:.1%} для усреднения")
            return None

        is_averaging = True
        logger.info(f"[Paper] 📉 {symbol}: УСРЕДНЕНИЕ (вход {n_entries+1}/{MAX_ENTRIES_PER_SYMBOL})")

    # ── Размер позиции ──
    trade_pct  = get_kelly_pct(symbol, confidence, cl, is_averaging)
    amount_rub = round(balance_data["balance"] * trade_pct, 2)

    can_open, reason = check_exposure_limits(portfolio, balance_data["balance"], symbol, amount_rub)
    if not can_open:
        logger.info(f"[Paper] {symbol}: лимит экспозиции — {reason}")
        return None

    qty = round(amount_rub / price, 4) if price > 0 else 0

    # ── SL для этого входа ──
    if atr > 0:
        raw_sl_pct = (atr * ATR_SL_MULT) / price
        sl_pct     = max(SL_FLOOR_PCT, min(raw_sl_pct, SL_CAP_PCT))
    else:
        sl_pct = STOP_LOSS_PCT
    sl = round(price * (1 - sl_pct), 4)

    entry = {
        "entry_id":   (len(pos["entries"]) + 1) if pos else 1,
        "price":      price,
        "amount_rub": amount_rub,
        "qty":        qty,
        "sl":         sl,
        "atr":        atr,
        "confidence": round(confidence, 4),
        "kelly_pct":  trade_pct,
        "opened_at":  _now(),
        "is_averaging": is_averaging,
        "trailing_active": False,
        "breakeven_hit":   False,
        "max_price":  price,
        **(extra_info or {}),
    }

    # ── Обновляем или создаём позицию ──
    if pos is None:
        portfolio[symbol] = {
            "symbol":     symbol,
            "status":     "OPEN",
            "entries":    [entry],
            "sell_levels_hit": [],   # какие уровни продажи уже исполнены
            "total_qty_sold":  0.0,
        }
    else:
        portfolio[symbol]["entries"].append(entry)

    # Обновляем среднюю цену
    portfolio[symbol] = _recalc_avg_price(portfolio[symbol])

    save_portfolio(portfolio)

    # Обновляем пик баланса
    if balance_data["balance"] > balance_data.get("peak_balance", INITIAL_BALANCE):
        balance_data["peak_balance"] = balance_data["balance"]
    save_balance(balance_data)

    name = SYMBOL_NAMES.get(symbol, symbol)
    logger.info(
        f"[Paper] {'📈 УСРЕДНЕНИЕ' if is_averaging else '🟢 ВХОД'} {name} | "
        f"Цена={price:.2f} ₽ | Qty={qty:.4f} | "
        f"Сумма={amount_rub:.0f} ₽ ({trade_pct:.1%}) | SL={sl:.2f} ₽ | "
        f"Ср.цена={portfolio[symbol]['avg_price']:.2f} ₽ | "
        f"Входов={len(portfolio[symbol]['entries'])}"
    )

    return {**entry, "symbol": symbol, "avg_price": portfolio[symbol]["avg_price"]}


def _recalc_avg_price(pos: dict) -> dict:
    """Пересчитывает среднюю цену входа по всем активным entries."""
    entries = pos.get("entries", [])
    if not entries:
        pos["avg_price"]    = 0.0
        pos["total_qty"]    = 0.0
        pos["total_amount"] = 0.0
        return pos
    total_amount = sum(e["amount_rub"] for e in entries)
    total_qty    = sum(e["qty"] for e in entries)
    pos["avg_price"]    = round(total_amount / total_qty, 4) if total_qty > 0 else 0.0
    pos["total_qty"]    = round(total_qty, 4)
    pos["total_amount"] = round(total_amount, 2)
    return pos


# ═══════════════════════════════════════════
# TRAILING STOP (на уровне позиции)
# ═══════════════════════════════════════════

def _update_trailing_for_pos(pos: dict, price: float) -> dict:
    """Обновляет trailing stop для всей позиции (используем avg_price как базу)."""
    if not TRAILING_ENABLED:
        return pos

    avg_price = pos.get("avg_price", 0)
    if avg_price <= 0:
        return pos

    pnl_pct = (price - avg_price) / avg_price

    for i, entry in enumerate(pos["entries"]):
        sl    = entry["sl"]
        atr   = entry.get("atr", 0)
        dist  = max(atr / avg_price * 0.8, TRAILING_DISTANCE_PCT) if atr > 0 else TRAILING_DISTANCE_PCT

        # Breakeven
        if not entry.get("breakeven_hit") and pnl_pct >= BREAKEVEN_ACTIVATION and sl < entry["price"]:
            new_sl = round(entry["price"] * 1.0005, 4)
            if new_sl > sl:
                pos["entries"][i]["sl"]            = new_sl
                pos["entries"][i]["breakeven_hit"] = True
                logger.info(f"[Paper] 🔄 {pos['symbol']} entry#{entry['entry_id']} BREAKEVEN: {sl:.2f}→{new_sl:.2f} ₽")

        # Trailing
        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            pos["entries"][i]["trailing_active"] = True
            trailing_sl = round(price * (1 - dist), 4)
            if trailing_sl > pos["entries"][i]["sl"]:
                pos["entries"][i]["sl"] = trailing_sl

        pos["entries"][i]["max_price"] = max(entry.get("max_price", entry["price"]), price)

    return pos


# ═══════════════════════════════════════════
# МОНИТОРИНГ: SL, УРОВНИ ПРОДАЖИ, TRAILING
# ═══════════════════════════════════════════

def monitor_trades(symbols: list = None) -> list:
    """
    Проверяет все открытые позиции.
    Для каждого тикера:
      1. Проверяет SL по каждому входу
      2. Проверяет уровни частичной продажи (+4/7/12%)
      3. Обновляет trailing stop
    Возвращает список событий (закрытия и частичные продажи).
    """
    if symbols is None:
        symbols = SYMBOLS_ALL

    portfolio    = load_portfolio()
    balance_data = load_balance()
    trades       = load_trades()
    events       = []

    for symbol in symbols:
        pos = portfolio.get(symbol)
        if pos is None or not pos.get("entries"):
            continue

        board, market = SYMBOL_BOARD.get(symbol, ("TQBR", "shares"))
        price = get_current_price(symbol, board, market)
        if price <= 0:
            logger.warning(f"[Paper] ⚠️ Нет цены для {symbol}")
            continue

        # Обновляем trailing
        pos = _update_trailing_for_pos(pos, price)

        # ── Проверяем SL по каждому входу ──
        entries_to_close = []
        for entry in pos["entries"]:
            sl = entry["sl"]
            if price <= sl:
                entries_to_close.append(entry)
                pnl_pct = (price - entry["price"]) / entry["price"] * 100
                pnl_rub = round(entry["amount_rub"] * pnl_pct / 100, 2)
                reason  = "SL_TRAILING" if entry.get("trailing_active") else "SL"

                balance_data["balance"]   = round(balance_data["balance"] + pnl_rub, 2)
                balance_data["total_pnl"] = round(balance_data.get("total_pnl", 0) + pnl_rub, 2)
                balance_data["trades_closed"] = balance_data.get("trades_closed", 0) + 1
                balance_data["losses"]    = balance_data.get("losses", 0) + 1
                balance_data["consecutive_losses"] = balance_data.get("consecutive_losses", 0) + 1

                event = {
                    "type":       "SL_CLOSE",
                    "symbol":     symbol,
                    "entry_id":   entry["entry_id"],
                    "price_open": entry["price"],
                    "price_close":price,
                    "pnl_pct":    round(pnl_pct, 2),
                    "pnl_rub":    pnl_rub,
                    "amount_rub": entry["amount_rub"],
                    "reason":     reason,
                    "closed_at":  _now(),
                }
                events.append(event)
                trades.append(event)
                logger.info(
                    f"[Paper] ❌ {symbol} entry#{entry['entry_id']} {reason}: "
                    f"{pnl_pct:+.2f}% | {pnl_rub:+.0f} ₽ | Баланс={balance_data['balance']:.0f} ₽"
                )

        # Удаляем закрытые входы
        if entries_to_close:
            closed_ids = {e["entry_id"] for e in entries_to_close}
            pos["entries"] = [e for e in pos["entries"] if e["entry_id"] not in closed_ids]
            pos = _recalc_avg_price(pos)

            if not pos["entries"]:
                # Позиция полностью закрыта по SL
                del portfolio[symbol]
                logger.info(f"[Paper] 🔴 {symbol}: позиция полностью закрыта по SL")
                continue

        # ── Проверяем уровни частичной продажи ──
        avg_price       = pos.get("avg_price", 0)
        levels_hit      = set(pos.get("sell_levels_hit", []))
        pnl_from_avg    = (price - avg_price) / avg_price if avg_price > 0 else 0

        for lvl in SELL_LEVELS:
            lvl_key = str(lvl["pct"])
            if lvl_key in levels_hit:
                continue
            if pnl_from_avg < lvl["pct"]:
                continue

            # Уровень достигнут — проверяем дивидендный фильтр
            if is_dividend_protected(symbol):
                logger.info(f"[Paper] 💰 {symbol}: уровень +{lvl['pct']*100:.0f}% достигнут, но защита дивидендов активна")
                continue

            # Считаем сколько продаём
            total_qty       = pos.get("total_qty", 0)
            keep_forever_qty = total_qty * SELL_KEEP_FOREVER_RATIO
            sellable_qty    = max(0, total_qty - keep_forever_qty - pos.get("total_qty_sold", 0))
            sell_qty        = round(sellable_qty * lvl["sell_ratio"], 4)

            if sell_qty <= 0:
                continue

            sell_amount = round(sell_qty * price, 2)
            avg_cost    = round(sell_qty * avg_price, 2)
            pnl_rub     = round(sell_amount - avg_cost, 2)
            pnl_pct     = round(pnl_from_avg * 100, 2)

            balance_data["balance"]   = round(balance_data["balance"] + pnl_rub, 2)
            balance_data["total_pnl"] = round(balance_data.get("total_pnl", 0) + pnl_rub, 2)
            balance_data["trades_closed"] = balance_data.get("trades_closed", 0) + 1
            balance_data["wins"]      = balance_data.get("wins", 0) + 1
            balance_data["consecutive_losses"] = 0

            if balance_data["balance"] > balance_data.get("peak_balance", INITIAL_BALANCE):
                balance_data["peak_balance"] = balance_data["balance"]

            pos["sell_levels_hit"].append(lvl_key)
            pos["total_qty_sold"] = round(pos.get("total_qty_sold", 0) + sell_qty, 4)

            # Уменьшаем qty во всех entries пропорционально
            for i, entry in enumerate(pos["entries"]):
                ratio = entry["qty"] / pos["total_qty"] if pos["total_qty"] > 0 else 0
                pos["entries"][i]["qty"]        = round(entry["qty"] - sell_qty * ratio, 4)
                pos["entries"][i]["amount_rub"] = round(entry["amount_rub"] - sell_amount * ratio, 2)

            pos = _recalc_avg_price(pos)

            event = {
                "type":        "PARTIAL_SELL",
                "symbol":      symbol,
                "level_pct":   lvl["pct"],
                "price":       price,
                "avg_price":   avg_price,
                "sell_qty":    sell_qty,
                "sell_amount": sell_amount,
                "pnl_pct":     pnl_pct,
                "pnl_rub":     pnl_rub,
                "sold_at":     _now(),
            }
            events.append(event)
            trades.append(event)

            logger.info(
                f"[Paper] ✅ {symbol} ЧАСТИЧНАЯ ПРОДАЖА +{pnl_pct:.1f}% | "
                f"Продано {sell_qty:.4f} шт = {sell_amount:.0f} ₽ | "
                f"P&L = {pnl_rub:+.0f} ₽ | Баланс={balance_data['balance']:.0f} ₽"
            )

        # Логируем текущее состояние позиции
        if pos.get("entries") and avg_price > 0:
            float_pnl_pct = (price - avg_price) / avg_price * 100
            _, dd = check_drawdown_guard(balance_data)
            dd_str = f" ⚠️ DD={dd:.1f}%" if dd >= 10 else ""
            logger.info(
                f"[Paper] ⏳ {symbol} | Цена={price:.2f} Ср.цена={avg_price:.2f} | "
                f"P&L={float_pnl_pct:+.2f}% | Входов={len(pos['entries'])} | "
                f"Продано уровней={len(pos.get('sell_levels_hit',[]))}{dd_str}"
            )

        portfolio[symbol] = pos

    save_portfolio(portfolio)
    save_balance(balance_data)
    save_trades(trades)

    return events


# ═══════════════════════════════════════════
# СТАТИСТИКА
# ═══════════════════════════════════════════

def get_stats() -> dict:
    balance_data = load_balance()
    portfolio    = load_portfolio()
    trades       = load_trades()

    total  = balance_data.get("trades_closed", 0)
    wins   = balance_data.get("wins", 0)
    losses = balance_data.get("losses", 0)
    winrate = round(wins / total * 100, 1) if total > 0 else 0

    closed_sl      = [t for t in trades if t.get("type") == "SL_CLOSE"]
    closed_partial = [t for t in trades if t.get("type") == "PARTIAL_SELL"]

    pnl_list    = [t["pnl_pct"] for t in closed_sl if t.get("pnl_pct") is not None]
    pnl_partial = [t["pnl_pct"] for t in closed_partial if t.get("pnl_pct") is not None]
    all_pnl     = pnl_list + pnl_partial

    avg_pnl     = round(sum(all_pnl) / len(all_pnl), 2) if all_pnl else 0
    best_trade  = round(max(all_pnl), 2) if all_pnl else 0
    worst_trade = round(min(all_pnl), 2) if all_pnl else 0

    growth_pct = round(
        (balance_data["balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2
    )
    _, current_dd = check_drawdown_guard(balance_data)

    # Открытые позиции
    open_positions = []
    for symbol, pos in portfolio.items():
        if pos.get("entries"):
            board, market = SYMBOL_BOARD.get(symbol, ("TQBR", "shares"))
            cur_price = get_current_price(symbol, board, market)
            avg_price = pos.get("avg_price", 0)
            pnl_float = (cur_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
            open_positions.append({
                "symbol":     symbol,
                "name":       SYMBOL_NAMES.get(symbol, symbol),
                "avg_price":  avg_price,
                "cur_price":  cur_price,
                "pnl_pct":    round(pnl_float, 2),
                "total_amt":  pos.get("total_amount", 0),
                "entries":    len(pos["entries"]),
                "levels_hit": len(pos.get("sell_levels_hit", [])),
            })

    return {
        "balance":            balance_data["balance"],
        "start_balance":      INITIAL_BALANCE,
        "growth_pct":         growth_pct,
        "total_pnl":          balance_data.get("total_pnl", 0),
        "trades_closed":      total,
        "wins":               wins,
        "losses":             losses,
        "winrate":            winrate,
        "avg_pnl":            avg_pnl,
        "best_trade":         best_trade,
        "worst_trade":        worst_trade,
        "open_positions":     open_positions,
        "n_open_positions":   len(open_positions),
        "partial_sells":      len(closed_partial),
        "consecutive_losses": balance_data.get("consecutive_losses", 0),
        "current_drawdown":   current_dd,
        "peak_balance":       balance_data.get("peak_balance", INITIAL_BALANCE),
        "currency":           "RUB",
    }


def format_stats_message(stats: dict) -> str:
    emoji  = "📈" if stats["growth_pct"] >= 0 else "📉"
    dd     = stats.get("current_drawdown", 0)
    cl     = stats.get("consecutive_losses", 0)
    dd_str = f" ⚠️ DD={dd:.1f}%" if dd >= 10 else f" ({dd:.1f}%)"
    cl_str = f"\n⚠️ Серия убытков: <b>{cl}</b>" if cl >= 2 else ""

    positions_str = ""
    for p in stats.get("open_positions", []):
        pnl_e = "🟢" if p["pnl_pct"] >= 0 else "🔴"
        positions_str += (
            f"\n  {pnl_e} <b>{p['name']}</b>: {p['pnl_pct']:+.1f}% | "
            f"Ср.цена {p['avg_price']:.2f} ₽ | "
            f"Входов: {p['entries']} | Продано ур-ней: {p['levels_hit']}"
        )

    return (
        f"📊 <b>Portfolio MOEX v2.0 — Статистика</b>\n\n"
        f"💰 Баланс:        <b>{stats['balance']:,.0f} ₽</b> {emoji}\n"
        f"📈 Рост:          <b>{stats['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L всего:     <b>{stats['total_pnl']:+,.0f} ₽</b>\n"
        f"📉 Просадка:      <b>{dd:.1f}%</b>{dd_str}\n\n"
        f"📋 Сделок закрыто: <b>{stats['trades_closed']}</b>\n"
        f"✅ Побед (Partial): <b>{stats['partial_sells']}</b>\n"
        f"❌ SL:             <b>{stats['losses']}</b>\n"
        f"🎯 Winrate:        <b>{stats['winrate']}%</b>\n"
        f"📊 Средний P&L:    <b>{stats['avg_pnl']:+.2f}%</b>\n"
        f"🏆 Лучшая сделка:  <b>{stats['best_trade']:+.2f}%</b>\n"
        f"💀 Худшая:         <b>{stats['worst_trade']:+.2f}%</b>\n\n"
        f"📂 Открытые позиции: <b>{stats['n_open_positions']}</b>"
        f"{positions_str}"
        f"{cl_str}"
    )
