"""
weekly_retrainer.py — Плановое переобучение модели по расписанию
Исправлено v3.1:
  - RETRAIN_DAY / RETRAIN_HOUR берутся из config корректно
  - train_model() возвращает модель — сохраняем её
  - Добавлен force_retrain() для ручного запуска
"""

import os
import time
import joblib
import schedule
import traceback
import logging
from datetime import datetime

from config import (
    RETRAIN_DAY, RETRAIN_HOUR,
    SYMBOL, TIMEFRAME,
    MODEL_PATH
)
from auto_trainer import train_model
from telegram_notify import send_message

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Сохранение / загрузка модели
# ─────────────────────────────────────────────
def save_model(model, metadata: dict = None) -> bool:
    """Сохраняет модель локально + опционально метаданные."""
    try:
        joblib.dump(model, MODEL_PATH)
        logger.info(f"[Retrainer] ✅ Модель сохранена: {MODEL_PATH}")

        if metadata:
            import json
            meta_path = MODEL_PATH.replace(".pkl", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"[Retrainer] ❌ Ошибка сохранения: {e}")
        return False


def load_model():
    """Загружает модель локально."""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"[Retrainer] ✅ Модель загружена: {MODEL_PATH}")
            return model
        logger.warning(f"[Retrainer] ⚠️ Модель не найдена: {MODEL_PATH}")
        return None
    except Exception as e:
        logger.error(f"[Retrainer] ❌ Ошибка загрузки: {e}")
        return None


# ─────────────────────────────────────────────
# Основная задача переобучения
# ─────────────────────────────────────────────
def retrain_job():
    started_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"[Retrainer] 🔄 Начало переобучения: {started_at}")

    try:
        send_message(
            f"🔄 <b>Начало переобучения модели</b>\n"
            f"📅 {started_at}\n"
            f"📊 {SYMBOL} | {TIMEFRAME}"
        )

        result = train_model()

        if result and result.get("success"):
            accuracy  = result.get("accuracy",  0)
            precision = result.get("precision", 0)
            recall    = result.get("recall",    0)
            n_samples = result.get("n_samples", 0)
            model     = result.get("model")  # ✅ модель теперь возвращается

            saved = False
            if model is not None:
                saved = save_model(
                    model=model,
                    metadata={
                        "trained_at": started_at,
                        "accuracy":   accuracy,
                        "precision":  precision,
                        "recall":     recall,
                        "n_samples":  n_samples,
                        "symbol":     SYMBOL,
                        "timeframe":  TIMEFRAME,
                    }
                )
            else:
                # train_model уже сохранил через joblib.dump — считаем ок
                saved = os.path.exists(MODEL_PATH)

            if saved:
                msg = (
                    f"✅ <b>Переобучение завершено!</b>\n\n"
                    f"📊 Accuracy:  <b>{accuracy:.1%}</b>\n"
                    f"🎯 Precision: <b>{precision:.1%}</b>\n"
                    f"🔁 Recall:    <b>{recall:.1%}</b>\n"
                    f"📚 Образцов:  <b>{n_samples}</b>\n\n"
                    f"💾 Модель сохранена локально ✅"
                )
            else:
                msg = (
                    f"⚠️ <b>Обучение прошло, но сохранение не удалось</b>\n"
                    f"Accuracy: {accuracy:.1%}"
                )
        else:
            error = result.get("error", "Нет результата") if result else "Нет результата"
            msg   = f"❌ <b>Ошибка переобучения</b>\nПричина: {error}"

        send_message(msg)
        logger.info(f"[Retrainer] {msg[:80]}")

    except Exception as e:
        error_msg = f"❌ <b>Критическая ошибка переобучения</b>\n{str(e)}"
        logger.error(f"[Retrainer] ОШИБКА: {e}")
        traceback.print_exc()
        send_message(error_msg)


# ─────────────────────────────────────────────
# Расписание
# ─────────────────────────────────────────────
def schedule_retraining():
    time_str = f"{RETRAIN_HOUR:02d}:00"

    day_map = {
        "sunday":   schedule.every().sunday,
        "monday":   schedule.every().monday,
        "saturday": schedule.every().saturday,
        "friday":   schedule.every().friday,
    }
    day_scheduler = day_map.get(RETRAIN_DAY.lower(), schedule.every().sunday)
    day_scheduler.at(time_str).do(retrain_job)

    logger.info(f"[Retrainer] ✅ Запланировано: {RETRAIN_DAY} в {time_str} UTC")


def run_retrainer_loop():
    """Бесконечный цикл планировщика."""
    schedule_retraining()
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"[Retrainer] Ошибка в цикле: {e}")
        time.sleep(60)


def force_retrain():
    """Принудительный запуск переобучения (CLI: python weekly_retrainer.py force)."""
    logger.info("[Retrainer] 🚀 Принудительный запуск...")
    retrain_job()


# ─────────────────────────────────────────────
# CLI запуск
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    if len(sys.argv) > 1 and sys.argv[1] == "force":
        force_retrain()
    else:
        logger.info("[Retrainer] Запуск планировщика...")
        run_retrainer_loop()