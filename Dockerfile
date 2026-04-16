FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (нужны для lightgbm, scipy)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Часовой пояс Москва (для корректного MOEX сессионного фильтра)
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Порт healthcheck
EXPOSE 8080

# Переменные окружения (задать в docker-compose или командной строке)
# TELEGRAM_TOKEN=...
# TELEGRAM_CHAT_ID=...
# TINKOFF_TOKEN=...  (для будущей реальной торговли)

CMD ["python", "app.py"]
