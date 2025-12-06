# Используем Python 3.10 slim образ
FROM python:3.10-slim

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения
COPY app.py .
COPY inference.py .
COPY tokenizer.py .
COPY tokenizer.pickle .

# Создание директории для модели (будет монтироваться через volume)
RUN mkdir -p /app/models

# Открытие порта Streamlit
EXPOSE 8501

# Healthcheck для проверки работы
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск приложения
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]