FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py composer.py conversation_handlers.py ./
ENV PORT=8080 PYTHONUNBUFFERED=1
# Single worker: all context/conversation state is in memory.
CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT} --workers 1"]
