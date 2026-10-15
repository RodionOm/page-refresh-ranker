FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/app/data/app.db

WORKDIR /app

# dependencies first: this layer stays cached while only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ranker/ ranker/
COPY app/ app/
COPY train.py ./
COPY models/ models/

# use your trained model if models/model.joblib exists; otherwise build a synthetic demo model
RUN mkdir -p data && ([ -f models/model.joblib ] || python train.py --synthetic 3000)

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
