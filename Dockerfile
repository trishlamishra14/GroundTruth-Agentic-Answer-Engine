FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, sample corpus, and the eval report (so the live demo shows benchmark stats)
COPY app ./app
COPY data ./data
COPY eval ./eval

EXPOSE 8000
# Hosts like Render/Railway inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
