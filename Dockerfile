# Romanian imbalance dashboard — Hugging Face Space (Docker SDK)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000 — give it a writable /app for the
# SQLite DB + logs the app creates at runtime.
RUN useradd -m -u 1000 user
WORKDIR /app

# Torch CPU wheel first (large, cached in its own layer). Pinned to the version
# the bundled models were saved with, so joblib/torch load cleanly.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R user:user /app
USER user

# HF routes traffic to app_port (see README); the app reads PORT.
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
