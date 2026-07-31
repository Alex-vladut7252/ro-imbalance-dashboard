# Romanian imbalance dashboard — runs on x86_64 and arm64 (Oracle Ampere A1).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# uid 1000 so the container does not run as root. Hugging Face Spaces also
# expects this uid; on a plain Docker host it is just good hygiene.
RUN useradd -m -u 1000 user
WORKDIR /app

# Torch first, in its own layer (it is by far the largest dependency, and it
# changes least often). Pinned to the version the bundled models were saved
# with, so joblib/torch load them cleanly.
#
# The wheel source differs by architecture:
#   x86_64  — PyPI ships a CUDA-linked build (~2.5 GB of nvidia-* deps we will
#             never use on a CPU host), so pull from PyTorch's CPU index.
#   aarch64 — that index has no ARM wheels, but the PyPI manylinux aarch64
#             wheel is already CPU-only. Plain PyPI is correct there.
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir torch==2.11.0; \
    else \
        pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The SQLite archive must outlive the container — the image is replaced on
# every redeploy. Mount a volume here (see README) or the 1-year history is
# rebuilt from scratch each time.
RUN mkdir -p /data && chown -R user:user /app /data
VOLUME /data
ENV DB_PATH=/data/energy_data.db

USER user

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
