# Lightweight Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user to match host permissions (typical UID 1000)
RUN groupadd -g 1000 ftuser && \
    useradd -u 1000 -g ftuser -m ftuser

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python deps – hyperliquid relies on websockets, etc.
RUN pip install --no-cache-dir \
    hyperliquid-python-sdk \
    websockets pandas pyarrow

# Copy your collector and wrapper
COPY hyperliquid_data_collector.py /app/hyperliquid_data_collector.py
COPY run_collector.py            /app/run_collector.py

# Set ownership to non-root user
RUN chown -R ftuser:ftuser /app

# Switch to non-root user
USER ftuser

# Default command (also set in compose)
CMD ["python", "run_collector.py"]
