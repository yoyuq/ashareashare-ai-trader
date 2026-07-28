FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,backtest]" && pip install --no-cache-dir uvicorn[standard]

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/sessions simulation_data reports

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Default: start API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
