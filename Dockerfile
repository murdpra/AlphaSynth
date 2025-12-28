
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency file first (for Docker cache)
COPY pyproject.toml ./

#  Install dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY src ./src
COPY app.py loader.py setup_data.py vector.py ./

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
