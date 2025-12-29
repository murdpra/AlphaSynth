FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml for dependency installation
COPY pyproject.toml ./

# Copy the application source code
# This layer will be invalidated if src/financial_analysis changes.
COPY src/financial_analysis ./src/financial_analysis

# Set PYTHONPATH to include the directory containing our top-level package
ENV PYTHONPATH=/app/src

# Install project and its dependencies from pyproject.toml
RUN pip install --no-cache-dir .

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "financial_analysis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]