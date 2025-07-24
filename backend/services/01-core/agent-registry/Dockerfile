# Dockerfile - Agent Registry Service
# Port: 8006

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Create non-root user BEFORE installing dependencies
RUN adduser --disabled-password --gecos '' appuser

# Copy requirements first
COPY requirements.txt .

# Install dependencies as appuser
USER appuser
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]