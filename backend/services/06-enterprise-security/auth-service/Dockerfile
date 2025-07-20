# backend/services/06-enterprise-security/auth-service/Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash nava-auth && \
    chown -R nava-auth:nava-auth /app

# Switch to non-root user
USER nava-auth

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Run the application
CMD ["python", "main.py"]