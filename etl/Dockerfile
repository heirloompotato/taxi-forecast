# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY etl_run.py parser_utils.py etl_utils.py forecast_utils.py ./

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start the service (using JSON format for CMD to handle signals properly)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "etl_run:app"]