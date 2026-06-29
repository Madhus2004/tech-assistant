# Use a slim Python image to keep the container small
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — this layer is cached unless requirements.txt changes
# This makes rebuilds much faster when you only change app code
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Render injects the PORT environment variable at runtime
ENV PORT=10000
EXPOSE 10000

# Run with gunicorn — production WSGI server, not Flask's dev server
# --workers 1: free tier has limited RAM, one worker is safest
# --threads 4: handles concurrent requests within that worker
# --timeout 120: LLM calls can take time, default 30s would kill requests
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]