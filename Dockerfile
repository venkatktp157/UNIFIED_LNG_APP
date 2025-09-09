# Use a slim Python base for small image size
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /unified_app

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all source code including sub-apps and shared DATA
COPY . .

# Expose unified app port
EXPOSE 8002

# Run the unified selector app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
