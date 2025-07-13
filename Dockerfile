# Use the official Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# GPU optimization environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Set work directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y ffmpeg git build-essential && rm -rf /var/lib/apt/lists/*

# Install build dependencies first
RUN pip install --upgrade pip wheel && pip install numpy==1.26.4 scipy==1.16.0 Cython==3.1.2

# Copy and install the rest of the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set the port for Cloud Run
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the main Flask app with Gunicorn
CMD gunicorn -b 0.0.0.0:$PORT app:app 