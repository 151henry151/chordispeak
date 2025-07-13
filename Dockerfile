# Use the official Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Install core scientific dependencies first (required for madmom build)
RUN pip install --upgrade pip && \
    pip install Cython numpy scipy

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set the port for Cloud Run
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"] 