# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Set up the Python Flask backend with PyTorch
FROM python:3.10-slim
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and ML model files
COPY backend/ ./backend/
COPY models/ ./models/

# Copy compiled frontend static assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose Flask default port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run the Flask app server
CMD ["python", "backend/server.py"]
