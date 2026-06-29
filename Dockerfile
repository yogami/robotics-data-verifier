FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code and static files
COPY . .

# Run FastAPI using uvicorn. Railway provides the $PORT environment variable.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
