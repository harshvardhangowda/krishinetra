FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISABLE_SQLALCHEMY_CEXT=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy source tree and models
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
