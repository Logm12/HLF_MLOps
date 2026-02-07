FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no heavy C libraries needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY calculator.py .

# Run with unbuffered output
CMD ["python", "-u", "calculator.py"]
