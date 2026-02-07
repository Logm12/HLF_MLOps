FROM python:3.9-slim

WORKDIR /app

# Install build dependencies mostly for asyncpg (if needed) but binary wheels usually work
# Keep image light
RUN pip install --no-cache-dir asyncpg aiokafka

COPY main.py .

CMD ["python", "main.py"]
