FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY bot.py .

# Expose stats API port
EXPOSE 8002

# Run the bot
CMD ["python", "-u", "bot.py"]
