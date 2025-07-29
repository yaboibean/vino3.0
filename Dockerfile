# Use official Python image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download large files at runtime (if needed)
RUN python download_data.py

# Expose port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]