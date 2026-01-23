# Base Python image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the API server
CMD ["python", "app.py"]