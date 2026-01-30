FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model during build OR runtime?
# For this lab, we'll train during container runtime (via pipeline), not at build.
EXPOSE 5000

CMD ["python", "app.py"]