# Dockerfile

# Stage 1: Build the environment with dependencies
FROM python:3.9-slim as builder
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Create the final, smaller production image
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY main.py .

# Command to run the application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]