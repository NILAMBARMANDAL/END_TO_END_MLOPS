# Dockerfile — packages the FastAPI prediction service so it runs identically anywhere.
FROM python:3.11-slim

WORKDIR /app

# install only what the SERVING app needs (lighter than the full training stack)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# copy the app and the exported model artifacts
COPY app.py .
COPY artifacts/ ./artifacts/

EXPOSE 8000

# uvicorn serves the FastAPI app; 0.0.0.0 so it's reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
