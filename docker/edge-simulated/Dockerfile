FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY edge/ ./edge/
COPY models/ ./models/
CMD ["python", "edge/inference.py"]