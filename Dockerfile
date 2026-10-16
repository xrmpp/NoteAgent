FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/ui/home.py", "--server.address=0.0.0.0", "--server.port=8501"]
