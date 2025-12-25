FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY harness ./harness
COPY data ./data

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 harness \
    && chown -R harness:harness /app
USER harness

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "harness.api:app", "--host", "0.0.0.0", "--port", "8000"]
