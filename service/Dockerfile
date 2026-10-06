# FraudLens scoring API. Build from the repository root:
#   docker build -f service/Dockerfile -t fraudlens-api .
# Trained artifacts are not baked in; mount them read-only at /artifacts (see compose.yaml).

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies first (cached layer), exactly as pinned in uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY fraudlens ./fraudlens
COPY service ./service
RUN uv sync --locked --no-dev

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin fraudlens
USER fraudlens

ENV PATH=/opt/venv/bin:$PATH \
    FRAUDLENS_ARTIFACTS=/artifacts

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# Access logs are off: the app writes its own structured JSON request log.
CMD ["uvicorn", "--factory", "service.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
