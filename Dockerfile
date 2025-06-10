ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --no-install-project --no-dev --no-editable

COPY . /app/
RUN uv sync --frozen --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm

WORKDIR /app

# Install libexpat1 for Kaleido
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
 
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["optuna-mcp"]
