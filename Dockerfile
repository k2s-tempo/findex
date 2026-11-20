# ---- етап 1: збірка (залежності за uv.lock, проєкт, індекс із корпусу) ----
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# залежності: шар кешується, поки не змінився uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# сам проєкт (--no-editable: у venv лежить копія пакета, src у фінальний образ не потрібен)
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# індекс будуємо тут, на Linux: шляхи документів у ньому будуть POSIX-шляхами
COPY corpus ./corpus
RUN .venv/bin/findex index corpus -o index.bin

# ---- етап 2: runtime (без uv і без збирального сміття) ----
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/corpus ./corpus
COPY --from=builder --chown=app:app /app/index.bin ./index.bin
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/index.bin \
    HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["findex", "serve"]
