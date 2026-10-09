FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    AI_AGENT_BACKEND_PORT=8010

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-noto-cjk \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/backend
COPY backend/pyproject.toml ./backend/pyproject.toml
RUN python -c "import pathlib, subprocess, sys, tomllib; deps = tomllib.loads(pathlib.Path('/app/backend/pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *deps])"

COPY backend ./backend
COPY data ./data
COPY docker/backend-entrypoint.sh /usr/local/bin/ai-agent-backend-entrypoint

RUN chmod +x /usr/local/bin/ai-agent-backend-entrypoint

EXPOSE 8010

CMD ["ai-agent-backend-entrypoint"]
