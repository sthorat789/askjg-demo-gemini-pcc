FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/custom_voice_agent

RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONPATH=/app
ENV PORT=8080

COPY pyproject.toml uv.lock requirements.txt README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY . /app/custom_voice_agent/

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/custom-voice-agent.conf

EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/custom-voice-agent.conf"]
