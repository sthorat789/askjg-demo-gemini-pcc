# =============================================================================
# Custom Voice Agent — Multi-Service Dockerfile
#
# Builds and runs both:
#   - Backend: Python voice agent (LiveKit + Gemini Live, no Pipecat)
#   - Frontend: Next.js UI (LiveKit React components)
#
# Build:
#   docker build -t custom-voice-agent ./custom_voice_agent
#
# Run:
#   docker run -p 3000:3000 -p 8080:8080 --env-file .env custom-voice-agent
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Build the Next.js frontend
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

# Copy frontend source
COPY frontend/ ./

# Build for production
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2 — Final image: Python backend + pre-built frontend
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Node 20 from NodeSource for a more recent version
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install Python dependencies from lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy Python backend source
COPY . /app/

# Copy pre-built Next.js frontend
COPY --from=frontend-builder /app/frontend/.next /app/frontend/.next
COPY --from=frontend-builder /app/frontend/node_modules /app/frontend/node_modules
COPY --from=frontend-builder /app/frontend/package.json /app/frontend/package.json
COPY --from=frontend-builder /app/frontend/public /app/frontend/public 2>/dev/null || true

# Supervisor configuration to run both services
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/custom-voice-agent.conf

# Expose ports
# 3000 — Next.js frontend
# 8080 — Backend HTTP (token/health endpoint, if any)
EXPOSE 3000 8080

# Start both services via supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/custom-voice-agent.conf"]
