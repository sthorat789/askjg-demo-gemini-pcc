FROM dailyco/pipecat-base:latest

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the application code
COPY ./bot/bot.py bot.py
COPY ./bot/core/ core/
COPY ./bot/pipelines/ pipelines/
COPY ./bot/processors/ processors/
COPY ./bot/prompts/ prompts/
COPY ./bot/transport_context.py transport_context.py
COPY ./bot/call_recorder.py call_recorder.py
COPY ./bot/end_of_call_reporter.py end_of_call_reporter.py
