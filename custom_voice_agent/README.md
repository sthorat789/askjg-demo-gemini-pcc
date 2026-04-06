# Custom Voice Agent — Gemini Live + LiveKit (No Pipecat)

A purpose-built voice AI agent that replicates Pipecat's interruption and barge-in
behavior using **only** `google-genai`, `livekit`, and `silero-vad` (via ONNX Runtime).

**Zero Pipecat dependency** — all patterns are reimplemented from scratch, inspired by
Pipecat's open-source Apache 2.0 codebase.

## Architecture

```
User (browser/phone)
    │
    ▼
┌─────────────────────────────────────────────┐
│               LiveKit Room                   │
│  (WebRTC audio transport — full duplex)      │
└─────────────┬───────────────────┬───────────┘
              │ Audio In          │ Audio Out
              ▼                   ▲
┌─────────────────────┐  ┌───────────────────┐
│   LiveKitInput      │  │  LiveKitOutput    │
│   (48kHz → 16kHz)   │  │  (24kHz → 48kHz) │
│   resampling        │  │  40ms chunking    │
└─────────┬───────────┘  │  barge-in cancel  │
          │               └───────┬───────────┘
          ▼                       ▲
┌─────────────────────┐           │
│    Silero VAD        │           │
│    (ONNX model)      │           │
│                      │           │
│  4-state machine:    │           │
│  QUIET → STARTING    │           │
│  → SPEAKING          │           │
│  → STOPPING          │           │
│                      │           │
│  On SPEAKING:        │           │
│    → InterruptionFrame           │
│    → cancel output ──────────────┘
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────┐
│      Gemini Live API            │
│      (WebSocket session)        │
│                                 │
│  send_realtime_input(audio)     │
│  → receive() async iterator    │
│  → audio chunks + text + turn   │
│                                 │
│  Implicit interruption:         │
│  New audio stops generation     │
└─────────────────────────────────┘
```

## Components

### 1. Frame System (`src/custom_voice_agent/frames.py`)
Simple dataclass-based frame types with `PriorityQueue` support.
`SystemFrame` (interruptions, control) always processes before `DataFrame` (audio).

### 2. Silero VAD (`src/custom_voice_agent/vad/silero_vad.py`)
Direct ONNX Runtime inference with the same 4-state machine as Pipecat:
- **QUIET → STARTING**: Voice detected (confidence ≥ 0.75, volume ≥ 0.6)
- **STARTING → SPEAKING**: After 0.2s of continuous speech → emits `SPEECH_STARTED`
- **SPEAKING → STOPPING**: Silence detected
- **STOPPING → QUIET**: After 0.2s of silence → emits `SPEECH_STOPPED`

Inference runs in a `ThreadPoolExecutor` (~2-5ms per frame, non-blocking).
Production deployments must provision the model locally via
`SILERO_VAD_MODEL_PATH` or by bundling `vad/silero_vad.onnx` in the image.

### 3. LiveKit Input (`src/custom_voice_agent/transport/livekit_input.py`)
Subscribes to remote audio tracks, resamples 48kHz → 16kHz, feeds VAD + Gemini.

### 4. LiveKit Output (`src/custom_voice_agent/transport/livekit_output.py`)
**The critical barge-in component:**
- Chunks audio into ≤40ms pieces before queuing
- Background task streams chunks to LiveKit
- `cancel_and_clear()`: Cancels task + drains queue in ~0ms
- Result: Audio stops within ~40ms of interruption trigger

### 5. Gemini Live Session (`src/custom_voice_agent/llm/gemini_live.py`)
Direct `google-genai` usage:
- `send_realtime_input()` for user audio
- `receive()` async iterator for bot audio/text
- `send_activity_start()`/`send_activity_end()` for client-side VAD signals
- Gemini handles interruption implicitly (new audio stops generation)

### 6. Voice Agent (`src/custom_voice_agent/agent.py`)
Central orchestrator replacing Pipecat's Pipeline + PipelineTask:
- Coordinates all components
- Manages user/bot speaking state
- Handles barge-in: VAD → cancel output → Gemini auto-stops
- Session timer (max duration) and idle timeout safeguards

## Dependencies

```
google-genai>=1.0.0        # Gemini Live API
livekit>=0.18.0            # LiveKit Python SDK
onnxruntime>=1.16.0        # Silero VAD inference
numpy>=1.24.0              # Audio processing
```

**Total: 4 core dependencies** (vs. Pipecat's 20+ transitive deps)

## Quick Start

```bash
# Enter the project directory
cd custom_voice_agent

# Install the package in editable mode
pip install -e .

# Set environment variables
export LIVEKIT_URL=wss://your-server.livekit.cloud
export LIVEKIT_API_KEY=your-key
export LIVEKIT_API_SECRET=your-secret
export GOOGLE_API_KEY=your-gemini-key      # For preview model
# OR for Vertex AI:
# export GOOGLE_VERTEX_CREDENTIALS='{"type": "service_account", ...}'
# export GOOGLE_CLOUD_PROJECT_ID=your-project

# Provision the Silero model locally (runtime downloads are disabled)
export SILERO_VAD_MODEL_PATH=/opt/models/silero_vad.onnx

# Run
python -m custom_voice_agent.main
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | — | LiveKit server URL |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |
| `LIVEKIT_ROOM` | `voice-agent` | Room name to join |
| `GEMINI_MODEL` | `ga` | `ga` (Vertex AI) or `preview` (Google AI) |
| `GOOGLE_API_KEY` | — | Google AI API key (preview model) |
| `GOOGLE_VERTEX_CREDENTIALS` | — | Vertex AI credentials JSON |
| `GOOGLE_CLOUD_PROJECT_ID` | — | GCP project ID (Vertex AI) |
| `BOT_NAME` | `Maya` | Bot's display name |
| `MAX_CALL_DURATION_SECS` | `840` | Maximum call duration |
| `USER_IDLE_TIMEOUT_SECS` | `120` | User idle timeout |
| `PORT` | `8080` | Health/readiness HTTP port |
| `SILERO_VAD_MODEL_PATH` | bundled path | Absolute path to the provisioned Silero ONNX model |
| `SILERO_VAD_MODEL_SHA256` | — | Optional checksum for the provisioned VAD model |
| `VAD_CONFIDENCE` | `0.75` | VAD confidence threshold |
| `VAD_START_SECS` | `0.2` | Speech start confirmation delay |
| `VAD_STOP_SECS` | `0.2` | Speech stop confirmation delay |
| `VAD_MIN_VOLUME` | `0.6` | Minimum volume threshold |

## Health Endpoints

- `GET /healthz` — liveness and diagnostic payload
- `GET /readyz` — returns `200` only after LiveKit is connected and the agent is running

Both endpoints are served by the Python backend on `PORT` (default `8080`).

## Frontend Scope

The `frontend/` directory remains an optional demo UI and is not packaged into the
production container. Production readiness for this service is defined around the
backend voice agent plus the health/readiness endpoint above.

## Project Layout

```text
custom_voice_agent/
├── src/custom_voice_agent/    # Python package source
├── tests/                     # Backend tests
├── frontend/                  # Optional demo UI
├── Dockerfile                 # Container build
├── pyproject.toml             # Package metadata
└── supervisord.conf           # Container runtime process config
```

## Barge-In Flow (The Critical Path)

```
1. User speaks           → LiveKit delivers audio frames
2. VAD detects voice     → confidence ≥ 0.75 for 0.2s
3. SPEECH_STARTED event  → agent._on_user_started_speaking()
4. If bot is speaking:
   a. send_activity_start()  → Gemini stops generating
   b. cancel_and_clear()     → Output queue drained, task cancelled
   c. Audio stops            → Within ~40ms (one chunk)
5. User audio forwarded  → Gemini processes new input
6. Gemini responds       → New audio queued and played
```

## Comparison with Pipecat

| Aspect | Pipecat | This Implementation |
|--------|---------|---------------------|
| Transport | Daily/SmallWebRTC | LiveKit |
| Dependencies | ~20+ transitive | 4 core |
| Framework | Generic pipeline | Purpose-built |
| Lines of code | ~50,000+ framework | ~1,000 |
| Interruption | Built-in, opaque | Transparent, tunable |
| Customization | Framework abstractions | Full control |
