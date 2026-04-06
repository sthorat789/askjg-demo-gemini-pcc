# AskJG Gemini Voice Demo

Voice AI demo assistant from askjohngeorge.com powered by Gemini Live native audio, deployed on Pipecat Cloud.

## Overview

This is a voice AI demo bot that showcases the capabilities of Gemini Live native audio. It helps visitors understand voice AI technology and can be customized for your specific use case.

**Key Features:**
- Native audio processing (no separate STT/TTS stages)
- Multilingual support (20+ languages)
- Affective dialog (emotional responsiveness)
- Call recording to Google Cloud Storage
- End-of-call reporting via webhook

## Tech Stack

- **Framework**: [Pipecat](https://github.com/pipecat-ai/pipecat)
- **Model**: Gemini 2.5 Flash Native Audio (via Vertex AI)
- **Transport**: Daily WebRTC / SmallWebRTC (local)
- **UI**: [Voice UI Kit](https://voiceuikit.pipecat.ai/)
- **Deployment**: Pipecat Cloud
- **Reporting**: Webhook (end-of-call reports)

---

## Table of Contents

1. [Quick Start (Local Development)](#quick-start-local-development)
2. [Set Up Google Cloud](#set-up-google-cloud)
3. [Deploy to Pipecat Cloud](#deploy-to-pipecat-cloud)
4. [Environment Variables Reference](#environment-variables-reference)
5. [Project Structure](#project-structure)
6. [Customization](#customization)

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Cloud credentials with Vertex AI access (see [Set Up Google Cloud](#set-up-google-cloud))

### Run Locally

```bash
# Clone the repository
git clone https://github.com/askjohngeorge/askjg-demo-gemini-pcc.git
cd askjg-demo-gemini-pcc

# Copy environment template
cp .env.example .env

# Edit .env with your credentials (see Environment Variables Reference)

# Run locally with SmallWebRTC transport
uv run python bot/bot.py
```

Open `http://localhost:7860` in your browser to connect.

---

## Set Up Google Cloud

This section walks you through setting up all the Google Cloud resources needed for the bot.

### Step 1: Create or Select a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click the project dropdown at the top
3. Either select an existing project or click **New Project**
4. If creating new:
   - **Project name**: e.g., `voice-bot-project`
   - Click **Create**
5. **Note your Project ID** - you'll need this later (visible in the project selector or on the dashboard)

### Step 2: Enable Required APIs

1. Go to [APIs & Services > Library](https://console.cloud.google.com/apis/library)
2. Search for and enable each of these APIs:
   - **Vertex AI API** - Required for Gemini Live
   - **Cloud Storage API** - Required if using call recording

Or use the gcloud CLI:
```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

### Step 3: Create a Service Account

1. Go to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click **+ Create Service Account**
3. Fill in:
   - **Service account name**: `voice-bot-service`
   - **Service account ID**: auto-fills (e.g., `voice-bot-service@your-project.iam.gserviceaccount.com`)
   - **Description**: "Service account for Ask JG voice bot"
4. Click **Create and Continue**

### Step 4: Assign IAM Roles

On the "Grant this service account access to project" step, add these roles:

| Role | Purpose |
|------|---------|
| **Vertex AI User** (`roles/aiplatform.user`) | Required - Access to Gemini Live API |
| **Storage Object Admin** (`roles/storage.objectAdmin`) | Only if using call recording |

Click **Continue**, then **Done**.

Or use gcloud CLI:
```bash
# Required for Gemini Live
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:voice-bot-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Only if using call recording
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:voice-bot-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Step 5: Create and Download Credentials Key

1. In the Service Accounts list, click on your new service account
2. Go to the **Keys** tab
3. Click **Add Key** → **Create new key**
4. Select **JSON** format
5. Click **Create**
6. **Save the downloaded JSON file securely** - this is your credentials file
   - Example: `google-credentials.json`
   - **Never commit this file to git!**

### Step 6: Create a GCS Bucket (Optional - For Call Recording)

If you want to record calls:

1. Go to [Cloud Storage > Buckets](https://console.cloud.google.com/storage/browser)
2. Click **+ Create**
3. Configure:
   - **Name**: e.g., `my-call-recordings` (must be globally unique)
   - **Location type**: Region
   - **Location**: `us-central1` (or same region as your bot)
   - **Storage class**: Standard
   - **Access control**: Uniform (recommended)
4. Click **Create**
5. **Note your bucket name** - you'll need this later

Or use gsutil:
```bash
gsutil mb -l us-central1 gs://my-call-recordings
```

### Step 7: Grant Bucket Permissions to Service Account

After creating the bucket, grant your service account permission to read, write, and delete objects:

```bash
gcloud storage buckets add-iam-policy-binding gs://my-call-recordings \
  --member="serviceAccount:voice-bot-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectUser"
```

This grants the `storage.objectUser` role which allows:
- Upload objects (for recording chunks)
- Compose objects (for merging chunks into final WAV)
- Delete objects (for cleanup after composition)
- List objects (for chunk management)

> **Note**: This is a bucket-level permission, more granular than the project-level `storage.objectAdmin` in Step 4. You can use either approach, but bucket-level is recommended for least-privilege access.

### Step 8: Recording Access Configuration

This repository stores recordings in GCS only. There is no bundled dashboard or
authenticated proxy service in the shipped production runtime, so access is
controlled entirely by your bucket policy.

#### Option A: Private Bucket (Recommended)

Keep the bucket private (default) and access recordings with your own
authenticated tooling or internal systems.

No additional bucket permissions are needed beyond Step 7.

#### Option B: Public Bucket (Simple but Less Secure)

If you want recording URLs to be directly accessible without authentication:

```bash
gsutil iam ch allUsers:objectViewer gs://my-call-recordings
```

> **Security Warning**: This makes all recordings publicly accessible via URL. The URLs are unguessable (contain session IDs and timestamps), but anyone with the link can access them. Only use this for testing or non-sensitive recordings.

### Step 9: Test Your Credentials Locally

Set up your local environment to verify everything works:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set these values:
```

Edit your `.env` file:
```bash
# Point to your downloaded credentials file
GOOGLE_VERTEX_CREDENTIALS_PATH=/path/to/google-credentials.json

# Your project ID from Step 1
GOOGLE_CLOUD_PROJECT_ID=your-project-id

# Region (us-central1 has best availability)
GOOGLE_CLOUD_LOCATION=us-central1

# Optional: Enable recording
RECORDING_MODE=stereo
GCS_BUCKET_NAME=my-call-recordings
```

Run the bot locally to test:
```bash
uv run python bot/bot.py
```

If you see "Starting Maya demo bot" without credential errors, you're ready to deploy!

---

## Deploy to Pipecat Cloud

### Prerequisites

- [Pipecat Cloud CLI](https://docs.pipecat.ai/deployment/pipecat-cloud/) installed
- Pipecat Cloud account (`pipecat cloud login`)
- Google Cloud setup completed (see [Set Up Google Cloud](#set-up-google-cloud))

### Step 1: Prepare Your Credentials File

You should have a JSON credentials file from the Google Cloud setup. Verify it exists:

```bash
cat /path/to/google-credentials.json | head -5
```

You should see something like:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  ...
```

### Step 2: Set Pipecat Cloud Secrets

Pipecat Cloud uses secret sets to securely store credentials. The `$(cat ...)` syntax reads your JSON file and passes it as a string.

**Basic deployment (Gemini Live only):**
```bash
pipecat cloud secrets set askjg-demo-gemini-pcc-secrets \
  GOOGLE_VERTEX_CREDENTIALS="$(cat /path/to/google-credentials.json)" \
  GOOGLE_CLOUD_PROJECT_ID="your-project-id" \
  GOOGLE_CLOUD_LOCATION="us-central1"
```

**With call recording:**
```bash
pipecat cloud secrets set askjg-demo-gemini-pcc-secrets \
  GOOGLE_VERTEX_CREDENTIALS="$(cat /path/to/google-credentials.json)" \
  GOOGLE_CLOUD_PROJECT_ID="your-project-id" \
  GOOGLE_CLOUD_LOCATION="us-central1" \
  RECORDING_MODE="stereo" \
  GCS_BUCKET_NAME="my-call-recordings"
```

**With webhook reporting:**
```bash
pipecat cloud secrets set askjg-demo-gemini-pcc-secrets \
  GOOGLE_VERTEX_CREDENTIALS="$(cat /path/to/google-credentials.json)" \
  GOOGLE_CLOUD_PROJECT_ID="your-project-id" \
  GOOGLE_CLOUD_LOCATION="us-central1" \
  RECORDING_MODE="stereo" \
  GCS_BUCKET_NAME="my-call-recordings" \
  WEBHOOK_URL="https://your-webhook-endpoint.com/api/call-reports" \
  WEBHOOK_API_KEY="your-api-key"
```

Verify your secrets were set:
```bash
pipecat cloud secrets list askjg-demo-gemini-pcc-secrets
```

### Step 3: Deploy

```bash
# Build and push Docker image to ghcr.io
pipecat cloud docker build-push

# Deploy the agent
pipecat cloud deploy pcc-deploy.toml
```

You should see output confirming the agent was deployed successfully.

### Step 4: Test the Deployment

Use the [Voice UI Kit](https://voiceuikit.pipecat.ai/) to connect to your deployed bot. You'll need the agent URL from the deployment output.

### Updating Secrets

To add or update individual secrets without re-setting everything:

```bash
# Add or update webhook configuration
pipecat cloud secrets set askjg-demo-gemini-pcc-secrets \
  WEBHOOK_URL="https://your-webhook-endpoint.com/api/call-reports" \
  WEBHOOK_API_KEY="your-api-key"
```

> **Note**: After updating secrets, you need to redeploy for changes to take effect.

### Updating the Deployment

After making code changes or updating secrets:

```bash
pipecat cloud docker build-push && pipecat cloud deploy pcc-deploy.toml
```

### View Logs

```bash
pipecat cloud logs askjg-demo-gemini-pcc
```

### Health Checks

The production container exposes:

- `GET /healthz` — liveness and diagnostic payload
- `GET /readyz` — readiness probe for the bot process

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| **Model Selection** ||||
| `GEMINI_MODEL` | No | `ga` | `ga` (Vertex AI) or `preview` (Google AI API) |
| **Vertex AI (required for `ga` model - default)** ||||
| `GOOGLE_VERTEX_CREDENTIALS` | Yes* | - | Service account JSON as string |
| `GOOGLE_VERTEX_CREDENTIALS_PATH` | Alt* | - | Path to credentials file (local dev) |
| `GOOGLE_CLOUD_PROJECT_ID` | Yes | - | Google Cloud project ID |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | Vertex AI region |
| **Google AI API (required for `preview` model)** ||||
| `GOOGLE_API_KEY` | Yes* | - | Google AI API key |
| `THINKING_BUDGET_TOKENS` | No | `0` | 0=disabled, 512-1024 for reasoning |
| **Bot Configuration** ||||
| `BOT_NAME` | No | `Maya` | Bot name in greeting |
| `PORT` | No | `8080` | Health/readiness probe port |
| `MAX_CALL_DURATION_SECS` | No | `840` | Max call duration (14 min) |
| `USER_IDLE_TIMEOUT_SECS` | No | `120` | Idle timeout before ending call |
| **Call Recording** ||||
| `RECORDING_MODE` | No | `disabled` | `disabled`, `mono`, `stereo`, `both` |
| `GCS_BUCKET_NAME` | If recording | - | GCS bucket name (not full URL) |
| **End-of-Call Reporting** ||||
| `WEBHOOK_URL` | No | - | Webhook endpoint URL for call reports |
| `WEBHOOK_API_KEY` | If webhook | - | API key for webhook authentication |
| `WEBHOOK_AUTH_TYPE` | No | `header` | `header` (X-API-Key) or `body` (wrapped in JSON) |
| `ENABLE_SUMMARY` | No | `true` | Generate LLM summary in reports |
| `ENABLE_TRANSCRIPT_POST_PROCESSING` | No | `true` | Process transcript (translate, fix STT) |

*One of `GOOGLE_VERTEX_CREDENTIALS` or `GOOGLE_VERTEX_CREDENTIALS_PATH` is required for the default `ga` model. For Pipecat Cloud, use `GOOGLE_VERTEX_CREDENTIALS` (the JSON string). For local development, `GOOGLE_VERTEX_CREDENTIALS_PATH` (file path) is often easier.

---

## Project Structure

```
bot/
  bot.py                    # Main entry point
  pipelines/
    base.py                 # PipelineConfig dataclass
    gemini.py               # Gemini Live pipeline
  core/
    health.py                # Health/readiness HTTP server
    observers.py            # RTVI observer for Voice UI Kit
    prompts.py              # Prompt loading utilities
  prompts/
    demo_system_prompt.md   # Maya's personality and knowledge
  call_recorder.py          # GCS call recording
  end_of_call_reporter.py   # Webhook reporting
  transport_context.py      # Call metadata

Dockerfile                  # Container for Pipecat Cloud
pcc-deploy.toml             # Deployment config
```

---

## Customization

### System Prompt

Edit `bot/prompts/demo_system_prompt.md` to modify:
- Personality and speaking style
- Knowledge base
- Conversation goals
- Demo guidance

### Voice Selection

Edit `bot/pipelines/gemini.py`, change `voice_id`:
```python
voice_id="Charon"  # Available: Aoede, Charon, Fenrir, Kore, Puck, etc.
```

### Pipeline Settings

Edit `bot/pipelines/gemini.py` to adjust:
- VAD sensitivity
- Thinking budget (affects latency)
- Affective dialog settings

---

## Architecture

### Native Audio Model

Gemini Live processes audio tokens directly without separate speech-to-text or text-to-speech stages:
- Low latency
- Natural prosody and expression
- Contextual understanding of tone and emotion

### Transport Flow

```
Browser                      Pipecat Cloud              Vertex AI
[Voice UI Kit] <--WebRTC--> [Daily Transport] <---> [Gemini Live]
   widget                      bot.py              Native Audio
                                 │
                                 ▼ (on call end)
                          [Webhook POST]
```

---

## License

MIT License - See LICENSE file for details.
