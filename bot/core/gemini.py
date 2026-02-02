#
# Standalone Gemini client for post-call processing
#
# Uses google-genai directly for:
# - Summary generation (text output)
# - Transcript processing (structured JSON output)
#

import json
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

# Ensure environment variables are loaded
load_dotenv(override=True)

from google.genai.types import GenerateContentConfig
from google.oauth2 import service_account
from loguru import logger

from core.prompts import load_summary_prompt, load_transcript_processing_prompt

# Model configuration
MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash-lite")
SUMMARY_MAX_TOKENS = 1024
PROCESSING_MAX_TOKENS = 20000  # ~15 min call capacity with buffer
PROCESSING_MAX_RETRIES = 3

# Singleton client
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Get or create the Gemini client singleton."""
    global _client
    if _client is not None:
        return _client

    credentials_json = os.getenv("GOOGLE_VERTEX_CREDENTIALS")
    credentials_path = os.getenv("GOOGLE_VERTEX_CREDENTIALS_PATH")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID must be set")

    if not credentials_json and not credentials_path:
        raise ValueError(
            "GOOGLE_VERTEX_CREDENTIALS or GOOGLE_VERTEX_CREDENTIALS_PATH must be set"
        )

    # Load credentials from JSON string or file path
    # Vertex AI requires cloud-platform scope
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    if credentials_json:
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes,
        )
    else:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=scopes,
        )

    _client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )
    logger.info(f"Gemini client initialized (project={project_id}, location={location})")
    return _client


async def generate_summary(transcript_text: str) -> Optional[str]:
    """Generate a summary of the call transcript.

    Args:
        transcript_text: Plain text transcript (no timestamps)

    Returns:
        Summary string, or None if generation failed
    """
    if not transcript_text.strip():
        return None

    try:
        client = _get_client()
        prompt = load_summary_prompt()

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=f"Transcript:\n\n{transcript_text}",
            config=GenerateContentConfig(
                system_instruction=prompt,
                max_output_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.3,
            ),
        )

        if response.text:
            return response.text.strip()
        return None
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return None


def _validate_transcript_schema(processed: list) -> bool:
    """Validate that processed transcript has correct schema.

    Each message must have: role, content, timestamp
    """
    if not isinstance(processed, list) or not processed:
        return False

    for msg in processed:
        if not isinstance(msg, dict):
            return False
        if not all(key in msg for key in ("role", "content", "timestamp")):
            return False

    return True


async def process_transcript(messages: list[dict]) -> Optional[list[dict]]:
    """Process transcript for translation and STT error correction.

    Uses structured JSON output with retry logic for reliability.

    Args:
        messages: List of transcript messages with role, content, timestamp

    Returns:
        List of processed messages, or None if processing failed after retries
    """
    if not messages:
        return None

    client = _get_client()
    prompt = load_transcript_processing_prompt()
    messages_json = json.dumps(messages)

    for attempt in range(1, PROCESSING_MAX_RETRIES + 1):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=messages_json,
                config=GenerateContentConfig(
                    system_instruction=prompt,
                    response_mime_type="application/json",
                    max_output_tokens=PROCESSING_MAX_TOKENS,
                ),
            )

            if not response.text:
                logger.warning(f"Empty response from transcript processing (attempt {attempt}/{PROCESSING_MAX_RETRIES})")
                continue

            # Parse JSON response
            processed = json.loads(response.text)

            # Validate schema
            if _validate_transcript_schema(processed):
                if attempt > 1:
                    logger.info(f"Transcript processing succeeded on attempt {attempt}")
                return processed
            else:
                logger.warning(f"Invalid schema in transcript response (attempt {attempt}/{PROCESSING_MAX_RETRIES})")
                logger.debug(f"Raw response:\n{response.text[:500]}")
                continue

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse transcript response (attempt {attempt}/{PROCESSING_MAX_RETRIES}): {e}")
            logger.debug(f"Raw response:\n{response.text}")
            continue
        except Exception as e:
            logger.error(f"Transcript processing failed (attempt {attempt}/{PROCESSING_MAX_RETRIES}): {e}")
            continue

    logger.error(f"Transcript processing failed after {PROCESSING_MAX_RETRIES} attempts")
    return None
