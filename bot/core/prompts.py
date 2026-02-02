#
# System prompt and summary prompt loading utilities
#

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class LoadedPrompt:
    """Container for loaded prompt with metadata."""

    text: str
    source: str  # "gcs" or "local"
    hash: str  # SHA256 hex


def _compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _load_local_prompt() -> str:
    """Load system prompt from local file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "demo_system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"System prompt not found: {prompt_path}")


async def load_system_prompt_async(
    bucket_name: Optional[str] = None,
    source: str = "local",
) -> LoadedPrompt:
    """Load system prompt from configured source.

    Args:
        bucket_name: GCS bucket name (required if source="gcs").
        source: "local" (default) or "gcs" (with local fallback).

    Returns:
        LoadedPrompt with text, source, and hash.
    """
    # Try GCS if explicitly requested and bucket name provided
    if source == "gcs" and bucket_name:
        try:
            from gcloud.aio.storage import Storage

            from core.gcs import get_gcs_credentials

            async with Storage(service_file=get_gcs_credentials()) as client:
                object_path = "prompts/demo_system_prompt.md"
                content = await client.download(bucket_name, object_path)
                # gcloud-aio returns bytes, decode to string
                text = content.decode("utf-8") if isinstance(content, bytes) else content
                logger.info(f"Loaded system prompt from GCS: gs://{bucket_name}/{object_path}")
                return LoadedPrompt(
                    text=text,
                    source="gcs",
                    hash=_compute_hash(text),
                )
        except Exception as e:
            logger.warning(f"GCS fetch failed ({e}), falling back to local")

    # Fallback to local file
    text = _load_local_prompt()
    logger.info("Loaded system prompt from local file")
    return LoadedPrompt(
        text=text,
        source="local",
        hash=_compute_hash(text),
    )


def load_system_prompt() -> str:
    """Load the demo system prompt.

    Returns:
        System prompt content

    Raises:
        FileNotFoundError: If prompt file not found
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "demo_system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"System prompt not found: {prompt_path}")


def load_summary_prompt() -> str:
    """Load the summary prompt for end-of-call summaries.

    Returns:
        Summary prompt content

    Raises:
        FileNotFoundError: If summary prompt not found
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "summary_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Summary prompt not found: {prompt_path}")


def load_transcript_processing_prompt() -> str:
    """Load the transcript processing prompt for translation and STT correction.

    Returns:
        Transcript processing prompt content

    Raises:
        FileNotFoundError: If transcript processing prompt not found
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "transcript_processing_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Transcript processing prompt not found: {prompt_path}")
