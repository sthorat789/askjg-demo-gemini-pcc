#
# Google Cloud Storage utilities
#

import io
import os
from typing import Optional, Union


def get_gcs_credentials() -> Optional[Union[str, io.StringIO]]:
    """Get GCS credentials from environment.

    Checks GOOGLE_VERTEX_CREDENTIALS (JSON string) first,
    then GOOGLE_VERTEX_CREDENTIALS_PATH (file path).

    Returns:
        StringIO of credentials JSON, path string, or None if not found.
    """
    credentials_json = os.getenv("GOOGLE_VERTEX_CREDENTIALS")
    if credentials_json:
        return io.StringIO(credentials_json)
    credentials_path = os.getenv("GOOGLE_VERTEX_CREDENTIALS_PATH")
    if credentials_path:
        return credentials_path
    return None
