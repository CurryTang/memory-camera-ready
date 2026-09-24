"""
Download helpers for dataset files.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

def download_file(
    url: str,
    destination: str | Path,
    timeout_seconds: int = 60,
    overwrite: bool = False,
) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        output_path.write_bytes(response.read())

    return output_path
