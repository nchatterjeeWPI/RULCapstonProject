import os
import zipfile
import requests
from pathlib import Path
from typing import Optional

DEFAULT_URL = "https://raw.githubusercontent.com/nchatterjeeWPI/CMAPSS-RUL/9937bcce8e4d97bf09f935a050bebe7e0138472e/CMaps.zip"

def download_and_extract(url: str, out_zip: Path, extract_to: Path, github_token: Optional[str] = None):
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(out_zip, "r") as z:
        z.extractall(extract_to)
