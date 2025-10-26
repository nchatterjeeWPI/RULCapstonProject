from pathlib import Path
import requests, zipfile

# Public mirror of CMAPSS CMaps.zip (adjust if needed)
DEFAULT_URL = "https://raw.githubusercontent.com/nchatterjeeWPI/CMAPSS-RUL/9937bcce8e4d97bf09f935a050bebe7e0138472e/CMaps.zip"

def fetch_cmaps(raw_data_dir: Path, url: str = DEFAULT_URL, github_token: str | None = None):
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    out_zip = raw_data_dir.parent / "CMaps.zip"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    print(f"[DOWNLOAD] {url} -> {out_zip}")
    r = requests.get(url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_zip, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    print(f"[DOWNLOAD] Saved: {out_zip}")
    with zipfile.ZipFile(out_zip, "r") as z:
        z.extractall(raw_data_dir)
    print(f"[DOWNLOAD] Extracted to: {raw_data_dir}")

