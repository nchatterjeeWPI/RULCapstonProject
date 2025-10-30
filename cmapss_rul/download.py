# ===============================================================
# cmapss_rul/fetch_cmaps.py
# ===============================================================
# This module automates downloading and extracting the CMAPSS dataset
# (the NASA turbofan engine degradation data) from a public mirror.
#
# What’s inside:
#   1) DEFAULT_URL: public link to the CMAPSS zip archive
#   2) fetch_cmaps(): downloads and unzips the dataset into a local folder
#
# Why this is useful:
#   - The official CMAPSS dataset is distributed as a ZIP file ("CMaps.zip")
#   - This script automatically downloads it if you don’t already have it
#   - It ensures the correct folder structure for downstream data loading
# ===============================================================

from pathlib import Path
import requests
import zipfile

# ===============================================================
# 1) DEFAULT DOWNLOAD LINK
# ===============================================================
# Default URL points to a GitHub mirror of "CMaps.zip" that contains:
#   • train_FD001.txt ... train_FD004.txt
#   • test_FD001.txt ... test_FD004.txt
#   • RUL_FD001.txt ... RUL_FD004.txt
#
# If this link becomes unavailable, you can pass a new one to fetch_cmaps().
# ---------------------------------------------------------------
DEFAULT_URL = (
    "https://raw.githubusercontent.com/nchatterjeeWPI/CMAPSS-RUL/"
    "9937bcce8e4d97bf09f935a050bebe7e0138472e/CMaps.zip"
)


# ===============================================================
# 2) FETCH AND EXTRACT CMAPSS DATA
# ===============================================================
# Downloads "CMaps.zip" and extracts it into your chosen data folder.
#
# Example:
#   from pathlib import Path
#   from cmapss_rul.fetch_cmaps import fetch_cmaps
#   fetch_cmaps(Path("data/raw"))
#
# This will:
#   • Create data/raw/ if it doesn’t exist
#   • Download the ZIP to ../CMaps.zip (one level above raw/)
#   • Unzip the contents into data/raw/
# ---------------------------------------------------------------
def fetch_cmaps(raw_data_dir: Path, url: str = DEFAULT_URL, github_token: str | None = None):
    """
    Download and extract the CMAPSS dataset archive.

    Parameters:
        raw_data_dir (Path):
            Path to the directory where you want the dataset extracted.
            Example: Path("data/raw")
        url (str):
            Direct URL to the CMaps.zip file (default = provided GitHub link)
        github_token (str | None):
            Optional GitHub token for authenticated downloads (prevents rate limiting)

    Returns:
        None. The function saves and extracts files locally.
    """

    # Ensure the target folder exists
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Path where the zip file will be saved (same parent directory)
    out_zip = raw_data_dir.parent / "CMaps.zip"

    # Optional GitHub token for private repositories or rate limits
    headers = {"Authorization": f"token {github_token}"} if github_token else {}

    # ---------- STEP 1: Download the ZIP file ----------
    print(f"[DOWNLOAD] {url} -> {out_zip}")

    # Stream download to handle large files efficiently
    r = requests.get(url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()  # Raise an error if the download fails (HTTP 4xx/5xx)

    # Write chunks to disk (saves memory compared to reading the full file)
    with open(out_zip, "wb") as f:
        for chunk in r.iter_content(8192):  # 8 KB chunks
            if chunk:
                f.write(chunk)

    print(f"[DOWNLOAD] Saved: {out_zip}")

    # ---------- STEP 2: Extract contents ----------
    with zipfile.ZipFile(out_zip, "r") as z:
        z.extractall(raw_data_dir)

    print(f"[DOWNLOAD] Extracted to: {raw_data_dir}")
