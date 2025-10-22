from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Paths:
    raw_data_dir: Path
    user_data_dir: Path
    gdrive_dir_raw: Path
    gdrive_dir_clean: Path
    gdrive_results_model: Path
    gdrive_results_figures: Path
    gdrive_src: Path

def make_paths(base_raw: str = "CMAPSS/RAW_DATA", gdrive_root: str = "") -> Paths:
    raw = Path(base_raw)
    # In the original notebook, user_data_dir ended up pointing to RAW_DATA/CMaps
    user = raw / "CMaps"
    if gdrive_root:
        groot = Path(gdrive_root)
    else:
        groot = Path("./_outputs")
    return Paths(
        raw_data_dir=raw,
        user_data_dir=user,
        gdrive_dir_raw=groot / "data/raw",
        gdrive_dir_clean=groot / "data/clean",
        gdrive_results_model=groot / "results/model",
        gdrive_results_figures=groot / "results/figures",
        gdrive_src=groot / "src",
    )

def ensure_dirs(p: Paths):
    for d in [
        p.raw_data_dir, p.user_data_dir.parent, p.gdrive_dir_raw, p.gdrive_dir_clean,
        p.gdrive_results_model, p.gdrive_results_figures, p.gdrive_src
    ]:
        d.mkdir(parents=True, exist_ok=True)
