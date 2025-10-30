# ===============================================================
# cmapss_rul/config.py
# ===============================================================
# This module defines global configuration settings for the CMAPSS
# Remaining Useful Life (RUL) prediction project.
#
# What’s inside:
#   1) Paths dataclass — defines key folder locations
#   2) make_paths() — creates a Paths object for file organization
#   3) ensure_dirs() — safely creates missing folders
#   4) ALL_FD — list of all available CMAPSS datasets
#   5) TrainingConfig dataclass — stores model training parameters
#   6) TRAINING — default training configuration instance
#
# Why this matters:
#   - Keeps your project organized and portable
#   - Ensures that directory paths are consistent across scripts
#   - Allows you to control model parameters from one place
# ===============================================================

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal



# ===============================================================
# 1) PATHS DATA CLASS
# ===============================================================
# This @dataclass defines a single structured object to store
# all relevant directory paths for your project.
#
# Each field is a Path object pointing to a specific folder.
# This design ensures you can easily update base locations in one place.
# ---------------------------------------------------------------
@dataclass
class Paths:
    raw_data_dir: Path             # Where the downloaded ZIP will be extracted
    user_data_dir: Path            # Folder containing the CMAPSS .txt files
    gdrive_dir_raw: Path           # Google Drive (or local) folder for raw data backup
    gdrive_dir_clean: Path         # Folder for cleaned/preprocessed data
    gdrive_results_model: Path     # Folder to store trained models
    gdrive_results_figures: Path   # Folder for evaluation figures/plots
    gdrive_src: Path               # Folder for project source code backups


# ===============================================================
# 2) MAKE PATHS FUNCTION
# ===============================================================
# Creates and returns a Paths object using the provided base directory.
#
# Example:
#   paths = make_paths("CMAPSS/RAW_DATA")
#   print(paths.raw_data_dir)
#
# Arguments:
#   base_raw:     local path where raw data will be stored
#   gdrive_root:  optional root folder (local or Google Drive)
#
# Returns:
#   Paths: a dataclass instance containing all project paths
# ---------------------------------------------------------------
def make_paths(base_raw: str = "CMAPSS/RAW_DATA", gdrive_root: str = "") -> Paths:
    # Define the base directories
    raw = Path(base_raw)
    user = raw / "CMaps"

    # Use Google Drive root if provided, otherwise default to ./_outputs
    groot = Path(gdrive_root) if gdrive_root else Path("./_outputs")

    # Return the full Paths object
    return Paths(
        raw_data_dir=raw,
        user_data_dir=user,
        gdrive_dir_raw=groot / "data/raw",
        gdrive_dir_clean=groot / "data/clean",
        gdrive_results_model=groot / "results/model",
        gdrive_results_figures=groot / "results/figures",
        gdrive_src=groot / "src",
    )


# ===============================================================
# 3) ENSURE DIRECTORIES FUNCTION
# ===============================================================
# Automatically creates all required directories if they don’t exist.
#
# Example:
#   paths = make_paths()
#   ensure_dirs(paths)
#
# After running, you’ll have all the folders ready for saving files.
# ---------------------------------------------------------------
def ensure_dirs(p: Paths):
    for d in [
        p.raw_data_dir,
        p.user_data_dir.parent,
        p.gdrive_dir_raw,
        p.gdrive_dir_clean,
        p.gdrive_results_model,
        p.gdrive_results_figures,
        p.gdrive_src,
    ]:
        d.mkdir(parents=True, exist_ok=True)  # Create folder safely (no error if already exists)


# ===============================================================
# 4) ALL FD LIST
# ===============================================================
# CMAPSS includes four standard datasets:
#   FD001, FD002, FD003, FD004
# Each represents a different combination of operating conditions
# and fault modes.
# ---------------------------------------------------------------
ALL_FD = ['FD001', 'FD002', 'FD003', 'FD004']


# ===============================================================
# 5) TRAINING CONFIGURATION
# ===============================================================
# This @dataclass defines default model training parameters.
#
# You can edit these values globally to control how all models
# are trained (CNN, LSTM, TCN, etc.).
#
# Fields:
#   arch:            model type to train ("cnn", "lstm", "tcn")
#   use_tuning:      whether to run hyperparameter tuning
#   epochs:          number of training epochs
#   sequence_length: how many cycles per input sequence
#   k:               number of KMeans clusters for operating regimes
#   cap:             maximum RUL cap for label clipping
#   datasets:        list of which CMAPSS subsets to train on
# ---------------------------------------------------------------
@dataclass
class TrainingConfig:
    arch: str = "lstm" # model types ("cnn", "lstm", "tcn")
    use_tuning: bool = False
    epochs: int = 1
    sequence_length: int = 50
    k: int = 6
    cap: int = 125
    datasets: List[str] = tuple(ALL_FD)

    # Uncertainty defaults
    uncertainty_method: Literal["none", "conformal", "mc"] = "conformal"
    # For conformal (coverage ≈ 1 - alpha) and MC intervals
    alpha: float = 0.10
    # For MC dropout: number of stochastic forward passes
    mc_samples: int = 50
    # Optional: clip negative predictions (RUL cannot be negative)
    clip_pred: bool = True

# ===============================================================
# 6) DEFAULT TRAINING INSTANCE
# ===============================================================
# This creates one ready-to-use instance of TrainingConfig.
# Import this in your main script to use shared defaults:
#   from cmapss_rul.config import TRAINING
# ---------------------------------------------------------------
TRAINING = TrainingConfig()
