"""
cli.py — command-line interface (argparse) isolated from program logic.

Usage from main:
    from cli import parse_args  # or build_parser
    args = parse_args()

Keep defaults as None so main can layer config (DEFAULTS) on top.
"""
from __future__ import annotations

import argparse
from typing import List

__all__ = ["build_parser", "parse_args"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CMAPSS Remaining Useful Life pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Lifecycle / IO
    parser.add_argument("--download", action="store_true",
                        help="Download datasets if missing")
    parser.add_argument("--github-token", default=None,
                        help="GitHub token for authenticated downloads (optional)")
    parser.add_argument("--out", default=None,
                        help="Output root directory (default: runs/<timestamp>)")

    # Core controls (None -> fall back to DEFAULTS in main)
    parser.add_argument("--arch", choices=["tcn", "lstm", "cnn", "all"], default=None,
                        help="Model architecture to run, or 'all' to run all three")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (complete passes over the dataset)")
    parser.add_argument("--sequence-length", type=int, dest="sequence_length", default=None,
                        help="Sliding window length for sequence models")
    parser.add_argument("--regimes-k", type=int, dest="regimes_k", default=None,
                        help="K for regime clustering / normalization")
    parser.add_argument("--val-size", type=float, dest="val_size", default=None,
                        help="Validation split fraction (e.g., 0.2)")
    parser.add_argument("--cap", type=int, default=None,
                        help="Optional RUL cap (e.g., 125); omit for uncapped")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run, e.g., FD001 FD003 FD004")

    # Tuning toggle
    parser.add_argument("--tuning", choices=["on", "off"], default=None,
                        help="Run hyperparameter tuning before training")

    # Sensor selection
    parser.add_argument("--use-common-sensors", action="store_true",
                        help="Run sensor analysis and use only top recommended sensors for feature selection")

    # Uncertainty controls
    parser.add_argument("--uncertainty", choices=["none", "conformal", "mc"], default=None,
                        help="Interval method: none, conformal residual quantile, or mc (Monte Carlo dropout)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="(1 - alpha) = target coverage; e.g., alpha=0.1 -> ~90%% interval")
    parser.add_argument("--mc-samples", type=int, default=None, help="T: number of MC dropout samples")

    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse args from argv (or sys.argv if None)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Print all parser fields
    print("Parser fields:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    return args


if __name__ == "__main__":
    # file discoverable if run directly (python cli.py)
    build_parser().print_help()
