"""
main.py — Entry point for CMAPSS RUL prediction pipeline.

This script orchestrates the high-level workflow:
1. Parse command-line arguments
2. Load data
3. Run the complete pipeline (preprocessing, training, evaluation)
"""

from pathlib import Path

from cmapss_rul.config import make_paths, ensure_dirs, DEFAULT
from cmapss_rul import download, load, explore
from cmapss_rul.pipeline import run_full_pipeline
from cmapss_rul.cli import parse_args
import tensorflow as tf

def main():
    """Main entry point for the RUL prediction pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Resolve configuration (CLI args override defaults)
    arch = args.arch or DEFAULT.arch
    use_tuning = (DEFAULT.use_tuning if args.tuning is None else (args.tuning == "on"))
    epochs = args.epochs if args.epochs is not None else DEFAULT.epochs
    sequence_length = args.sequence_length if args.sequence_length is not None else DEFAULT.sequence_length
    K = args.regimes_k if args.regimes_k is not None else DEFAULT.k
    cap_val = args.cap if args.cap is not None else DEFAULT.cap
    val_size = args.val_size if args.val_size is not None else DEFAULT.val_size
    datasets = args.datasets or list(DEFAULT.datasets)
    use_common_sensors = args.use_common_sensors

    # Determine architectures to run
    if arch == "all":
        architectures = ["tcn", "lstm", "cnn"]
    else:
        architectures = [arch]
    
    print("\n" + "="*70)
    print("CMAPSS RUL PREDICTION PIPELINE")
    print("="*70)
    print(f"Architectures: {architectures}")
    print(f"Datasets: {datasets}")
    print(f"Epochs: {epochs}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Regimes (K): {K}")
    print(f"RUL Cap: {cap_val}")
    print(f"Val Size: {val_size}")
    print(f"Tuning: {'ON' if use_tuning else 'OFF'}")
    print(f"Use Common Sensors: {'YES' if use_common_sensors else 'NO'}")
    print("="*70)
    
    # Setup paths and directories
    paths = make_paths()
    ensure_dirs(paths)
    output_dir = Path(args.out) if args.out else Path("./_outputs/results")
    
    # Download data if requested
    if args.download:
        print("\n[INFO] Downloading datasets...")
        download.fetch_cmaps(paths.raw_data_dir, github_token=args.github_token)
    
    # Load data
    print("\n[INFO] Loading datasets...")
    train_data, test_data, rul_data = load.load_all(paths.user_data_dir, datasets)
    
    # Basic data inspection
    print("\n[INFO] Inspecting data...")
    explore.inspect(train_data)
    missing_dupes = explore.missing_and_dupes_report(train_data, test_data)
    print(f"Missing/Duplicate report: {missing_dupes}")
    
    # Identify non-constant sensors
    sensors_to_keep = explore.non_constant_sensors(train_data)
    if sensors_to_keep:
        print(f"Non-constant sensors: {len(sensors_to_keep)}")
    else:
        print("All sensors will be kept")
        sensors_to_keep = None
    
    # Run the complete pipeline
    print("\n[INFO] Starting pipeline...")
    results = run_full_pipeline(
        train_data=train_data,
        test_data=test_data,
        rul_data=rul_data,
        architectures=architectures,
        datasets=datasets,
        epochs=epochs,
        sequence_length=sequence_length,
        K=K,
        cap_val=cap_val,
        val_size=val_size,
        use_tuning=use_tuning,
        sensors_to_keep=sensors_to_keep,
        run_sensor_analysis_flag=True,
        output_dir=output_dir
    )

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Architectures trained: {list(results.keys())}")


if __name__ == "__main__":
    main()
