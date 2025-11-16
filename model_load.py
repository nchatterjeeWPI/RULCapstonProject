from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from cmapss_rul import config, load, pipeline, sequences, eval as eval_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a saved CMAPSS RUL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        choices=["tcn", "lstm", "cnn"],
        default="tcn",
        help="Which architecture's saved model to load.",
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default="FD001",
        help="Which CMAPSS dataset to run inference on.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing <arch>_final.keras and <arch>_final.meta.json. "
             "Defaults to ./_outputs/results/final_model relative to this script.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for inference CSV. "
             "Defaults to ./_outputs/results/inference relative to this script.",
    )
    return parser


def main(argv=None):
    # ------------------------
    # 0) CLI
    # ------------------------
    parser = build_parser()
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent

    # ------------------------
    # 1) Locate model + meta
    # ------------------------
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = base_dir / "_outputs" / "results" / "final_model"

    arch = args.arch
    dataset_name = args.dataset

    model_path = model_dir / f"{arch}_final.keras"
    meta_path = model_dir / f"{arch}_final.meta.json"

    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] Loading metadata from: {meta_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    # Load model
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Loaded model:", model)

    # Load metadata
    with meta_path.open() as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    seq_len = int(meta["sequence_length"])
    clip_pred = bool(meta.get("clip_pred", True))
    meta_K = int(meta.get("K", config.DEFAULT.k))

    print(
        f"[INFO] Metadata: arch={meta.get('architecture')} | "
        f"seq_len={seq_len} | n_features={len(feature_cols)} | K={meta_K}"
    )

    # Sanity check vs model input shape
    _, m_seq_len, m_n_features = model.input_shape
    if m_seq_len != seq_len or m_n_features != len(feature_cols):
        raise ValueError(
            f"Model input shape {model.input_shape} does not match metadata "
            f"(seq_len={seq_len}, n_features={len(feature_cols)})."
        )

    # ------------------------
    # 2) Load + preprocess data
    # ------------------------
    cfg = config.DEFAULT
    paths = config.make_paths()
    datasets = [dataset_name]

    print(f"[INFO] Loading raw data for {dataset_name}...")
    train_data, test_data, rul_data = load.load_all(paths.user_data_dir, datasets)

    print("[INFO] Preprocessing data...")
    train_data, test_data = pipeline.preprocess_data(
        train_data,
        test_data,
        rul_data,
        datasets,
        cfg.cap,
        sensors_to_keep=None,
    )

    print("[INFO] Creating train/val split...")
    train_df, val_df = pipeline.train_val_split(
        train_data,
        datasets,
        val_size=cfg.val_size,
    )

    print("[INFO] Running regime clustering and scaling...")
    train_df, val_df, test_data, setting_cols, sensor_cols = pipeline.regime_clustering(
        train_df,
        val_df,
        test_data,
        datasets,
        meta_K,
    )

    # ------------------------
    # 3) Add regime one-hot + build TEST sequences
    # ------------------------
    # Mirror what pipeline.sequence_generation does, but use feature_cols
    # from metadata and only build TEST sequences.
    from cmapss_rul import sequences as seq_mod

    print("[INFO] Adding regime one-hot encoding...")
    seq_mod.add_regime_onehot(train_df, meta_K)
    seq_mod.add_regime_onehot(val_df, meta_K)
    for fd in datasets:
        seq_mod.add_regime_onehot(test_data[fd], meta_K)

    print(f"[INFO] Building test sequences for {dataset_name} using metadata feature_cols...")
    X_test_dict, y_test_dict, engine_ids_test_dict, last_idx_map = (
        seq_mod.build_test_sequences_per_dataset(
            test_data_dict=test_data,
            seq_len=seq_len,
            feature_cols=feature_cols,
        )
    )

    if dataset_name not in X_test_dict:
        raise KeyError(
            f"{dataset_name} not found in X_test_dict. "
            "Check test_data keys and preprocessing."
        )

    print(f"[INFO] X_test_dict['{dataset_name}'] shape: {X_test_dict[dataset_name].shape}")

    # ------------------------
    # 4) Run inference & save results
    # ------------------------
    print(f"[INFO] Running inference on {dataset_name} with {arch.upper()} model...")
    final_df = eval_module.build_final_engine_table(
        model,
        X_test_dict,
        y_test_dict,
        engine_ids_test_dict,
        last_idx_map,
        clip_pred=clip_pred,
    )

    print("[INFO] Sample predictions:")
    print(final_df.head())

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = base_dir / "_outputs" / "results" / "inference"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{dataset_name.lower()}_{arch}_inference.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved inference results to: {out_csv}")


if __name__ == "__main__":
    main()
