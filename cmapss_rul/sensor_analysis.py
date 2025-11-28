"""
sensor_analysis.py — Sensor importance analysis for RUL prediction.

This module provides multiple methods to analyze which sensors have the most impact on RUL:
1. Correlation analysis with RUL
2. Feature importance from tree-based models
3. Permutation importance
4. Sequential feature selection
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Optional GPU-based Random Forest via RAPIDS cuML (Colab)
USE_GPU_RF = False  # set to True manually *and* install cuML to enable

try:
    from cuml.ensemble import RandomForestRegressor as CUMLRandomForestRegressor  # type: ignore
    GPU_RF_AVAILABLE = True
except Exception:
    CUMLRandomForestRegressor = None  # type: ignore
    GPU_RF_AVAILABLE = False

if USE_GPU_RF and GPU_RF_AVAILABLE:
    print("[INFO] Using GPU-based Random Forest (cuML)")
elif USE_GPU_RF and not GPU_RF_AVAILABLE:
    print("[WARNING] USE_GPU_RF=True but cuML is not available; falling back to CPU scikit-learn.")
    print("[INFO] Using CPU-based Random Forest (scikit-learn)")
else:
    print("[INFO] Using CPU-based Random Forest (scikit-learn)")


# import seaborn as sns


def correlation_analysis(
        df: pd.DataFrame,
        sensor_cols: List[str],
        rul_col: str = "RUL"
) -> pd.DataFrame:
    """
    Compute correlation between each sensor and RUL.

    Args:
        df: DataFrame containing sensor data and RUL
        sensor_cols: List of sensor column names
        rul_col: Name of the RUL column

    Returns:
        DataFrame with sensors ranked by absolute correlation with RUL
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlations = []
    for sensor in sensor_cols:
        corr = df[[sensor, rul_col]].corr().iloc[0, 1]
        correlations.append({
            'sensor': sensor,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation',
                                                     ascending=False)

    print("\nTop 10 sensors by correlation with RUL:")
    print(corr_df.head(10).to_string(index=False))

    return corr_df


def random_forest_importance(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        rul_col: str = "RUL",
        n_estimators: int = 100,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Train a Random Forest model and extract feature importances.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        rul_col: Name of the RUL column
        n_estimators: Number of trees in the forest
        random_state: Random seed

    Returns:
        DataFrame with sensors ranked by Random Forest feature importance
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    X_train = train_df[sensor_cols].values
    y_train = train_df[rul_col].values
    X_val = val_df[sensor_cols].values
    y_val = val_df[rul_col].values

    print(f"Training Random Forest with {n_estimators} trees...")

    if USE_GPU_RF and GPU_RF_AVAILABLE and CUMLRandomForestRegressor is not None:
        print("[INFO] Using GPU-based Random Forest (cuML)")
        rf = CUMLRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=15,
        )
    else:
        print("[INFO] Using CPU-based Random Forest (scikit-learn)")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            max_depth=15,
            min_samples_split=10,
        )

    rf.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAE: {mae:.2f}")

    # Feature importances
    importances = []
    for sensor, importance in zip(sensor_cols, rf.feature_importances_):
        importances.append({
            'sensor': sensor,
            'importance': importance
        })

    importance_df = pd.DataFrame(importances).sort_values('importance',
                                                          ascending=False)

    print("\nTop 10 sensors by Random Forest importance:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def permutation_importance_analysis(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        rul_col: str = "RUL",
        n_repeats: int = 10,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance using a Random Forest model.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        rul_col: Name of the RUL column
        n_repeats: Number of times to permute each feature
        random_state: Random seed

    Returns:
        DataFrame with sensors ranked by permutation importance
    """
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 60)

    X_train = train_df[sensor_cols].values
    y_train = train_df[rul_col].values
    X_val = val_df[sensor_cols].values
    y_val = val_df[rul_col].values

    print("Training model...")
    print("[INFO] Using CPU-based Random Forest (scikit-learn)")
    
    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=random_state,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=10
    )

    rf.fit(X_train, y_train)

    print(f"Computing permutation importance (n_repeats={n_repeats})...")
    perm_importance = permutation_importance(
        rf, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importances = []
    for sensor, imp_mean, imp_std in zip(
            sensor_cols,
            perm_importance.importances_mean,
            perm_importance.importances_std
    ):
        importances.append({
            'sensor': sensor,
            'importance_mean': imp_mean,
            'importance_std': imp_std
        })

    perm_df = pd.DataFrame(importances).sort_values('importance_mean',
                                                    ascending=False)

    print("\nTop 10 sensors by permutation importance:")
    print(perm_df.head(10).to_string(index=False))

    return perm_df


def ablation_study(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        rul_col: str = "RUL",
        top_n: int = 10,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Perform ablation study: remove one sensor at a time and measure performance drop.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        rul_col: Name of the RUL column
        top_n: Number of top sensors to analyze (to save time)
        random_state: Random seed

    Returns:
        DataFrame with sensors ranked by performance drop when removed
    """
    print("\n" + "=" * 60)
    print(f"ABLATION STUDY (Top {top_n} sensors)")
    print("=" * 60)

    X_train = train_df[sensor_cols].values
    y_train = train_df[rul_col].values
    X_val = val_df[sensor_cols].values
    y_val = val_df[rul_col].values

    print("Training baseline model with all sensors...")

    if USE_GPU_RF and GPU_RF_AVAILABLE and CUMLRandomForestRegressor is not None:
        print("[INFO] Using GPU-based Random Forest (cuML)")
        rf_baseline = CUMLRandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            max_depth=15,
        )
    else:
        print("[INFO] Using CPU-based Random Forest (scikit-learn)")
        rf_baseline = RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=10,
        )

    rf_baseline.fit(X_train, y_train)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_val, rf_baseline.predict(X_val)))
    print(f"Baseline RMSE: {baseline_rmse:.2f}")

    # Get top sensors from feature importance
    baseline_importances = sorted(
        zip(sensor_cols, rf_baseline.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    top_sensors = [s for s, _ in baseline_importances[:top_n]]

    print(f"\nTesting ablation on top {top_n} sensors...")
    ablation_results = []

    for sensor in top_sensors:
        # Remove one sensor
        cols_without = [s for s in sensor_cols if s != sensor]
        X_train_ablated = train_df[cols_without].values
        X_val_ablated = val_df[cols_without].values

        print("Training baseline model with all sensors...")

        if USE_GPU_RF and GPU_RF_AVAILABLE and CUMLRandomForestRegressor is not None:
            print("[INFO] Using GPU-based Random Forest (cuML)")
            rf_baseline = CUMLRandomForestRegressor(
                n_estimators=50,
                random_state=random_state,
                max_depth=15,
            )
        else:
            print("[INFO] Using CPU-based Random Forest (scikit-learn)")
            rf_baseline = RandomForestRegressor(
                n_estimators=50,
                random_state=random_state,
                n_jobs=-1,
                max_depth=15,
                min_samples_split=10,
            )

        rf.fit(X_train_ablated, y_train)
        rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val_ablated)))

        ablation_results.append({
            'sensor': sensor,
            'rmse_without': rmse,
            'rmse_increase': rmse - baseline_rmse,
            'percent_increase': ((rmse - baseline_rmse) / baseline_rmse) * 100
        })

    ablation_df = pd.DataFrame(ablation_results).sort_values('rmse_increase',
                                                             ascending=False)

    print("\nTop sensors by performance drop when removed:")
    print(ablation_df.to_string(index=False))

    return ablation_df


def plot_sensor_importance(
        corr_df: pd.DataFrame,
        rf_importance_df: pd.DataFrame,
        perm_importance_df: pd.DataFrame,
        output_dir: Path,
        top_n: int = 15
):
    """
    Create visualizations of sensor importance from different methods.

    Args:
        corr_df: Correlation analysis results
        rf_importance_df: Random Forest importance results
        perm_importance_df: Permutation importance results
        output_dir: Directory to save plots
        top_n: Number of top sensors to display
    """
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Correlation plot
    top_corr = corr_df.head(top_n).copy()
    axes[0].barh(top_corr['sensor'], top_corr['abs_correlation'])
    axes[0].set_xlabel('Absolute Correlation with RUL')
    axes[0].set_title('Top Sensors by Correlation')
    axes[0].invert_yaxis()

    # RF importance plot
    top_rf = rf_importance_df.head(top_n).copy()
    axes[1].barh(top_rf['sensor'], top_rf['importance'])
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Top Sensors by RF Importance')
    axes[1].invert_yaxis()

    # Permutation importance plot
    top_perm = perm_importance_df.head(top_n).copy()
    axes[2].barh(top_perm['sensor'], top_perm['importance_mean'])
    axes[2].set_xlabel('Permutation Importance')
    axes[2].set_title('Top Sensors by Permutation Importance')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plot_path = output_dir / "sensor_importance_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()


def ensemble_sensor_selection(
        results: Dict[str, pd.DataFrame],
        method_weights: Dict[str, float] = None,
        top_n: int = 10
) -> pd.DataFrame:
    """
    Select sensors using ensemble ranking across all methods.
    
    Args:
        results: Dictionary of analysis results from run_full_analysis
        method_weights: Optional weights for each method (higher = more important)
        top_n: Number of top sensors to return
        
    Returns:
        DataFrame with sensors ranked by ensemble score
    """
    print("\n" + "=" * 60)
    print("ENSEMBLE SENSOR SELECTION")
    print("=" * 60)
    
    # Default weights (can be tuned based on which method you trust more)
    if method_weights is None:
        method_weights = {
            'rf_importance': 0.30,      # Model-based, good for actual usage
            'perm_importance': 0.35,     # Most reliable for predictive power
            'ablation': 0.25,            # Shows true impact
            'correlation': 0.10          # Less weight - only linear relationships
        }
    
    print(f"Method weights: {method_weights}")
    
    # Get all unique sensors
    all_sensors = set()
    for method_name, df in results.items():
        if method_name not in ['common_sensors'] and isinstance(df, pd.DataFrame):
            all_sensors.update(df['sensor'].tolist())
    
    # For each sensor, calculate normalized rank score from each method
    ensemble_results = []
    
    for sensor in all_sensors:
        scores = {}
        
        # Correlation
        if 'correlation' in results:
            corr_df = results['correlation']
            sensor_row = corr_df[corr_df['sensor'] == sensor]
            if not sensor_row.empty:
                # Normalize: highest correlation gets 1.0, lowest gets 0.0
                max_val = corr_df['abs_correlation'].max()
                min_val = corr_df['abs_correlation'].min()
                val = sensor_row['abs_correlation'].values[0]
                scores['correlation'] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            else:
                scores['correlation'] = 0.0
        
        # RF Importance
        if 'rf_importance' in results:
            rf_df = results['rf_importance']
            sensor_row = rf_df[rf_df['sensor'] == sensor]
            if not sensor_row.empty:
                max_val = rf_df['importance'].max()
                min_val = rf_df['importance'].min()
                val = sensor_row['importance'].values[0]
                scores['rf_importance'] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            else:
                scores['rf_importance'] = 0.0
        
        # Permutation Importance
        if 'perm_importance' in results:
            perm_df = results['perm_importance']
            sensor_row = perm_df[perm_df['sensor'] == sensor]
            if not sensor_row.empty:
                max_val = perm_df['importance_mean'].max()
                min_val = perm_df['importance_mean'].min()
                val = sensor_row['importance_mean'].values[0]
                scores['perm_importance'] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            else:
                scores['perm_importance'] = 0.0
        
        # Ablation
        if 'ablation' in results:
            abl_df = results['ablation']
            sensor_row = abl_df[abl_df['sensor'] == sensor]
            if not sensor_row.empty:
                max_val = abl_df['percent_increase'].max()
                min_val = abl_df['percent_increase'].min()
                val = sensor_row['percent_increase'].values[0]
                scores['ablation'] = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            else:
                scores['ablation'] = 0.0
        
        # Calculate weighted ensemble score
        ensemble_score = sum(scores.get(method, 0) * method_weights.get(method, 0) 
                           for method in method_weights.keys())
        
        ensemble_results.append({
            'sensor': sensor,
            'ensemble_score': ensemble_score,
            **{f'{method}_score': scores.get(method, 0) for method in method_weights.keys()}
        })
    
    # Sort by ensemble score
    ensemble_df = pd.DataFrame(ensemble_results).sort_values('ensemble_score', ascending=False)
    
    print(f"\nTop {top_n} sensors by ensemble ranking:")
    print(ensemble_df.head(top_n).to_string(index=False))
    
    return ensemble_df


def find_common_sensors(
        results: Dict[str, pd.DataFrame],
        top_n: int = 5,
        min_methods: int = 3
) -> List[str]:
    """
    Find sensors that appear in top-N rankings across multiple methods.

    Args:
        results: Dictionary of analysis results from run_full_analysis
        top_n: Number of top sensors to consider from each method
        min_methods: Minimum number of methods a sensor must appear in

    Returns:
        List of common sensors sorted by frequency
    """
    print("\n" + "=" * 60)
    print("FINDING COMMON SENSORS ACROSS METHODS")
    print("=" * 60)

    # Extract top N sensors from each method
    top_sensors = {
        'correlation': set(
            results['correlation'].head(top_n)['sensor'].tolist()),
        'rf_importance': set(
            results['rf_importance'].head(top_n)['sensor'].tolist()),
        'perm_importance': set(
            results['perm_importance'].head(top_n)['sensor'].tolist()),
        'ablation': set(results['ablation'].head(top_n)['sensor'].tolist())
    }

    # Count appearances of each sensor
    from collections import Counter
    sensor_counts = Counter()
    for method_sensors in top_sensors.values():
        sensor_counts.update(method_sensors)

    # Filter sensors that appear in at least min_methods
    common_sensors = [
        sensor for sensor, count in sensor_counts.most_common()
        if count >= min_methods
    ]

    print(f"\nSensors appearing in top {top_n} of ALL 4 methods:")
    all_methods = [s for s, c in sensor_counts.items() if c == 4]
    if all_methods:
        print(", ".join(sorted(all_methods)))
    else:
        print("None")

    print(
        f"\nSensors appearing in top {top_n} of at least {min_methods} methods:")
    if common_sensors:
        print(", ".join(sorted(common_sensors)))

        # Show which methods each common sensor appears in
        print("\nDetailed breakdown:")
        for sensor in sorted(common_sensors):
            methods = [name for name, sensors in top_sensors.items() if
                       sensor in sensors]
            print(
                f"  {sensor}: {sensor_counts[sensor]}/4 methods ({', '.join(methods)})")
    else:
        print("None")

    return common_sensors


def run_full_analysis(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        sensor_cols: List[str],
        output_dir: Path,
        rul_col: str = "RUL",
        save_results: bool = True,
        top_n_for_common: int = 5,
        min_methods_for_common: int = 3,
        recommended_sensor_count: int = 4
) -> Dict[str, Any]:
    """
    Run all sensor importance analyses.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        sensor_cols: List of sensor column names
        output_dir: Directory to save results
        rul_col: Name of the RUL column
        save_results: Whether to save results to CSV
        top_n_for_common: Number of top sensors to consider for common sensor analysis
        min_methods_for_common: Minimum methods a sensor must appear in to be considered common

    Returns:
        Dictionary containing all analysis results and common sensors
    """
    print("\n" + "=" * 60)
    print("SENSOR IMPORTANCE ANALYSIS")
    print("=" * 60)
    print(f"Analyzing {len(sensor_cols)} sensors")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    if USE_GPU_RF and GPU_RF_AVAILABLE:
        print("GPU-accelerated Random Forest: ENABLED (cuML)")
    elif USE_GPU_RF and not GPU_RF_AVAILABLE:
        print(
            "GPU-accelerated Random Forest: REQUESTED, but cuML not available — using scikit-learn CPU")
    else:
        print(
            "GPU-accelerated Random Forest: DISABLED (using scikit-learn CPU)")

    results = {}

    # 1. Correlation analysis
    results['correlation'] = correlation_analysis(train_df, sensor_cols,
                                                  rul_col)

    # 2. Random Forest importance
    results['rf_importance'] = random_forest_importance(train_df, val_df,
                                                        sensor_cols, rul_col)

    # 3. Permutation importance
    results['perm_importance'] = permutation_importance_analysis(
        train_df, val_df, sensor_cols, rul_col, n_repeats=10
    )

    # 4. Ablation study (on top sensors only to save time)
    results['ablation'] = ablation_study(train_df, val_df, sensor_cols,
                                         rul_col, top_n=10)

    # Create visualizations
    plot_sensor_importance(
        results['correlation'],
        results['rf_importance'],
        results['perm_importance'],
        output_dir
    )

    # Save results
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            csv_path = output_dir / f"sensor_{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: TOP 5 SENSORS BY EACH METHOD")
    print("=" * 60)
    print("\nBy Correlation:")
    print(", ".join(results['correlation'].head(5)['sensor'].tolist()))
    print("\nBy RF Importance:")
    print(", ".join(results['rf_importance'].head(5)['sensor'].tolist()))
    print("\nBy Permutation Importance:")
    print(", ".join(results['perm_importance'].head(5)['sensor'].tolist()))
    print("\nBy Ablation Study:")
    print(", ".join(results['ablation'].head(5)['sensor'].tolist()))

    # Find common sensors
    common_sensors = find_common_sensors(
        results,
        top_n=top_n_for_common,
        min_methods=min_methods_for_common
    )

    # Add common sensors to results
    results['common_sensors'] = common_sensors

    # NEW: Ensemble ranking for recommended sensors
    ensemble_df = ensemble_sensor_selection(results, top_n=10)
    results['ensemble_ranking'] = ensemble_df
    
    # Extract recommended sensor list
    recommended_sensors = ensemble_df.head(recommended_sensor_count)['sensor'].tolist()
    results['recommended_sensors'] = recommended_sensors
    
    # Print recommendation
    print("\n" + "=" * 60)
    print("AUTOMATED SENSOR RECOMMENDATION (Ensemble Ranking)")
    print("=" * 60)
    print(f"\n✅ RECOMMENDED TOP {recommended_sensor_count} SENSORS:")
    print(f"   {', '.join(recommended_sensors)}")
    print(f"\nThese sensors provide the best combination of:")
    print("  • Predictive power (permutation importance: 35%)")
    print("  • Model usage (RF importance: 30%)")
    print("  • True impact (ablation: 25%)")
    print("  • Linear correlation (correlation: 10%)")

    # Save common sensors list
    if save_results and common_sensors:
        common_sensors_path = output_dir / "common_sensors.txt"
        with open(common_sensors_path, 'w') as f:
            f.write(
                "# Common sensors appearing in top rankings across multiple methods\n")
            f.write(
                f"# Criteria: Top {top_n_for_common} in at least {min_methods_for_common} methods\n\n")
            for sensor in sorted(common_sensors):
                f.write(f"{sensor}\n")
        print(f"\nSaved common sensors: {common_sensors_path}")
    
    # Save recommended sensors list
    if save_results:
        recommended_path = output_dir / "recommended_sensors.txt"
        with open(recommended_path, 'w') as f:
            f.write("# Recommended sensors via Ensemble Ranking\n")
            f.write(f"# Top {recommended_sensor_count} sensors based on weighted combination of all methods\n")
            f.write("# Weights: Permutation (35%), RF Importance (30%), Ablation (25%), Correlation (10%)\n\n")
            for sensor in recommended_sensors:
                f.write(f"{sensor}\n")
        print(f"Saved recommended sensors: {recommended_path}")
        
        # Save full ensemble ranking
        ensemble_csv_path = output_dir / "sensor_ensemble_ranking.csv"
        ensemble_df.to_csv(ensemble_csv_path, index=False)
        print(f"Saved ensemble ranking: {ensemble_csv_path}")

    return results