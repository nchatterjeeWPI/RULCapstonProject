# ===============================================================
# cmapss_rul/uncertainty.py
# ===============================================================
# This file contains two methods for estimating uncertainty in model predictions:
#   1. Conformal Prediction – builds a fixed-width tolerance range around predictions
#   2. Monte Carlo (MC) Dropout – estimates uncertainty using repeated stochastic passes
# Both methods help estimate the confidence or "tolerance band" of Remaining Useful Life (RUL) predictions.
# ===============================================================

from __future__ import annotations
import numpy as np
import tensorflow as tf


# ===============================================================
# 1. CONFORMAL PREDICTION
# ===============================================================
# This approach computes a single “tolerance width” (q-hat) based on residuals from the validation set.
# The idea: find how far off predictions usually are, then build symmetric intervals around new predictions.
# ---------------------------------------------------------------

def conformal_calibrate(model, X_val, y_val, alpha: float = 0.1, clip_pred: bool = True) -> float:
    """
    Compute a single-sided absolute-residual quantile q̂ (q-hat) on a held-out validation set.
    The goal is to achieve roughly (1 - alpha) coverage for prediction intervals [ŷ - q̂, ŷ + q̂].
    For example, alpha=0.1 targets ~90% confidence.
    """

    # Convert validation labels to numpy array (ensure correct shape)
    y_val = np.asarray(y_val).reshape(-1)

    # Predict RUL on validation set using the trained model
    yhat_val = model.predict(X_val, verbose=0).reshape(-1)

    # Clip predictions to ensure RUL is not negative
    if clip_pred:
        yhat_val = np.clip(yhat_val, 0, None)

    # Compute absolute residuals |y_true - y_pred|
    r = np.abs(y_val - yhat_val)

    # Compute quantile threshold q̂ that covers (1 - alpha) of residuals
    # The formula below uses a finite-sample correction for small datasets
    n = r.shape[0]
    if n == 0:
        # Edge case: if no validation samples exist, return 0 (no uncertainty)
        return 0.0

    # Compute the rank index for the quantile cutoff
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = np.clip(rank, 0, n - 1)  # Ensure the rank is within valid range

    # Sort residuals and extract the q-hat value at that rank
    qhat = np.partition(np.sort(r), rank)[rank]

    # Return q-hat as a single scalar float
    return float(qhat)


def conformal_interval(y_pred: np.ndarray, qhat: float, clip_pred: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Given model predictions and q-hat, compute lower and upper tolerance limits.
    The output intervals are symmetric around the prediction:
        lower = y_pred - qhat
        upper = y_pred + qhat
    """

    # Ensure array shape is 1D
    y_pred = y_pred.reshape(-1)

    # Clip negative values to zero (RUL cannot be negative)
    if clip_pred:
        y_pred = np.clip(y_pred, 0, None)

    # Build interval bounds
    lo = y_pred - qhat
    hi = y_pred + qhat

    # Clip the lower bound to zero (since RUL ≥ 0)
    lo = np.clip(lo, 0, None)

    # Return lower and upper interval arrays
    return lo, hi


# ===============================================================
# 2. MONTE CARLO (MC) DROPOUT
# ===============================================================
# This approach keeps dropout “on” during prediction.
# By running the same input through the model multiple times,
# we get slightly different predictions each time (because dropout randomly disables neurons).
# The spread between these predictions represents the model’s uncertainty.
# ---------------------------------------------------------------

def mc_predict(model, X, T: int = 50, batch_size: int = 2048) -> np.ndarray:
    """
    Run T stochastic forward passes with dropout 'on' at inference.
    Returns array of shape (T, N) with T predictions per sample.
    Uses batching to limit peak memory.
    """
    import numpy as np
    import tensorflow as tf

    X = np.asarray(X)
    N = X.shape[0]
    preds_TN = np.empty((T, N), dtype=np.float32)

    for t in range(T):
        # Build predictions in batches with dropout active
        out_chunks = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            y = model(X[start:end], training=True)  # dropout ON
            y = tf.reshape(y, [-1]).numpy().astype(np.float32, copy=False)
            out_chunks.append(y)
        preds_TN[t, :] = np.concatenate(out_chunks, axis=0)
    return preds_TN


def mc_interval_from_samples(
    samples: np.ndarray,
    alpha: float = 0.1,
    clip_pred: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given MC dropout samples of shape (T, N), compute:
        - Mean prediction across all runs
        - Lower and upper (1 - alpha) prediction interval
    The interval is derived using empirical percentiles of the samples.
    """

    # Compute mean prediction across all MC runs
    mean = samples.mean(axis=0)
    if clip_pred:
        mean = np.clip(mean, 0, None)  # Ensure mean RULs are non-negative

    # Compute percentile-based bounds for the chosen confidence level
    lower = np.percentile(samples, 100 * (alpha / 2), axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

    # Clip bounds to keep all RUL values non-negative
    if clip_pred:
        lower = np.clip(lower, 0, None)
        upper = np.clip(upper, 0, None)

    # Return mean prediction and uncertainty interval bounds
    return mean, lower, upper
