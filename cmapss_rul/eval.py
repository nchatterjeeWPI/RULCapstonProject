from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def cmapss_score(rul_true, rul_pred):
    diff = np.array(rul_pred) - np.array(rul_true)
    return float(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1).sum())

def per_dataset_metrics(model, X_te_dict, y_te_dict, datasets):
    rows = []
    for fd in datasets:
        Xte = X_te_dict.get(fd, None)
        yte = y_te_dict.get(fd, None)
        if Xte is None or Xte.shape[0] == 0:
            rows.append((fd, np.nan, np.nan, 0))
            continue
        yhat = model.predict(Xte, verbose=0).flatten()
        yhat = np.clip(yhat, 0, None)
        rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
        score = cmapss_score(yte, yhat)
        rows.append((fd, rmse, score, int(Xte.shape[0])))
    return pd.DataFrame(rows, columns=["dataset","RMSE","CMAPSS","n_windows"])

def build_final_engine_table(model, X_te_dict, y_te_dict, engine_ids_te_dict, last_idx_map, clip_pred=True):
    rows = []
    for fd, Xte in X_te_dict.items():
        if fd not in last_idx_map or Xte.shape[0] == 0:
            continue
        yhat_all = model.predict(Xte, verbose=0).flatten()
        if clip_pred: yhat_all = np.clip(yhat_all, 0, None)
        ytrue_all = y_te_dict[fd]
        eids_all  = engine_ids_te_dict[fd]
        final_idxs = sorted(last_idx_map[fd].values())
        for idx in final_idxs:
            y_true = float(ytrue_all[idx])
            y_pred = float(yhat_all[idx])
            delta  = y_pred - y_true
            rows.append({
                "dataset":  fd,
                "engine_id": int(eids_all[idx]),
                "y_true":   y_true,
                "y_pred":   y_pred,
                "delta":    float(delta),
                "abs_delta": float(abs(delta)),
                "pct_abs_error": float(abs(delta) / y_true * 100.0) if y_true > 0 else np.nan,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("abs_delta", ascending=False, ignore_index=True)
