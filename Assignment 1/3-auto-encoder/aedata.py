from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Constants for nutriments
NUTRI_BOUNDS = {
    "energy-kcal_100g":   (0.0, 900.0),
    "fat_100g":           (0.0, 100.0),
    "saturated-fat_100g": (0.0, 60.0),
    "carbohydrates_100g": (0.0, 100.0),
    "sugars_100g":        (0.0, 100.0),
    "fiber_100g":         (0.0, 30.0),
    "proteins_100g":      (0.0, 60.0),
    "salt_100g":          (0.0, 5.0),
}
POS_HEAVY = ["price_nok_total", "carbon_kg_total"]   # log1p + min–max
UNIT01    = ["taste_balance", "price_norm", "carbon_norm", "macro_balance"]
HEALTH    = "health_score"                           # this part for staying between 1 and 5

#  Basics
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def batches(X: np.ndarray, batch_size: int):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i:i + batch_size]

# Scaling spec
def _fit_log_minmax(x: pd.Series, q_low=1.0, q_hi=99.0):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    v = np.log1p(x)
    lo = float(np.percentile(v, q_low))
    hi = float(np.percentile(v, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    return {"mode": "log_minmax", "lo": lo, "hi": hi}

def _fit_minmax_fixed(lo, hi):
    return {"mode": "minmax", "lo": float(lo), "hi": float(hi)}

def _fit_unit():
    return {"mode": "unit", "lo": 0.0, "hi": 1.0}

def _fit_health15():
    
    return {"mode": "health15", "lo": 1.0, "hi": 5.0}

def fit_spec(df: pd.DataFrame) -> dict:
    spec = {}
    for col in df.columns:
        if col == HEALTH:
            spec[col] = _fit_health15()
        elif col in NUTRI_BOUNDS:
            lo, hi = NUTRI_BOUNDS[col]
            spec[col] = _fit_minmax_fixed(lo, hi)
        elif col in POS_HEAVY:
            spec[col] = _fit_log_minmax(df[col])
        elif col in UNIT01:
            spec[col] = _fit_unit()
        else:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if s.min() >= 0.0 and s.max() <= 1.0:
                spec[col] = _fit_unit()
            else:
                spec[col] = _fit_log_minmax(s)
    return spec

def apply_spec(df: pd.DataFrame, spec: dict) -> np.ndarray:
    Xn = np.zeros((len(df), len(df.columns)), dtype=np.float32)
    for j, col in enumerate(df.columns):
        rule = spec[col]; mode = rule["mode"]
        if mode == "minmax":
            lo, hi = rule["lo"], rule["hi"]
            x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lo, hi).to_numpy()
            Xn[:, j] = ((x - lo) / (hi - lo)).astype(np.float32)
        elif mode == "log_minmax":
            lo, hi = rule["lo"], rule["hi"]
            x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
            v = np.log1p(x); v = np.clip(v, lo, hi)
            Xn[:, j] = ((v - lo) / (hi - lo)).astype(np.float32)
        elif mode == "health15":
            x = pd.to_numeric(df[col], errors="coerce").to_numpy()
            x = np.clip(x, 1.0, 5.0)
            Xn[:, j] = ((5.0 - x) / 4.0).astype(np.float32)  # A->1 ... E->0
        else:  # unit
            x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
            Xn[:, j] = np.clip(x, 0.0, 1.0).astype(np.float32)
    return Xn

def invert_spec(Xn: np.ndarray, cols: list[str], spec: dict) -> np.ndarray:
    X = np.zeros_like(Xn, dtype=np.float32)
    for j, col in enumerate(cols):
        rule = spec[col]; mode = rule["mode"]
        y = np.clip(Xn[:, j], 0.0, 1.0)
        if mode == "minmax":
            lo, hi = rule["lo"], rule["hi"]
            X[:, j] = (y * (hi - lo) + lo).astype(np.float32)
        elif mode == "log_minmax":
            lo, hi = rule["lo"], rule["hi"]
            v = y * (hi - lo) + lo
            X[:, j] = np.expm1(v).astype(np.float32)
        elif mode == "health15":
            X[:, j] = (5.0 - 4.0 * y).astype(np.float32)  # ~1..5
        else:
            X[:, j] = y.astype(np.float32)
    return X

# Labels & sampling
def compute_class_weights_1to5(hp: pd.Series):
    counts = hp.value_counts().reindex([1,2,3,4,5], fill_value=0)
    inv = counts.replace(0, np.nan)
    inv = 1.0 / inv
    if inv.isna().all():
        inv = pd.Series([1,1,1,1,1], index=[1,2,3,4,5], dtype=float)
    else:
        inv = inv.fillna(inv[~inv.isna()].max())
    inv = inv / inv.mean()
    return inv.to_dict(), counts.to_dict()

def balanced_indices_per_epoch(labels_1to5: np.ndarray, rng: np.random.Generator):
    idxs_by_c = {c: np.where(labels_1to5 == c)[0] for c in range(1,6)}
    counts = {c: len(idxs_by_c[c]) for c in range(1,6)}
    if max(counts.values()) == 0:
        return np.arange(len(labels_1to5))
    max_n = max(counts.values())
    all_idxs = []
    for c in range(1,6):
        idxs = idxs_by_c[c]
        if len(idxs) == 0:
            continue
        take = rng.choice(idxs, size=max_n, replace=True)
        all_idxs.append(take)
    cat = np.concatenate(all_idxs) if all_idxs else np.arange(len(labels_1to5))
    rng.shuffle(cat)
    return cat
