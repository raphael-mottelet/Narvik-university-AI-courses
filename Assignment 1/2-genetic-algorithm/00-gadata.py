from __future__ import annotations
import os
import numpy as np
import pandas as pd

CORE_DEFAULT = os.path.join("..", "data", "mlready", "step12-vector-core.csv")


def load_core(path: str = CORE_DEFAULT) -> pd.DataFrame:
    # Load core feature CSV and normalize expected columns
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)

    # Required columns
    for c in ["taste_balance", "price_norm", "health_score"]:
        if c not in df.columns:
            raise KeyError(f"Missing column in core CSV: {c}")

    # Optional environmental impact
    if "carbon_norm" not in df.columns:
        df["carbon_norm"] = 0.0

    # Clean types and ranges
    df["taste_balance"] = pd.to_numeric(df["taste_balance"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["price_norm"]    = pd.to_numeric(df["price_norm"],    errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["health_score"]  = pd.to_numeric(df["health_score"],  errors="coerce").fillna(3).round().clip(1, 5).astype(int)
    df["carbon_norm"]   = pd.to_numeric(df["carbon_norm"],   errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df


def clip01(x: np.ndarray) -> np.ndarray:
    # Clip array to [0,1]
    return np.minimum(1.0, np.maximum(0.0, x))


def clip_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # Clip array to [lo,hi]
    return np.minimum(hi, np.maximum(lo, x))


def health_to_unit(health_like: float | np.ndarray) -> np.ndarray:
    # Map health in [1..5] to [1..0] (A=1 -> 1.0, E=5 -> 0.0)
    h = np.rint(health_like).astype(float)
    return (5.0 - clip_range(h, 1.0, 5.0)) / 4.0


def fitness_core(chrom: np.ndarray, wt=1.0, wc=1.0, wh=1.0, wd=1.0) -> float:
    # Scalar fitness for physical/core chromosome
    # Chromosome: [taste (0..1), price (0..1), health (1..5), carbon (0..1)]
    # fitness = wt * taste + wc * (1 - price) + wh * health_unit - wd * carbon
    taste  = float(chrom[0])
    price  = float(chrom[1])
    health = float(chrom[2])
    carbon = float(chrom[3]) if len(chrom) > 3 else 0.0

    taste  = float(clip01(np.array([taste]))[0])
    price  = float(clip01(np.array([price]))[0])
    h_u    = float(health_to_unit(health))
    carbon = float(clip01(np.array([carbon]))[0])

    return wt * taste + wc * (1.0 - price) + wh * h_u - wd * carbon


def seed_core_population(
    df: pd.DataFrame,
    ingredients: list[str],
    pop_size: int,
    jitter: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Seed initial population by sampling rows and adding small Gaussian noise.

    Args:
        df: DataFrame containing ingredient features.
        ingredients: List of column names to use as features.
        pop_size: Number of individuals in the population.
        jitter: Standard deviation of Gaussian noise for numeric features.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (pop_size, len(ingredients))
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=pop_size)

    # Extract base feature values from DataFrame
    base = df.iloc[idx][ingredients].to_numpy(dtype=float)

    # Apply jitter / normalization if applicable
    noisy = base.copy()
    for j, col in enumerate(ingredients):
        col_data = base[:, j]
        noise = rng.normal(0, jitter, size=pop_size)

        # Handle specific ranges based on heuristics
        if "health" in col.lower():
            noisy[:, j] = clip_range(col_data + rng.normal(0, 0.25, size=pop_size), 1.0, 5.0)
        else:
            noisy[:, j] = clip01(col_data + noise)

    return noisy
