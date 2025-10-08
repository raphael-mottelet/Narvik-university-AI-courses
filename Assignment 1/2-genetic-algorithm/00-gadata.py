from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

FULL_DEFAULT = os.path.join("..", "data", "openfoodfacts", "step13-vector-full.csv")


def load_full(path: str = FULL_DEFAULT, ingredients_col: str = "") -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)

    candidates = []
    if ingredients_col:
        candidates.append(ingredients_col)
    candidates += ["ingredients_tags", "ingredients", "ingredients_text", "ingredients_list", "ingredients_en", "ingredients_tags_en"]

    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if not found:
        raise KeyError(f"No ingredients column found. Tried: {candidates}")

    if "ingredients_tags" not in df.columns:
        df["ingredients_tags"] = df[found]

    for c in ["taste_balance", "price_norm", "health_score"]:
        if c not in df.columns:
            raise KeyError(f"Missing column in full CSV: {c}")
    if "carbon_norm" not in df.columns:
        df["carbon_norm"] = 0.0

    df["taste_balance"] = pd.to_numeric(df["taste_balance"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["price_norm"]    = pd.to_numeric(df["price_norm"],    errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["health_score"]  = pd.to_numeric(df["health_score"],  errors="coerce").fillna(3).round().clip(1, 5).astype(int)
    df["carbon_norm"]   = pd.to_numeric(df["carbon_norm"],   errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df


def clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def clip_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(hi, np.maximum(lo, x))


def health_to_unit(health_like: float | np.ndarray) -> np.ndarray:
    h = np.rint(health_like).astype(float)
    return (5.0 - clip_range(h, 1.0, 5.0)) / 4.0


def tokenize_ingredients(s: str) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    parts = re.split(r"[|,]\s*", str(s).strip().lower())
    out = []
    for p in parts:
        t = re.sub(r"[^a-z0-9\-_\s]", "", p).strip()
        if t:
            out.append(t)
    return out


def build_ingredient_stats(df: pd.DataFrame, min_count: int = 5, top_k: int | None = None, blacklist: set[str] | None = None):
    counts = {}
    for v in df["ingredients_tags"].fillna(""):
        for tok in tokenize_ingredients(v):
            counts[tok] = counts.get(tok, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if min_count > 1:
        items = [x for x in items if x[1] >= min_count]
    if blacklist:
        items = [x for x in items if x[0] not in blacklist]
    if top_k is not None:
        items = items[:top_k]
    vocab = [w for w, _ in items]
    if not vocab:
        raise ValueError("Empty ingredient vocabulary after filtering")

    idx = {w: i for i, w in enumerate(vocab)}
    sums = np.zeros((len(vocab), 4), dtype=float)
    cnts = np.zeros(len(vocab), dtype=float)

    for _, row in df.iterrows():
        toks = [t for t in tokenize_ingredients(row.get("ingredients_tags", "")) if t in idx]
        if not toks:
            continue
        for t in set(toks):
            j = idx[t]
            sums[j, 0] += float(row["taste_balance"])
            sums[j, 1] += float(row["price_norm"])
            sums[j, 2] += float(row["health_score"])
            sums[j, 3] += float(row["carbon_norm"])
            cnts[j] += 1.0

    cnts[cnts == 0] = 1.0
    stats = sums / cnts[:, None]
    return vocab, stats, counts


def seed_ingredient_population(vocab_size: int, pop_size: int, min_k: int = 3, max_k: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.zeros((pop_size, vocab_size), dtype=float)
    for i in range(pop_size):
        k = int(rng.integers(min_k, max_k + 1))
        idx = rng.choice(vocab_size, size=k, replace=False)
        pop[i, idx] = 1.0
    return pop


def parse_banlist(path: str, vocab: list[str]) -> list[frozenset[int]]:
    if not path or not os.path.isfile(path):
        return []
    idx = {w: i for i, w in enumerate(vocab)}
    banned = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = [t.strip().lower() for t in line.split("+") if t.strip()]
        ids = []
        for t in toks:
            if t in idx:
                ids.append(idx[t])
        if ids:
            banned.append(frozenset(ids))
    return banned


def is_banned(bits: np.ndarray, banned_sets: list[frozenset[int]]) -> bool:
    if not banned_sets:
        return False
    sel = set(np.nonzero(bits > 0.5)[0].tolist())
    for b in banned_sets:
        if b.issubset(sel):
            return True
    return False


def aggregate_components(bits: np.ndarray, stats: np.ndarray) -> tuple[float, float, int, float]:
    sel = np.nonzero(bits > 0.5)[0]
    if len(sel) == 0:
        return 0.0, 1.0, 5, 1.0
    taste = float(np.clip(stats[sel, 0].mean(), 0.0, 1.0))
    price = float(np.clip(stats[sel, 1].mean(), 0.0, 1.0))
    health = int(np.rint(np.clip(stats[sel, 2].mean(), 1.0, 5.0)))
    carbon = float(np.clip(stats[sel, 3].mean(), 0.0, 1.0))
    return taste, price, health, carbon


def fitness_ingredients(bits: np.ndarray, stats: np.ndarray, wt=1.0, wc=1.0, wh=1.0, wd=1.0, min_k: int = 1, max_k: int = 999, banned: list[frozenset[int]] | None = None) -> float:
    if banned and is_banned(bits, banned):
        return -1e12
    k = int((bits > 0.5).sum())
    if k < min_k or k > max_k:
        return -1e10
    taste, price, health, carbon = aggregate_components(bits, stats)
    fit = wt * taste + wc * (1.0 - price) + wh * float(health_to_unit(health)) - wd * carbon
    return float(fit)
