import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd

# elpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

# decode for latent mode
class LatentDecoder:
    
#deterministic proxy: three linear heads + sigmoid -> [0,1].
 #Replace this with your actual autoencoder decoder if available.

    def __init__(self, zdim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.Wt = rng.normal(0, 0.7, size=(zdim,))
        self.bp = rng.normal(0, 0.2, size=())  # bias for price
        self.Wp = rng.normal(0, 0.7, size=(zdim,))
        self.bt = rng.normal(0, 0.2, size=())  # bias for taste
        self.Wh = rng.normal(0, 0.7, size=(zdim,))
        self.bh = rng.normal(0, 0.2, size=())  # bias for health

    def __call__(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Z: (N, zdim)
        taste  = sigmoid(Z @ self.Wt + self.bt)     # higher is better
        price  = sigmoid(Z @ self.Wp + self.bp)     # higher -> more expensive (worse)
        health = sigmoid(Z @ self.Wh + self.bh)     # higher is healthier (better)
        return clip01(taste), clip01(price), clip01(health)

# GA operators

def init_pop(rng, pop: int, dim: int, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, size=(pop, dim))

def tournament_select(rng, fitness: np.ndarray, k: int) -> int:
    # Return index of winner
    n = fitness.shape[0]
    idxs = rng.integers(0, n, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)

def blx_alpha_cx(rng, p1: np.ndarray, p2: np.ndarray, alpha: float, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    # BLX-α: sample from [min - α*range, max + α*range]
    cmin = np.minimum(p1, p2)
    cmax = np.maximum(p1, p2)
    span = cmax - cmin
    low  = cmin - alpha * span
    high = cmax + alpha * span
    c1 = rng.uniform(low, high)
    c2 = rng.uniform(low, high)
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

def gaussian_mut(rng, x: np.ndarray, pmut: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    mask = rng.random(x.shape) < pmut
    noise = rng.normal(0, sigma, size=x.shape)
    x_new = np.where(mask, x + noise, x)
    return np.clip(x_new, lo, hi)

#fitness wrappers 

def eval_physical(X: np.ndarray, wt: float, wc: float, wh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # X: (N, 3): [taste_balance, price_norm, health_score] all in [0,1]
    taste  = clip01(X[:, 0])
    price  = clip01(X[:, 1])
    health = clip01(X[:, 2])
    fit = wt * taste + wc * (1 - price) + wh * health
    return taste, price, health, fit

def eval_latent(Z: np.ndarray, decoder: LatentDecoder, wt: float, wc: float, wh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    taste, price, health = decoder(Z)
    fit = wt * taste + wc * (1 - price) + wh * health
    return taste, price, health, fit

#  main GA loop 

def run_ga(args):
    rng = np.random.default_rng(args.seed)

    if args.mode == "physical":
        dim, lo, hi = 3, 0.0, 1.0
    else:
        dim, lo, hi = args.zdim, args.zmin, args.zmax
        decoder = LatentDecoder(zdim=dim, seed=args.seed)

    X = init_pop(rng, args.pop, dim, lo, hi)

    # Evaluate
    if args.mode == "physical":
        taste, price, health, fitness = eval_physical(X, args.wt, args.wc, args.wh)
    else:
        taste, price, health, fitness = eval_latent(X, decoder, args.wt, args.wc, args.wh)

    best_fit = float(np.max(fitness))
    print("\n=== Run GA ===")
    print(f"[GA] gen 0001  best {best_fit:.6f}")

    for gen in range(2, args.gens + 1):
        # Elitism
        elite_idx = np.argsort(-fitness)[:args.elite]
        elites = X[elite_idx].copy()

        # New offspring
        children = []
        while len(children) < args.pop - args.elite:
            i = tournament_select(rng, fitness, args.tour)
            j = tournament_select(rng, fitness, args.tour)
            if rng.random() < args.cx:
                c1, c2 = blx_alpha_cx(rng, X[i], X[j], args.alpha, lo, hi)
            else:
                c1, c2 = X[i].copy(), X[j].copy()
            c1 = gaussian_mut(rng, c1, args.mut, args.sigma, lo, hi)
            c2 = gaussian_mut(rng, c2, args.mut, args.sigma, lo, hi)
            children.append(c1)
            if len(children) < args.pop - args.elite:
                children.append(c2)

        X = np.vstack([elites, np.array(children)])
        # Re-evaluate
        if args.mode == "physical":
            taste, price, health, fitness = eval_physical(X, args.wt, args.wc, args.wh)
        else:
            taste, price, health, fitness = eval_latent(X, decoder, args.wt, args.wc, args.wh)

        if gen % args.log_every == 0 or gen == args.gens:
            print(f"[GA] gen {gen:04d}  best {np.max(fitness):.6f}")

    #save
    out = os.path.abspath(f"./ga-{args.mode}.csv")
    order = np.argsort(-fitness)
    Xs = X[order]
    taste_s, price_s, health_s, fit_s = taste[order], price[order], health[order], fitness[order]

    if args.mode == "physical":
        df = pd.DataFrame({
            "taste_balance": taste_s,
            "price_norm":    price_s,
            "health_score":  health_s,
            "fitness":       fit_s
        })
    else:
        z_cols = {f"z{i+1}": Xs[:, i] for i in range(Xs.shape[1])}
        df = pd.DataFrame(z_cols)
        df["taste_balance"] = taste_s
        df["price_norm"]    = price_s
        df["health_score"]  = health_s
        df["fitness"]       = fit_s

    if args.save_unique:
        before = len(df)
        df = df.drop_duplicates()
        print(f"[GA] dedup: {before} -> {len(df)} rows")

    df.to_csv(out, index=False)
    mode = args.mode
    best = df.iloc[0].to_dict()
    print(f"\n[GA] mode={mode}  saved -> {out}")
    print(f"[GA] best: taste={best['taste_balance']:.4f}  price={best['price_norm']:.4f}  health={best['health_score']:.4f}  fitness={best['fitness']:.6f}")

# arguments that was at first not in the program but i needed to gain time

def build_parser():
    p = argparse.ArgumentParser(description="Minimal real-coded GA for physical & latent spaces.")
    p.add_argument("--mode", choices=["physical", "latent"], default="physical")
    # Fitness weights
    p.add_argument("--wt", type=float, default=1.0, help="weight for taste (higher is better)")
    p.add_argument("--wc", type=float, default=1.0, help="weight for cost via (1 - price_norm)")
    p.add_argument("--wh", type=float, default=1.0, help="weight for health")
    # GA params
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gens", type=int, default=120)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--tour", type=int, default=3)
    p.add_argument("--cx", type=float, default=0.9, help="crossover prob")
    p.add_argument("--mut", type=float, default=0.1, help="per-gene mutation prob")
    p.add_argument("--alpha", type=float, default=0.5, help="BLX-alpha")
    p.add_argument("--sigma", type=float, default=0.08, help="Gaussian mutation sigma")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-unique", action="store_true", help="drop duplicate rows before saving")
    # Latent specifics
    p.add_argument("--zdim", type=int, default=8)
    p.add_argument("--zmin", type=float, default=-3.0)
    p.add_argument("--zmax", type=float, default= 3.0)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_ga(args)
