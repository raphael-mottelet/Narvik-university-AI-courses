from __future__ import annotations
import os
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ingredient-combination GA runner
# Initializes binary ingredient chromosomes
# Scores fitness by aggregating ingredient components with simple averages
# Exports CSV without fitness

def _load_module(filename: str, alias: str):
    here = Path(__file__).resolve().parent
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


gadata = _load_module("00-gadata.py", "gadata")
gaengine = _load_module("01-gaengine.py", "gaengine")

DEF_FULL = os.path.join("..", "data", "openfoodfacts", "step13-vector-full.csv")
DEF_OUT = os.path.join("..", "data", "mlready", "ga-output.csv")


def main():
    ap = argparse.ArgumentParser(description="Ingredient-combination GA")
    ap.add_argument("--full-csv", default=DEF_FULL)
    ap.add_argument("--ingredients-col", default="")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--wt", type=float, default=1.0)
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wh", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--tour", type=int, default=3)
    ap.add_argument("--cx", type=float, default=0.9)
    ap.add_argument("--mut", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--top-vocab", type=int, default=200)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--banlist", default="")
    ap.add_argument("--kill-age", type=int, default=20)
    ap.add_argument("--kill-penalty", type=float, default=1e12)
    args = ap.parse_args()

    df = gadata.load_full(args.full_csv, ingredients_col=args.ingredients_col)
    blacklist = {"vegetable", "syrup"}  # keep common generic tokens from dominating
    vocab, stats, counts = gadata.build_ingredient_stats(df, min_count=args.min_count, top_k=args.top_vocab, blacklist=blacklist)
    banned_sets = gadata.parse_banlist(args.banlist, vocab)

    bounds = [(0.0, 1.0)] * len(vocab)
    init_pop = gadata.seed_ingredient_population(len(vocab), args.pop, min_k=args.min_k, max_k=args.max_k, seed=args.seed)

    def fit_fn(bits):
        return gadata.fitness_ingredients(bits, stats, wt=args.wt, wc=args.wc, wh=args.wh, wd=args.wd, min_k=args.min_k, max_k=args.max_k, banned=banned_sets)

    params = gaengine.GAParams(
        pop_size=args.pop,
        gens=args.gens,
        elite=args.elite,
        tournament=args.tour,
        crossover_rate=args.cx,
        mutation_rate=args.mut,
        blx_alpha=0.2,
        mut_sigma=0.1,
        seed=args.seed,
        log_every=args.log_every,
        binary=True,
        kill_copy_age=args.kill_age,
        kill_penalty=args.kill_penalty,
    )

    ga = gaengine.GA(fitness_fn=fit_fn, bounds=bounds, params=params, init_pop=init_pop)
    ga.run()

    order = np.argsort(-ga.fit)[: max(1, min(50, ga.pop.shape[0]))]
    top = ga.pop[order]

    rows = []
    for i in range(top.shape[0]):
        bits = top[i]
        sel_idx = np.nonzero(bits > 0.5)[0].tolist()
        ing_list = [vocab[j] for j in sel_idx]
        t, p, h, c = gadata.aggregate_components(bits, stats)
        rows.append(
            {
                "ingredients_combo": " + ".join(ing_list),
                "taste_balance": t,
                "price_norm": p,
                "health_score": h,
                "carbon_norm": c,
            }
        )

    out = pd.DataFrame(rows)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[GA] mode=ingredients  saved -> {args.out}")


if __name__ == "__main__":
    main()
