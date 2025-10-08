from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
import sys

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

CONFIG = {
    "full_csv": os.path.join("..", "data", "openfoodfacts", "step13-vector-full.csv"),
    "ingredients_col": "",
    "out": os.path.join("..", "data", "mlready", "ga-output.csv"),
    "wt": 1.5,
    "wc": 1.0,
    "wh": 1.0,
    "wd": 1.0,
    "pop": 200,
    "gens": 300,
    "elite": 2,
    "tournament": 3,
    "crossover_rate": 0.9,
    "mutation_rate": 0.08,
    "seed": 42,
    "log_every": 10,
    "min_count": 5,
    "top_vocab": 300,
    "min_k": 4,
    "max_k": 10,
    "soft_cap": 10,
    "soft_penalty": 0.5,
    "banlist": "",
    "kill_age": 20,
    "kill_penalty": 1e12,
    "blacklist": {"vegetable", "syrup"},
}

def main():
    print("=== Run GA ===")
    print("All good for now")

    df = gadata.load_full(CONFIG["full_csv"], ingredients_col=CONFIG["ingredients_col"])
    vocab, stats, counts = gadata.build_ingredient_stats(
        df,
        min_count=CONFIG["min_count"],
        top_k=CONFIG["top_vocab"],
        blacklist=CONFIG["blacklist"],
    )
    banned_sets = gadata.parse_banlist(CONFIG["banlist"], vocab)

    bounds = [(0.0, 1.0)] * len(vocab)
    init_pop = gadata.seed_ingredient_population(
        len(vocab),
        CONFIG["pop"],
        min_k=CONFIG["min_k"],
        max_k=CONFIG["max_k"],
        seed=CONFIG["seed"],
    )

    def fit_fn(bits):
        return gadata.fitness_ingredients(
            bits,
            stats,
            wt=CONFIG["wt"],
            wc=CONFIG["wc"],
            wh=CONFIG["wh"],
            wd=CONFIG["wd"],
            min_k=CONFIG["min_k"],
            max_k=CONFIG["max_k"],
            banned=banned_sets,
            soft_cap=CONFIG["soft_cap"],
            soft_penalty=CONFIG["soft_penalty"],
            hard_penalty=CONFIG["kill_penalty"],
        )

    params = gaengine.GAParams(
        pop_size=CONFIG["pop"],
        gens=CONFIG["gens"],
        elite=CONFIG["elite"],
        tournament=CONFIG["tournament"],
        crossover_rate=CONFIG["crossover_rate"],
        mutation_rate=CONFIG["mutation_rate"],
        blx_alpha=0.2,
        mut_sigma=0.1,
        seed=CONFIG["seed"],
        log_every=CONFIG["log_every"],
        binary=True,
        kill_copy_age=CONFIG["kill_age"],
        kill_penalty=CONFIG["kill_penalty"],
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
    Path(Path(CONFIG["out"]).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(CONFIG["out"], index=False)
    print(f"[GA] mode=ingredients  saved -> {CONFIG['out']}")

if __name__ == "__main__":
    main()
