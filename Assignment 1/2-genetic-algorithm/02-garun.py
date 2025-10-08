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
    "full_csv": os.path.join("..", "data", "openfoodfacts", "step13-vector-full.csv"),  # input vectors with ingredients_tags + taste/price/health/carbon
    "ingredients_col": "",                                                               # leave empty to auto-detect; set a column name to force it
    "out": os.path.join("..", "data", "mlready", "ga-output.csv"),                       # output file path (kept exactly as you asked)

    "wt": 1.5,    # weight for taste (higher = prioritize taste)
    "wc": 1.0,    # weight for cost via (1 - price_norm) (higher = prioritize cheaper)
    "wh": 1.0,    # weight for health (A→1.0 … E→0.0) (higher = prioritize healthier)
    "wd": 1.0,    # weight for environmental impact penalty (higher = penalize carbon more)

    "pop": 200,       # population size
    "gens": 300,      # generations (epochs)
    "elite": 2,       # number of elites copied directly each generation
    "tournament": 3,  # tournament size for parent selection (3 is a mild pressure)
    "crossover_rate": 0.9,  # probability of crossover (binary uniform)
    "mutation_rate": 0.06,  # mutation probability per gene (lower to avoid blowing up bit-count)

    "seed": 42,        # RNG seed for reproducibility
    "log_every": 10,   # print progress every N generations

    "min_count": 10,   # ignore very-rare tokens (<10 occurrences) to drop noise like 'birmingham', 'vegetali'
    "top_vocab": 200,  # keep only the top-N most frequent tokens after filtering to stabilize combos
    "min_k": 4,        # minimum ingredients per combo
    "max_k": 10,       # hard cap on ingredients (violations get a huge negative fitness)
    "soft_cap": 8,     # soft discouragement threshold (above this, apply soft_penalty)
    "soft_penalty": 2.0,  # strength of quadratic penalty for k > soft_cap (pushes toward 6–8 items)

    "banlist": "",        # optional path to a text file with banned “a + b + c” patterns (one per line); empty disables
    "kill_age": 20,       # if the same champion persists this many generations, kill its clones (diversity protection)
    "kill_penalty": 1e12, # magnitude for hard penalties (size violations, banned combos, clone-kill)

    "blacklist": {        # tokens to exclude from the vocabulary (normalize language noise, additives, packaging terms)
        "packet", "birmingham", "aro", "mam", "vegetali", "vegetale", "palme", "palmfat",
        "palmkernel", "palmiste", "palmist", "oleomargarina", "buttert", "sweetner", "sugart",
        "ricestarch", "cornstrach", "cornstarch", "seasalt", "unsweetened", "monooleate",
        "monoestearate", "stearoyl", "stearate", "tartrazina", "molybdate", "guanylate",
        "aspartamo", "menaquinone", "curcumine", "glicerol", "glicerolo", "glicerina",
        "shortening", "butterscotch", "brownie", "waffle", "cupcake", "cheesecake",
        "vegetal", "palma", "palme", "palma", "papa", "patata", "pellet", "seedless"
    }, 
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
