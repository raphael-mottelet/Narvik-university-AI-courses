from __future__ import annotations
import os
import json
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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

DEF_CORE = os.path.join("..", "data", "mlready", "step12-vector-core.csv")
DEF_ZCSV = os.path.join("..", "data", "mlready", "X_latent.csv")
DEF_AEDIR = os.path.join("..", "models", "autoencoder")
DEF_OUTCORE = os.path.join("..", "data", "mlready", "ga-core.csv")
DEF_OUTLAT = os.path.join("..", "data", "mlready", "ga-latent.csv")


def load_autoencoder(ae_dir: str):
    cfg_path = Path(ae_dir) / "config.json"
    pt_path = Path(ae_dir) / "autoencoder.pt"
    if not cfg_path.exists() or not pt_path.exists():
        raise FileNotFoundError("Autoencoder files not found. Train AE first (config.json, autoencoder.pt).")

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    cols = cfg["columns"]
    zdim = int(cfg["z_dim"])
    hidden = list(cfg["hidden"])

    aemodel = _load_module("01-aemodel.py", "aemodel")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = aemodel.Autoencoder(
        in_dim=len(cols),
        hidden=hidden,
        z_dim=zdim,
        health_head_hidden=cfg.get("health_head_hidden", []),
    )
    ae.load_state_dict(torch.load(pt_path, map_location=device))
    ae.to(device).eval()

    idx_taste = cols.index("taste_balance")
    idx_price = cols.index("price_norm")
    idx_carbon = cols.index("carbon_norm") if "carbon_norm" in cols else None

    return ae, device, zdim, idx_taste, idx_price, idx_carbon


def decode_z_to_core(ae, device, idx_taste: int, idx_price: int, Z: np.ndarray, idx_carbon: int | None = None):
    with torch.no_grad():
        z_t = torch.from_numpy(Z.astype(np.float32)).to(device)
        d = ae.decoder(z_t) if hasattr(ae, "decoder") else z_t
        xhat_n = ae.out_act(ae.dec_out(d))
        logits = ae.health_head(z_t)
        health_pred = (logits.argmax(dim=1) + 1).detach().cpu().numpy().astype(int)
        xhat_n = xhat_n.detach().cpu().numpy()

    taste = xhat_n[:, idx_taste]
    price = xhat_n[:, idx_price]
    if idx_carbon is not None:
        carbon = xhat_n[:, idx_carbon]
    else:
        carbon = np.zeros(Z.shape[0], dtype=float)
    return taste, price, health_pred, carbon


def main():
    ap = argparse.ArgumentParser(description="Run GA in core or latent space")
    ap.add_argument("--mode", choices=["core", "latent"], default="core")

    ap.add_argument("--wt", type=float, default=1.0)
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--wh", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1.0)

    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--tour", type=int, default=3)
    ap.add_argument("--cx", type=float, default=0.9)
    ap.add_argument("--mut", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)

    ap.add_argument("--core-csv", default=DEF_CORE)
    ap.add_argument("--z-csv", default=DEF_ZCSV)
    ap.add_argument("--ae-dir", default=DEF_AEDIR)
    ap.add_argument("--out-core", default=DEF_OUTCORE)
    ap.add_argument("--out-latent", default=DEF_OUTLAT)
    ap.add_argument("--topk", type=int, default=50)

    args = ap.parse_args()

    if args.mode == "core":
        df = gadata.load_core(args.core_csv)
        bounds = [(0.0, 1.0), (0.0, 1.0), (1.0, 5.0), (0.0, 1.0)]
        init_pop = gadata.seed_core_population(df, args.pop, seed=args.seed)

        def fit_fn(chrom):
            return gadata.fitness_core(chrom, wt=args.wt, wc=args.wc, wh=args.wh, wd=args.wd)

        params = gaengine.GAParams(
            pop_size=args.pop,
            gens=args.gens,
            elite=args.elite,
            tournament=args.tour,
            crossover_rate=args.cx,
            mutation_rate=args.mut,
            blx_alpha=args.alpha,
            mut_sigma=args.sigma,
            seed=args.seed,
            log_every=args.log_every,
        )
        ga = gaengine.GA(fitness_fn=fit_fn, bounds=bounds, params=params, init_pop=init_pop)
        best, best_fit, _ = ga.run()

        order = np.argsort(-ga.fit)[: max(1, min(args.topk, ga.pop.shape[0]))]
        top = ga.pop[order]
        out = pd.DataFrame(
            {
                "taste_balance": np.clip(top[:, 0], 0.0, 1.0),
                "price_norm": np.clip(top[:, 1], 0.0, 1.0),
                "health_score": np.rint(np.clip(top[:, 2], 1.0, 5.0)).astype(int),
                "carbon_norm": np.clip(top[:, 3], 0.0, 1.0),
            }
        )
        Path(Path(args.out_core).parent).mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_core, index=False)
        print(f"[GA] mode=core  saved -> {args.out_core}")

    else:
        ae, device, zdim, idx_taste, idx_price, idx_carbon = load_autoencoder(args.ae_dir)

        if os.path.isfile(args.z_csv):
            Zall = pd.read_csv(args.z_csv).to_numpy(dtype=float)
            if Zall.shape[1] != zdim:
                raise ValueError(f"Z dim mismatch: file has {Zall.shape[1]}, AE expects {zdim}")
        else:
            rng = np.random.default_rng(args.seed)
            Zall = rng.normal(0.0, 1.0, size=(max(1000, args.pop), zdim))

        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(Zall), size=args.pop)
        init_pop = Zall[idx].copy()

        z_lo = Zall.min(axis=0)
        z_hi = Zall.max(axis=0)
        margin = 0.05 * (z_hi - z_lo + 1e-6)
        bounds = list(zip((z_lo - margin).tolist(), (z_hi + margin).tolist()))

        def fit_fn(z_vec):
            z = np.asarray(z_vec, dtype=float).reshape(1, -1)
            taste, price, health, carbon = decode_z_to_core(ae, device, idx_taste, idx_price, z, idx_carbon)
            taste_v = float(np.clip(taste[0], 0.0, 1.0))
            price_v = float(np.clip(price[0], 0.0, 1.0))
            health_v = int(health[0])
            carbon_v = float(np.clip(carbon[0], 0.0, 1.0))
            h_u = float(gadata.health_to_unit(health_v))
            return args.wt * taste_v + args.wc * (1.0 - price_v) + args.wh * h_u - args.wd * carbon_v

        params = gaengine.GAParams(
            pop_size=args.pop,
            gens=args.gens,
            elite=args.elite,
            tournament=args.tour,
            crossover_rate=args.cx,
            mutation_rate=args.mut,
            blx_alpha=args.alpha,
            mut_sigma=args.sigma,
            seed=args.seed,
            log_every=args.log_every,
        )
        ga = gaengine.GA(fitness_fn=fit_fn, bounds=bounds, params=params, init_pop=init_pop)
        best, best_fit, _ = ga.run()

        order = np.argsort(-ga.fit)[: max(1, min(args.topk, ga.pop.shape[0]))]
        Ztop = ga.pop[order]
        taste, price, health, carbon = decode_z_to_core(ae, device, idx_taste, idx_price, Ztop, idx_carbon)

        out = pd.DataFrame(Ztop, columns=[f"z{i+1}" for i in range(Ztop.shape[1])])
        out["taste_balance"] = np.clip(taste, 0.0, 1.0)
        out["price_norm"] = np.clip(price, 0.0, 1.0)
        out["health_score"] = np.asarray(health, dtype=int)
        out["carbon_norm"] = np.clip(carbon, 0.0, 1.0)

        Path(Path(args.out_latent).parent).mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_latent, index=False)
        print(f"[GA] mode=latent  saved -> {args.out_latent}")


if __name__ == "__main__":
    main()
