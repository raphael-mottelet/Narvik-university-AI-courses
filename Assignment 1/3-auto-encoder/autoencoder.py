import argparse
from aetrain import run_pipeline

# I put my argumetns here because code starting to get too messy

DEF_INPUT  = "../data/mlready/step13-vector-full.csv"
DEF_OUTDIR = "../models/autoencoder"
DEF_LATENT = "../data/mlready/X_latent.csv"
DEF_RECON  = "../data/mlready/X_recon_full.csv"

def parse_args():
    ap = argparse.ArgumentParser(description="Autoencoder + health classifier")
    ap.add_argument("--input", default=DEF_INPUT)
    ap.add_argument("--outdir", default=DEF_OUTDIR)
    ap.add_argument("--latents", default=DEF_LATENT)
    ap.add_argument("--recon", default=DEF_RECON)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--hidden", nargs="*", type=int, default=[64, 32])
    ap.add_argument("--health-head-hidden", nargs="*", type=int, default=[])
    ap.add_argument("--val-split", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lambda-health-mse", type=float, default=1.0)
    ap.add_argument("--lambda-health-ce", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10, help="print every N epochs")
    return ap.parse_args()

def main():
    a = parse_args()
    run_pipeline(
        input_csv=a.input, outdir=a.outdir, latents_csv=a.latents, recon_csv=a.recon,
        epochs=a.epochs, batch=a.batch, zdim=a.zdim, hidden=a.hidden,
        health_head_hidden=a.health_head_hidden, val_split=a.val_split, lr=a.lr,
        patience=a.patience, lambda_health_mse=a.lambda_health_mse,
        lambda_health_ce=a.lambda_health_ce, seed=a.seed, log_every=a.log_every,
    )

if __name__ == "__main__":
    main()
