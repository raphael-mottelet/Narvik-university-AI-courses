from __future__ import annotations
import time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aemodel import Autoencoder
from aedata import (
    HEALTH, load_csv, fit_spec, apply_spec, invert_spec, set_seed, batches,
    compute_class_weights_1to5, balanced_indices_per_epoch
)

def run_pipeline(
    input_csv: str,
    outdir: str,
    latents_csv: str,
    recon_csv: str,
    epochs: int = 400,
    batch: int = 256,
    zdim: int = 8,
    hidden: list[int] = (64, 32),
    health_head_hidden: list[int] | None = None,
    val_split: float = 0.10,
    lr: float = 1e-3,
    patience: int = 20,
    lambda_health_mse: float = 1.0,
    lambda_health_ce: float = 4.0,
    seed: int = 42,
    log_every: int = 10,
) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(Path(latents_csv).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(recon_csv).parent).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_csv(input_csv)
    cols = list(df.columns)
    if HEALTH not in cols:
        raise KeyError("Input CSV must contain 'health_score' (integers 1..5).")

    spec = fit_spec(df)
    Xn = apply_spec(df, spec).astype(np.float32)

    hp = pd.to_numeric(df[HEALTH], errors="coerce").round().clip(1,5).astype(int)
    class_w, counts = compute_class_weights_1to5(hp)
    class_w_vec = torch.tensor([class_w[k] for k in [1,2,3,4,5]], dtype=torch.float32, device=device)

    # Split
    n = Xn.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(n * val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Xva = Xn[tr_idx], Xn[val_idx]
    Htr, Hva = hp.to_numpy()[tr_idx], hp.to_numpy()[val_idx]

    # Model/opt/loss
    in_dim = Xn.shape[1]
    ae = Autoencoder(in_dim, list(hidden), zdim, health_head_hidden=health_head_hidden or []).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=class_w_vec)
    health_idx = cols.index(HEALTH)

    def total_loss(x_hat, x, health_logits, labels_0to4):
        diff2 = (x_hat - x) ** 2
        base = diff2.mean()
        health_mse = diff2[:, health_idx].mean()
        ce = ce_loss(health_logits, labels_0to4)
        return base + lambda_health_mse * health_mse + lambda_health_ce * ce, base.item(), health_mse.item(), ce.item()

    best_loss, best_epoch, best_state = float("inf"), -1, None
    no_improve = 0

    # --- Train ---
    for ep in range(1, epochs + 1):
        ae.train()
        # Balanced oversampling each epoch
        sample_idx = balanced_indices_per_epoch(Htr, rng)
        Xtr_ep = Xtr[sample_idx]
        Htr_ep = Htr[sample_idx]
        # Shuffle
        perm = rng.permutation(len(Xtr_ep))
        Xtr_ep, Htr_ep = Xtr_ep[perm], Htr_ep[perm]

        tr_total = tr_base = tr_hmse = tr_ce = 0.0
        for i in range(0, len(Xtr_ep), batch):
            xb = Xtr_ep[i:i+batch]
            yb = (Htr_ep[i:i+batch] - 1).astype(np.int64)  # 0..4
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            opt.zero_grad(set_to_none=True)
            x_hat, z, h_logits = ae(xb_t)
            loss, base, hmse, ce = total_loss(x_hat, xb_t, h_logits, yb_t)
            loss.backward()
            opt.step()

            n_b = xb.shape[0]
            tr_total += loss.item() * n_b
            tr_base  += base * n_b
            tr_hmse  += hmse * n_b
            tr_ce    += ce * n_b

        denom = max(1, len(Xtr_ep))
        tr_total /= denom; tr_base /= denom; tr_hmse /= denom; tr_ce /= denom

        # Validation
        ae.eval()
        with torch.no_grad():
            va_t = torch.from_numpy(Xva).to(device)
            yv_t = torch.from_numpy((Hva - 1).astype(np.int64)).to(device)
            xh, zv, hv_logits = ae(va_t)
            va_total, va_base, va_hmse, va_ce = total_loss(xh, va_t, hv_logits, yv_t)
            hv_pred = (hv_logits.argmax(dim=1) + 1).detach().cpu().numpy()
            health_acc_val = float((hv_pred == Hva).mean())
            va_total = float(va_total)

        # Minimal logging: every log_every epochs and on improvement
        if (ep % log_every == 0) or (va_total + 1e-9 < best_loss) or (ep == 1):
            print(f"[{ep:03d}] train={tr_total:.6f}  val={va_total:.6f}  val_health_acc={health_acc_val:.4f}")

        if va_total + 1e-9 < best_loss:
            best_loss, best_epoch = va_total, ep
            best_state = {k: v.detach().cpu() for k, v in ae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {ep} (best {best_epoch})")
                break

    # Save best model
    ae.load_state_dict(best_state)
    torch.save(ae.state_dict(), Path(outdir, "autoencoder.pt"))

    # --- Inference on full dataset ---
    ae.eval()
    Z_list, Xhat_list, Hlogits_list = [], [], []
    with torch.no_grad():
        for xb in batches(Xn, 4096):
            xb_t = torch.from_numpy(xb).to(device)
            xh, z, hlog = ae(xb_t)
            Z_list.append(z.cpu().numpy())
            Xhat_list.append(xh.cpu().numpy())
            Hlogits_list.append(hlog.cpu().numpy())
    Z = np.vstack(Z_list)
    Xhat_n = np.vstack(Xhat_list)
    Hlogits = np.vstack(Hlogits_list)

    recon_rmse = float(np.sqrt(((Xn - Xhat_n) ** 2).mean()))
    recon_mae  = float(np.abs(Xn - Xhat_n).mean())

    # Save latents
    pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])]).to_csv(latents_csv, index=False)

    # Invert to original units
    Xrec = invert_spec(Xhat_n, cols, spec)

    # Health from classifier argmax (keeps your fix)
    h_pred = Hlogits.argmax(axis=1) + 1
    Xrec[:, cols.index(HEALTH)] = h_pred
    df_rec = pd.DataFrame(Xrec, columns=cols)
    df_rec[HEALTH] = df_rec[HEALTH].astype(int)
    df_rec.to_csv(recon_csv, index=False)

    # Full-dataset health accuracy
    health_acc_full = float((hp.to_numpy() == h_pred).mean())

    # Config dump
    cfg = dict(
        data=input_csv, outdir=outdir, latents=latents_csv, recon=recon_csv,
        input_dim=in_dim, columns=cols, hidden=list(hidden), z_dim=zdim,
        health_head_hidden=list(health_head_hidden or []),
        epochs=epochs, batch=batch, lr=lr, val_split=val_split, patience=patience,
        lambda_health_mse=lambda_health_mse, lambda_health_ce=lambda_health_ce,
        class_weights={str(k): float(v) for k, v in class_w.items()},
        class_counts=counts, recon_rmse=recon_rmse, recon_mae=recon_mae,
        health_accuracy=health_acc_full, seconds=round(time.time() - t0, 2),
        scaler_spec=spec,
    )
    Path(outdir, "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Final concise summary
    print(f"[OK] best_epoch={best_epoch}  recon_RMSE={recon_rmse:.6f}  recon_MAE={recon_mae:.6f}  health_acc={health_acc_full:.4f}")
    print(f"[OK] Latents -> {latents_csv}")
    print(f"[OK] Model   -> {Path(outdir, 'autoencoder.pt')}")
    print(f"[OK] Recon   -> {recon_csv}")
