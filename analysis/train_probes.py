"""Linear probes on pi-0.5 activations predicting human safety ratings.

Loads per-inference activations (prefix + action streams, 18 layers each),
joins per-episode safety ratings, trains ridge-regression probes per layer,
reports test R^2 and Spearman, and plots probe quality vs layer.

Run from vlsa-aegis/:
    python train_probes.py --acts-dir /tmp/acts_task1 --gt-dir data/libero/groundtruth --ratings safety_ratings.csv
"""
import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
import ml_dtypes
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit


def load_activation(path):
    """Load one .npz, decode bfloat16 -> float32. Returns dict of arrays."""
    d = np.load(path)
    out = {}
    for k in d.keys():
        a = d[k]
        if a.dtype == np.dtype("V2"):  # bfloat16 stored as void-2
            a = a.view(ml_dtypes.bfloat16).astype(np.float32)
        else:
            a = a.astype(np.float32)
        out[k] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge regularization")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="probe")
    args = ap.parse_args()

    # --- Load ground truth, concatenate all episodes ---
    gt_files = sorted(glob.glob(f"{args.gt_dir}/*.parquet"))
    gt = pd.concat([pd.read_parquet(f) for f in gt_files], ignore_index=True)
    gt = gt.sort_values("global_infer_id").reset_index(drop=True)
    print(f"Loaded {len(gt)} ground-truth rows from {len(gt_files)} episodes")

    # --- Load ratings, clean the messy header/columns ---
    ratings = pd.read_csv(args.ratings, skiprows=1)
    ratings = ratings.loc[:, ~ratings.columns.str.contains("Unnamed")]
    ratings = ratings[["episode_id", "task_success", "safety_rating"]].dropna(subset=["safety_rating"])
    ratings["episode_id"] = ratings["episode_id"].astype(int)
    ratings["safety_rating"] = ratings["safety_rating"].astype(float)
    print(f"Loaded ratings for {len(ratings)} episodes, range "
          f"{ratings['safety_rating'].min():.0f}-{ratings['safety_rating'].max():.0f}")

    # --- Join ratings onto every step by episode_id ---
    gt = gt.merge(ratings[["episode_id", "safety_rating"]], on="episode_id", how="inner")
    print(f"After join: {len(gt)} rows with safety ratings")

    # --- Load activations, aligned by global_infer_id ---
    # discover shapes from first file
    first = load_activation(sorted(glob.glob(f"{args.acts_dir}/act_*.npz"))[0])
    streams = list(first.keys())  # e.g. ['hidden_prefix', 'hidden_action']
    print(f"Activation streams: {streams}, shapes: {[first[s].shape for s in streams]}")
    n_layers = first[streams[0]].shape[0]

    # build arrays: for each stream, (N, n_layers, width)
    N = len(gt)
    acts = {s: np.zeros((N, *first[s].shape), dtype=np.float32) for s in streams}
    missing = 0
    for i, row in enumerate(gt.itertuples()):
        gid = int(row.global_infer_id)
        path = pathlib.Path(args.acts_dir) / f"act_{gid:05d}.npz"
        if not path.exists():
            missing += 1
            continue
        a = load_activation(path)
        for s in streams:
            acts[s][i] = a[s]
    if missing:
        print(f"WARNING: {missing} activation files missing")

    y = gt["safety_rating"].values
    groups = gt["episode_id"].values  # split by episode to avoid leakage

    # --- Episode-level train/test split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(gss.split(np.zeros(N), y, groups))
    print(f"Train: {len(train_idx)} steps / {len(np.unique(groups[train_idx]))} episodes; "
          f"Test: {len(test_idx)} steps / {len(np.unique(groups[test_idx]))} episodes")

    # --- Per-stream, per-layer probe ---
    results = {s: {"r2": [], "spearman": []} for s in streams}
    steering_vectors = {}  # (stream, best_layer) -> weight vector

    for s in streams:
        best_r2, best_layer, best_w = -np.inf, None, None
        for layer in range(n_layers):
            X = acts[s][:, layer, :]  # (N, width)
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            probe = Ridge(alpha=args.alpha).fit(Xtr_s, ytr)
            pred = probe.predict(Xte_s)

            ss_res = np.sum((yte - pred) ** 2)
            ss_tot = np.sum((yte - yte.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rho = spearmanr(yte, pred).correlation if len(np.unique(yte)) > 1 else 0.0

            results[s]["r2"].append(r2)
            results[s]["spearman"].append(rho)

            if r2 > best_r2:
                best_r2, best_layer = r2, layer
                # steering direction in original (unscaled) space:
                # probe weights are in standardized space; map back via 1/scale
                best_w = probe.coef_ / scaler.scale_

        steering_vectors[s] = {"layer": best_layer, "w": best_w, "r2": best_r2}
        print(f"\n[{s}] best layer = {best_layer}, test R2 = {best_r2:.3f}")
        print(f"  per-layer R2: " + " ".join(f"{r:.2f}" for r in results[s]["r2"]))
        print(f"  per-layer rho: " + " ".join(f"{r:.2f}" for r in results[s]["spearman"]))

    # --- Save steering vectors ---
    np.savez(f"{args.out_prefix}_steering.npz",
             **{f"{s}_w": steering_vectors[s]["w"] for s in streams},
             **{f"{s}_layer": steering_vectors[s]["layer"] for s in streams})
    print(f"\nSaved steering vectors to {args.out_prefix}_steering.npz")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for s in streams:
            axes[0].plot(range(n_layers), results[s]["r2"], marker="o", label=s)
            axes[1].plot(range(n_layers), results[s]["spearman"], marker="o", label=s)
        axes[0].set_title("Probe R² vs layer"); axes[0].set_xlabel("layer"); axes[0].set_ylabel("test R²")
        axes[1].set_title("Spearman ρ vs layer"); axes[1].set_xlabel("layer"); axes[1].set_ylabel("test ρ")
        axes[0].axhline(0, color="gray", lw=0.5); axes[1].axhline(0, color="gray", lw=0.5)
        for ax in axes: ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_layers.png", dpi=120)
        print(f"Saved plot to {args.out_prefix}_layers.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()