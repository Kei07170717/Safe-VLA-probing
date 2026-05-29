"""Linear probes on pi-0.5 activations predicting WHICH obstacle is present.

Companion to train_probes.py. Where that script predicted a per-episode human
safety rating (regression), this one predicts the *object identity* -- which of
the six declared obstacles is actually present in the episode's scene -- as a
6-way classification, per layer, per activation stream.

It also (optionally) builds a CLEAN per-object knock/collision signal: instead of
the all-obstacles OR, it looks only at the obstacle that is actually present in
that episode (per your label), ignoring the five phantom columns.

CSV format expected (your safety_ratings.csv):
    a leading junk column, real header on row 2, columns:
        episode_id, task_success, safety_rating, object present
    where 'object present' is a code 1-6:
        1=moka_pot 2=white_storage_box 3=milk
        4=wine_bottle 5=red_coffee_mug 6=yellow_book

Run from the analysis dir:
    python train_probe_object.py --acts-dir acts_task1 --gt-dir groundtruth --ratings safety_ratings.csv
"""
import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
import ml_dtypes
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# code -> obstacle name (matches BDDL object prefixes and parquet column stems)
OBJECT_CODE_TO_NAME = {
    1: "moka_pot",
    2: "white_storage_box",
    3: "milk",
    4: "wine_bottle",
    5: "red_coffee_mug",
    6: "yellow_book",
}


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


def load_ratings(path):
    """Parse the messy CSV: leading junk col + header on second row."""
    r = pd.read_csv(path, skiprows=1)
    r = r.loc[:, ~r.columns.str.contains("Unnamed")]
    # normalize column names
    r.columns = [c.strip() for c in r.columns]
    keep = ["episode_id", "task_success", "safety_rating", "object present"]
    keep = [c for c in keep if c in r.columns]
    r = r[keep].dropna(subset=["episode_id"])
    r["episode_id"] = r["episode_id"].astype(int)
    if "task_success" in r:
        r["task_success"] = r["task_success"].astype(str).str.strip().str.lower()
    if "safety_rating" in r:
        r["safety_rating"] = pd.to_numeric(r["safety_rating"], errors="coerce")
    r = r.rename(columns={"object present": "object_code"})
    r["object_code"] = pd.to_numeric(r["object_code"], errors="coerce").astype("Int64")
    r["object_name"] = r["object_code"].map(OBJECT_CODE_TO_NAME)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge alpha (knock regression)")
    ap.add_argument("--C", type=float, default=1.0, help="logistic-regression inverse reg")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--out-prefix", default="probe_object")
    args = ap.parse_args()

    # --- ground truth ---
    gt_files = sorted(glob.glob(f"{args.gt_dir}/*.parquet"))
    gt = pd.concat([pd.read_parquet(f) for f in gt_files], ignore_index=True)
    gt = gt.sort_values("global_infer_id").reset_index(drop=True)
    print(f"Loaded {len(gt)} GT rows from {len(gt_files)} episodes")

    # --- ratings / object labels ---
    ratings = load_ratings(args.ratings)
    print(f"Loaded labels for {len(ratings)} episodes")
    print("object distribution:")
    print(ratings["object_name"].value_counts().to_string())

    if args.success_only:
        before = len(ratings)
        ratings = ratings[ratings["task_success"].isin(["s", "success"])]
        print(f"--success-only: kept {len(ratings)}/{before} episodes")

    # join object label onto every step
    gt = gt.merge(ratings[["episode_id", "object_code", "object_name", "safety_rating"]],
                  on="episode_id", how="inner")
    gt = gt.sort_values("global_infer_id").reset_index(drop=True)
    print(f"After join: {len(gt)} rows / {gt['episode_id'].nunique()} episodes")

    # --- clean per-object knock signal: only the present obstacle's column ---
    # for each row, look up {object_name}_obstacle_1_knocked
    def relevant_knock(row):
        col = f"{row['object_name']}_obstacle_1_knocked"
        return bool(row[col]) if col in gt.columns else np.nan
    gt["relevant_knocked"] = gt.apply(relevant_knock, axis=1)
    n_relknock = gt["relevant_knocked"].sum(skipna=True)
    print(f"relevant_knocked positive rate: {gt['relevant_knocked'].mean():.3f} "
          f"({int(n_relknock)}/{len(gt)})")
    if "is_colliding" in gt.columns:
        print(f"(for reference) is_colliding positive rate: {gt['is_colliding'].mean():.3f}")

    # --- load activations aligned by global_infer_id ---
    act_files = sorted(glob.glob(f"{args.acts_dir}/act_*.npz"))
    if not act_files:
        raise SystemExit(f"No activations in {args.acts_dir}")
    first = load_activation(act_files[0])
    streams = list(first.keys())
    n_layers = first[streams[0]].shape[0]
    print(f"streams={streams} shapes={[first[s].shape for s in streams]}")

    N = len(gt)
    acts = {s: np.zeros((N, *first[s].shape), dtype=np.float32) for s in streams}
    missing = 0
    for i, row in enumerate(gt.itertuples()):
        p = pathlib.Path(args.acts_dir) / f"act_{int(row.global_infer_id):05d}.npz"
        if not p.exists():
            missing += 1
            continue
        a = load_activation(p)
        for s in streams:
            acts[s][i] = a[s]
    if missing:
        print(f"WARNING: {missing} activation files missing")

    groups = gt["episode_id"].values
    y_obj = gt["object_code"].astype(int).values            # 1..6 classification
    y_knock = gt["relevant_knocked"].astype(float).values   # 0/1 clean knock

    # episode-level split (shared across all targets for comparability)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    train_idx, test_idx = next(gss.split(np.zeros(N), y_obj, groups))
    print(f"Train {len(train_idx)} steps / {len(np.unique(groups[train_idx]))} eps; "
          f"Test {len(test_idx)} steps / {len(np.unique(groups[test_idx]))} eps")

    # baseline accuracy for the object task = predicting the majority class
    from collections import Counter
    maj = Counter(y_obj[test_idx]).most_common(1)[0][1] / len(test_idx)
    print(f"object-id majority-class baseline (test): {maj:.3f}\n")

    results = {s: {"obj_acc": [], "obj_balacc": [], "knock_auc": []} for s in streams}
    steering = {}

    for s in streams:
        best_balacc, best_layer, best_w = -np.inf, None, None
        for layer in range(n_layers):
            X = acts[s][:, layer, :]
            Xtr, Xte = X[train_idx], X[test_idx]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            # ---- object identity: 6-way logistic regression ----
            clf = LogisticRegression(max_iter=2000, C=args.C, multi_class="multinomial")
            clf.fit(Xtr_s, y_obj[train_idx])
            pred = clf.predict(Xte_s)
            acc = accuracy_score(y_obj[test_idx], pred)
            balacc = balanced_accuracy_score(y_obj[test_idx], pred)
            results[s]["obj_acc"].append(acc)
            results[s]["obj_balacc"].append(balacc)

            if balacc > best_balacc:
                best_balacc, best_layer = balacc, layer
                # store the per-class weight matrix as the "object directions"
                best_w = clf.coef_ / scaler.scale_  # (n_classes, width)

        steering[s] = {"layer": best_layer, "w": best_w, "balacc": best_balacc}
        print(f"[{s}] OBJECT-ID best layer={best_layer} balacc={best_balacc:.3f}")
        print("  per-layer acc:    " + " ".join(f"{a:.2f}" for a in results[s]["obj_acc"]))
        print("  per-layer balacc: " + " ".join(f"{a:.2f}" for a in results[s]["obj_balacc"]))
        print()

    np.savez(f"{args.out_prefix}_directions.npz",
             **{f"{s}_w": steering[s]["w"] for s in streams},
             **{f"{s}_layer": np.array(steering[s]["layer"]) for s in streams},
             object_code_to_name=np.array(
                 [OBJECT_CODE_TO_NAME[i] for i in range(1, 7)], dtype=object))
    print(f"Saved object directions to {args.out_prefix}_directions.npz")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for s in streams:
            ax.plot(range(n_layers), results[s]["obj_balacc"], marker="o", label=f"{s} (balacc)")
        ax.axhline(1 / 6, color="gray", ls="--", lw=1, label="chance (1/6)")
        ax.set_title("Object-identity probe vs layer" +
                     (" (success-only)" if args.success_only else ""))
        ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_layers.png", dpi=120)
        print(f"Saved plot to {args.out_prefix}_layers.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()