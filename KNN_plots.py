#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
import warnings
import scipy.cluster.hierarchy as sch

warnings.filterwarnings("ignore", category=UserWarning)


# Helpers
def find_shards_under(base_dir: Path, splits=("train", "val", "test")):
    shards = []
    if not base_dir.exists():
        return []
    for sp in splits:
        p = base_dir / sp
        if p.exists():
            shards.extend(sorted([x for x in p.rglob("*.npy") if x.is_file()]))
    if not shards:
        shards = sorted([x for x in base_dir.rglob("*.npy") if x.is_file()])
    shards = [s for s in shards if not s.name.endswith("_vectors_meta.json")]
    return shards


def load_meta_single(meta_path: Path):
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        return [data]
    return []


def safe_get_label(meta):
    if not isinstance(meta, dict):
        return "UNK"
    return str(meta.get("label") or meta.get("primary_scp") or meta.get("super_class") or "UNK")


def load_embeddings_and_meta(shard_paths, max_samples=None):
    arrays = []
    metas = []
    total = 0
    for p in shard_paths:
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            continue
        nrows = int(arr.shape[0])
        meta_p = p.with_name(p.stem + "_meta.json")
        meta_list = load_meta_single(meta_p) if meta_p.exists() else []
        available = min(nrows, len(meta_list)) if len(meta_list) > 0 else nrows
        if available < nrows:
            print(f"[WARN] truncating {p.name}: npy_rows={nrows} meta_rows={len(meta_list)} -> using {available}")
        take = available
        if max_samples is not None and total + take > max_samples:
            take = max(0, max_samples - total)
        if take <= 0:
            break
        arrays.append(np.asarray(arr[:take]))
        if len(meta_list) >= take:
            metas.extend(meta_list[:take])
        else:
            metas.extend(meta_list + [{}] * (take - len(meta_list)))
        total += take
        if max_samples is not None and total >= max_samples:
            break
    if len(arrays) == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    X = np.concatenate(arrays, axis=0)
    return X, metas


def build_label_index_map(metas):
    idx_map = defaultdict(list)
    for i, m in enumerate(metas):
        lbl = safe_get_label(m)
        idx_map[lbl].append(i)
    return dict(idx_map)


def to_serializable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# k-NN affinity matrix
def compute_knn_affinity_matrix(X, metas, classes, sample_per_class=300, k=5, min_per_class=1, rng_seed=42, metric="cosine"):
    rng = np.random.RandomState(rng_seed)
    idx_map = build_label_index_map(metas)
    classes = [c for c in classes if c in idx_map and len(idx_map[c]) >= min_per_class]
    if len(classes) < 2:
        raise RuntimeError("Not enough classes with min_per_class.")

    sampled_idx = {}
    for c in classes:
        idxs = idx_map[c]
        if len(idxs) <= sample_per_class:
            chosen = idxs.copy()
        else:
            chosen = list(rng.choice(idxs, size=sample_per_class, replace=False))
        sampled_idx[c] = chosen

    sel_idxs = []
    sel_classes = []
    for c in classes:
        sel_idxs.extend(sampled_idx[c])
        sel_classes.extend([c] * len(sampled_idx[c]))

    if len(sel_idxs) == 0:
        raise RuntimeError("No samples selected.")

    Xsel = X[sel_idxs]
    le = {c: i for i, c in enumerate(classes)}
    y = np.array([le[c] for c in sel_classes], dtype=int)

    total_pts = Xsel.shape[0]
    neigh_n = min(k + 1, total_pts)
    nbrs = NearestNeighbors(n_neighbors=neigh_n, metric=metric, algorithm="auto", n_jobs=-1)
    nbrs.fit(Xsel)
    _, neighbors = nbrs.kneighbors(Xsel, return_distance=True)

    n_classes = len(classes)
    mat = np.zeros((n_classes, n_classes), dtype=float)

    for i in range(total_pts):
        neigh = neighbors[i]
        neigh_filtered = neigh[neigh != i]
        if neigh_filtered.shape[0] > k:
            neigh_filtered = neigh_filtered[:k]
        if neigh_filtered.shape[0] == 0:
            continue
        neigh_labels = y[neigh_filtered]
        for cls_idx in range(n_classes):
            frac = float((neigh_labels == cls_idx).sum()) / float(neigh_filtered.shape[0])
            mat[y[i], cls_idx] += frac

    counts_per_class = np.array([len(sampled_idx[c]) for c in classes], dtype=float)
    for i in range(n_classes):
        if counts_per_class[i] > 0:
            mat[i, :] = mat[i, :] / counts_per_class[i]
        else:
            mat[i, :] = np.nan

    df = pd.DataFrame(mat, index=classes, columns=classes, dtype=float)
    return df


# k-NN accuracy curve
def compute_knn_accuracy_curve(X, metas, classes, sample_per_class=300, k_max=10, min_per_class=1, rng_seed=42, metric="cosine"):
    rng = np.random.RandomState(rng_seed)
    idx_map = build_label_index_map(metas)
    classes = [c for c in classes if c in idx_map and len(idx_map[c]) >= min_per_class]
    if len(classes) < 2:
        raise RuntimeError("Not enough classes with min_per_class for accuracy calc.")

    sampled_idx = {}
    for c in classes:
        idxs = idx_map[c]
        if len(idxs) <= sample_per_class:
            chosen = idxs.copy()
        else:
            chosen = list(rng.choice(idxs, size=sample_per_class, replace=False))
        sampled_idx[c] = chosen

    sel_idxs = []
    sel_classes = []
    for c in classes:
        sel_idxs.extend(sampled_idx[c])
        sel_classes.extend([c] * len(sampled_idx[c]))

    if len(sel_idxs) == 0:
        raise RuntimeError("No samples selected for accuracy calculation.")

    Xsel = X[sel_idxs]
    le = {c: i for i, c in enumerate(classes)}
    y = np.array([le[c] for c in sel_classes], dtype=int)
    total_pts = Xsel.shape[0]

    neigh_n = min(k_max + 1, total_pts)
    nbrs = NearestNeighbors(n_neighbors=neigh_n, metric=metric, algorithm="auto", n_jobs=-1)
    nbrs.fit(Xsel)
    _, neighbors = nbrs.kneighbors(Xsel, return_distance=True)

    accuracies = {}
    k_values = list(range(1, k_max + 1))
    for k in k_values:
        correct = 0
        for i in range(total_pts):
            neigh = neighbors[i]
            neigh_filtered = neigh[neigh != i]
            if neigh_filtered.shape[0] > k:
                neigh_filtered = neigh_filtered[:k]
            if neigh_filtered.shape[0] == 0:
                continue
            neigh_labels = y[neigh_filtered]
            counts = np.bincount(neigh_labels, minlength=len(classes))
            pred = int(np.argmax(counts))
            if pred == y[i]:
                correct += 1
        accuracy = float(correct) / float(total_pts) if total_pts > 0 else float("nan")
        accuracies[k] = accuracy

    details = {
        "k_values": k_values,
        "accuracies": [accuracies[k] for k in k_values],
        "total_points": total_pts,
        "classes": classes,
    }
    return accuracies, details


# Plots
def plot_accuracy_curve(accuracies_dict, out_png, title=None):
    ks = sorted(accuracies_dict.keys())
    accs = [accuracies_dict[k] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, accs, marker="o", linewidth=2)
    plt.xticks(ks)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    if title:
        plt.title(title, fontsize=18, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_heatmap_numeric_summary(df: pd.DataFrame):
    data = df.values.astype(float)
    n = data.shape[0]
    overall_mean = float(np.nanmean(data))
    diag = np.diag(data) if n > 0 else np.array([])
    diag_mean = float(np.nanmean(diag)) if diag.size > 0 else float("nan")
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_diag_mean = float(np.nanmean(data[mask]))
    else:
        off_diag_mean = float("nan")
    row_means = np.nanmean(data, axis=1)
    row_mean_excl = []
    purity = []
    row_sums = []
    for i in range(n):
        row = data[i, :]
        row_sum = float(np.nansum(row))
        row_sums.append(row_sum)
        self_val = float(row[i]) if not np.isnan(row[i]) else float("nan")
        if n > 1:
            excl = np.nanmean(np.delete(row, i))
        else:
            excl = float("nan")
        row_mean_excl.append(excl)
        purity_val = float(self_val / row_sum) if (row_sum and not np.isnan(self_val)) else float("nan")
        purity.append(purity_val)
    per_class_df = pd.DataFrame(
        {
            "row_mean_including_diag": row_means,
            "row_mean_excl_diag": np.array(row_mean_excl, dtype=float),
            "self_affinity": diag,
            "row_sum": np.array(row_sums, dtype=float),
            "purity_diag_over_row_sum": np.array(purity, dtype=float),
        },
        index=df.index,
    )
    mean_row_mean = float(np.nanmean(row_means)) if row_means.size > 0 else float("nan")
    mean_row_mean_excl = float(np.nanmean(per_class_df["row_mean_excl_diag"])) if n > 0 else float("nan")
    diag_vs_off_ratio = float(diag_mean / (off_diag_mean + 1e-12)) if (not np.isnan(diag_mean) and not np.isnan(off_diag_mean)) else float("nan")
    summary = {
        "overall_mean": overall_mean,
        "diag_mean": diag_mean,
        "off_diag_mean": off_diag_mean,
        "mean_row_mean_including_diag": mean_row_mean,
        "mean_row_mean_excl_diag": mean_row_mean_excl,
        "diag_vs_off_diag_ratio": diag_vs_off_ratio,
        "n_classes": n,
    }
    return summary, per_class_df


def plot_heatmap(df, out_png, title=None, cmap="viridis", vmin=None, vmax=None, reorder=False):
    labels = list(df.index)
    data = df.values.astype(float)
    if reorder:
        filled = data.copy()
        col_mean = np.nanmean(filled, axis=0)
        inds = np.where(np.isnan(filled))
        for r, c in zip(*inds):
            filled[r, c] = 0.0 if np.isnan(col_mean[c]) else col_mean[c]
        try:
            linkage = sch.linkage(filled, method="average", metric="euclidean")
            order = sch.leaves_list(linkage)
            data = data[order][:, order]
            labels = [labels[i] for i in order]
        except Exception:
            order = list(range(len(labels)))
    plt.figure(figsize=(14, 12))
    plt.imshow(data, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=14)
    plt.yticks(ticks, labels, fontsize=14)
    if title is not None:
        plt.title(title, fontsize=25, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# Combined plot
def plot_combined_accuracy(outdir, tag1="h", tag2="clean", out_name="combined_knn_accuracy.png"):
    p1 = Path(outdir) / tag1 / "knn_accuracy.csv"
    p2 = Path(outdir) / tag2 / "knn_accuracy.csv"
    missing = []
    if not p1.exists():
        missing.append(str(p1))
    if not p2.exists():
        missing.append(str(p2))
    if missing:
        print(f"[WARN] cannot create combined accuracy plot; missing: {', '.join(missing)}")
        return
    try:
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        plt.figure(figsize=(6, 4))
        plt.plot(df1["k"], df1["accuracy"], marker="o", linewidth=2, label=tag1)
        plt.plot(df2["k"], df2["accuracy"], marker="o", linewidth=2, label=tag2)
        all_ks = sorted(set(df1["k"].tolist() + df2["k"].tolist()))
        plt.xticks(all_ks)
        plt.ylim(0.0, 1.0)
        plt.xlabel("k (number of neighbors)")
        plt.ylabel("Accuracy")
        plt.title(f"Combined k-NN accuracy: {tag1} vs {tag2}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        out_png = Path(outdir) / out_name
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[INFO] saved combined accuracy plot to {out_png}")
    except Exception as e:
        print(f"[WARN] failed saving combined accuracy plot: {e}")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dir", type=str, default="./out/contrastive_reps/h_vectors")
    parser.add_argument("--segments_root", type=str, default="./Segments/ptbxl")
    parser.add_argument("--splits", type=str, default="test", help="comma-separated splits to analyze (e.g. test or train,val,test)")
    parser.add_argument("--max_load", type=int, default=80000)
    parser.add_argument("--sample_per_class", type=int, default=300)
    parser.add_argument("--min_per_class", type=int, default=1)
    parser.add_argument("--per_class_plot", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="./knn_plots")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--normalize", type=str, default="l2", choices=["none", "zscore", "l2"], help="normalize embeddings before metrics")
    parser.add_argument("--reorder_heatmap", action="store_true", help="reorder heatmaps with hierarchical clustering")
    parser.add_argument("--knn_k", type=int, default=10, help="k for k-NN affinity heatmap")
    parser.add_argument("--accuracy_k_max", type=int, default=10, help="maximum k (inclusive) for k-NN accuracy curve")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.random_state)

    h_base = Path(args.h_dir)
    print(f"[INFO] searching h shards under splits: {splits}")
    h_shards = find_shards_under(h_base, splits=splits)
    if len(h_shards) == 0:
        print("[ERR] no h shards found.")
        return
    print(f"[INFO] loading up to {args.max_load} h vectors ...")
    Xh, metas_h = load_embeddings_and_meta(h_shards, max_samples=args.max_load)
    if Xh.size == 0 or len(metas_h) == 0:
        print("[ERR] failed to load h.")
        return
    idx_map_h = build_label_index_map(metas_h)
    classes_all_h = sorted(idx_map_h.keys(), key=lambda c: len(idx_map_h[c]), reverse=True)
    print(f"[INFO] h representation: found {len(classes_all_h)} classes.")

    rep_configs = [
        ("h", Path(args.h_dir), Xh, metas_h),
        ("clean", Path(args.segments_root), None, None),
    ]

    for tag, base, Xref, metas_ref in rep_configs:
        print(f"\n[INFO] ==== PROCESSING representation: {tag} ====")
        rep_out = outdir / tag
        rep_out.mkdir(parents=True, exist_ok=True)

        if tag == "h":
            Xrep, metas_rep = Xref, metas_ref
        else:
            seg_shards = find_shards_under(base, splits=splits)
            if len(seg_shards) == 0:
                print(f"[WARN] no segment shards under {base}, skipping clean.")
                continue
            arrays = []
            metas = []
            total = 0
            for p in seg_shards:
                try:
                    arr = np.load(p, mmap_mode="r")
                except Exception:
                    continue
                nrows = int(arr.shape[0])
                meta_p = p.with_name(p.stem + "_meta.json")
                meta_list = load_meta_single(meta_p) if meta_p.exists() else []
                available = min(nrows, len(meta_list)) if len(meta_list) > 0 else nrows
                if available < nrows:
                    print(f"[WARN] clean meta shorter than npy for {p.name} -> trunc {available}")
                take = available
                if args.max_load is not None and total + take > args.max_load:
                    take = max(0, args.max_load - total)
                if take <= 0:
                    break
                arr_slice = arr[:take]
                if arr_slice.ndim == 3 and arr_slice.shape[2] >= 12:
                    sel = arr_slice[:, :, [0, 1, 7]].astype(np.float32)
                    sel = np.transpose(sel, (0, 2, 1))
                elif arr_slice.ndim == 3 and arr_slice.shape[1] >= 12:
                    sel = arr_slice[:, [0, 1, 7], :].astype(np.float32)
                else:
                    sel = np.zeros((arr_slice.shape[0], 3, 256), dtype=np.float32)
                flat = sel.reshape(sel.shape[0], -1)
                arrays.append(flat)
                if len(meta_list) >= take:
                    metas.extend(meta_list[:take])
                else:
                    metas.extend(meta_list + [{}] * (take - len(meta_list)))
                total += take
                if args.max_load is not None and total >= args.max_load:
                    break
            if len(arrays) == 0:
                print("[WARN] no clean arrays loaded.")
                continue
            Xrep = np.concatenate(arrays, axis=0)
            metas_rep = metas

        if Xrep.shape[0] != len(metas_rep):
            mn = min(Xrep.shape[0], len(metas_rep))
            Xrep = Xrep[:mn]
            metas_rep = metas_rep[:mn]

        if args.normalize == "zscore":
            scaler = StandardScaler()
            try:
                Xrep = scaler.fit_transform(Xrep)
            except Exception:
                Xrep = Xrep
        elif args.normalize == "l2":
            try:
                Xrep = normalize(Xrep, norm="l2")
            except Exception:
                Xrep = Xrep

        idx_map = build_label_index_map(metas_rep)
        classes_all = sorted(idx_map.keys(), key=lambda c: len(idx_map[c]), reverse=True)

        classes_selected = [c for c in classes_all if len(idx_map[c]) >= args.min_per_class]
        if len(classes_selected) < 2:
            print(f"[WARN] not enough classes in {tag} meeting min_per_class ({args.min_per_class}). Found {len(classes_selected)}. Skipping.")
            continue
        print(f"[INFO] {tag}: {len(classes_all)} total classes, {len(classes_selected)} meet min_per_class")

        print(f"[INFO] computing k-NN affinity matrix (k={args.knn_k}) for {tag} ...")
        try:
            mats = compute_knn_affinity_matrix(
                Xrep,
                metas_rep,
                classes_selected,
                sample_per_class=args.sample_per_class,
                k=args.knn_k,
                min_per_class=args.min_per_class,
                rng_seed=args.random_state,
                metric="cosine",
            )
            class_counts = {c: len(idx_map[c]) for c in mats.index}
            sorted_classes = sorted(mats.index, key=lambda c: class_counts[c])
            mats = mats.loc[sorted_classes, sorted_classes]
        except Exception as e:
            print(f"[ERR] failed computing k-NN matrix for {tag}: {e}")
            continue

        try:
            print(f"[INFO] computing k-NN accuracy curve (k=1..{args.accuracy_k_max}) for {tag} ...")
            accuracies, acc_details = compute_knn_accuracy_curve(
                Xrep,
                metas_rep,
                classes_selected,
                sample_per_class=args.sample_per_class,
                k_max=args.accuracy_k_max,
                min_per_class=args.min_per_class,
                rng_seed=args.random_state,
                metric="cosine",
            )
            acc_csv_path = rep_out / "knn_accuracy.csv"
            acc_df = pd.DataFrame(
                {
                    "k": acc_details["k_values"],
                    "accuracy": acc_details["accuracies"],
                }
            )
            acc_df.to_csv(acc_csv_path, index=False)

            acc_txt_path = rep_out / "knn_accuracy.txt"
            with open(acc_txt_path, "w", encoding="utf-8") as fh:
                fh.write(f"Total sampled points: {acc_details['total_points']}\n")
                fh.write("k,accuracy\n")
                for k in acc_details["k_values"]:
                    fh.write(f"{k},{accuracies[k]:.6f}\n")

            acc_png = rep_out / "knn_accuracy.png"
            plot_accuracy_curve(accuracies, acc_png, title=f"{tag} k-NN accuracy")
            print(f"[INFO] saved k-NN accuracy CSV to {acc_csv_path}, text to {acc_txt_path}, plot to {acc_png}")
        except Exception as e:
            print(f"[WARN] failed computing/saving k-NN accuracy for {tag}: {e}")

        try:
            summary, per_class_df = compute_heatmap_numeric_summary(mats)
            stats_json_path = rep_out / "pairwise_knn_affinity_stats.json"
            with open(stats_json_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False, default=to_serializable)
            stats_csv_path = rep_out / "pairwise_knn_affinity_per_class.csv"
            per_class_df.to_csv(stats_csv_path, index=True)
            print(f"[INFO] saved numeric heatmap summary to {stats_json_path} and per-class CSV to {stats_csv_path}")
        except Exception as e:
            print(f"[WARN] failed computing/saving numeric heatmap stats for {tag}: {e}")

        npy_path = rep_out / "pairwise_knn_affinity.npy"
        np.save(npy_path, mats.values.astype(np.float32))

        labels_path = rep_out / "pairwise_knn_affinity_labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(list(mats.index), f, indent=2)

        print(f"[INFO] saved heatmap matrix to {npy_path}")

        vmin = 0.0
        vmax = 1.0
        pngp = rep_out / "pairwise_knn_affinity.png"
        try:
            plot_heatmap(mats, pngp, title=f"{tag} (k={args.knn_k})", cmap="Reds", vmin=vmin, vmax=vmax, reorder=args.reorder_heatmap)
            print(f"[INFO] saved heatmap to {pngp}")
        except Exception as e:
            print(f"[WARN] failed plotting heatmap for {tag}: {e}")

        sel_summary = {
            "representation": tag,
            "splits": splits,
            "normalize": args.normalize,
            "min_per_class": args.min_per_class,
            "sample_per_class": args.sample_per_class,
            "per_class_plot": args.per_class_plot,
            "classes_total": len(classes_all),
            "classes_meet_min_per_class": len(classes_selected),
            "outputs": {
                "pairwise_knn_png": str(pngp.name),
                "pairwise_knn_stats_json": str("pairwise_knn_affinity_stats.json"),
                "pairwise_knn_per_class_csv": str("pairwise_knn_affinity_per_class.csv"),
                "knn_accuracy_csv": str("knn_accuracy.csv"),
                "knn_accuracy_txt": str("knn_accuracy.txt"),
                "knn_accuracy_png": str("knn_accuracy.png"),
            },
        }
        with open(rep_out / "selection_summary.json", "w", encoding="utf-8") as fh:
            json.dump(sel_summary, fh, indent=2, ensure_ascii=False, default=to_serializable)

        print(f"[INFO] finished {tag}; outputs in {rep_out}")

    try:
        plot_combined_accuracy(outdir, tag1="h", tag2="clean", out_name="combined_knn_accuracy.png")
    except Exception as e:
        print(f"[WARN] combined accuracy plotting failed: {e}")

    print("\n[DONE] All requested reps processed. Check the outdir for PNG heatmaps and k-NN accuracy outputs.")


if __name__ == "__main__":
    main()