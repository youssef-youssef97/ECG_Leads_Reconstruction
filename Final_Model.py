
import os, json, math, random, gc
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- config --------------------
DATASET = "ptbxl"
SEGMENTS_ROOT = Path("Segments")
DATA_DIR = SEGMENTS_ROOT / DATASET
CONTRASTIVE_REPS_ROOT = Path("out/contrastive_reps")
OUTPUT_DIR = Path(f"Recons_{DATASET}_Clean_h")

# create outputs
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "preds").mkdir(parents=True, exist_ok=True)

INPUT_LEADS = ["I", "II", "V2"]
TARGET_LEADS = ["V1", "V3", "V4", "V5", "V6"]
ALL_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
SEGMENT_LENGTH = 256

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SHARD_PATTERN = "all_shard_*.npy"
EPS = 1e-8

# -------------------- utils --------------------
def list_fold_dirs(base_split_dir):
    if not base_split_dir.exists(): return []
    return sorted([p for p in base_split_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])

def discover_rep_meta_from_shards(shard_paths):
    if not shard_paths: return {"shards": [], "num_samples": 0, "sample_shape": None}
    seen = []
    total = 0
    sample_shape = None
    for p in shard_paths:
        if p in seen: continue
        seen.append(p)
        arr = np.load(p, mmap_mode="r")
        if sample_shape is None:
            sample_shape = list(arr.shape[1:]) if arr.ndim>1 else [int(arr.shape[1]) if arr.ndim>1 else 1]
        total += int(arr.shape[0])
        del arr
    return {"shards": seen, "num_samples": int(total), "sample_shape": sample_shape}

def find_rep_shards_for_split(rep_name, split):
    base = CONTRASTIVE_REPS_ROOT / rep_name
    split_dir = base / split

    candidates = []
    if split_dir.exists():
        candidates = sorted([str(p) for p in split_dir.rglob(SHARD_PATTERN)])
        if not candidates:
            candidates = sorted([str(p) for p in split_dir.rglob("*.npy")])
    else:
        matches = list(base.rglob(split))
        for m in matches:
            if m.is_dir():
                candidates = sorted([str(p) for p in Path(m).rglob(SHARD_PATTERN)])
                if not candidates:
                    candidates = sorted([str(p) for p in Path(m).rglob("*.npy")])
                if candidates:
                    break

    if not candidates and base.exists():
        candidates = sorted([str(p) for p in base.rglob(f"*{split}*") if p.is_file() and p.suffix == ".npy"])

    return discover_rep_meta_from_shards(candidates) if candidates else None

def collect_shards_for_foldnames(base_dir: Path, fold_names):
    shards = []
    for fn in fold_names:
        d = base_dir / fn
        if d.exists():
            shards.extend(sorted([str(p) for p in d.glob(SHARD_PATTERN)]))
        else:
            matches = list(base_dir.rglob(fn))
            for m in matches:
                if m.is_dir():
                    shards.extend(sorted([str(p) for p in Path(m).glob(SHARD_PATTERN)]))
    shards = sorted(list(dict.fromkeys(shards)))
    return shards

# -------------------- dataset helpers --------------------
def load_sample_from_shards(shard_paths, idx):
    acc = 0
    for p in shard_paths:
        arr = np.load(p, mmap_mode="r")
        l = int(arr.shape[0])
        if idx < acc + l:
            local = idx - acc
            s = np.array(arr[local], dtype=np.float32); del arr
            return s
        del arr
        acc += l
    if len(shard_paths) == 0:
        return np.zeros((SEGMENT_LENGTH, len(ALL_LEADS)), dtype=np.float32)
    arr0 = np.load(shard_paths[0], mmap_mode="r")
    out = np.zeros(arr0.shape[1:], dtype=np.float32); del arr0
    return out

class MultiRepDataset(torch.utils.data.Dataset):
    def __init__(self, rep_info_map, y_shard_paths, reps, target_lead, N_max=None):
        self.rep_info_map = rep_info_map; self.reps = reps; self.target_lead = target_lead
        N_all = min(int(rep_info_map[r]["num_samples"]) for r in reps)
        self.N = min(N_all, N_max) if N_max is not None else N_all
        self.y_shards = y_shard_paths; self.rep_shards = {r: rep_info_map[r]["shards"] for r in reps}
    def __len__(self): return int(self.N)
    def __getitem__(self, idx):
        sample = {}
        for r in self.reps:
            shards = self.rep_shards[r]; arr = load_sample_from_shards(shards, idx)
            sample[r] = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        seg = load_sample_from_shards(self.y_shards, idx)
        if seg.ndim == 2 and seg.shape[1] == len(ALL_LEADS):
            y = seg[:, ALL_LEADS.index(self.target_lead)]
        elif seg.ndim == 2 and seg.shape[0] == len(ALL_LEADS):
            y = seg[ALL_LEADS.index(self.target_lead), :]
        else:
            y = np.array(seg).ravel()[:SEGMENT_LENGTH]
        return sample, np.asarray(y, dtype=np.float32)

# -------------------- collate / dataset stats / normalization --------------------
def normalize_arr(a):
    denom = a.std() + 1e-8
    return ((a - a.mean()) / denom).astype(np.float32) if denom != 0 else a.astype(np.float32)

def _ensure_channels_time(arr):
    a = np.array(arr, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        if a.shape[0] == SEGMENT_LENGTH and a.shape[1] != SEGMENT_LENGTH:
            return a.T
        if a.shape[1] == SEGMENT_LENGTH and a.shape[0] != SEGMENT_LENGTH:
            return a
        return a
    # fallback
    return a.reshape(a.shape[0], -1)

def collate_batch(batch, reps, rep_stats, training=False):
    batch_inputs = {r: [] for r in reps}; batch_y = []
    for sample, y in batch:
        for r in reps:
            a = sample.get(r)
            if a is None:
                a = np.zeros((len(INPUT_LEADS), SEGMENT_LENGTH), dtype=np.float32)
            # normalize shape to (channels, seq_len)
            a = _ensure_channels_time(a)

            # special handling for clean_signal to select input leads
            if r == "clean_signal":
                lead_indices = [ALL_LEADS.index(l) for l in INPUT_LEADS]
                if a.shape[0] == len(ALL_LEADS):  # (12,256)
                    a = a[lead_indices, :]
                elif a.shape[1] == len(ALL_LEADS):  # (256,12)
                    a = a[:, lead_indices].T

            # dataset-level normalization if stats provided for this rep
            stats = rep_stats.get(r)
            if stats is not None:
                mean = stats["mean"]
                std = stats["std"]
                # try to broadcast correctly
                if mean.size == 1:
                    a = (a - float(mean)) / (float(std) + EPS)
                else:
                    # ensure channel dimension matches
                    if a.shape[0] != mean.shape[0] and a.shape[1] == mean.shape[0]:
                        a = a.T
                    if a.shape[0] == mean.shape[0]:
                        a = (a - mean.reshape(-1,1)) / (std.reshape(-1,1) + EPS)
                    else:
                        # fallback to per-sample normalize if shapes don't align
                        a = normalize_arr(a)
            else:
                if r == "clean_signal":
                    a = normalize_arr(a)
            batch_inputs[r].append(a)
        batch_y.append(y)
    coll = {}
    for r in reps:
        arrs = np.stack(batch_inputs[r], axis=0).astype(np.float32)
        coll[r] = torch.tensor(arrs)
    y_t = torch.tensor(np.stack(batch_y, axis=0).astype(np.float32))
    return coll, y_t

def _shard_array_to_nCT(shard_arr):
    a = shard_arr
    if a.ndim == 3:
        N, d1, d2 = a.shape
        if d1 == SEGMENT_LENGTH:
            # (N, seq_len, channels)
            return a.transpose(0,2,1)  # -> (N, C, T)
        elif d2 == SEGMENT_LENGTH:
            # (N, channels, seq_len)
            return a
        else:
            # fallback assume (N, C, T)
            return a
    elif a.ndim == 2:
        N, d1 = a.shape
        if d1 == SEGMENT_LENGTH:
            # (N, seq_len) treat as (N, 1, seq_len)
            return a.reshape(N, 1, d1)
        else:
            # (N, channels) treat as (N, C, 1)
            return a.reshape(N, d1, 1)
    elif a.ndim == 1:
        N = a.shape[0]
        return a.reshape(N, 1, 1)
    else:
        # unknown
        return a.reshape(a.shape[0], -1, 1)

def compute_rep_dataset_stats(shard_paths):
    sums = None; sqs = None; total_pts = 0
    for p in tqdm(shard_paths, desc="Computing stats", leave=False):
        arr = np.load(p, mmap_mode="r")
        nCT = _shard_array_to_nCT(arr)  # (N, C, T)
        N, C, T = nCT.shape
        # sum over N and T -> per-channel sums
        s = nCT.sum(axis=(0,2)).astype(np.float64)
        s2 = (nCT.astype(np.float64) ** 2).sum(axis=(0,2))
        pts = N * T
        if sums is None:
            sums = s
            sqs = s2
        else:
            sums += s
            sqs += s2
        total_pts += pts
        del arr, nCT
    if total_pts == 0:
        # fallback scalar zeros
        return {"mean": np.array([0.0], dtype=np.float32), "std": np.array([1.0], dtype=np.float32)}
    mean = sums / float(total_pts)
    var = (sqs / float(total_pts)) - (mean ** 2)
    var[var < 0] = 0.0
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

# -------------------- models --------------------
class SimplePerRepProj(nn.Module):
    def __init__(self, in_ch, per_branch_dim, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, per_branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(per_branch_dim),
            nn.GELU(),
            nn.Conv1d(per_branch_dim, per_branch_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        ); self.seq_len = seq_len
    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] != self.seq_len:
            out = nn.functional.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out

class StackedLatentDecoder(nn.Module):
    def __init__(self, reps, rep_in_ch_map, per_branch_dim=128, seq_len=SEGMENT_LENGTH):
        super().__init__()
        self.reps = list(reps); self.seq_len = seq_len; self.per_branch_dim = per_branch_dim
        self.proj = nn.ModuleDict()
        for r in self.reps:
            in_ch = int(rep_in_ch_map.get(r, 1)); in_ch = max(1, in_ch)
            self.proj[r] = SimplePerRepProj(in_ch, per_branch_dim, seq_len)
        self.fusion = nn.Conv1d(len(self.reps)*per_branch_dim, per_branch_dim, kernel_size=3, padding=1, bias=False)
        self.decoder = nn.Sequential(
            nn.Conv1d(per_branch_dim, max(32, per_branch_dim//2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(max(32, per_branch_dim//2), 1, kernel_size=1)
        )
    def forward(self, coll):
        B = None; feats = []
        for r in self.reps:
            x = coll.get(r)
            if x is None:
                x = torch.zeros((B if B is not None else 1, 1, self.seq_len), device=next(self.parameters()).device)
            if B is None: B = x.shape[0]
            if x.dim() == 2: x = x.unsqueeze(-1)
            if x.dim() == 3 and x.shape[-1] == self.seq_len and x.shape[1] == self.seq_len:
                x = x.permute(0,2,1)
            out = self.proj[r](x.float()); feats.append(out)
        cat = torch.cat(feats, dim=1); fused = self.fusion(cat); out = self.decoder(fused)
        return out.squeeze(1)

# -------------------- metrics / training --------------------
def pearson_mean(y_true, y_pred):
    vals = []
    for i in range(y_true.shape[0]):
        if np.std(y_true[i]) > 0:
            try: vals.append(pearsonr(y_true[i], y_pred[i])[0])
            except: vals.append(0.0)
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

def compute_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "pearson": 0.0}

    rmses = []
    r2s = []

    for i in range(y_true.shape[0]):
        yt = y_true[i].ravel()
        yp = y_pred[i].ravel()

        if yt.size == 0:
            continue

        rmses.append(math.sqrt(mean_squared_error(yt, yp)))

        try:
            r2_i = r2_score(yt, yp)
            if np.isfinite(r2_i):
                r2s.append(r2_i)
        except Exception:
            pass

    rmse = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
    r2 = float(np.mean(r2s)) if len(r2s) > 0 else float("nan")

    return {
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson_mean(y_true, y_pred),
    }

def train_model(model, train_loader, val_loader, epochs, lr, device, model_tag, report_fh):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf"); best_state = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=False)
    patience = 12; patience_ctr = 0
    for epoch in range(1, epochs+1):
        model.train(); epoch_loss = 0.0; n = 0
        with tqdm(train_loader, desc=f"Train {model_tag} Epoch {epoch}/{epochs}", leave=False) as tbar:
            for coll, y in tbar:
                for k in coll: coll[k] = coll[k].to(device)
                y = y.to(device)
                pred = model(coll)
                loss = 0.8*loss_fn(pred, y) + 0.2*torch.nn.functional.l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                bs = pred.shape[0]; epoch_loss += float(loss.item()) * bs; n += bs
                tbar.set_postfix({"loss": f"{(epoch_loss/max(1,n)):.6f}"})
        train_loss = epoch_loss / max(1, n)
        # validation
        model.eval(); val_loss_sum = 0.0; val_preds=[]; val_trues=[]
        with torch.no_grad():
            for coll, y in tqdm(val_loader, desc=f"Val {model_tag} Epoch {epoch}/{epochs}", leave=False):
                for k in coll: coll[k] = coll[k].to(device)
                y = y.to(device); pred = model(coll)
                val_loss_sum += float(nn.MSELoss()(pred, y).item()) * pred.shape[0]
                val_preds.append(pred.cpu().numpy()); val_trues.append(y.cpu().numpy())
        n_val = len(val_loader.dataset); val_loss = val_loss_sum / max(1, n_val)
        val_preds = np.vstack(val_preds) if len(val_preds)>0 else np.zeros((0, SEGMENT_LENGTH))
        val_trues = np.vstack(val_trues) if len(val_trues)>0 else np.zeros((0, SEGMENT_LENGTH))
        m = compute_metrics(val_trues, val_preds)
        line = f"[{model_tag}] Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_rmse={m['rmse']:.6f} val_r2={m['r2']:.6f} val_pear={m['pearson']:.6f}"
        print(line)
        if report_fh is not None: report_fh.write(line+"\n")
        try: scheduler.step(val_loss)
        except: pass
        if val_loss < best_val - 1e-9:
            best_val = val_loss; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping."); break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def predict_model_numpy(model, loader, device=DEVICE):
    model.eval(); preds=[]
    with torch.no_grad():
        for coll, y in loader:
            for k in coll: coll[k] = coll[k].to(device)
            out = model(coll).cpu().numpy(); preds.append(out)
    return np.vstack(preds) if len(preds)>0 else np.zeros((0, SEGMENT_LENGTH))

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    seg_root = DATA_DIR
    if (seg_root / "by_patient").exists():
        base_train = seg_root / "by_patient" / "train"
        base_val = seg_root / "by_patient" / "val"
        base_test = seg_root / "by_patient" / "test"
    else:
        base_train = seg_root / "train"
        base_val = seg_root / "val"
        base_test = seg_root / "test"

    # target fold names
    train_fold_names = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6", "fold_7", "fold_8"]
    val_fold_names = ["fold_9"]
    test_fold_names = ["fold_10"]

    # collect shard paths for each split based on requested folds
    train_shards = collect_shards_for_foldnames(base_train, train_fold_names)
    val_shards = collect_shards_for_foldnames(base_val, val_fold_names)
    test_shards = collect_shards_for_foldnames(base_test, test_fold_names)

    if len(train_shards) == 0:
        raise RuntimeError(f"No train shards found for folds {train_fold_names} under {base_train}")
    if len(val_shards) == 0:
        raise RuntimeError(f"No val shards found for folds {val_fold_names} under {base_val}")
    if len(test_shards) == 0:
        print(f"[WARN] No test shards found for folds {test_fold_names} under {base_test}. Proceeding with zero test set.")

    # build rep_info maps
    meta_train = discover_rep_meta_from_shards(train_shards)
    meta_val = discover_rep_meta_from_shards(val_shards)
    meta_test = discover_rep_meta_from_shards(test_shards)

    # find contrastive reps
    h_train = find_rep_shards_for_split("h_vectors", "train"); h_val = find_rep_shards_for_split("h_vectors", "val"); h_test = find_rep_shards_for_split("h_vectors", "test")
    z_train = find_rep_shards_for_split("z_vectors", "train"); z_val = find_rep_shards_for_split("z_vectors", "val"); z_test = find_rep_shards_for_split("z_vectors", "test")

    # rep info maps per split
    rep_info_train = {"clean_signal": meta_train}
    rep_info_val = {"clean_signal": meta_val}
    rep_info_test = {"clean_signal": meta_test}

    # (optional) add h/z into rep_info if found for those splits
    if h_train is not None: rep_info_train["h_vectors"] = h_train
    if h_val is not None: rep_info_val["h_vectors"] = h_val
    if h_test is not None: rep_info_test["h_vectors"] = h_test
    if z_train is not None: rep_info_train["z_vectors"] = z_train
    if z_val is not None: rep_info_val["z_vectors"] = z_val
    if z_test is not None: rep_info_test["z_vectors"] = z_test

    # infer rep input channels from sample_shape
    rep_in_ch_map = {}
    for r, m in rep_info_train.items():
        rep_in_ch_map["clean_signal"] = len(INPUT_LEADS)
        shape = m.get("sample_shape")

        if shape is None:
            rep_in_ch_map[r] = 1

        elif len(shape) == 2:
            if shape[0] == SEGMENT_LENGTH:
                rep_in_ch_map[r] = shape[1]
            elif shape[1] == SEGMENT_LENGTH:
                rep_in_ch_map[r] = shape[0]
            else:
                rep_in_ch_map[r] = min(shape)

        elif len(shape) == 1:
            rep_in_ch_map[r] = shape[0]

        else:
            rep_in_ch_map[r] = shape[0]

    # expanded reps used for training (clean_signal + h_vectors)
    expanded_reps = ["clean_signal", "h_vectors"]

    # compute counts
    N_train = int(rep_info_train["clean_signal"]["num_samples"])
    N_val = int(rep_info_val["clean_signal"]["num_samples"])
    N_test = int(rep_info_test["clean_signal"]["num_samples"]) if "clean_signal" in rep_info_test else 0
    print(f"Dataset sizes -> train={N_train} val={N_val} test={N_test}")

    # ---------------- compute dataset-level stats for reps in train ----------------
    rep_stats = {}
    # We'll compute stats for any rep present in rep_info_train (clean and h)
    for rep_name, info in rep_info_train.items():
        shards = info.get("shards", [])
        if shards:
            print(f"Computing dataset stats for rep: {rep_name} (num_shards={len(shards)})")
            rep_stats[rep_name] = compute_rep_dataset_stats(shards)
            # save stats for inspection
            np.save(out_dir / f"{rep_name}_train_mean.npy", rep_stats[rep_name]["mean"])
            np.save(out_dir / f"{rep_name}_train_std.npy", rep_stats[rep_name]["std"])
        else:
            print(f"No shards for rep {rep_name}, skipping stats.")
    # ------------------------------------------------------------------------------

    metrics_across_leads = {lead: None for lead in TARGET_LEADS}

    # loop per target lead
    for lead in TARGET_LEADS:
        print(f"\n--- Training for lead {lead} ---")
        report_path = out_dir / "reports" / f"evaluation_report_lead_{lead}.txt"
        with open(report_path, "w") as report:
            report.write(f"Fixed-splits training - train: {train_fold_names}, val: {val_fold_names}, test: {test_fold_names}\n")
            report.write(f"Reps: {expanded_reps}\n")
            report.write(f"Sizes: N_train={N_train} N_val={N_val} N_test={N_test}\n")

            ds_train = MultiRepDataset(rep_info_train, train_shards, expanded_reps, lead, N_max=None)
            ds_val   = MultiRepDataset(rep_info_val, val_shards, expanded_reps, lead, N_max=None)
            ds_test  = MultiRepDataset(rep_info_test, test_shards, expanded_reps, lead, N_max=None)

            report.write(f"N_train={len(ds_train)} N_val={len(ds_val)} N_test={len(ds_test)}\n")

            train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=lambda b: collate_batch(b, expanded_reps, rep_stats, training=True),
                                      num_workers=4, pin_memory=True)
            val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=lambda b: collate_batch(b, expanded_reps, rep_stats, training=False),
                                      num_workers=4, pin_memory=True)
            test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=lambda b: collate_batch(b, expanded_reps, rep_stats, training=False),
                                      num_workers=4, pin_memory=True)

            per_branch_dim_used = 128
            model = StackedLatentDecoder(expanded_reps, rep_in_ch_map, per_branch_dim=per_branch_dim_used, seq_len=SEGMENT_LENGTH)
            report.write(f"Model params: {sum(p.numel() for p in model.parameters())}\n")
            model_tag = f"Stacked_{lead}_fixed_splits"
            trained = train_model(model, train_loader, val_loader, args.epochs, LEARNING_RATE, DEVICE, model_tag, report)
            torch.save(trained.state_dict(), out_dir / "models" / f"{model_tag}.pt")

            preds_test = predict_model_numpy(trained, test_loader, device=DEVICE)
            # build y_test from test_shards (index locally 0..N_test-1)
            y_list = []
            for i in tqdm(range(len(ds_test)), desc=f"Building y_test for lead {lead}", leave=False):
                seg = load_sample_from_shards(test_shards, i)
                if seg.ndim == 2 and seg.shape[1] == len(ALL_LEADS):
                    y_list.append(seg[:, ALL_LEADS.index(lead)])
                elif seg.ndim == 2 and seg.shape[0] == len(ALL_LEADS):
                    y_list.append(seg[ALL_LEADS.index(lead), :])
                else:
                    y_list.append(np.array(seg).ravel()[:SEGMENT_LENGTH])
            y_test = np.stack(y_list, axis=0).astype(np.float32) if len(y_list)>0 else np.zeros((0, SEGMENT_LENGTH), dtype=np.float32)
            m = compute_metrics(y_test, preds_test)
            line = f"[FINAL_TEST] Lead {lead} RMSE={m['rmse']:.6f} R2={m['r2']:.6f} Pearson={m['pearson']:.6f}"
            print(line); report.write(line + "\n")
            metrics_across_leads[lead] = m

            out_pred_dir = out_dir / "preds" / f"{lead}_fixed_splits"; out_pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_pred_dir / f"y_test.npy", y_test); np.save(out_pred_dir / f"preds_test.npy", preds_test)

            # plots
            if y_test.shape[0] > 0:
                rmse_per_point = np.sqrt(np.mean((y_test - preds_test) ** 2, axis=0))
                plt.figure(figsize=(10,4)); plt.plot(rmse_per_point); plt.title(f"RMSE per timepoint - Lead {lead}")
                plt.xlabel("Time index"); plt.ylabel("RMSE"); plt.grid(True); plt.tight_layout()
                plt.savefig(out_dir / "plots" / f"rmse_lead_{lead}.png"); plt.close()
                n_show = min(8, y_test.shape[0])
                fig, axes = plt.subplots(n_show, 1, figsize=(10, 2*n_show))
                for i in range(n_show):
                    axes[i].plot(y_test[i], label="true"); axes[i].plot(preds_test[i], linestyle='--', label="pred")
                    axes[i].set_xlim(0, SEGMENT_LENGTH-1); axes[i].legend(fontsize=6)
                plt.tight_layout(); plt.savefig(out_dir / "plots" / f"{model_tag}_overlay_first{n_show}.png", dpi=200); plt.close()
            report.write(f"Saved predictions + plots for lead {lead} in {out_pred_dir}\n")
        print(f"Finished lead {lead}. Report: {report_path}")

    # summary
    summary_path = out_dir / "reports" / "evaluation_report_summary.txt"
    with open(summary_path, "w") as summ:
        summ.write("Fixed-splits Summary\n")
        for lead, m in metrics_across_leads.items():
            if m is None:
                summ.write(f"{lead}: no metrics collected\n"); continue
            line = (f"{lead}: RMSE={m['rmse']:.6f} | R2={m['r2']:.6f} | Pearson={m['pearson']:.6f}")
            print(line); summ.write(line + "\n")
    print("Run finished. Results in", out_dir)

if __name__ == "__main__":
    main()