
# usage:
# python extract_contrastive_reps_ptbxl.py --checkpoint ./contrastive_out/best_encoder.pt --out ./out/contrastive_reps --batch 256 --device cuda

import argparse, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# params
SEGMENTS_ROOT = Path("./Segments")
LEAD_INDICES = [0, 1, 7]
BATCH = 256
HIDDEN = 256
PROJ_DIM = 128
SEGMENT_LEN = 256

# model
class ResNet1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
        else:
            self.down = nn.Identity()
    def forward(self, x):
        r = self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)

class ResNet1DEncoder(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, proj_dim=PROJ_DIM, depth=4):
        super().__init__()
        layers = []
        ch = 32
        layers.append(nn.Conv1d(in_ch, ch, 7, padding=3))
        layers.append(nn.BatchNorm1d(ch)); layers.append(nn.ReLU()); layers.append(nn.MaxPool1d(2))
        for i in range(depth):
            layers.append(ResNet1DBlock(ch, ch*2 if i%2==0 else ch))
            if i%2==0:
                ch = ch*2
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.body = nn.Sequential(*layers)
        self.hidden = ch
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, proj_dim)
        )
    def forward(self, x):
        h = self.body(x).squeeze(-1)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z
    def encode_h(self, x):
        h = self.body(x).squeeze(-1)
        return h

# ckpt loader
def robust_load_checkpoint(path):
    ck = torch.load(str(path), map_location="cpu")
    if isinstance(ck, dict):
        for k in ("model_state","model_state_dict","state_dict","model"):
            if k in ck:
                return ck[k]
        return ck
    return ck

# list shards preserving relative folder tree
def list_shards_for_split(dataset, split):
    base = SEGMENTS_ROOT / dataset
    if not base.exists():
        return []
    shards = []
    if dataset.lower() == "ptbxl":
        # search under train/fold_*/ etc or val/test
        if split in ("train","training"):
            fold_dirs = [base/"train"/f"fold_{i}" for i in range(1,9)]
        elif split in ("val","validation"):
            fold_dirs = [base/"val"/"fold_9"]
        elif split in ("test","testing"):
            fold_dirs = [base/"test"/"fold_10"]
        else:
            # custom split -> try direct folder
            candidate = base / split
            fold_dirs = [candidate] if candidate.exists() else []
        for fd in fold_dirs:
            if not fd.exists(): continue
            found = sorted([p for p in fd.glob("all_shard_*.npy")]) or sorted([p for p in fd.glob("*.npy")])
            for s in found:
                shards.append(s)
    else:
        cand = base / split
        if cand.exists():
            found = sorted([p for p in cand.glob("all_shard_*.npy")]) or sorted([p for p in cand.glob("*.npy")])
            shards = found
    return shards

# find meta for shard
def meta_for_shard(shard_path: Path):
    meta = shard_path.with_name(shard_path.stem + "_meta.json")
    if meta.exists(): return meta
    candidates = list(shard_path.parent.glob(shard_path.stem + "*_meta.json"))
    if candidates: return candidates[0]
    any_meta = list(shard_path.parent.glob("*_meta.json"))
    return any_meta[0] if any_meta else None

# process single shard -> produce matching outputs
def process_shard_one2one(shard_path: Path, encoder, out_base: Path, dataset: str, split: str,
                          batch_size= BATCH, device="cpu", save_h=True, save_z=True):
    # prepare mirror dirs
    rel = shard_path.relative_to(SEGMENTS_ROOT / dataset)
    out_h_dir = out_base / "h_vectors" / split / rel.parent
    out_z_dir = out_base / "z_vectors" / split / rel.parent
    out_h_dir.mkdir(parents=True, exist_ok=True)
    out_z_dir.mkdir(parents=True, exist_ok=True)

    # load data & meta
    try:
        arr = np.load(str(shard_path), mmap_mode="r")
    except Exception as e:
        return {"shard": str(shard_path), "status": "error", "reason": f"cannot_load_shard: {e}"}
    meta_p = meta_for_shard(shard_path)
    metas = None
    if meta_p is not None:
        try:
            with open(meta_p, "r", encoding="utf-8") as fh:
                metas = json.load(fh)
        except Exception:
            metas = None
    N = int(arr.shape[0])
    if metas is not None:
        n_meta = len(metas)
        n_local = min(N, n_meta)
    else:
        n_local = N

    device = torch.device(device)
    encoder.to(device)
    encoder.eval()

    # buffers for shard-level outputs
    h_out = []
    z_out = []
    out_meta_list = []

    # batch infer on this shard only
    start = 0
    with torch.no_grad():
        while start < n_local:
            end = min(start + batch_size, n_local)
            batch = arr[start:end]
            # select leads -> (B, C, T)
            if batch.ndim == 3 and batch.shape[2] >= max(LEAD_INDICES)+1:
                sel = batch[:, :, LEAD_INDICES].astype(np.float32)
                sel = np.transpose(sel, (0, 2, 1))
            elif batch.ndim == 3 and batch.shape[1] >= max(LEAD_INDICES)+1:
                sel = batch[:, LEAD_INDICES, :].astype(np.float32)
            else:
                Bf = batch.shape[0]
                sel = np.zeros((Bf, len(LEAD_INDICES), SEGMENT_LEN), dtype=np.float32)
            x = torch.from_numpy(sel).to(device=device)
            h = encoder.encode_h(x) if save_h else None
            z = encoder(x) if save_z else None
            if save_h:
                h_out.append(h.cpu().numpy().astype(np.float32))
            if save_z:
                z_out.append(z.cpu().numpy().astype(np.float32))
            # build meta entries for these local samples
            for li in range(start, end):
                m = {}
                if metas is not None and li < len(metas):
                    meta_entry = metas[li]
                    m["rec_id"] = meta_entry.get("rec_id") or meta_entry.get("record_id") or None
                    m["label"] = meta_entry.get("super_class") or meta_entry.get("primary_scp") or None
                    m["primary_vector"] = meta_entry.get("primary_vector", None)
                else:
                    m["rec_id"] = None
                    m["label"] = None
                    m["primary_vector"] = None
                m["shard"] = str(shard_path.name)
                m["local_idx"] = int(li)
                out_meta_list.append(m)
            start = end

    # stack and save
    saved = {"shard": str(shard_path), "saved_h": None, "saved_z": None, "meta": None}
    if save_h:
        if len(h_out) > 0:
            h_array = np.concatenate(h_out, axis=0)
        else:
            h_array = np.zeros((0, encoder.hidden), dtype=np.float32)
        outp = out_h_dir / shard_path.name
        np.save(str(outp), h_array)
        saved["saved_h"] = str(outp)
        # save meta
        meta_outp = outp.with_name(outp.stem + "_meta.json")
        with open(meta_outp, "w", encoding="utf-8") as fh:
            json.dump(out_meta_list, fh, ensure_ascii=False)
        saved["meta"] = str(meta_outp)

    if save_z:
        if len(z_out) > 0:
            z_array = np.concatenate(z_out, axis=0)
        else:
            z_array = np.zeros((0, PROJ_DIM), dtype=np.float32)
        outp_z = out_z_dir / shard_path.name
        np.save(str(outp_z), z_array)
        saved["saved_z"] = str(outp_z)
        meta_z_outp = outp_z.with_name(outp_z.stem + "_meta.json")
        with open(meta_z_outp, "w", encoding="utf-8") as fh:
            json.dump(out_meta_list, fh, ensure_ascii=False)

    return {"shard": str(shard_path), "status": "ok", "saved": saved, "n_samples": n_local}

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ptbxl")
    parser.add_argument("--checkpoint", type=str, default="./contrastive_out/best_encoder.pt")
    parser.add_argument("--out", type=str, default="./out/contrastive_reps")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-h", action="store_true", default=True)
    parser.add_argument("--save-z", action="store_true", default=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # build encoder and load weights
    encoder = ResNet1DEncoder(in_ch=len(LEAD_INDICES), hidden=HIDDEN, proj_dim=PROJ_DIM, depth=4)
    state = robust_load_checkpoint(Path(args.checkpoint))
    try:
        encoder.load_state_dict(state, strict=True)
    except Exception:
        fixed = {}
        for k,v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            fixed[nk] = v
        encoder.load_state_dict(fixed, strict=False)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    summary = {"processed_shards": 0, "errors": [], "per_split": {}}
    for sp in splits:
        shards = list_shards_for_split(args.dataset, sp)
        summary["per_split"][sp] = {"num_input_shards": len(shards), "num_output_shards": 0, "samples": 0}
        if not shards:
            print(f"[INFO] no shards found for split {sp}, skipping.")
            continue
        print(f"[INFO] split {sp}: {len(shards)} input shard files")
        for s in tqdm(shards, desc=f"processing {sp}"):
            res = process_shard_one2one(s, encoder, out_base, args.dataset, sp,
                                        batch_size=args.batch, device=device, save_h=args.save_h, save_z=args.save_z)
            if res.get("status") != "ok":
                summary["errors"].append(res)
            else:
                summary["processed_shards"] += 1
                summary["per_split"][sp]["num_output_shards"] += 1
                summary["per_split"][sp]["samples"] += res.get("n_samples", 0)
        print(f"[INFO] done split {sp}. produced {summary['per_split'][sp]['num_output_shards']} shard outputs")
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
