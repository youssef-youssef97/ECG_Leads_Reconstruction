"""
Usage
 python extract_contrastive_reps_ptb.py \
    --checkpoint ./contrastive_out/best_encoder.pt \
    --out ./out/contrastive_reps_ptb \
    --batch 256 \
    --device cuda \
    --leads 0 1 7

"""
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Default params
# ----------------------------
SEGMENTS_ROOT = Path("./Segments")
DEFAULT_DATASET = "ptb"
BATCH = 256
HIDDEN = 256
PROJ_DIM = 128
SEGMENT_LEN = 256

# ----------------------------
# Model definition
# ----------------------------
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

# ----------------------------
# Utilities
# ----------------------------
def robust_load_checkpoint(path):
    ck = torch.load(str(path), map_location="cpu")
    if isinstance(ck, dict):
        for k in ("model_state","model_state_dict","state_dict","model"):
            if k in ck:
                return ck[k]
        return ck
    return ck

def find_shards_for_ptb(root: Path, dataset: str):
    base = root / dataset
    if not base.exists():
        return []
    # find all numpy files that look like shards
    candidates = sorted([p for p in base.rglob("*.npy") if p.is_file()])
    return candidates

def meta_for_shard(shard_path: Path):
    meta = shard_path.with_name(shard_path.stem + "_meta.json")
    if meta.exists():
        return meta
    candidates = list(shard_path.parent.glob(shard_path.stem + "*_meta.json"))
    if candidates:
        return candidates[0]
    # any meta in same folder
    any_meta = list(shard_path.parent.glob("*_meta.json"))
    return any_meta[0] if any_meta else None

def process_shard_one2one(shard_path: Path, encoder, out_base: Path, dataset: str, split_name: str,
                          batch_size= BATCH, device="cpu", lead_indices=(0,1,7), save_h=True, save_z=True):
    rel = shard_path.relative_to(SEGMENTS_ROOT / dataset)
    # we'll write under out_base/h_vectors/all/<rel.parent>/<shard_name>
    out_h_dir = out_base / "h_vectors" / split_name / rel.parent
    out_z_dir = out_base / "z_vectors" / split_name / rel.parent
    out_h_dir.mkdir(parents=True, exist_ok=True)
    out_z_dir.mkdir(parents=True, exist_ok=True)

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

    h_out = []
    z_out = []
    out_meta_list = []

    start = 0
    with torch.no_grad():
        while start < n_local:
            end = min(start + batch_size, n_local)
            batch = arr[start:end]

            if batch.ndim == 3 and batch.shape[2] >= max(lead_indices)+1:
                # (B, T, C) -> transpose to (B, C, T)
                sel = batch[:, :, list(lead_indices)].astype(np.float32)
                sel = np.transpose(sel, (0, 2, 1))
            elif batch.ndim == 3 and batch.shape[1] >= max(lead_indices)+1:
                # (B, C, T)
                sel = batch[:, list(lead_indices), :].astype(np.float32)
            else:
                # fallback: zero-pad shape (B, C, T)
                Bf = batch.shape[0]
                sel = np.zeros((Bf, len(lead_indices), SEGMENT_LEN), dtype=np.float32)

            x = torch.from_numpy(sel).to(device=device)
            h = encoder.encode_h(x) if save_h else None
            z = encoder(x) if save_z else None
            if save_h:
                h_out.append(h.cpu().numpy().astype(np.float32))
            if save_z:
                z_out.append(z.cpu().numpy().astype(np.float32))

            # build simple meta entries aligned to local indices
            for li in range(start, end):
                m = {}
                if metas is not None and li < len(metas):
                    meta_entry = metas[li]
                    m["rec_id"] = meta_entry.get("rec_id") or meta_entry.get("record_id") or None
                    # try a few fields for labels present in PTB meta
                    m["label"] = meta_entry.get("super_class") or meta_entry.get("primary_scp") or meta_entry.get("diagnose") or meta_entry.get("primary_diagnosis") or None
                    m["primary_vector"] = meta_entry.get("primary_vector", None)
                else:
                    m["rec_id"] = None
                    m["label"] = None
                    m["primary_vector"] = None
                m["shard"] = str(shard_path.name)
                m["local_idx"] = int(li)
                out_meta_list.append(m)

            start = end

    saved = {"shard": str(shard_path), "saved_h": None, "saved_z": None, "meta": None}
    if save_h:
        if len(h_out) > 0:
            h_array = np.concatenate(h_out, axis=0)
        else:
            h_array = np.zeros((0, encoder.hidden), dtype=np.float32)
        outp = out_h_dir / shard_path.name
        np.save(str(outp), h_array)
        saved["saved_h"] = str(outp)
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
        # write same meta for z folder too
        with open(meta_z_outp, "w", encoding="utf-8") as fh:
            json.dump(out_meta_list, fh, ensure_ascii=False)

    return {"shard": str(shard_path), "status": "ok", "saved": saved, "n_samples": n_local}

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract contrastive representations for PTB segments")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="dataset folder under Segments/ (default: ptb)")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to encoder checkpoint")
    parser.add_argument("--out", type=str, default="./out/contrastive_reps_ptb", help="output base folder")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-h", action="store_true", default=True, help="save hidden vectors (encoder body output)")
    parser.add_argument("--save-z", action="store_true", default=True, help="save projection vectors (normalized z)")
    parser.add_argument("--leads", type=int, nargs="+", default=[0,1,7], help="lead indices to select (space separated). default 0 1 7")
    parser.add_argument("--ext", type=str, default=".npy", help="file extension to treat as shards (default .npy)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # build encoder and load checkpoint
    encoder = ResNet1DEncoder(in_ch=len(args.leads), hidden=HIDDEN, proj_dim=PROJ_DIM, depth=4)
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

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    # For PTB: list all shards recursively
    shards = find_shards_for_ptb(SEGMENTS_ROOT, args.dataset)
    if not shards:
        print(f"[ERROR] no shard files found under {SEGMENTS_ROOT / args.dataset}. Exiting.")
        return

    print(f"[INFO] Found {len(shards)} shard files under {SEGMENTS_ROOT / args.dataset}.")
    summary = {"processed_shards": 0, "errors": [], "samples": 0}

    for s in tqdm(shards, desc="Processing shards"):
        res = process_shard_one2one(s, encoder, out_base, args.dataset, "all",
                                    batch_size=args.batch, device=device,
                                    lead_indices=tuple(args.leads), save_h=args.save_h, save_z=args.save_z)
        if res.get("status") != "ok":
            summary["errors"].append(res)
        else:
            summary["processed_shards"] += 1
            summary["samples"] += res.get("n_samples", 0)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
