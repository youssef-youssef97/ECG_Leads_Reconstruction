import os
import sys
import argparse
import numpy as np
import math
import random
import json
from pathlib import Path
import importlib.util
import inspect
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------ USER-CONFIG (change paths here if needed) ------------------
DEFAULTS = {
    'clean_only_dir': 'Recons_ptbxl_Clean',
    'clean_h_dir': 'Recons_ptbxl_Clean_h',
    'test_shards_dir': 'Segments/ptb/all',
    'rep_h_shards_dir': 'out/contrastive_reps_ptb/h_vectors/all/all',
    'shard_pattern': 'all_shard_{:04d}.npy',
    'shard_start': 0,
    'shard_end': 3,
    'output_dir': 'cross_dataset_testing',
    'per_branch_dim': 128,
    'seq_len': 256,   # expected segment length
    'input_leads': ['I', 'II', 'V2'],
    'output_leads': ['V1', 'V3', 'V4', 'V5', 'V6'],
    'all_leads_order': ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_plots': 300,
    'random_seed': 42,
}


SEGMENT_LENGTH = DEFAULTS['seq_len']
EPS = 1e-8

# ------------------ Helpers ------------------
TRAINING_PY_CANDIDATES = [
    './Final_Model.py'
]


def import_training_module():
    for path in TRAINING_PY_CANDIDATES:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location('Final_Model_user', path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print(f"Imported training module from: {path}")
                return mod
            except Exception as e:
                print(f"Found {path} but failed to import it: {e}")
    print("Could not import Final_Model from default candidate paths. Continuing with local fallbacks.")
    return None


# ------------------ Fallback minimal model definitions ------------------
class FallbackSimplePerRepProj(nn.Module):
    def __init__(self, in_ch, out_ch, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        # x: (B,C,T)
        if x.shape[-1] != self.seq_len:
            x = nn.functional.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        return self.net(x)

class FallbackStackedLatentDecoder(nn.Module):
    def __init__(self, rep_in_ch_map, per_branch_dim=128, seq_len=256, reps=None):
        super().__init__()
        self.seq_len = seq_len
        self.reps = list(reps) if reps is not None else list(rep_in_ch_map.keys())
        self.per_branch_dim = per_branch_dim
        self.proj = nn.ModuleDict()
        for r in self.reps:
            in_ch = int(rep_in_ch_map[r])
            self.proj[r] = FallbackSimplePerRepProj(in_ch, per_branch_dim, seq_len)
        total_ch = per_branch_dim * len(self.reps)
        self.fusion = nn.Conv1d(total_ch, per_branch_dim, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv1d(per_branch_dim, max(32, per_branch_dim // 2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(max(32, per_branch_dim // 2), 1, kernel_size=3, padding=1)
        )
    def forward(self, reps_dict):
        feats = []
        for r in self.reps:
            x = reps_dict.get(r)
            if x is None:
                # create zeros
                b = next(iter(reps_dict.values())).shape[0]
                x = torch.zeros((b, 1, self.seq_len), device=next(self.parameters()).device)
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            if x.shape[-1] != self.seq_len:
                x = nn.functional.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
            feats.append(self.proj[r](x))
        x = torch.cat(feats, dim=1)
        x = self.fusion(x)
        x = self.decoder(x).squeeze(1)
        return x


# ------------------ collate / normalization helpers from training ------------------

def normalize_arr(a):
    denom = a.std() + 1e-8
    return ((a - a.mean()) / denom).astype(np.float32) if denom != 0 else a.astype(np.float32)

def _ensure_channels_time(arr):
    a = np.array(arr, dtype=np.float32)
    if a.ndim == 1:
        # treat as (seq_len,) -> (1, seq_len)
        return a.reshape(1, -1)
    if a.ndim == 2:
        # detect which axis is channels vs time
        if a.shape[0] == SEGMENT_LENGTH and a.shape[1] != SEGMENT_LENGTH:
            # (seq_len, channels)
            return a.T
        if a.shape[1] == SEGMENT_LENGTH and a.shape[0] != SEGMENT_LENGTH:
            # (channels, seq_len)
            return a
        # fallback: assume first dim channels
        return a
    # fallback
    return a.reshape(a.shape[0], -1)

def collate_batch(batch, reps, rep_stats, training=False):
    batch_inputs = {r: [] for r in reps}; batch_y = []
    for sample, y in batch:
        for r in reps:
            a = sample.get(r)
            if a is None:
                a = np.zeros((len(DEFAULTS['input_leads']), SEGMENT_LENGTH), dtype=np.float32)
            # normalize shape to (channels, seq_len)
            a = _ensure_channels_time(a)

            # special handling for clean_signal to select input leads
            if r == "clean_signal":
                lead_indices = [DEFAULTS['all_leads_order'].index(l) for l in DEFAULTS['input_leads']]
                if a.shape[0] == len(DEFAULTS['all_leads_order']):  # (12,256)
                    a = a[lead_indices, :]
                elif a.shape[1] == len(DEFAULTS['all_leads_order']):  # (256,12)
                    a = a[:, lead_indices].T

            # dataset-level normalization if stats provided for this rep
            stats = rep_stats.get(r) if rep_stats is not None else None
            if stats is not None:
                mean = stats["mean"]
                std = stats["std"]
                # try to broadcast correctly
                if np.asarray(mean).size == 1:
                    a = (a - float(mean)) / (float(std) + EPS)
                else:
                    mean = np.asarray(mean)
                    std = np.asarray(std)
                    # ensure channel dimension matches
                    if a.shape[0] != mean.shape[0] and a.shape[1] == mean.shape[0]:
                        a = a.T
                    if a.shape[0] == mean.shape[0]:
                        a = (a - mean.reshape(-1,1)) / (std.reshape(-1,1) + EPS)
                    else:
                        a = normalize_arr(a)
            else:
                # fallback
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

# ------------------ Other helpers ------------------

def load_stats_np(path_mean, path_std):
    mean = None
    std = None
    if os.path.exists(path_mean):
        mean = np.load(path_mean)
    if os.path.exists(path_std):
        std = np.load(path_std)
    return mean, std

def find_shard_paths(shard_dir, pattern, start, end):
    paths = []
    for i in range(start, end+1):
        p = os.path.join(shard_dir, pattern.format(i))
        if not os.path.exists(p):
            raise FileNotFoundError(f"Shard not found: {p}")
        paths.append(p)
    return paths

def load_shard_np(path):
    arr = np.load(path)
    if arr.ndim == 3:
        # convert (N,12,T) -> (N,T,12) or leave (N,T,12)
        if arr.shape[1] == 12 and arr.shape[2] != 12:
            arr = arr.transpose(0,2,1)  # (N,T,12)
    return arr

# ------------------ Model helpers ------------------

def instantiate_model_from_module(mod, rep_in_ch_map, cfg):
    ModelClass = None
    if mod is not None and hasattr(mod, 'StackedLatentDecoder'):
        ModelClass = mod.StackedLatentDecoder
    else:
        ModelClass = FallbackStackedLatentDecoder

    sig = inspect.signature(ModelClass.__init__)
    kwargs = {}
    if 'rep_in_ch_map' in sig.parameters:
        kwargs['rep_in_ch_map'] = rep_in_ch_map
    if 'per_branch_dim' in sig.parameters:
        kwargs['per_branch_dim'] = cfg['per_branch_dim']
    if 'seq_len' in sig.parameters:
        kwargs['seq_len'] = cfg['seq_len']
    if 'reps' in sig.parameters:
        kwargs['reps'] = list(rep_in_ch_map.keys())
    try:
        model = ModelClass(**kwargs)
    except Exception:
        try:
            model = ModelClass(list(rep_in_ch_map.keys()), cfg['per_branch_dim'], cfg['seq_len'])
        except Exception:
            model = ModelClass(rep_in_ch_map, cfg['per_branch_dim'], cfg['seq_len'], list(rep_in_ch_map.keys()))
    return model

def load_checkpoint_into_model(ckpt_path, model, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            if all(isinstance(k, str) for k in ckpt.keys()):
                state = ckpt
    else:
        state = ckpt
    if state is None:
        raise RuntimeError(f"Could not find model state dict in {ckpt_path}")
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k,v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state)
    return model

# ------------------ small utilities ------------------
class SegmentShardIterator:
    def __init__(self, segment_shard_paths, h_shard_paths=None):
        self.segment_shard_paths = segment_shard_paths
        self.h_shard_paths = h_shard_paths
    def __iter__(self):
        for i, sp in enumerate(self.segment_shard_paths):
            segs = load_shard_np(sp)
            harr = None
            if self.h_shard_paths is not None:
                harr = np.load(self.h_shard_paths[i])
            yield sp, segs, harr


shard_generator = lambda seg_paths, h_paths=None: SegmentShardIterator(seg_paths, h_paths)

# ------------------ Metrics------------------

def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        return math.sqrt(mean_squared_error(y_true, y_pred))

    # expect shape (N, T)
    try:
        # compute MSE per segment, then sqrt to get RMSE per segment
        mse_per = np.mean((y_true - y_pred) ** 2, axis=1)
        rmse_per = np.sqrt(mse_per)
        return float(np.mean(rmse_per))
    except Exception:
        # fallback
        return math.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        try:
            return r2_score(y_true, y_pred)
        except Exception:
            return float('nan')

    vals = []
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        if np.std(t) == 0:
            # skip constant ground-truth segments
            continue
        try:
            vals.append(r2_score(t, p))
        except Exception:
            continue
    if len(vals) == 0:
        return float('nan')
    return float(np.mean(vals))

def pearson_mean(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        if np.std(y_true) == 0:
            return 0.0
        return float(pearsonr(y_true, y_pred)[0])
    vals = []
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        if np.std(t) == 0:
            continue
        try:
            vals.append(pearsonr(t, p)[0])
        except Exception:
            continue
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

# ------------------ helper to extract primary from meta ------------------
def get_primary_from_meta(meta_entry):
    if not isinstance(meta_entry, dict):
        return 'UNK'
    for key in ('primary_scp', 'primary', 'super_class', 'primary_vector'):
        if key in meta_entry and meta_entry.get(key):
            val = meta_entry.get(key)
            # primary_vector is numeric vector, skip
            if key == 'primary_vector':
                continue
            return str(val)
    # try scp_codes_raw highest scoring key
    scp_raw = meta_entry.get('scp_codes_raw') or {}
    if isinstance(scp_raw, dict) and len(scp_raw) > 0:
        # pick code with max value
        try:
            best = max(scp_raw.items(), key=lambda kv: kv[1])[0]
            return str(best)
        except Exception:
            pass
    return 'UNK'

# ------------------ Main inference loop ------------------

def run_inference(cfg):
    training_mod = import_training_module()

    shard_paths = find_shard_paths(cfg['test_shards_dir'], cfg['shard_pattern'], cfg['shard_start'], cfg['shard_end'])
    h_shard_paths = find_shard_paths(cfg['rep_h_shards_dir'], cfg['shard_pattern'], cfg['shard_start'], cfg['shard_end'])

    # Preload all meta entries in order so we can map idx -> primary label
    all_meta = []
    for sp in shard_paths:
        meta_path = sp.replace('.npy', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as fh:
                try:
                    shard_meta = json.load(fh)
                except Exception:
                    shard_meta = []
            # ensure this is a list with same length as segments
            if isinstance(shard_meta, list):
                all_meta.extend(shard_meta)
            else:
                # unexpected format -> replicate empty dicts
                # try to determine segment count
                try:
                    segs = np.load(sp)
                    n = segs.shape[0]
                except Exception:
                    n = 0
                all_meta.extend([{}]*n)
        else:
            # no meta file -> fill placeholders
            try:
                segs = np.load(sp)
                n = segs.shape[0]
            except Exception:
                n = 0
            all_meta.extend([{}]*n)

    # load stats (as tuples mean/std)
    clean_only_stats = load_stats_np(os.path.join(cfg['clean_only_dir'], 'clean_signal_train_mean.npy'),
                                    os.path.join(cfg['clean_only_dir'], 'clean_signal_train_std.npy'))
    cleanh_stats_clean = load_stats_np(os.path.join(cfg['clean_h_dir'], 'clean_signal_train_mean.npy'),
                                       os.path.join(cfg['clean_h_dir'], 'clean_signal_train_std.npy'))
    cleanh_stats_h = load_stats_np(os.path.join(cfg['clean_h_dir'], 'h_vectors_train_mean.npy'),
                                   os.path.join(cfg['clean_h_dir'], 'h_vectors_train_std.npy'))

    # convert to rep_stats dicts that collate_batch expects
    rep_stats_clean_only = {}
    if clean_only_stats[0] is not None:
        rep_stats_clean_only['clean_signal'] = {'mean': np.asarray(clean_only_stats[0]), 'std': np.asarray(clean_only_stats[1])}

    rep_stats_cleanh = {}
    if cleanh_stats_clean[0] is not None:
        rep_stats_cleanh['clean_signal'] = {'mean': np.asarray(cloud_safe := cleanh_stats_clean[0]), 'std': np.asarray(cloud_safe if False else cleanh_stats_clean[1])}
    if cleanh_stats_h[0] is not None:
        rep_stats_cleanh['h_vectors'] = {'mean': np.asarray(cleanh_stats_h[0]), 'std': np.asarray(cleanh_stats_h[1])}

    # prepare per-lead results containers
    results = {}
    for lead in cfg['output_leads']:
        results[lead] = {
            'y_true': [],
            'pred_clean_only': [],
            'pred_clean_h': []
        }

    total_samples = 0
    C_h = None

    # first pass: accumulate all shards predictions per lead
    global_idx_offset = 0
    for shard_idx, (seg_path, segs, h_arr) in enumerate(shard_generator(shard_paths, h_shard_paths)):
        print(f"Processing shard {shard_idx}: {seg_path}")
        # segs: (N, T, 12)
        if segs.ndim != 3 or segs.shape[2] != 12:
            raise RuntimeError(f"Unexpected segments shape {segs.shape} in {seg_path}")
        N = segs.shape[0]
        total_samples += N

        if h_arr is not None:
            # h_arr expected shape (N, C_h) or (N, C_h, 1)
            arr = np.asarray(h_arr)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:,:,0]
            if C_h is None:
                C_h = arr.shape[1]
        else:
            arr = None

        # For each batch, build collates once (clean-only and clean+h) and reuse for all leads
        for start in range(0, N, cfg['batch_size']):
            end = min(N, start + cfg['batch_size'])
            seg_batch = segs[start:end]  # (B,T,12)
            h_batch = arr[start:end] if arr is not None else None
            B = seg_batch.shape[0]

            batch_list_clean = []
            batch_list_cleanh = []
            for i in range(B):
                sample = {}
                sample['clean_signal'] = seg_batch[i]
                if h_batch is not None:
                    sample['h_vectors'] = h_batch[i]
                y_placeholder = np.zeros(SEGMENT_LENGTH, dtype=np.float32)
                batch_list_clean.append((sample, y_placeholder))
                batch_list_cleanh.append((sample, y_placeholder))

            # collate for clean-only (reps = ['clean_signal'])
            coll_co, _ = collate_batch(batch_list_clean, ['clean_signal'], rep_stats_clean_only, training=False)
            # collate for clean+h (reps = ['clean_signal', 'h_vectors'])
            coll_ch, _ = collate_batch(batch_list_cleanh, ['clean_signal', 'h_vectors'], rep_stats_cleanh, training=False)

            # For each output lead, instantiate/load model once per lead per config
            for lead in cfg['output_leads']:
                lead_idx = cfg['all_leads_order'].index(lead)
                y_batch = seg_batch[:, :, lead_idx].astype(np.float32)  # (B, T)

                entry = results[lead]

                # lazy init model objects and device
                if 'models' not in entry:
                    entry['models'] = {}

                # CLEAN-ONLY model
                if 'clean_only' not in entry['models']:
                    rep_map = {'clean_signal': len(cfg['input_leads'])}
                    model = instantiate_model_from_module(training_mod, rep_map, cfg)
                    ckpt_path = os.path.join(cfg['clean_only_dir'], 'models', f'Stacked_{lead}_fixed_splits.pt')
                    if not os.path.exists(ckpt_path):
                        raise FileNotFoundError(f"Checkpoint for clean-only not found: {ckpt_path}")
                    model = load_checkpoint_into_model(ckpt_path, model, device=cfg['device'])
                    model.to(cfg['device']).eval()
                    entry['models']['clean_only'] = model

                # CLEAN+H model
                if 'clean_h' not in entry['models']:
                    if C_h is None:
                        raise RuntimeError("h_vectors channels could not be inferred from shards. Ensure h shard files exist and contain arrays.")
                    rep_map = {'clean_signal': len(cfg['input_leads']), 'h_vectors': C_h}
                    model = instantiate_model_from_module(training_mod, rep_map, cfg)
                    ckpt_path = os.path.join(cfg['clean_h_dir'], 'models', f'Stacked_{lead}_fixed_splits.pt')
                    if not os.path.exists(ckpt_path):
                        raise FileNotFoundError(f"Checkpoint for clean+h not found: {ckpt_path}")
                    model = load_checkpoint_into_model(ckpt_path, model, device=cfg['device'])
                    model.to(cfg['device']).eval()
                    entry['models']['clean_h'] = model

                # run inference
                with torch.no_grad():
                    # clean-only forward
                    model_co = entry['models']['clean_only']
                    inp_co = {'clean_signal': coll_co['clean_signal'].to(cfg['device'])}
                    pred_co = model_co(inp_co).cpu().numpy()  # (B, T)

                    # clean+h forward
                    model_ch = entry['models']['clean_h']
                    # ensure coll_ch contains both keys
                    inp_ch = {k: v.to(cfg['device']) for k, v in coll_ch.items()}
                    pred_ch = model_ch(inp_ch).cpu().numpy()

                # store
                entry['y_true'].append(y_batch)
                entry['pred_clean_only'].append(pred_co)
                entry['pred_clean_h'].append(pred_ch)

        global_idx_offset += N


    # concatenate per-lead
    report_lines = []
    for lead in cfg['output_leads']:
        entry = results[lead]
        y_true = np.concatenate(entry['y_true'], axis=0)
        p_co = np.concatenate(entry['pred_clean_only'], axis=0)
        p_ch = np.concatenate(entry['pred_clean_h'], axis=0)

        # ensure shapes
        assert y_true.shape == p_co.shape == p_ch.shape

        # compute metrics
        rmse_co = compute_rmse(y_true, p_co)
        r2_co = compute_r2(y_true, p_co)
        pearson_co = pearson_mean(y_true, p_co)

        rmse_ch = compute_rmse(y_true, p_ch)
        r2_ch = compute_r2(y_true, p_ch)
        pearson_ch = pearson_mean(y_true, p_ch)

        report_lines.append(f"Lead {lead} — Clean-only: RMSE={rmse_co:.5f}, R2={r2_co:.5f}, Pearson={pearson_co:.5f}\n")
        report_lines.append(f"Lead {lead} — Clean+h   : RMSE={rmse_ch:.5f}, R2={r2_ch:.5f}, Pearson={pearson_ch:.5f}\n")

        # save arrays
        np.save(os.path.join(cfg['output_dir'], f'y_true_{lead}.npy'), y_true)
        np.save(os.path.join(cfg['output_dir'], f'pred_clean_only_{lead}.npy'), p_co)
        np.save(os.path.join(cfg['output_dir'], f'pred_clean_h_{lead}.npy'), p_ch)

        # store back
        entry['y_true_all'] = y_true
        entry['p_co_all'] = p_co
        entry['p_ch_all'] = p_ch

    # write report
    report_path = os.path.join(cfg['output_dir'], 'inference_report.txt')
    with open(report_path, 'w') as fh:
        fh.write(''.join(report_lines))
    print(''.join(report_lines))
    print(f"Saved report to {report_path}")

    # ------------------ Create plots ------------------
    # build global sample count from any lead
    sample_count = list(results.values())[0]['y_true_all'].shape[0]
    rng = random.Random(cfg['random_seed'])
    chosen_indices = list(range(sample_count))
    if cfg['n_plots'] < sample_count:
        chosen_indices = rng.sample(range(sample_count), cfg['n_plots'])

    time_ax = np.arange(cfg['seq_len'])
    for i, idx in enumerate(tqdm(chosen_indices, desc='Plotting')):
        # get primary label from preloaded all_meta
        primary_label = 'UNK'
        if 0 <= idx < len(all_meta):
            primary_label = get_primary_from_meta(all_meta[idx])
        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        for ax_i, lead in enumerate(cfg['output_leads']):
            y = results[lead]['y_true_all'][idx]
            pco = results[lead]['p_co_all'][idx]
            pch = results[lead]['p_ch_all'][idx]
            ax = axes[ax_i]
            ax.plot(time_ax, y, label='GT')
            ax.plot(time_ax, pco, label='Clean-only', alpha=0.8)
            ax.plot(time_ax, pch, label='Clean+h', alpha=0.8)
            ax.set_title(f"{lead} | {primary_label}")
            if ax_i == 0:
                ax.legend(loc='upper right')
        fig.tight_layout()
        out_png = os.path.join(cfg['output_dir'], 'plots', f'compare_{i:04d}.png')
        fig.savefig(out_png)
        plt.close(fig)

    print(f"Saved {len(chosen_indices)} comparison plots to {os.path.join(cfg['output_dir'],'plots')}")
    return report_path

# ------------------ CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--clean-only-dir', default=DEFAULTS['clean_only_dir'])
    p.add_argument('--clean-h-dir', default=DEFAULTS['clean_h_dir'])
    p.add_argument('--test-shards-dir', default=DEFAULTS['test_shards_dir'])
    p.add_argument('--rep-h-shards-dir', default=DEFAULTS['rep_h_shards_dir'])
    p.add_argument('--shard-start', type=int, default=DEFAULTS['shard_start'])
    p.add_argument('--shard-end', type=int, default=DEFAULTS['shard_end'])
    p.add_argument('--shard-pattern', default=DEFAULTS['shard_pattern'])
    p.add_argument('--output-dir', default=DEFAULTS['output_dir'])
    p.add_argument('--seq-len', type=int, default=DEFAULTS['seq_len'])
    p.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    p.add_argument('--n-plots', type=int, default=DEFAULTS['n_plots'])
    p.add_argument('--random-seed', type=int, default=DEFAULTS['random_seed'])

    args = p.parse_args()
    cfg = DEFAULTS.copy()
    cfg.update({
        'clean_only_dir': args.clean_only_dir,
        'clean_h_dir': args.clean_h_dir,
        'test_shards_dir': args.test_shards_dir,
        'rep_h_shards_dir': args.rep_h_shards_dir,
        'shard_start': args.shard_start,
        'shard_end': args.shard_end,
        'shard_pattern': args.shard_pattern,
        'output_dir': args.output_dir,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'n_plots': args.n_plots,
        'random_seed': args.random_seed,
    })

    os.makedirs(cfg['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['output_dir'], 'plots'), exist_ok=True)

    run_inference(cfg)