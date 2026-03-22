# imports
from pathlib import Path
import math
import gc
import json
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

# params
ROOT = Path("./Cleaned_Datasets")
OUT_ROOT = Path("./Segments")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 100
SEGMENT_LENGTH = 256
HOP = 64
FINAL_SHARD_FLUSH = 2000
SAMPLE_FOR_THRESHOLDS = 12000

LOW_PCT = 0.0
HIGH_PCT = 100.0

DEFAULT_WORKERS = max(1, min(32, (mp.cpu_count() or 2) - 1))
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BAD_LEADS_MAX = 3
SAMPLES_FOR_PLOTTING = 100000

# helper
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# files
def list_npy_files(folder: Path, recursive: bool = False) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        return sorted([p for p in folder.rglob("*.npy") if p.is_file()])
    return sorted([p for p in folder.glob("*.npy") if p.is_file()])

# meta path
def meta_path_for_npy(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.stem + "_meta.json")

# json dump
def safe_json_dump(obj, path: Path):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
    except Exception:
        try:
            with open(path, "w") as fh:
                json.dump(obj, fh, indent=2)
        except Exception:
            pass

# constants
_RHYTHM_CODES = set(["AFIB", "AFLT", "STACH", "SBRAD", "PSVT", "SVTAC", "SR", "SB"])
_MORPHOLOGY_INDICATORS = set([
    "ABQRS", "QWAVE", "LVOLT", "HVOLT", "LOWT", "LVH", "IVCD", "LBBB", "RBBB"
])

# normalize
def _normalize_label_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    tl = t.lower()
    if "myocardial" in tl or "infarct" in tl or tl == "mi":
        return "Myocardial infarction"
    if "myocarditis" in tl:
        return "Myocarditis"
    if "hypertrophy" in tl or "myocardial hypertrophy" in tl or "mh" in tl:
        return "Myocardial hypertrophy"
    if "bundle" in tl or "bbb" in tl or "branch block" in tl:
        return "Bundle branch block"
    if "cardiomyopathy" in tl or "cardiomyopath" in tl:
        return "Cardiomyopathy"
    if "dysrhythm" in tl or "arrhythm" in tl or "dysrhyth" in tl:
        return "Dysrhythmia"
    if "valvular" in tl or "valve" in tl:
        return "Valvular heart disease"
    if "healthy" in tl or "control" in tl or "normal" in tl:
        return "Healthy control"
    return t if t[0].isupper() else t.capitalize()

# infer
def infer_statement_type(record_meta: Dict) -> str:
    try:
        sc = record_meta.get("super_class")
        ps = record_meta.get("primary_scp")
        scp_keys = list(record_meta.get("scp_codes", {}).keys()) if record_meta.get("scp_codes") else []
        if sc is not None and str(sc).strip() != "":
            return "diagnostic"
        if ps and str(ps).upper() in _RHYTHM_CODES:
            return "rhythm"
        if any(k.upper() in _MORPHOLOGY_INDICATORS for k in scp_keys):
            return "morphology"
        acrs = record_meta.get("diagnosis_acronyms", []) or []
        for a in acrs:
            if a.upper() in _RHYTHM_CODES:
                return "rhythm"
            if a.upper() in _MORPHOLOGY_INDICATORS:
                return "morphology"
        return "unknown"
    except Exception:
        return "unknown"

# read
def read_record_meta(npy_path: Path) -> Dict:
    mpth = meta_path_for_npy(npy_path)
    base_fallback = {
        "rec_id": npy_path.stem,
        "patient_id": npy_path.stem,
        "fs": None,
        "age": None,
        "sex": None,
        "source_path": str(npy_path),
        "scp_codes": {},
        "scp_codes_raw": None,
        "primary_scp": None,
        "super_class": None,
        "diagnostic_subclass": None,
        "primary_vector": None,
        "strat_fold": None,
    }
    if not mpth.exists():
        base_fallback["statement_type"] = infer_statement_type(base_fallback)
        return base_fallback
    try:
        with open(mpth, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        base_fallback["statement_type"] = infer_statement_type(base_fallback)
        return base_fallback

    ptb_hdr = d.get("ptb_header_parsed", {}) or {}
    primary = d.get("primary_scp", None)
    superc = d.get("super_class", None)

    if not primary:
        reason = ptb_hdr.get("Reason for admission") or ptb_hdr.get("Diagnose") or ptb_hdr.get("Diagnosis")
        primary = reason if reason else None
    if not superc:
        superc = primary if primary else None

    primary_norm = _normalize_label_text(primary)
    super_norm = _normalize_label_text(superc)

    # parse weighted vector robustly
    pv_in = d.get("primary_vector", None)
    pv_list = None
    if pv_in is not None:
        try:
            if isinstance(pv_in, list):
                pv_list = [float(x) for x in pv_in]
            elif isinstance(pv_in, (str,)):
                try:
                    pv_list = json.loads(pv_in)
                    pv_list = [float(x) for x in pv_list]
                except Exception:
                    pv_list = None
            elif hasattr(pv_in, "tolist"):
                pv_list = list(map(float, pv_in.tolist()))
        except Exception:
            pv_list = None

    out = {
        "rec_id": d.get("rec_id", base_fallback["rec_id"]),
        "patient_id": d.get("patient_id", base_fallback["patient_id"]),
        "fs": d.get("fs", base_fallback["fs"]),
        "age": d.get("age", base_fallback["age"]),
        "sex": d.get("sex", base_fallback["sex"]),
        "source_path": d.get("source_path", base_fallback["source_path"]),
        "scp_codes": d.get("scp_codes", {}) or {},
        "scp_codes_raw": d.get("scp_codes_raw", None),
        "primary_scp": primary_norm,
        "super_class": super_norm,
        "diagnostic_subclass": d.get("diagnostic_subclass", None),
        "diagnosis_snomed_codes": d.get("diagnosis_snomed_codes", []) or [],
        "diagnosis_acronyms": d.get("diagnosis_acronyms", []) or [],
        "diagnosis_fullnames": d.get("diagnosis_fullnames", []) or [],
        "primary_vector": pv_list,
        "strat_fold": d.get("strat_fold", None),
    }

    if ptb_hdr:
        if "Additional diagnoses" in ptb_hdr:
            out["additional_diagnoses"] = ptb_hdr.get("Additional diagnoses")
        if "Smoker" in ptb_hdr:
            out["smoker"] = ptb_hdr.get("Smoker")
        if "Number of coronary vessels involved" in ptb_hdr:
            out["num_coronary_vessels"] = ptb_hdr.get("Number of coronary vessels involved")
        if "Acute infarction (localization)" in ptb_hdr:
            out["acute_infarction_localization"] = ptb_hdr.get("Acute infarction (localization)")
        if "Infarction date (acute)" in ptb_hdr:
            out["infarction_date_acute"] = ptb_hdr.get("Infarction date (acute)")

    out["statement_type"] = infer_statement_type(out)
    return out

# ensure leads
def ensure_12_leads(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return None
    T, C = arr.shape
    if C < 12:
        pad = np.zeros((T, 12 - C), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    elif C > 12:
        arr = arr[:, :12]
    return arr

# pass1
def sample_stats_from_file(args: Tuple[str, int, int, int]):
    npy_path_str, seg_len, hop, sample_per_file = args
    try:
        npy_path = Path(npy_path_str)
        arr = np.load(npy_path, mmap_mode="r")
        arr = ensure_12_leads(arr)
        if arr is None:
            return []
        T, C = arr.shape
        if T <= 0:
            return []
        starts = list(range(0, T, hop))
        valid_starts = [s for s in starts if (s + seg_len) <= T]
        if not valid_starts:
            return []
        if len(valid_starts) <= sample_per_file:
            chosen = valid_starts
        else:
            chosen = list(np.random.choice(valid_starts, size=sample_per_file, replace=False))
        out = []
        for s in chosen:
            end = s + seg_len
            seg = arr[s:end, :]
            if not np.all(np.isfinite(seg)):
                continue
            amp = np.max(np.abs(seg), axis=0).astype(np.float32)
            rms = np.sqrt(np.mean(seg**2, axis=0)).astype(np.float32)
            out.append((amp, rms))
        del arr
        gc.collect()
        return out
    except Exception:
        return []

# pass2
def extract_segments_sliding_from_file(args: Tuple[str, int, int]):
    npy_path_str, seg_len, hop = args
    try:
        npy_path = Path(npy_path_str)
        record_meta = read_record_meta(npy_path)
        arr = np.load(npy_path, mmap_mode="r")
        arr = ensure_12_leads(arr)
        if arr is None:
            return []
        T, C = arr.shape
        if T <= 0:
            return []
        segments_with_meta = []
        start = 0
        while (start + seg_len) <= T:
            end = start + seg_len
            seg = arr[start:end, :]
            if not np.all(np.isfinite(seg)):
                start += hop
                continue
            seg = seg.astype(np.float32, copy=False)

            # attach fold/split info if strat_fold present
            strat = record_meta.get("strat_fold", None)
            split = None
            fold_label = None
            try:
                if strat is not None:
                    si = int(str(strat))
                    if 1 <= si <= 8:
                        split = "train"
                        fold_label = f"fold_{si}"
                    elif si == 9:
                        split = "val"
                        fold_label = "fold_9"
                    elif si == 10:
                        split = "test"
                        fold_label = "fold_10"
            except Exception:
                split = None
                fold_label = None
            # fallback: try to infer from path (parent names)
            if split is None:
                parents = [p.name.lower() for p in npy_path.parents]
                if "train" in parents:
                    split = "train"
                elif "val" in parents or "validation" in parents:
                    split = "val"
                elif "test" in parents:
                    split = "test"
                else:
                    split = "train"
                # try to find fold_# in parents
                fl = next((p for p in [pp.name for pp in npy_path.parents] if p.lower().startswith("fold_")), None)
                fold_label = fl if fl else "fold_1"

            seg_meta = {
                "rec_id": record_meta.get("rec_id"),
                "patient_id": record_meta.get("patient_id"),
                "source_path": record_meta.get("source_path"),
                "fs": record_meta.get("fs"),
                "age": record_meta.get("age"),
                "sex": record_meta.get("sex"),
                "start_sample": int(start),
                "end_sample": int(end),
                "scp_codes": record_meta.get("scp_codes", {}),
                "scp_codes_raw": record_meta.get("scp_codes_raw", None),
                "primary_scp": record_meta.get("primary_scp", None),
                "super_class": record_meta.get("super_class", None),
                "diagnostic_subclass": record_meta.get("diagnostic_subclass", None),
                "statement_type": record_meta.get("statement_type", infer_statement_type(record_meta)),
                "diagnosis_snomed_codes": record_meta.get("diagnosis_snomed_codes", []),
                "diagnosis_acronyms": record_meta.get("diagnosis_acronyms", []),
                "diagnosis_fullnames": record_meta.get("diagnosis_fullnames", []),
                "primary_vector": record_meta.get("primary_vector", None),
                "strat_fold": record_meta.get("strat_fold", None),
                "split": split,
                "fold_label": fold_label,
            }
            segments_with_meta.append((seg, seg_meta))
            start += hop
        del arr
        gc.collect()
        return segments_with_meta
    except Exception:
        return []

# thresholds
def compute_percentile_thresholds_from_samples(
    amp_samples: List[np.ndarray],
    rms_samples: List[np.ndarray],
    low_pct=LOW_PCT,
    high_pct=HIGH_PCT,
):
    thresholds = {}
    if not amp_samples or not rms_samples:
        for li in range(12):
            thresholds[li] = {
                "amp_low": 0.0,
                "amp_high": float("inf"),
                "rms_low": 0.0,
                "rms_high": float("inf"),
            }
        return thresholds
    amps = np.vstack(amp_samples)
    rmss = np.vstack(rms_samples)
    for li in range(amps.shape[1]):
        a_col = amps[:, li]
        r_col = rmss[:, li]
        amp_low = float(np.percentile(a_col, low_pct))
        amp_high = float(np.percentile(a_col, high_pct))
        rms_low = float(np.percentile(r_col, low_pct))
        rms_high = float(np.percentile(r_col, high_pct))
        thresholds[li] = {
            "amp_low": amp_low,
            "amp_high": amp_high,
            "rms_low": rms_low,
            "rms_high": rms_high,
        }
    return thresholds

# reports
def make_report_dirs(dataset_name: str):
    base = OUT_ROOT / dataset_name
    reports = base / "reports"
    ensure_dir(reports)
    return base, reports

# plotting
def plot_hist_before_after(
    before_vals, after_vals, xlabel, outpath_before, outpath_after, bins=80
):
    try:
        plt.figure(figsize=(10, 4))
        plt.hist(before_vals, bins=bins)
        plt.title("Before (sampled) - " + xlabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(outpath_before, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.hist(after_vals, bins=bins)
        plt.title("After QC (sampled) - " + xlabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(outpath_after, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

# heatmap
def plot_per_lead_heatmap(counts_per_lead, title, outpath):
    if sns is None:
        return
    try:
        arr = np.array(counts_per_lead)
        plt.figure(figsize=(8, 2))
        sns.heatmap(arr[None, :], annot=True, fmt="d", cmap="Reds")
        plt.yticks([])
        plt.xticks(np.arange(12) + 0.5, [f"L{i+1}" for i in range(12)])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

# boxplot
def plot_boxplot(values, ylabel, outpath):
    try:
        plt.figure(figsize=(6, 4))
        plt.boxplot(values)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

# two-pass
def process_dataset_two_passes(
    dataset_name: str,
    dataset_folder: Path,
    workers: int = DEFAULT_WORKERS,
    seg_len: int = SEGMENT_LENGTH,
    hop: int = HOP,
    sample_target: int = SAMPLE_FOR_THRESHOLDS,
    shard_flush: int = FINAL_SHARD_FLUSH,
    bad_leads_max: int = BAD_LEADS_MAX,
    samples_for_plotting: int = SAMPLES_FOR_PLOTTING,
):
    print(f"\n=== Processing dataset: {dataset_name}  (folder: {dataset_folder}) ===")

    # determine file listing (PTB-XL may be organized in train/val/test/fold_x)
    if dataset_name.lower() == "ptbxl":
        files = list_npy_files(dataset_folder, recursive=True)
    else:
        files = list_npy_files(dataset_folder, recursive=False)

    # filter to only files that look like cleaned-records (avoid shard outputs if rerun)
    files = [p for p in files if not p.name.startswith("all_shard_")]

    print(f"Found {len(files)} .npy files in {dataset_folder}")
    if len(files) == 0:
        print("No files, skipping.")
        return

    # PASS 1: sample for thresholds
    per_file_k = max(1, int(math.ceil(sample_target / max(1, len(files)))))
    amp_samples = []
    rms_samples = []

    args = [(str(p), seg_len, hop, per_file_k) for p in files]
    with mp.Pool(processes=workers) as pool:
        for file_samples in tqdm(
            pool.imap(sample_stats_from_file, args),
            total=len(args),
            desc="Sampling files",
        ):
            if not file_samples:
                continue
            for amp, rms in file_samples:
                amp_samples.append(amp)
                rms_samples.append(rms)
                if len(amp_samples) >= sample_target:
                    break
            if len(amp_samples) >= sample_target:
                break

    print(f"Collected sample_count={len(amp_samples)} for threshold estimation")
    thresholds = compute_percentile_thresholds_from_samples(amp_samples, rms_samples)

    base, reports_dir = make_report_dirs(dataset_name)
    try:
        with open(base / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
    except Exception:
        pass

    # prepare output structure
    if dataset_name.lower() == "ptbxl":
        for fnum in range(1, 9):
            ensure_dir(base / "train" / f"fold_{fnum}")
        ensure_dir(base / "val" / "fold_9")
        ensure_dir(base / "test" / "fold_10")
    else:
        ensure_dir(base / "all")

    # buffers per fold (for ptbxl) or single buffer for others
    buffers: Dict[Tuple[str, str], List] = {}
    meta_buffers: Dict[Tuple[str, str], List] = {}
    shard_idx_map: Dict[Tuple[str, str], int] = {}

    def write_shard_for_key(key):
        segs = buffers.get(key, [])
        metas = meta_buffers.get(key, [])
        if not segs:
            return
        split, fold_label = key
        if dataset_name.lower() == "ptbxl":
            out_dir = base / split / fold_label
        else:
            out_dir = base / "all"
        ensure_dir(out_dir)
        idx = shard_idx_map.get(key, 0)
        out_path = out_dir / f"all_shard_{idx:04d}.npy"
        np.save(out_path, np.array(segs, dtype=np.float32))
        meta_out_path = out_dir / f"all_shard_{idx:04d}_meta.json"
        safe_json_dump(metas, meta_out_path)
        shard_idx_map[key] = idx + 1
        buffers[key] = []
        meta_buffers[key] = []
        gc.collect()

    # counters
    rejected_reason_counter = Counter()
    rejected_per_lead = [0] * 12
    accepted_per_lead_counts = [0] * 12

    before_amp_samples_flat = []
    before_rms_samples_flat = []
    after_amp_samples_flat = []
    after_rms_samples_flat = []

    total_seen = 0
    accepted = 0
    rejected = 0

    primary_counter = Counter()
    super_counter = Counter()

    # PASS 2: sliding-window extract (workers)
    args2 = [(str(p), seg_len, hop) for p in files]
    with mp.Pool(processes=workers) as pool:
        for segments_with_meta in tqdm(
            pool.imap(extract_segments_sliding_from_file, args2),
            total=len(args2),
            desc="Sliding-window extract (pass2)",
        ):
            if not segments_with_meta:
                continue
            for seg, seg_meta in segments_with_meta:
                total_seen += 1
                if seg is None or seg.size == 0 or not np.all(np.isfinite(seg)):
                    rejected += 1
                    rejected_reason_counter["nonfinite_or_empty"] += 1
                    continue
                amp = np.max(np.abs(seg), axis=0)
                rms = np.sqrt(np.mean(seg**2, axis=0))

                if len(before_amp_samples_flat) < samples_for_plotting:
                    before_amp_samples_flat.extend(list(amp))
                if len(before_rms_samples_flat) < samples_for_plotting:
                    before_rms_samples_flat.extend(list(rms))

                per_lead_flags = [False] * 12
                per_lead_reasons = []
                for li in range(12):
                    thr = thresholds.get(li, None)
                    if thr is None:
                        continue
                    a = float(amp[li])
                    r = float(rms[li])
                    reason_keys = []
                    if a > thr["amp_high"]:
                        reason_keys.append("amp_high")
                    if a < thr["amp_low"]:
                        reason_keys.append("amp_low")
                    if r > thr["rms_high"]:
                        reason_keys.append("rms_high")
                    if r < thr["rms_low"]:
                        reason_keys.append("rms_low")
                    if reason_keys:
                        per_lead_flags[li] = True
                        per_lead_reasons.append((li, reason_keys))

                bad_leads = sum(1 for f in per_lead_flags if f)

                if bad_leads > bad_leads_max:
                    rejected += 1
                    rejected_reason_counter["too_many_bad_leads"] += 1
                    for li, flag in enumerate(per_lead_flags):
                        if flag:
                            rejected_per_lead[li] += 1
                    continue

                # determine key (split, fold_label)
                split = seg_meta.get("split", "train")
                fold_label = seg_meta.get("fold_label", "fold_1")
                key = (split, fold_label) if dataset_name.lower() == "ptbxl" else ("all", "all")

                # init buffers for key if needed
                if key not in buffers:
                    buffers[key] = []
                    meta_buffers[key] = []
                    shard_idx_map[key] = 0

                buffers[key].append(seg.astype(np.float32))
                seg_meta_qc = dict(seg_meta)
                seg_meta_qc.update(
                    {
                        "qc_rejected": False,
                        "qc_bad_leads_count": int(bad_leads),
                        "qc_per_lead_flags": per_lead_flags,
                        "qc_per_lead_reasons": {str(li): reasons for (li, reasons) in per_lead_reasons},
                    }
                )
                if "qc_per_lead_reasons" in seg_meta_qc and isinstance(seg_meta_qc["qc_per_lead_reasons"], dict):
                    pass
                else:
                    seg_meta_qc["qc_per_lead_reasons"] = {str(li): reasons for (li, reasons) in per_lead_reasons} if per_lead_reasons else {}

                meta_buffers[key].append(seg_meta_qc)

                accepted += 1

                for li in range(12):
                    if not per_lead_flags[li]:
                        accepted_per_lead_counts[li] += 1

                if len(after_amp_samples_flat) < samples_for_plotting:
                    after_amp_samples_flat.extend(list(amp))
                if len(after_rms_samples_flat) < samples_for_plotting:
                    after_rms_samples_flat.extend(list(rms))

                ps = seg_meta.get("primary_scp")
                sc = seg_meta.get("super_class")
                if ps:
                    ps_norm = _normalize_label_text(ps)
                    primary_counter[ps_norm] += 1
                if sc:
                    sc_norm = _normalize_label_text(sc)
                    super_counter[sc_norm] += 1

                # flush per-key shards
                if len(buffers[key]) >= shard_flush:
                    write_shard_for_key(key)

    # final flush all remaining buffers
    for k in list(buffers.keys()):
        write_shard_for_key(k)

    # summary
    summary = {
        "shards_written_per_key": {f"{k[0]}|{k[1]}": int(shard_idx_map.get(k, 0)) for k in shard_idx_map.keys()},
        "accepted_segments": int(accepted),
        "rejected_segments": int(rejected),
        "total_seen_segments": int(total_seen),
        "rejected_reason_counts": dict(rejected_reason_counter.most_common()),
        "rejected_per_lead": rejected_per_lead,
        "accepted_per_lead_counts": accepted_per_lead_counts,
        "primary_scp_counts": dict(primary_counter.most_common()),
        "super_class_counts": dict(super_counter.most_common()),
    }
    safe_json_dump(summary, base / "shard_summary.json")

    print(
        f"Finished dataset {dataset_name}: shards_summary={summary.get('shards_written_per_key')}, accepted={accepted}, rejected={rejected}, total_seen_segments={total_seen}"
    )

    # plotting and reports
    try:
        before_amp = (
            np.array(before_amp_samples_flat) if before_amp_samples_flat else np.array([])
        )
        before_rms = (
            np.array(before_rms_samples_flat) if before_rms_samples_flat else np.array([])
        )
        after_amp = (
            np.array(after_amp_samples_flat) if after_amp_samples_flat else np.array([])
        )
        after_rms = (
            np.array(after_rms_samples_flat) if after_rms_samples_flat else np.array([])
        )

        if before_amp.size > 0 and after_amp.size > 0:
            plot_hist_before_after(
                before_amp,
                after_amp,
                "Max abs amplitude (flattened)",
                reports_dir / "amp_before.png",
                reports_dir / "amp_after.png",
            )
        if before_rms.size > 0 and after_rms.size > 0:
            plot_hist_before_after(
                before_rms,
                after_rms,
                "RMS (flattened)",
                reports_dir / "rms_before.png",
                reports_dir / "rms_after.png",
            )

        plot_per_lead_heatmap(
            rejected_per_lead,
            "Rejected segments per lead",
            reports_dir / "rejected_per_lead_heatmap.png",
        )

        if before_amp.size > 0:
            plot_boxplot(before_amp, "amp_before", reports_dir / "amp_before_box.png")
        if after_amp.size > 0:
            plot_boxplot(after_amp, "amp_after", reports_dir / "amp_after_box.png")

        try:
            keys = list(summary["rejected_reason_counts"].keys())
            vals = list(summary["rejected_reason_counts"].values())
            if keys:
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(vals)), vals)
                plt.xticks(range(len(vals)), keys, rotation=45, ha="right")
                plt.title("Rejected segments by reason")
                plt.tight_layout()
                plt.savefig(reports_dir / "rejection_reasons_bar.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(6, 6))
                plt.pie(vals, labels=keys, autopct="%1.1f%%")
                plt.title("Rejection reasons pie")
                plt.savefig(reports_dir / "rejection_reasons_pie.png", bbox_inches="tight")
                plt.close()
        except Exception:
            pass

        try:
            accept_rates = []
            for li in range(12):
                tot = accepted_per_lead_counts[li] + rejected_per_lead[li]
                rate = (accepted_per_lead_counts[li] / tot) if tot > 0 else 0.0
                accept_rates.append(rate)
            plt.figure(figsize=(8, 3))
            plt.bar(range(12), accept_rates)
            plt.xticks(range(12), [f"L{i+1}" for i in range(12)])
            plt.ylim(0, 1.0)
            plt.title("Per-lead accept rate (accepted / (accepted+rejected))")
            plt.tight_layout()
            plt.savefig(reports_dir / "per_lead_accept_rate.png", bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        try:
            import pandas as pd

            df_summary = pd.DataFrame(
                {
                    "lead": [f"L{i+1}" for i in range(12)],
                    "rejected_count": rejected_per_lead,
                    "accepted_count": accepted_per_lead_counts,
                }
            )
            df_summary.to_csv(reports_dir / "per_lead_summary.csv", index=False)
        except Exception:
            pass

        try:
            if thresholds:
                amp_lows = [thresholds[i]["amp_low"] for i in range(12)]
                amp_highs = [thresholds[i]["amp_high"] for i in range(12)]
                plt.figure(figsize=(8, 3))
                plt.errorbar(
                    range(12),
                    [(l + h) / 2.0 for l, h in zip(amp_lows, amp_highs)],
                    yerr=[(h - l) / 2.0 for l, h in zip(amp_lows, amp_highs)],
                    fmt="o",
                )
                plt.xticks(range(12), [f"L{i+1}" for i in range(12)])
                plt.title("Per-lead amplitude thresholds (low/high)")
                plt.tight_layout()
                plt.savefig(reports_dir / "amp_thresholds.png", bbox_inches="tight")
                plt.close()
        except Exception:
            pass

    except Exception:
        pass

    print(f"Reports and shard summary saved to: {reports_dir.resolve()}")

# main
if __name__ == "__main__":
    ptbxl_folder = ROOT / "ptbxl_clean"
    ptb_folder = ROOT / "ptb_clean"
    any_processed = False
    if ptbxl_folder.exists():
        process_dataset_two_passes("ptbxl", ptbxl_folder, workers=DEFAULT_WORKERS)
        any_processed = True
    else:
        print("No PTB-XL cleaned folder found under Cleaned_Datasets. Skipping ptbxl.")
    if ptb_folder.exists():
        process_dataset_two_passes("ptb", ptb_folder, workers=DEFAULT_WORKERS)
        any_processed = True
    else:
        print("No PTB cleaned folder found under Cleaned_Datasets. Skipping ptb.")
    if not any_processed:
        print("No cleaned folders found under Cleaned_Datasets. Exiting.")
        raise SystemExit(0)
    print("\nDone. Segmented shards + QC reports under ./Segments/<dataset>/")
