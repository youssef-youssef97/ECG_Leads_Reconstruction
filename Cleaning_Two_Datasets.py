# Usage: python Cleaning_Two_Datasets.py --ptbxl --ptb

# imports
from pathlib import Path
import os
import json
import gc
import argparse
import ast
import re
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import wfdb
except Exception:
    wfdb = None
from scipy.signal import butter, filtfilt, medfilt, decimate, resample_poly, iirnotch
from math import gcd

# params
TARGET_LENGTH = 1000
OUT_ROOT = Path("./Cleaned_Datasets").resolve()
REPORTS_ROOT = OUT_ROOT / "reports"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

PTBXL_PATH = Path("../../../ptbxl")
DATABASE_CSV = PTBXL_PATH / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = PTBXL_PATH / "scp_statements.csv"
PTBXL_OUT_DIR = OUT_ROOT / "ptbxl_clean"
PTBXL_OUT_DIR.mkdir(parents=True, exist_ok=True)
PTBXL_FS = 100.0
PTBXL_USE_LR = True

# create fold dirs for ptbxl
for fold in range(1, 9):
    (PTBXL_OUT_DIR / "train" / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
(PTBXL_OUT_DIR / "val" / "fold_9").mkdir(parents=True, exist_ok=True)
(PTBXL_OUT_DIR / "test" / "fold_10").mkdir(parents=True, exist_ok=True)

PTB_PATH = Path("../../../ptb")
PTB_OUT_DIR = OUT_ROOT / "ptb_clean"
PTB_OUT_DIR.mkdir(parents=True, exist_ok=True)

POWERLINE_FREQS = (50.0, 60.0)
NOTCH_Q = 30.0
DEFAULT_WORKERS = max(1, min(32, (multiprocessing.cpu_count() or 2) - 1))

# helpers
def _choose_powerline_freq(fs: float):
    nyq = fs / 2.0
    freqs = [f for f in POWERLINE_FREQS if f < nyq]
    if not freqs:
        return None
    return freqs[0] if len(freqs) == 1 else (50.0 if 50.0 < nyq else 60.0)

def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 0.9999
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0, padlen=3 * (max(len(a), len(b))))

def notch_filter(data: np.ndarray, fs: float, freq: float = None, q: float = NOTCH_Q):
    if freq is None:
        freq = _choose_powerline_freq(fs)
    if freq is None:
        return data
    nyq = fs / 2.0
    w0 = freq / nyq
    if not (0 < w0 < 1.0):
        return data
    b, a = iirnotch(w0, q)
    try:
        return filtfilt(b, a, data, axis=0, padlen=3 * (max(len(a), len(b))))
    except Exception:
        out = np.zeros_like(data)
        for ch in range(data.shape[1]):
            out[:, ch] = filtfilt(b, a, data[:, ch], padlen=3 * (max(len(a), len(b))))
        return out

def baseline_wander_removal(data: np.ndarray, fs: float = 100.0, window_sec: float = 0.6):
    window_size = int(round(window_sec * fs))
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    baseline = np.zeros_like(data)
    for ch in range(data.shape[1]):
        baseline[:, ch] = medfilt(data[:, ch], kernel_size=window_size)
    return data - baseline

def pad_or_trim(ecg: np.ndarray, targ_len: int = TARGET_LENGTH) -> np.ndarray:
    if ecg is None:
        return np.zeros((targ_len, 12), dtype=np.float32)
    t, c = ecg.shape
    if t > targ_len:
        return ecg[:targ_len, :]
    elif t < targ_len:
        pad = np.zeros((targ_len - t, c), dtype=ecg.dtype)
        return np.concatenate([ecg, pad], axis=0)
    return ecg

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

# mapping
def load_scp_statements_map(scp_csv: Path):
    mapping = {}
    if not scp_csv.exists():
        return mapping
    try:
        df = pd.read_csv(scp_csv, dtype=str).fillna("")
        code_col = None
        super_col = None
        for c in df.columns:
            lc = c.lower()
            if "code" in lc or "scp" in lc:
                code_col = c
            if "super" in lc or "superclass" in lc or "super_class" in lc:
                super_col = c
        if code_col is None and len(df.columns) >= 1:
            code_col = df.columns[0]
        if super_col is None:
            for c in df.columns:
                if df[c].nunique() < 500:
                    super_col = c
                    break
        if code_col is None:
            return mapping
        for _, r in df.iterrows():
            code = str(r.get(code_col, "")).strip()
            sc = str(r.get(super_col, "")).strip() if super_col else ""
            if code:
                mapping[code] = sc if sc else ""
    except Exception:
        pass
    return mapping

SCP_TO_SUPER = load_scp_statements_map(SCP_STATEMENTS_CSV) if SCP_STATEMENTS_CSV.exists() else {}

# readers
def _read_ptbxl_record_local(filename: Path):
    if wfdb is None:
        raise RuntimeError("wfdb not available to read PTB-XL local records")
    record = wfdb.rdrecord(str(filename))
    sig = record.p_signal.astype(np.float32)
    if sig.shape[1] >= 12:
        sig = sig[:, :12]
    else:
        pad = np.zeros((sig.shape[0], 12 - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)
    return sig

def _read_ptb_record_local(filename: Path):
    if wfdb is None:
        raise RuntimeError("wfdb not available to read PTB records")
    base = Path(filename)
    if base.suffix == ".hea" or base.suffix == ".dat":
        base = base.with_suffix("")
    record = wfdb.rdrecord(str(base))
    sig = record.p_signal.astype(np.float32)
    fs = None
    if hasattr(record, "fs"):
        try:
            fs = float(record.fs)
        except Exception:
            pass
    if fs is None and hasattr(record, "sample_rate"):
        try:
            fs = float(record.sample_rate)
        except Exception:
            pass
    if fs is None:
        fs = PTBXL_FS
    return sig, fs

# process ptbxl worker
def _process_ptbxl_worker(args):
    # worker
    idx, fn_path, out_base_dir, meta_info, keep_noisy = args
    try:
        rec_id = f"ptbxl_{idx:07d}"
        sig = _read_ptbxl_record_local(Path(fn_path))
        sig_clean = notch_filter(sig, fs=PTBXL_FS)
        sig_clean = butter_bandpass_filter(sig_clean, fs=PTBXL_FS)
        sig_clean = baseline_wander_removal(sig_clean, fs=PTBXL_FS)
        sig_clean = pad_or_trim(sig_clean, targ_len=TARGET_LENGTH)

        # decide fold output dir
        strat = meta_info.get("strat_fold", None)
        try:
            strat_int = int(str(strat)) if strat is not None and not pd.isna(strat) else None
        except Exception:
            strat_int = None

        if strat_int is None or strat_int < 1 or strat_int > 10:
            # fallback: put in train/fold_1
            target_dir = Path(out_base_dir) / "train" / "fold_1"
        else:
            if 1 <= strat_int <= 8:
                target_dir = Path(out_base_dir) / "train" / f"fold_{strat_int}"
            elif strat_int == 9:
                target_dir = Path(out_base_dir) / "val" / "fold_9"
            else:  # 10
                target_dir = Path(out_base_dir) / "test" / "fold_10"

        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{rec_id}.npy"
        np.save(out_path, sig_clean.astype(np.float32))

        # meta compose
        meta = {
            "rec_id": rec_id,
            "dataset": "ptbxl",
            "source_path": str(fn_path),
            "patient_id": None,
            "age": None,
            "sex": None,
            "fs": PTBXL_FS,
            "strat_fold": None,
        }
        try:
            if isinstance(meta_info, dict):
                meta["patient_id"] = meta_info.get("patient_id")
                meta["age"] = meta_info.get("age")
                meta["sex"] = meta_info.get("sex")
                meta["strat_fold"] = meta_info.get("strat_fold")
                if meta_info.get("scp_codes_raw"):
                    meta["scp_codes_raw"] = meta_info.get("scp_codes_raw")
                if meta_info.get("primary_scp"):
                    meta["primary_scp"] = meta_info.get("primary_scp")
                if meta_info.get("super_class"):
                    meta["super_class"] = meta_info.get("super_class")
                if meta_info.get("primary_vector") is not None:
                    meta["primary_vector"] = meta_info.get("primary_vector")
        except Exception:
            pass

        meta_path = target_dir / f"{rec_id}_meta.json"
        safe_json_dump(meta, meta_path)

        del sig, sig_clean
        gc.collect()
        return {"status": "kept", "rec_id": rec_id, "fold": meta.get("strat_fold")}
    except Exception as e:
        return {"status": "error", "reason": str(e), "rec_id": None}

# process ptb worker
def _process_ptb_worker(args):
    # worker
    idx, patient_dir, out_dir = args
    try:
        patient_dir = Path(patient_dir)
        patient_name = patient_dir.name
        m = re.search(r"(\d+)", patient_name)
        patient_id = m.group(1) if m else patient_name
        results = []
        hea_files = sorted([p for p in patient_dir.glob("*.hea")])
        if not hea_files:
            return {"status": "empty", "patient_dir": str(patient_dir), "processed": 0}
        for hea in hea_files:
            stem = hea.stem
            try:
                # header
                diag_map = {}
                try:
                    with open(hea, "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("#"):
                                txt = line.lstrip("#").strip()
                                if ":" in txt:
                                    k, v = txt.split(":", 1)
                                    diag_map[k.strip()] = v.strip()
                                else:
                                    # fallback: add as raw
                                    diag_map.setdefault("header_comments", []).append(txt)
                except Exception:
                    pass

                sig, fs = _read_ptb_record_local(hea)

                # --- resample PTB recordings to PTBXL_FS (100 Hz) if needed ---
                if fs is None:
                    fs = PTBXL_FS
                # If sampling rate differs from target (PTBXL_FS), resample before filtering
                if abs(float(fs) - float(PTBXL_FS)) > 1e-6:
                    # compute integer ratio up/down and reduce by gcd
                    up = int(round(PTBXL_FS))
                    down = int(round(fs))
                    if down <= 0:
                        down = int(round(fs)) if int(round(fs)) > 0 else 1
                    g = gcd(up, down) if (up > 0 and down > 0) else 1
                    up_r = up // g
                    down_r = down // g
                    try:
                        sig = resample_poly(sig, up_r, down_r, axis=0)
                        fs = PTBXL_FS
                    except Exception:
                        # fallback: if down/up is integer decimation
                        try:
                            if down_r % up_r == 0 and down_r // up_r > 0:
                                factor = down_r // up_r
                                sig = decimate(sig, factor, axis=0)
                                fs = PTBXL_FS
                        except Exception:
                            # if resampling fails, keep original sig and fs (will be processed at original fs)
                            pass
                # --- end resample ---

                sig_clean = notch_filter(sig, fs=fs)
                sig_clean = butter_bandpass_filter(sig_clean, fs=fs)
                sig_clean = baseline_wander_removal(sig_clean, fs=fs)
                sig_clean = pad_or_trim(sig_clean, targ_len=TARGET_LENGTH)
                rec_id = f"ptb_{patient_id}_{stem}"
                out_path = Path(out_dir) / f"{rec_id}.npy"
                np.save(out_path, sig_clean.astype(np.float32))

                # meta
                meta = {
                    "rec_id": rec_id,
                    "dataset": "ptb",
                    "patient_id": patient_id,
                    "ptb_header_parsed": diag_map
                }
                try:
                    # map Reason for admission to primary_scp for consistency with PTB-XL metadata fields
                    reason = diag_map.get("Reason for admission") or diag_map.get("Diagnose") or diag_map.get("Diagnosis")
                    if reason:
                        meta["primary_scp"] = reason
                        meta["super_class"] = reason
                    # keep some common keys as top-level if present
                    if "Additional diagnoses" in diag_map:
                        meta["additional_diagnoses"] = diag_map.get("Additional diagnoses")
                    if "Smoker" in diag_map:
                        meta["smoker"] = diag_map.get("Smoker")
                    if "Number of coronary vessels involved" in diag_map:
                        meta["num_coronary_vessels"] = diag_map.get("Number of coronary vessels involved")
                    if "Acute infarction (localization)" in diag_map:
                        meta["acute_infarction_localization"] = diag_map.get("Acute infarction (localization)")
                    if "Infarction date (acute)" in diag_map:
                        meta["infarction_date_acute"] = diag_map.get("Infarction date (acute)")
                except Exception:
                    pass

                meta_path = Path(out_dir) / f"{rec_id}_meta.json"
                safe_json_dump(meta, meta_path)
                results.append({"status": "kept", "rec_id": rec_id})
                del sig, sig_clean
                gc.collect()
            except Exception as e:
                results.append({"status": "error", "reason": str(e), "rec_id": f"ptb_{patient_id}_{stem}"})
        return {"status": "done", "patient_dir": str(patient_dir), "processed": len(results), "results": results}
    except Exception as e:
        return {"status": "error", "reason": str(e), "patient_dir": str(patient_dir)}

# report
def summarize_and_report(dataset_name: str, out_dir: Path, results: list):
    # report
    reports_dir = REPORTS_ROOT / dataset_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    total = len(results)
    kept = sum(1 for r in results if r and r.get("status") == "kept")
    errors = sum(1 for r in results if r and r.get("status") == "error")
    reasons = {}
    for r in results:
        if r and r.get("status") == "error":
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
    summary = {
        "dataset": dataset_name,
        "total_records": total,
        "kept": kept,
        "errors": errors,
        "rejection_reasons": reasons,
    }
    safe_json_dump(summary, reports_dir / f"{dataset_name}_summary.json")
    try:
        pd.DataFrame(results).to_csv(reports_dir / f"{dataset_name}_records_summary.csv", index=False)
    except Exception:
        pass
    print(f"Reports saved to: {reports_dir}")

# run ptbxl
def run_ptbxl(keep_noisy: bool, workers: int = DEFAULT_WORKERS):
    # load meta
    print("\n=== PTB-XL: Loading metadata and enumerating records ===")
    if not DATABASE_CSV.exists():
        print("[ERROR] PTBXL CSV not found at", DATABASE_CSV)
        return
    df = pd.read_csv(DATABASE_CSV)
    expected_cols = ["validated_by_human", "filename_lr", "filename_hr", "electrodes_problems", "pacemaker", "burst_noise", "static_noise", "patient_id", "age", "sex", "scp_codes", "strat_fold"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    if keep_noisy:
        df_sel = df[df["filename_lr"].notnull() if PTBXL_USE_LR else df["filename_hr"].notnull()]
    else:
        df_sel = df[(df.get("validated_by_human", False) == True) & (df["filename_lr"].notnull())]
        df_sel = df_sel[(df_sel["electrodes_problems"] == 0) & (df_sel["pacemaker"] == 0) & (df_sel["burst_noise"] == 0) & (df_sel["static_noise"] == 0)]
    file_paths = []
    meta_infos = []
    for i, (_, row) in enumerate(df_sel.iterrows()):
        rel_path = row["filename_lr"] if PTBXL_USE_LR else row["filename_hr"]
        full_path = PTBXL_PATH / rel_path
        file_paths.append(str(full_path))
        scp_raw = row.get("scp_codes", None)
        scp_dict = {}
        primary_scp = None
        super_class = None
        if isinstance(scp_raw, str) and scp_raw.strip():
            try:
                parsed = ast.literal_eval(scp_raw)
                if isinstance(parsed, dict) and parsed:
                    scp_dict = parsed
                    primary_scp = max(parsed, key=parsed.get)
                    super_class = SCP_TO_SUPER.get(primary_scp, None)
            except Exception:
                toks = re.findall(r"[A-Z0-9_]{2,10}", str(scp_raw))
                if toks:
                    primary_scp = toks[0]
                    super_class = SCP_TO_SUPER.get(primary_scp, None)

        # strat fold parsing
        strat_val = row.get("strat_fold", None)
        strat_fold = None
        try:
            if not pd.isna(strat_val):
                strat_fold = int(str(strat_val))
        except Exception:
            strat_fold = None

        meta_infos.append({
            "patient_id": (None if pd.isna(row.get("patient_id")) else row.get("patient_id")),
            "age": None if pd.isna(row.get("age")) else row.get("age"),
            "sex": None if pd.isna(row.get("sex")) else row.get("sex"),
            "scp_codes_raw": scp_dict,
            "primary_scp": primary_scp,
            "super_class": super_class,
            "strat_fold": strat_fold,
        })

    # build vocab and weighted vectors
    all_codes = set()
    for m in meta_infos:
        scp = m.get("scp_codes_raw") or {}
        for k in scp.keys():
            all_codes.add(k)
    all_codes = sorted(list(all_codes))
    safe_json_dump(all_codes, PTBXL_OUT_DIR / "ptbxl_primary_scp_vocab.json")

    # create weighted vectors (value scaled to [0,1])
    for m in meta_infos:
        scp = m.get("scp_codes_raw") or {}
        vec = []
        for c in all_codes:
            v = scp.get(c, 0)
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            # clip and normalize to [0,1]
            fv = max(0.0, min(100.0, fv)) / 100.0
            vec.append(fv)
        m["primary_vector"] = vec

    print(f"PTB-XL: {len(file_paths)} records to process (workers={workers})")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        futures = []
        for i, fp in enumerate(file_paths):
            futures.append(exe.submit(_process_ptbxl_worker, (i, fp, str(PTBXL_OUT_DIR), meta_infos[i], keep_noisy)))
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="PTB-XL"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"status": "error", "reason": str(e)})

    # save a mapping summary of rec_id -> fold and some meta
    try:
        mapping = []
        for r in results:
            if r and r.get("status") == "kept":
                rec = r.get("rec_id")
                # attempt to read meta
                # search train/val/test directories for meta file (fast check)
                found_meta = None
                for base in ["train", "val", "test"]:
                    # if many folds, we can search relevant dirs
                    base_dir = PTBXL_OUT_DIR / base
                    if not base_dir.exists():
                        continue
                    for p in base_dir.rglob(f"{rec}_meta.json"):
                        try:
                            with open(p, "r", encoding="utf-8") as fh:
                                found_meta = json.load(fh)
                                break
                        except Exception:
                            continue
                    if found_meta is not None:
                        break
                if found_meta is not None:
                    mapping.append(found_meta)
        safe_json_dump(mapping, PTBXL_OUT_DIR / "ptbxl_records_mapping.json")
    except Exception:
        pass

    summarize_and_report("ptbxl", PTBXL_OUT_DIR, results)

# run ptb
def run_ptb(workers: int = DEFAULT_WORKERS):
    # ptb
    print("\n=== PTB: Enumerating patient directories ===")
    if not PTB_PATH.exists():
        print("[ERROR] PTB path not found at", PTB_PATH)
        return
    patient_dirs = sorted([p for p in PTB_PATH.iterdir() if p.is_dir()])
    if not patient_dirs:
        print("[ERROR] No patient directories found in", PTB_PATH)
        return
    print(f"PTB: {len(patient_dirs)} patient directories to process (workers={workers})")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        futures = []
        for i, pd_dir in enumerate(patient_dirs):
            futures.append(exe.submit(_process_ptb_worker, (i, str(pd_dir), str(PTB_OUT_DIR))))
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="PTB"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"status": "error", "reason": str(e)})
    summarize_and_report("ptb", PTB_OUT_DIR, results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl", action="store_true")
    p.add_argument("--ptb", action="store_true")
    p.add_argument("--keep-noisy", dest="keep_noisy", action="store_true")
    p.add_argument("--no-keep-noisy", dest="keep_noisy", action="store_false")
    p.set_defaults(keep_noisy=True)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()
    if not args.ptbxl and not args.ptb:
        print("Nothing requested. Use --ptbxl or --ptb")
        raise SystemExit(0)
    if args.ptbxl:
        run_ptbxl(keep_noisy=args.keep_noisy, workers=args.workers)
    if args.ptb:
        run_ptb(workers=args.workers)
