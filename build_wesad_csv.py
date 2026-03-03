import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


@dataclass(frozen=True)
class WESADConfig:
    wesad_dir: str
    out_csv: str
    label_sf: int = 700
    bvp_sf: int = 64
    eda_sf: int = 4
    win_sec: int = 60
    step_sec: int = 30

    # WESAD original labels (per dataset docs)
    baseline_label: int = 1
    stress_label: int = 2
    meditation_label: int = 4
    invalid_labels: tuple = (0, 3, 5, 6, 7)  # includes amusement=3


def _bandpass(x: np.ndarray, fs: int, low: float = 0.5, high: float = 4.0, order: int = 2) -> np.ndarray:
    # Typical heart-rate band for BVP
    nyq = 0.5 * fs
    lowc = low / nyq
    highc = high / nyq
    b, a = butter(order, [lowc, highc], btype="bandpass")
    return filtfilt(b, a, x)


def _compute_hr_rmssd_from_bvp(bvp: np.ndarray, fs: int) -> tuple[float, float] | tuple[None, None]:
    bvp = np.asarray(bvp).reshape(-1)
    if bvp.size < fs * 10:  # need at least ~10s
        return None, None

    bvp = np.nan_to_num(bvp, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    try:
        bvp_f = _bandpass(bvp, fs=fs)
    except Exception:
        bvp_f = bvp

    # Peak detection (conservative defaults)
    # min distance ~0.3s -> supports up to ~200 bpm
    min_dist = max(1, int(0.30 * fs))
    peaks, _ = find_peaks(bvp_f, distance=min_dist, prominence=np.std(bvp_f) * 0.25)
    if peaks.size < 3:
        return None, None

    rr_s = np.diff(peaks) / float(fs)
    rr_s = rr_s[(rr_s > 0.30) & (rr_s < 2.0)]  # 30–200 bpm
    if rr_s.size < 2:
        return None, None

    hr = 60.0 / float(np.mean(rr_s))
    rr_ms = rr_s * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if rr_ms.size >= 2 else None
    return float(hr), float(rmssd) if rmssd is not None else None


def _mode_label(labels: np.ndarray) -> int | None:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return None
    # fast mode for small int range
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def build_wesad_csv(cfg: WESADConfig) -> pd.DataFrame:
    rows: list[dict] = []

    if not os.path.isdir(cfg.wesad_dir):
        raise FileNotFoundError(f"WESAD dir not found: {cfg.wesad_dir}")

    # subjects 2..17 excluding 12 (as in original notebooks)
    subjects = [f"S{i}" for i in range(2, 18) if i != 12]

    for subj in subjects:
        pkl_path = os.path.join(cfg.wesad_dir, subj, f"{subj}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[WARN] Missing {pkl_path} (skipping)")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        labels = np.asarray(data.get("label")).reshape(-1)
        wrist = data.get("signal", {}).get("wrist", {})
        bvp = np.asarray(wrist.get("BVP")).reshape(-1)
        eda = np.asarray(wrist.get("EDA")).reshape(-1)

        # Determine usable duration (seconds) across label & sensors
        dur_sec_labels = int(labels.size // cfg.label_sf)
        dur_sec_bvp = int(bvp.size // cfg.bvp_sf)
        dur_sec_eda = int(eda.size // cfg.eda_sf) if eda.size else 0
        total_sec = min(dur_sec_labels, dur_sec_bvp, dur_sec_eda)
        if total_sec < cfg.win_sec:
            print(f"[WARN] {subj}: too short ({total_sec}s), skipping")
            continue

        for start in range(0, total_sec - cfg.win_sec + 1, cfg.step_sec):
            end = start + cfg.win_sec

            lab_seg = labels[start * cfg.label_sf : end * cfg.label_sf]
            lab = _mode_label(lab_seg)
            if lab is None or lab in cfg.invalid_labels:
                continue

            # map meditation to baseline, then to binary 0/1
            if lab == cfg.meditation_label:
                lab = cfg.baseline_label
            if lab not in (cfg.baseline_label, cfg.stress_label):
                continue

            stress = 1 if lab == cfg.stress_label else 0

            bvp_seg = bvp[start * cfg.bvp_sf : end * cfg.bvp_sf]
            eda_seg = eda[start * cfg.eda_sf : end * cfg.eda_sf]
            hr, rmssd = _compute_hr_rmssd_from_bvp(bvp_seg, fs=cfg.bvp_sf)
            if hr is None or rmssd is None:
                continue

            scl = float(np.nanmean(eda_seg)) if eda_seg.size else None
            if scl is None or not np.isfinite(scl):
                continue

            rows.append(
                {
                    "id": subj.replace("S", ""),
                    "HR": float(hr),
                    "RMSSD": float(rmssd),
                    "SCL": float(scl),
                    "stress": int(stress),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No samples extracted from WESAD. Check data and window settings.")

    # Save
    out_dir = os.path.dirname(cfg.out_csv)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(cfg.out_csv, index=False)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wesad_dir = os.path.join(base_dir, "WESAD")
    out_csv = os.path.join(base_dir, "Final_CSVs", "wesad_new_with1.csv")
    cfg = WESADConfig(wesad_dir=wesad_dir, out_csv=out_csv)
    df = build_wesad_csv(cfg)
    print("\nSaved:", out_csv)
    print("Rows:", len(df))
    print("Class distribution (stress):")
    print(df["stress"].value_counts(dropna=False).sort_index())
