"""STEP 5-2: reproducible Criteo split (final test / validation / nested training candidates). No training.

1. Reads only treatment and conversion (int8) to build the 4 strata (treatment x conversion).
2. Proportional stratified random split with seed 42: test 20% first, then validation 20%, then the
   remaining 60% training pool in random order; nested training candidates 100k / 300k / 1M are
   prefixes of that order (see src/criteo_data.py).
3. Saves sorted original row numbers as int32 .npy under data/criteo/splits/ (Git-ignored). Existing
   index files are never overwritten: they must equal the regenerated split or the script stops.
4. Verifies index integrity and compares every split with the full data in one chunked pass
   (rates, conversions per arm, feature SMD vs. full data).

Usage (from the project root):
    .venv/bin/python scripts/criteo_prepare.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.criteo_data import (  # noqa: E402
    DATA_PATH,
    DATA_SHA256,
    FEATURES,
    N_ROWS,
    OUTCOME,
    POST_ASSIGNMENT,
    SPLIT_DIR,
    TREATMENT,
    check_splits,
    make_splits,
    strata,
)

TABLES = PROJECT_ROOT / "results" / "tables"
SEED = 42
TEST_FRAC = 0.2
VALID_FRAC = 0.2
TRAIN_SIZES = [100_000, 300_000, 1_000_000]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rss_mb() -> float:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def read_assignment(path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.empty(N_ROWS, dtype=np.int8)
    y = np.empty(N_ROWS, dtype=np.int8)
    pos = 0
    for chunk in pd.read_csv(path, usecols=[TREATMENT, OUTCOME], dtype=np.int8, chunksize=chunksize):
        t[pos:pos + len(chunk)] = chunk[TREATMENT].to_numpy()
        y[pos:pos + len(chunk)] = chunk[OUTCOME].to_numpy()
        pos += len(chunk)
    if pos != N_ROWS:
        raise SystemExit(f"row count {pos} != {N_ROWS}")
    return t, y


def save_or_verify(splits: dict[str, np.ndarray]) -> dict:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, idx in splits.items():
        path = SPLIT_DIR / f"{name}.npy"
        arr = idx.astype(np.int32)
        if path.exists():
            if not np.array_equal(np.load(path), arr):
                raise SystemExit(f"{path} exists and differs from the regenerated split; refusing to overwrite")
            status = "existing file verified identical (not rewritten)"
        else:
            np.save(path, arr)
            status = "written"
        files[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "n_rows": int(len(arr)), "dtype": "int32",
                       "sorted": True, "bytes": path.stat().st_size, "sha256": sha256(path), "status": status}
    return files


def compare_pass(path: Path, splits: dict[str, np.ndarray], chunksize: int) -> dict:
    """One chunked pass: per-split counts and feature moments (full data included)."""
    names = ["full"] + list(splits)
    masks = {k: np.zeros(N_ROWS, dtype=bool) for k in splits}
    for k, idx in splits.items():
        masks[k][idx] = True
    acc = {k: {"n": 0, "n_t": 0, "y_t": 0, "y_c": 0, "visit": 0, "exposure": 0,
               "s": np.zeros(12), "ss": np.zeros(12)} for k in names}
    pos = 0
    cols = FEATURES + [TREATMENT, OUTCOME, *POST_ASSIGNMENT]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        X = chunk[FEATURES].to_numpy(dtype=np.float64)
        t, y = chunk[TREATMENT].to_numpy(), chunk[OUTCOME].to_numpy()
        v, e = chunk["visit"].to_numpy(), chunk["exposure"].to_numpy()
        for k in names:
            m = slice(None) if k == "full" else masks[k][pos:pos + len(chunk)]
            a = acc[k]
            tm, ym = t[m], y[m]
            a["n"] += int(len(tm))
            a["n_t"] += int(tm.sum())
            a["y_t"] += int((tm * ym).sum())
            a["y_c"] += int(((1 - tm) * ym).sum())
            a["visit"] += int(v[m].sum())
            a["exposure"] += int(e[m].sum())
            a["s"] += X[m].sum(axis=0)
            a["ss"] += (X[m] ** 2).sum(axis=0)
        pos += len(chunk)
    return acc


def summarize(acc: dict, splits: dict[str, np.ndarray], stratum: np.ndarray) -> tuple[pd.DataFrame, dict]:
    full = acc["full"]
    mu_f = full["s"] / full["n"]
    sd_f = np.sqrt(full["ss"] / full["n"] - mu_f ** 2)
    pop_share = np.bincount(stratum, minlength=4) / len(stratum)
    rows, smd = [], {}
    for k, a in acc.items():
        n, n_t = a["n"], a["n_t"]
        n_c = n - n_t
        mu = a["s"] / n
        d = (mu - mu_f) / sd_f
        smd[k] = dict(zip(FEATURES, d.round(6).tolist()))
        cell = np.bincount(stratum[splits[k]], minlength=4) if k != "full" else np.bincount(stratum, minlength=4)
        expected = n * pop_share
        rows.append({
            "split": k, "n_rows": n, "share_of_full": n / full["n"],
            "n_treated": n_t, "n_control": n_c, "treatment_ratio": n_t / n,
            "conversions": a["y_t"] + a["y_c"], "conversion_rate": (a["y_t"] + a["y_c"]) / n,
            "conv_treated": a["y_t"], "conv_control": a["y_c"],
            "conv_rate_treated": a["y_t"] / n_t, "conv_rate_control": a["y_c"] / n_c,
            "rate_diff_treated_minus_control": a["y_t"] / n_t - a["y_c"] / n_c,
            "treatment_ratio_diff_vs_full": n_t / n - full["n_t"] / full["n"],
            "conversion_rate_diff_vs_full": (a["y_t"] + a["y_c"]) / n - (full["y_t"] + full["y_c"]) / full["n"],
            "max_abs_stratum_count_dev_vs_proportional": float(np.abs(cell - expected).max()),
            "max_abs_feature_smd_vs_full": float(np.abs(d).max()),
            "max_abs_feature_smd_feature": FEATURES[int(np.abs(d).argmax())],
            "visit_rate": a["visit"] / n, "exposure_rate_treated": a["exposure"] / n_t,
        })
    return pd.DataFrame(rows), smd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    if sha256(DATA_PATH) != DATA_SHA256:
        raise SystemExit("data file SHA256 mismatch")
    t, y = read_assignment(DATA_PATH, args.chunksize)
    stratum = strata(t, y)
    del t, y
    splits, alloc = make_splits(stratum, TEST_FRAC, VALID_FRAC, TRAIN_SIZES, SEED)
    checks = check_splits(splits, N_ROWS, TRAIN_SIZES)
    if not checks["all_ok"]:
        raise SystemExit(f"split integrity check failed: {checks}")
    print(f"split built in {time.perf_counter() - t0:.1f}s (peak RSS {rss_mb():.0f} MB)", flush=True)
    files = save_or_verify(splits)
    acc = compare_pass(DATA_PATH, splits, args.chunksize)
    table, smd = summarize(acc, splits, stratum)
    split_order = ["full", "test", "valid", "train_pool"] + [f"train_{s}" for s in TRAIN_SIZES]
    table = table.set_index("split").loc[split_order].reset_index()
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "criteo_split_summary.csv", index=False, float_format="%.8g")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": "STEP 5-2 data split and evaluation pipeline preparation (no model training)",
        "data": {"path": str(DATA_PATH.relative_to(PROJECT_ROOT)), "sha256": DATA_SHA256, "n_rows": N_ROWS},
        "research_variables": {"X": FEATURES, "T": TREATMENT, "Y": OUTCOME, "estimand": "ITT of campaign assignment",
                               "never_used_as_X_or_T": list(POST_ASSIGNMENT)},
        "split_design": {
            "seed": SEED, "rng": "numpy.random.default_rng(42); one permutation per stratum, strata in order 0..3",
            "strata": "2 * treatment + conversion (0: T0Y0, 1: T0Y1, 2: T1Y0, 3: T1Y1)",
            "allocation": "largest remainder, proportional to full-data stratum counts",
            "order": "per stratum permutation -> first test, then validation, rest = training pool (random order)",
            "test_frac": TEST_FRAC, "valid_frac": VALID_FRAC, "train_sizes": TRAIN_SIZES,
            "nested": "train_100000 ⊂ train_300000 ⊂ train_1000000 ⊂ train_pool (prefixes of the pool order)",
            "row_numbers": "0-based data-row positions in the CSV (header excluded)",
            "usage": ("test: final evaluation only, once; valid: model / setting choices; "
                      "train_*: fitting only. Validation rows are never fitted."),
            "duplicates": "identical feature rows may fall into different splits; not removed (no user id)",
        },
        "allocation_by_stratum": alloc.to_dict(orient="records"),
        "index_files": files,
        "checks": checks,
        "feature_smd_vs_full": smd,
        "processing": {"chunksize": args.chunksize, "peak_rss_mb": rss_mb(), "wall_time_sec": time.perf_counter() - t0},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: version(p) for p in ["numpy", "pandas"]}},
    }
    (TABLES / "criteo_split_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(table.to_string(index=False))
    print(f"checks all_ok={checks['all_ok']}  done in {time.perf_counter() - t0:.0f}s, peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
