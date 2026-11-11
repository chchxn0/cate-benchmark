"""STEP 5-1: Criteo Uplift v2.1 structure check and EDA (no model training).

Reads data/criteo/criteo-research-uplift-v2.1.csv.gz in chunks, never holding the whole table in
memory. Feature columns are spilled to on-disk float64 memmaps (one column in RAM at a time) so that
exact quantiles and unique counts can be computed; duplicate rows are detected via 64-bit row hashes.

Research variables (fixed): X = f0..f11, T = treatment, Y = conversion. visit and exposure are
described here but are never features or the primary outcome.

Usage (from the project root):
    .venv/bin/python scripts/criteo_eda.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
TABLES = PROJECT_ROOT / "results" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "criteo_eda"

FEATURES = [f"f{i}" for i in range(12)]
BINARY = ["treatment", "conversion", "visit", "exposure"]
COLUMNS = FEATURES + BINARY
EXPECTED_ROWS = 13_979_592
EXPECTED_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
SAMPLE_SIZES = [100_000, 300_000, 500_000, 1_000_000]
TEST_SIZE = 0.2
TOP_FRACTIONS = [0.1, 0.2, 0.3]
BLOCK = 100_000  # row-order diagnostic block size

SOURCE = {
    "official_page": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
    "official_page_download_link": "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
    "official_page_download_status": ("2026-11-11: redirects to criteostorage.blob.core.windows.net, "
                                      "which does not resolve (DNS NXDOMAIN); download failed"),
    "downloaded_from": "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "downloaded_from_note": ("Hugging Face organization 'criteo' (Verified, 'Criteo AI Lab', links to ailab.criteo.com); "
                             "same file name as the official page link; SHA256 equals the value published on the file page"),
    "version": "v2.1 (un-biased version released after the ERRATUM; the original 25,309,483-row version is not used)",
    "license": "CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)",
    "citation": ("Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). A Large Scale Benchmark for Uplift "
                 "Modeling. Proceedings of the AdKDD and TargetAd Workshop, KDD 2018, London, United Kingdom."),
    "official_key_figures_v2_1": {"rows": 13_979_592, "visit_rate": 0.046992, "conversion_rate": 0.00292,
                                  "treatment_ratio": 0.85, "size_compressed": "297M"},
}

KOREAN_FONTS = ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
C_TREATED, C_CONTROL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#5f5e58", "#e4e3df"


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


def scan(path: Path, cache: Path, chunksize: int) -> dict:
    """Single streaming pass: counts, NaN/inf, per-group moments, row hashes, feature memmaps."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != COLUMNS:
        raise ValueError(f"unexpected columns: {header}")
    mm = {f: np.lib.format.open_memmap(cache / f"{f}.npy", mode="w+", dtype=np.float64, shape=(EXPECTED_ROWS,))
          for f in FEATURES}
    treat = np.empty(EXPECTED_ROWS, dtype=np.int8)
    hash_row = np.empty(EXPECTED_ROWS, dtype=np.uint64)
    hash_x = np.empty(EXPECTED_ROWS, dtype=np.uint64)
    nan = dict.fromkeys(COLUMNS, 0)
    inf = dict.fromkeys(COLUMNS, 0)
    nonbinary = dict.fromkeys(BINARY, 0)
    joint = {}
    grp = {g: {"n": 0, "s": np.zeros(12), "ss": np.zeros(12)} for g in (0, 1)}
    raw_dtypes = None
    pos = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if raw_dtypes is None:
            raw_dtypes = {c: str(t) for c, t in chunk.dtypes.items()}
        k = len(chunk)
        if pos + k > EXPECTED_ROWS:
            raise ValueError("more rows than expected; aborting before overflowing the memmaps")
        for c in COLUMNS:
            v = chunk[c].to_numpy(dtype=np.float64)
            nan[c] += int(np.isnan(v).sum())
            inf[c] += int(np.isinf(v).sum())
        for c in BINARY:
            nonbinary[c] += int((~chunk[c].isin([0, 1])).sum())
        X = chunk[FEATURES].to_numpy(dtype=np.float64)
        t = chunk["treatment"].to_numpy(dtype=np.int8)
        for i, f in enumerate(FEATURES):
            mm[f][pos:pos + k] = X[:, i]
        treat[pos:pos + k] = t
        hash_row[pos:pos + k] = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        hash_x[pos:pos + k] = pd.util.hash_pandas_object(chunk[FEATURES], index=False).to_numpy()
        for g in (0, 1):
            Xg = X[t == g]
            grp[g]["n"] += len(Xg)
            grp[g]["s"] += Xg.sum(axis=0)
            grp[g]["ss"] += (Xg ** 2).sum(axis=0)
        for key, cnt in chunk.groupby(BINARY).size().items():
            joint[key] = joint.get(key, 0) + int(cnt)
        pos += k
        print(f"  rows read: {pos:,}  (peak RSS {rss_mb():.0f} MB)", flush=True)
    for f in FEATURES:
        mm[f].flush()
    return dict(n_rows=pos, raw_dtypes=raw_dtypes, nan=nan, inf=inf, nonbinary=nonbinary, joint=joint, grp=grp,
                treat=treat[:pos], hash_row=hash_row[:pos], hash_x=hash_x[:pos])


def duplicate_stats(h: np.ndarray) -> dict:
    _, counts = np.unique(h, return_counts=True)
    dup_groups = counts[counts > 1]
    return {"distinct": int(len(counts)), "rows_in_duplicate_groups": int(dup_groups.sum()),
            "extra_copies": int((dup_groups - 1).sum()), "duplicate_groups": int(len(dup_groups)),
            "largest_group": int(counts.max())}


def feature_summary(cache: Path, treat: np.ndarray, grp: dict) -> tuple[pd.DataFrame, dict]:
    rows, hists = [], {}
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    for i, f in enumerate(FEATURES):
        v = np.load(cache / f"{f}.npy", mmap_mode="r")
        v = np.asarray(v)  # one column (~112 MB) in RAM
        q = np.quantile(v, qs)
        uniq, cnt = np.unique(v, return_counts=True)
        mode_i = int(cnt.argmax())
        q1, q3 = q[3], q[5]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n1, n0 = grp[1]["n"], grp[0]["n"]
        m1, m0 = grp[1]["s"][i] / n1, grp[0]["s"][i] / n0
        v1 = (grp[1]["ss"][i] - n1 * m1 ** 2) / (n1 - 1)
        v0 = (grp[0]["ss"][i] - n0 * m0 ** 2) / (n0 - 1)
        sd = v.std(ddof=1)
        rows.append({
            "feature": f, "count": len(v), "mean": v.mean(), "std": sd, "min": v.min(),
            **{f"p{p * 100:g}": val for p, val in zip(qs, q)}, "max": v.max(),
            "n_unique": len(uniq), "mode_value": uniq[mode_i], "mode_share": cnt[mode_i] / len(v),
            "iqr": iqr, "iqr_outlier_share": float(((v < lo) | (v > hi)).mean()) if iqr > 0 else np.nan,
            "share_beyond_5sd": float((np.abs(v - v.mean()) > 5 * sd).mean()),
            "skew": float(stats.skew(v)),
            "mean_treated": m1, "mean_control": m0,
            "smd_treated_vs_control": (m1 - m0) / np.sqrt((v1 + v0) / 2),
        })
        lo_h, hi_h = q[0], q[-1]
        if hi_h <= lo_h:
            lo_h, hi_h = v.min(), v.max()
        edges = np.linspace(lo_h, hi_h, 61)
        clipped = np.clip(v, lo_h, hi_h)
        hists[f] = (edges, np.histogram(clipped[treat == 1], edges)[0], np.histogram(clipped[treat == 0], edges)[0])
        del v, clipped
        print(f"  feature {f} summarised (peak RSS {rss_mb():.0f} MB)", flush=True)
    return pd.DataFrame(rows), hists


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def group_table(joint: dict, n_rows: int) -> pd.DataFrame:
    j = pd.Series(joint)
    j.index = j.index.set_names(BINARY)
    out = []
    for label, mask in (("control (treatment=0)", 0), ("treated (treatment=1)", 1), ("total", None)):
        part = j if mask is None else j.xs(mask, level="treatment", drop_level=False)
        n = int(part.sum())
        rec = {"group": label, "n": n, "share": n / n_rows}
        for y in ("conversion", "visit", "exposure"):
            k = int(part[part.index.get_level_values(y) == 1].sum())
            lo, hi = wilson(k, n)
            rec.update({f"{y}_count": k, f"{y}_rate": k / n, f"{y}_rate_ci_low": lo, f"{y}_rate_ci_high": hi})
        out.append(rec)
    return pd.DataFrame(out)


def sample_plan(n_total: int, n1: int, n0: int, c1: int, c0: int) -> pd.DataFrame:
    p_t, r1, r0 = n1 / n_total, c1 / n1, c0 / n0
    rows = []
    for n in SAMPLE_SIZES + [n_total]:
        e1, e0 = n * p_t, n * (1 - p_t)
        te1, te0 = e1 * TEST_SIZE, e0 * TEST_SIZE
        se_test = np.sqrt(r1 * (1 - r1) / te1 + r0 * (1 - r0) / te0)
        rec = {
            "n_rows": n, "label": "full data" if n == n_total else f"{n:,}",
            "exp_treated": e1, "exp_control": e0,
            "exp_conv_treated": e1 * r1, "exp_conv_control": e0 * r0,
            "exp_train_conv_treated": e1 * r1 * (1 - TEST_SIZE), "exp_train_conv_control": e0 * r0 * (1 - TEST_SIZE),
            "exp_test_conv_treated": te1 * r1, "exp_test_conv_control": te0 * r0,
            "test_ate_se": se_test, "test_ate_over_se": (r1 - r0) / se_test,
            "x_float64_mb": n * 12 * 8 / 1024**2, "x_float32_mb": n * 12 * 4 / 1024**2,
        }
        for k in TOP_FRACTIONS:
            # Expected counts if the top-k% segment had average rates (a random ordering); real segments differ.
            rec[f"top{int(k * 100)}_test_conv_control_if_avg"] = te0 * k * r0
            rec[f"top{int(k * 100)}_test_uplift_se_if_avg"] = np.sqrt(r1 * (1 - r1) / (te1 * k) + r0 * (1 - r0) / (te0 * k))
        rows.append(rec)
    return pd.DataFrame(rows)


def set_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams.update({"font.family": name, "axes.unicode_minus": False, "pdf.fonttype": 42})
            return name
    raise RuntimeError(f"No Korean font found among {KOREAN_FONTS}")


def style(ax, title, xlabel, ylabel, sub=None):
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_rates(groups: pd.DataFrame) -> None:
    g = groups.set_index("group")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (y, label) in zip(axes, [("conversion", "conversion 비율"), ("visit", "visit 비율"), ("exposure", "exposure 비율")]):
        names = ["control (treatment=0)", "treated (treatment=1)"]
        vals = [g.loc[n, f"{y}_rate"] for n in names]
        err = [[g.loc[n, f"{y}_rate"] - g.loc[n, f"{y}_rate_ci_low"] for n in names],
               [g.loc[n, f"{y}_rate_ci_high"] - g.loc[n, f"{y}_rate"] for n in names]]
        ax.bar([0, 1], vals, color=[C_CONTROL, C_TREATED], width=0.55, yerr=err, capsize=4, zorder=3,
               error_kw=dict(ecolor=INK, elinewidth=1))
        for x, v in zip([0, 1], vals):
            ax.annotate(f"{v * 100:.3f}%", (x, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["대조군\n(treatment=0)", "처치군\n(treatment=1)"])
        ax.set_ylim(0, max(vals) * 1.25)
        style(ax, label, "", "비율")
    fig.suptitle("Criteo v2.1: 처치 배정별 conversion · visit · exposure 비율 (전체 13,979,592행)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.90, "오차막대: Wilson 95% 신뢰구간 · exposure는 처치군에서만 발생 가능한 실제 노출 여부(분석 변수 아님)",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"rates_by_treatment.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def plot_features(hists: dict, fs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    for ax, f in zip(axes.ravel(), FEATURES):
        edges, h1, h0 = hists[f]
        centers = (edges[:-1] + edges[1:]) / 2
        w = edges[1] - edges[0]
        ax.step(centers, h1 / h1.sum() / w, where="mid", color=C_TREATED, lw=1.4, label="처치군")
        ax.step(centers, h0 / h0.sum() / w, where="mid", color=C_CONTROL, lw=1.4, ls="--", label="대조군")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        r = fs.set_index("feature").loc[f]
        ax.set_title(f"{f}  (최빈값 비중 {r.mode_share * 100:.0f}%, SMD {r.smd_treated_vs_control:+.3f})", fontsize=9.5,
                     color=INK, loc="left")
        ax.tick_params(labelsize=7.5, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Criteo v2.1: 특성 f0~f11 분포 (처치군 vs 대조군, 밀도·로그 축)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.945, "x축: 0.1%~99.9% 분위수 범위(범위 밖 값은 양 끝 구간에 포함) · SMD: 표준화 평균차(처치 - 대조)",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"feature_distributions.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def plot_row_order(treat: np.ndarray) -> np.ndarray:
    n_blocks = int(np.ceil(len(treat) / BLOCK))
    share = np.array([treat[i * BLOCK:(i + 1) * BLOCK].mean() for i in range(n_blocks)])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.arange(n_blocks) * BLOCK / 1e6, share, color=C_TREATED, lw=1.5)
    ax.axhline(treat.mean(), color=MUTED, ls="--", lw=1)
    ax.text(0.1, treat.mean() - 0.02, f"전체 평균 {treat.mean():.3f}", va="top", fontsize=8.5, color=MUTED)
    ax.set_ylim(-0.03, 1.05)
    style(ax, "원본 파일의 행 순서별 처치군 비율", "행 위치 (백만 행)", "처치군 비율",
          f"{BLOCK:,}행 구간별 treatment 평균 · 파일 앞부분만 잘라 쓰면 무작위 표본이 아님")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"row_order_treatment_share.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    return share


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "criteo" / "eda_cache",
                    help="temporary memmap folder (deleted at the end unless --keep-cache)")
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print("Checking file hash ...", flush=True)
    digest = sha256(args.data)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA256 mismatch: {digest}")
    print("Streaming pass ...", flush=True)
    sc = scan(args.data, args.cache, args.chunksize)
    if sc["n_rows"] != EXPECTED_ROWS:
        raise SystemExit(f"row count {sc['n_rows']} != expected {EXPECTED_ROWS}")

    groups = group_table(sc["joint"], sc["n_rows"])
    fs, hists = feature_summary(args.cache, sc["treat"], sc["grp"])
    dup_row, dup_x = duplicate_stats(sc["hash_row"]), duplicate_stats(sc["hash_x"])
    g = groups.set_index("group")
    c_, t_ = g.loc["control (treatment=0)"], g.loc["treated (treatment=1)"]
    n1, n0, c1, c0 = int(t_.n), int(c_.n), int(t_.conversion_count), int(c_.conversion_count)
    r1, r0 = c1 / n1, c0 / n0
    ate, ate_se = r1 - r0, np.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    j = pd.Series(sc["joint"])
    j.index = j.index.set_names(BINARY)
    conv_wo_visit = int(j[(j.index.get_level_values("conversion") == 1) & (j.index.get_level_values("visit") == 0)].sum())
    exposed_control = int(j[(j.index.get_level_values("treatment") == 0) & (j.index.get_level_values("exposure") == 1)].sum())
    visits = j.groupby(level="visit").sum()
    conv_given_visit = int(j[(j.index.get_level_values("visit") == 1) & (j.index.get_level_values("conversion") == 1)].sum()) / int(visits[1])
    row_share = plot_row_order(sc["treat"])

    file_size = args.data.stat().st_size
    summary = [
        ("n_rows", sc["n_rows"]), ("n_columns", len(COLUMNS)), ("file_size_bytes", file_size), ("sha256", digest),
        ("total_nan", sum(sc["nan"].values())), ("total_inf", sum(sc["inf"].values())),
        ("binary_columns_non_01_values", sum(sc["nonbinary"].values())),
        ("duplicate_full_rows_extra_copies", dup_row["extra_copies"]),
        ("duplicate_full_rows_rows_in_groups", dup_row["rows_in_duplicate_groups"]),
        ("duplicate_full_rows_distinct", dup_row["distinct"]),
        ("duplicate_full_rows_largest_group", dup_row["largest_group"]),
        ("duplicate_features_extra_copies", dup_x["extra_copies"]),
        ("duplicate_features_rows_in_groups", dup_x["rows_in_duplicate_groups"]),
        ("duplicate_features_distinct", dup_x["distinct"]),
        ("duplicate_features_largest_group", dup_x["largest_group"]),
        ("n_treated", n1), ("n_control", n0), ("treatment_ratio", n1 / sc["n_rows"]),
        ("conversion_rate_total", g.loc["total", "conversion_rate"]),
        ("conversion_rate_treated", r1), ("conversion_rate_control", r0),
        ("conversions_treated", c1), ("conversions_control", c0),
        ("ate_itt_conversion_diff_in_means", ate), ("ate_itt_se", ate_se),
        ("ate_itt_ci_low", ate - 1.959964 * ate_se), ("ate_itt_ci_high", ate + 1.959964 * ate_se),
        ("relative_lift_treated_over_control", r1 / r0 - 1),
        ("visit_rate_total", g.loc["total", "visit_rate"]),
        ("visit_rate_treated", t_.visit_rate), ("visit_rate_control", c_.visit_rate),
        ("conversion_rate_given_visit", conv_given_visit), ("conversions_without_visit", conv_wo_visit),
        ("exposure_rate_treated", t_.exposure_rate), ("exposed_in_control", exposed_control),
        ("max_abs_smd_features", float(fs.smd_treated_vs_control.abs().max())),
        ("row_order_block_treatment_share_min", float(row_share.min())),
        ("row_order_block_treatment_share_max", float(row_share.max())),
    ]
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(TABLES / "criteo_eda_summary.csv", index=False)

    groups.to_csv(TABLES / "criteo_treatment_conversion.csv", index=False, float_format="%.8g")
    fs.to_csv(TABLES / "criteo_feature_summary.csv", index=False, float_format="%.8g")
    plan = sample_plan(sc["n_rows"], n1, n0, c1, c0)
    plan.to_csv(TABLES / "criteo_sample_size_plan.csv", index=False, float_format="%.6g")
    plot_rates(groups)
    plot_features(hists, fs)

    joint_records = [dict(zip(BINARY, map(int, k)), count=v) for k, v in sorted(sc["joint"].items())]
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": "STEP 5-1 data acquisition, structure check and EDA (no model training)",
        "file": {"path": str(args.data.relative_to(PROJECT_ROOT)), "size_bytes": file_size, "sha256": digest,
                 "sha256_expected_from_source": EXPECTED_SHA256, "git": "ignored by .gitignore rule data/criteo/*"},
        "source": SOURCE,
        "structure": {"n_rows": sc["n_rows"], "n_rows_expected": EXPECTED_ROWS, "columns": COLUMNS,
                      "raw_dtypes_pandas_inferred": sc["raw_dtypes"], "nan_by_column": sc["nan"],
                      "inf_by_column": sc["inf"], "non_01_values_in_binary_columns": sc["nonbinary"]},
        "research_variables": {"X": FEATURES, "T": "treatment", "Y": "conversion", "estimand": "ITT (campaign assignment)",
                               "excluded_from_X": ["visit", "exposure"],
                               "notes": "exposure is post-assignment and is not used as treatment or feature; no true CATE exists"},
        "joint_counts_treatment_conversion_visit_exposure": joint_records,
        "duplicates": {"method": "pandas.util.hash_pandas_object 64-bit row hashes (collisions possible but negligible)",
                       "full_rows": dup_row, "features_only": dup_x, "action": "none (rows are not removed)"},
        "row_order": {"block_size": BLOCK, "block_treatment_share_min": float(row_share.min()),
                      "block_treatment_share_max": float(row_share.max()),
                      "first_block_treatment_share": float(row_share[0]), "last_block_treatment_share": float(row_share[-1])},
        "processing": {"chunksize": args.chunksize, "peak_rss_mb": rss_mb(), "wall_time_sec": time.perf_counter() - t0},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib"]}},
    }
    (TABLES / "criteo_dataset_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float) + "\n",
                                                         encoding="utf-8")
    if not args.keep_cache:
        shutil.rmtree(args.cache)
    print(f"font: {set_font()}")
    print(f"done in {time.perf_counter() - t0:.0f}s, peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    set_font()
    main()
