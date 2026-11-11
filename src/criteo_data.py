"""Criteo Uplift v2.1: research variables, reproducible stratified splits and chunked row loading.

X = f0..f11, T = treatment (campaign assignment, ITT), Y = conversion. visit and exposure are
post-assignment variables and are never returned as model inputs.

Splits are drawn per stratum (treatment x conversion, 4 cells) from one random permutation of
each stratum, in this order: final test, validation, then the remaining training pool in random
order. Training candidates of increasing size are prefixes of that order, so they are nested
(100k in 300k in 1M) and each keeps the full-data stratum proportions (largest-remainder
allocation). The test set is fixed first and does not depend on any later choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
SPLIT_DIR = PROJECT_ROOT / "data" / "criteo" / "splits"
DATA_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
N_ROWS = 13_979_592

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT, OUTCOME = "treatment", "conversion"
POST_ASSIGNMENT = ("visit", "exposure")   # descriptive only; never X, never T


def strata(treatment, conversion) -> np.ndarray:
    """Stratum code 0..3 = 2 * treatment + conversion."""
    return (2 * np.asarray(treatment, dtype=np.int8) + np.asarray(conversion, dtype=np.int8)).astype(np.int8)


def largest_remainder(total: int, shares: np.ndarray) -> np.ndarray:
    """Integer counts summing to `total`, proportional to `shares` (Hamilton method)."""
    raw = total * np.asarray(shares, dtype=float) / np.sum(shares)
    counts = np.floor(raw).astype(np.int64)
    short = total - counts.sum()
    counts[np.argsort(-(raw - counts), kind="stable")[:short]] += 1
    return counts


def make_splits(stratum: np.ndarray, test_frac: float, valid_frac: float, train_sizes: list[int],
                seed: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return {name: sorted row indices} for test, valid, train_pool and train_<size>, plus allocations."""
    stratum = np.asarray(stratum)
    n = len(stratum)
    cells = np.arange(4)
    pop = np.array([(stratum == s).sum() for s in cells])
    alloc = {"test": largest_remainder(round(test_frac * n), pop),
             "valid": largest_remainder(round(valid_frac * n), pop)}
    for size in train_sizes:
        alloc[f"train_{size}"] = largest_remainder(size, pop)
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in ["test", "valid", "train_pool"] + [f"train_{s}" for s in train_sizes]}
    for s in cells:
        perm = rng.permutation(np.flatnonzero(stratum == s))
        a_test, a_valid = alloc["test"][s], alloc["valid"][s]
        parts["test"].append(perm[:a_test])
        parts["valid"].append(perm[a_test:a_test + a_valid])
        pool = perm[a_test + a_valid:]          # random order -> prefixes are random samples
        parts["train_pool"].append(pool)
        for size in train_sizes:
            k = alloc[f"train_{size}"][s]
            if k > len(pool):
                raise ValueError(f"train_{size} needs {k} rows of stratum {s}, pool has {len(pool)}")
            parts[f"train_{size}"].append(pool[:k])
    splits = {name: np.sort(np.concatenate(p)) for name, p in parts.items()}
    table = pd.DataFrame({"stratum": cells, "treatment": cells // 2, "conversion": cells % 2, "population": pop,
                          **{f"alloc_{k}": v for k, v in alloc.items()}})
    return splits, table


def check_splits(splits: dict[str, np.ndarray], n_total: int, train_sizes: list[int]) -> dict:
    """Index integrity: no duplicates, disjoint test/valid/train_pool covering all rows, nesting."""
    out = {"no_duplicate_indices": {k: bool(len(np.unique(v)) == len(v)) for k, v in splits.items()},
           "in_range": {k: bool(len(v) == 0 or (v.min() >= 0 and v.max() < n_total)) for k, v in splits.items()}}
    disjoint = {}
    names = ["test", "valid", "train_pool"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            disjoint[f"{a}&{b}"] = int(len(np.intersect1d(splits[a], splits[b], assume_unique=True)))
    for size in train_sizes:
        for other in ("test", "valid"):
            disjoint[f"train_{size}&{other}"] = int(len(np.intersect1d(splits[f"train_{size}"], splits[other],
                                                                       assume_unique=True)))
    out["intersection_sizes"] = disjoint
    union = np.concatenate([splits[k] for k in names])
    out["test_valid_pool_partition_all_rows"] = bool(len(union) == n_total and len(np.unique(union)) == n_total)
    nested = {}
    chain = [f"train_{s}" for s in sorted(train_sizes)] + ["train_pool"]
    for small, big in zip(chain[:-1], chain[1:]):
        nested[f"{small}⊂{big}"] = bool(np.isin(splits[small], splits[big], assume_unique=True).all())
    out["nested"] = nested
    out["all_ok"] = (all(out["no_duplicate_indices"].values()) and all(out["in_range"].values())
                     and not any(disjoint.values()) and out["test_valid_pool_partition_all_rows"]
                     and all(nested.values()))
    return out


def read_rows(indices: np.ndarray, path: Path = DATA_PATH, columns: list[str] | None = None,
              chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read only the given original row numbers (0-based, data rows) without loading the whole file.

    Returns rows in ascending original order with a `row_id` column; features as float32, binary
    columns as int8.
    """
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    columns = columns or FEATURES + [TREATMENT, OUTCOME]
    dtypes = {c: (np.float32 if c in FEATURES else np.int8) for c in columns}
    frames, start = [], 0
    for chunk in pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunksize):
        lo, hi = np.searchsorted(indices, [start, start + len(chunk)])
        if hi > lo:
            sel = chunk.iloc[indices[lo:hi] - start]
            frames.append(sel.assign(row_id=indices[lo:hi]))
        start += len(chunk)
        if hi == len(indices):
            break
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns + ["row_id"])
    if len(out) != len(indices):
        raise ValueError(f"requested {len(indices)} rows, read {len(out)}")
    return out[["row_id"] + columns]


def load_split(name: str, split_dir: Path = SPLIT_DIR, **kwargs) -> pd.DataFrame:
    """Load X, T, Y (and nothing post-assignment) for a saved split, e.g. 'train_100000' or 'test'."""
    return read_rows(np.load(split_dir / f"{name}.npy"), **kwargs)
