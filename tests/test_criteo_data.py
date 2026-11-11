import numpy as np
import pandas as pd
import pytest

from src.criteo_data import (
    FEATURES,
    check_splits,
    largest_remainder,
    make_splits,
    read_rows,
    strata,
)

SIZES = [1000, 3000, 10000]


@pytest.fixture(scope="module")
def stratum():
    rng = np.random.default_rng(0)
    n = 20000
    t = (rng.random(n) < 0.85).astype(int)
    y = (rng.random(n) < 0.01).astype(int)
    return strata(t, y)


def test_largest_remainder_sums_and_is_proportional():
    shares = np.array([0.1499, 0.0003, 0.8475, 0.0023])
    counts = largest_remainder(100_000, shares)
    assert counts.sum() == 100_000
    assert np.abs(counts - 100_000 * shares / shares.sum()).max() < 1


def test_splits_are_disjoint_nested_and_cover_all_rows(stratum):
    splits, alloc = make_splits(stratum, 0.2, 0.2, SIZES, seed=42)
    checks = check_splits(splits, len(stratum), SIZES)
    assert checks["all_ok"], checks
    assert len(splits["test"]) == 4000 and len(splits["valid"]) == 4000
    for size in SIZES:
        assert len(splits[f"train_{size}"]) == size


def test_each_split_keeps_stratum_proportions(stratum):
    splits, alloc = make_splits(stratum, 0.2, 0.2, SIZES, seed=42)
    pop = np.bincount(stratum, minlength=4)
    for name in ["test", "valid"] + [f"train_{s}" for s in SIZES]:
        got = np.bincount(stratum[splits[name]], minlength=4)
        np.testing.assert_array_equal(got, alloc[f"alloc_{name}"])
        assert np.abs(got - len(splits[name]) * pop / pop.sum()).max() < 1


def test_test_set_is_fixed_first_and_reproducible(stratum):
    a, _ = make_splits(stratum, 0.2, 0.2, SIZES, seed=42)
    b, _ = make_splits(stratum, 0.2, 0.2, [500], seed=42)
    c, _ = make_splits(stratum, 0.2, 0.2, SIZES, seed=43)
    np.testing.assert_array_equal(a["test"], b["test"])
    np.testing.assert_array_equal(a["train_1000"], make_splits(stratum, 0.2, 0.2, SIZES, seed=42)[0]["train_1000"])
    assert not np.array_equal(a["test"], c["test"])


def test_too_large_training_sample_is_refused(stratum):
    with pytest.raises(ValueError):
        make_splits(stratum, 0.2, 0.2, [15000], seed=42)


def test_read_rows_selects_exact_rows_across_chunks(tmp_path):
    rng = np.random.default_rng(5)
    n = 53
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=FEATURES)
    for c in ("treatment", "conversion", "visit", "exposure"):
        df[c] = rng.integers(0, 2, n)
    path = tmp_path / "mini.csv.gz"
    df.to_csv(path, index=False)
    idx = np.array([50, 0, 7, 8, 21, 52])
    out = read_rows(idx, path=path, chunksize=7)
    np.testing.assert_array_equal(out["row_id"], np.sort(idx))
    np.testing.assert_allclose(out[FEATURES].to_numpy(), df.iloc[np.sort(idx)][FEATURES].to_numpy(), rtol=1e-6)
    np.testing.assert_array_equal(out["treatment"], df.iloc[np.sort(idx)]["treatment"])
    assert "visit" not in out and "exposure" not in out
    assert out[FEATURES].dtypes.eq(np.float32).all()
