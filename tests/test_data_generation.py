import numpy as np
import pandas as pd
import pytest

from src.data_generation import (
    EFFECT_FUNCTIONS,
    SimulationConfig,
    covariate_columns,
    effect_diagnostics,
    generate_data,
    mu0,
    save_dataset,
)

EFFECTS = list(EFFECT_FUNCTIONS)


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[covariate_columns()].to_numpy()


# 1. Row count -----------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 1000, 10000])
def test_row_count_matches_n(n):
    assert len(generate_data(SimulationConfig(n=n))) == n


# 2. Required columns ------------------------------------------------------------
def test_required_columns_exist():
    df = generate_data(SimulationConfig())
    for col in covariate_columns() + ["T", "Y", "true_cate", "mu0"]:
        assert col in df.columns
    assert set(df["T"].unique()) <= {0, 1}


# 3. Treatment share -------------------------------------------------------------
@pytest.mark.parametrize("p", [0.5, 0.3, 0.1])
@pytest.mark.parametrize("n", [1000, 100000])
def test_treated_share_within_binomial_bounds(n, p):
    df = generate_data(SimulationConfig(n=n, treat_prob=p))
    # 4-sigma binomial band: false-failure rate ~6e-5 per check
    tol = 4 * np.sqrt(p * (1 - p) / n)
    assert abs(df["T"].mean() - p) < tol


# 4. Reproducibility -------------------------------------------------------------
@pytest.mark.parametrize("effect", EFFECTS)
def test_same_seed_same_data(effect):
    cfg = SimulationConfig(n=500, treat_prob=0.3, effect=effect, seed=123)
    pd.testing.assert_frame_equal(generate_data(cfg), generate_data(cfg))


def test_different_seed_different_data():
    a = generate_data(SimulationConfig(seed=1))
    b = generate_data(SimulationConfig(seed=2))
    assert not np.allclose(a["Y"], b["Y"])


def test_saved_csv_roundtrip(tmp_path):
    cfg = SimulationConfig(n=200)
    df = generate_data(cfg)
    csv_path, meta_path = save_dataset(df, cfg, tmp_path)
    reloaded = pd.read_csv(csv_path)
    np.testing.assert_allclose(reloaded.to_numpy(dtype=float), df.to_numpy(dtype=float), rtol=1e-9)
    assert meta_path.exists()


# 5. Outcome follows Y = mu0 + T*tau + eps ----------------------------------------
@pytest.mark.parametrize("effect", EFFECTS)
def test_outcome_exact_without_noise(effect):
    df = generate_data(SimulationConfig(n=2000, effect=effect, noise_sd=0.0))
    X = _X(df)
    expected = mu0(X) + df["T"].to_numpy() * EFFECT_FUNCTIONS[effect](X)
    np.testing.assert_allclose(df["Y"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(df["mu0"], mu0(X), rtol=0, atol=1e-12)


@pytest.mark.parametrize("noise_sd", [0.5, 1.0, 2.0])
def test_outcome_residual_is_noise(noise_sd):
    n = 50000
    df = generate_data(SimulationConfig(n=n, noise_sd=noise_sd))
    resid = df["Y"] - df["mu0"] - df["T"] * df["true_cate"]
    assert abs(resid.mean()) < 4 * noise_sd / np.sqrt(n)
    assert resid.std(ddof=1) == pytest.approx(noise_sd, rel=0.02)
    # residual must be independent of treatment assignment and covariates
    assert abs(np.corrcoef(resid, df["T"])[0, 1]) < 0.02
    assert abs(np.corrcoef(resid, df["X1"])[0, 1]) < 0.02


# 6. true_cate matches the chosen effect function ----------------------------------
@pytest.mark.parametrize("effect", EFFECTS)
def test_true_cate_matches_effect_function(effect):
    df = generate_data(SimulationConfig(n=2000, effect=effect))
    np.testing.assert_allclose(df["true_cate"], EFFECT_FUNCTIONS[effect](_X(df)), rtol=0, atol=1e-12)


def test_linear_effect_formula():
    df = generate_data(SimulationConfig(effect="linear"))
    np.testing.assert_allclose(df["true_cate"], 0.5 + 0.3 * df["X1"], atol=1e-12)


def test_effects_share_scale_but_differ_in_shape():
    X = np.random.default_rng(0).standard_normal((200000, 10))
    diag = effect_diagnostics(X)
    for effect in EFFECTS:
        assert diag.loc[effect, "mean"] == pytest.approx(0.5, abs=0.005)
        assert diag.loc[effect, "sd"] == pytest.approx(0.3, abs=0.005)
    assert diag.loc["linear", "linear_r2"] == pytest.approx(1.0)
    assert diag.loc["nonlinear", "linear_r2"] < 0.05
    assert diag.loc["interaction", "linear_r2"] < 0.01


# 7. No missing / infinite values -------------------------------------------------
@pytest.mark.parametrize("effect", EFFECTS)
@pytest.mark.parametrize("p", [0.5, 0.1])
def test_no_missing_or_infinite(effect, p):
    df = generate_data(SimulationConfig(n=5000, treat_prob=p, effect=effect))
    values = df.to_numpy(dtype=float)
    assert not np.isnan(values).any()
    assert np.isfinite(values).all()


# Config validation ---------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [{"n": 0}, {"treat_prob": 0.0}, {"treat_prob": 1.0}, {"effect": "cubic"}, {"noise_sd": -1.0}],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)
